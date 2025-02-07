import os

import casadi as ca
import numpy as np
import toml
from scipy.spatial.transform import Rotation as Rot

from lsy_drone_racing.mpc_utils.planners import HermiteSpline
from lsy_drone_racing.mpc_utils.utils import (
    W1,
    quaternion_product,
    quaternion_rotation,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    W2,
)

from .drone_dynamics import DroneDynamics


class MPCCppDynamics(DroneDynamics):
    def __init__(self, initial_obs, initial_info, dynamics_info, constraints_info, cost_info):
        super().__init__(
            initial_obs,
            initial_info,
            dynamics_info=dynamics_info,
            constraints_info=constraints_info,
            cost_info=cost_info,
        )
        self.Wn = self.constraints_info.get("Wn", 0.3)
        self.Wgate = self.constraints_info.get("Wgate", 0.1)
        self.baseDynamics = "MPCC"
        self.controlType = "Thrusts"
        self.useControlRates = True
        self.last_theta = 0

        # Setup the Dynamics, returns expressions for the continuous dynamics
        self.setup_dynamics()
        # Defines the bounds and scaling factors for the states and controls, and which states/controls have slack variables
        self.setupBoundsAndScals()
        # Init the path Planner
        self.pathPlanner = HermiteSpline(
            initial_obs["pos"],
            initial_obs["rpy"],
            initial_obs["gates_pos"],
            initial_obs["gates_rpy"],
        )
        # Setup nonlinear constraints
        self.setupNLConstraints()
        # Setup the cost function
        self.setupMPCCCosts()
        # Last step
        super().setupCasadiFunctions()
        if self.ukf_info["useUKF"]:
            self.initUKF()

    def setup_dynamics(self):
        self.modelName = "MPCC"
        # States
        pos = ca.MX.sym("pos", 3)  # position in world frame
        vel = ca.MX.sym("vel", 3)  # velocity in world frame
        quat = ca.MX.sym("quat", 4)  # [qx, qy, qz,qw] quaternion rotation from body to world
        w = ca.MX.sym("w", 3)  # angular velocity in body frame
        f = ca.MX.sym("f", 4)  # individual rotor thrusts, actual control
        progress = ca.MX.sym("progress", 1)  # progress along the path
        dprogress = ca.MX.sym("dprogress", 1)  # progress rate
        x = ca.vertcat(pos, vel, quat, w, f, progress, dprogress)  # state vector

        self.state_indices = {
            "pos": np.arange(0, 3),
            "vel": np.arange(3, 6),
            "quat": np.arange(6, 10),
            "w": np.arange(10, 13),
            "f": np.arange(13, 17),
            "progress": np.arange(17, 18),
            "dprogress": np.arange(18, 19),
        }

        # Controls
        df = ca.MX.sym("df", 4)  # individual rotor thrust rates, virtual control
        ddprogress = ca.MX.sym("ddprogress", 1)  # progress rate, virtual control
        u = ca.vertcat(df, ddprogress)  # control vector
        self.control_indices = {"df": np.arange(0, 4), "ddprogress": np.arange(4, 5)}

        torques = self.thrustsToTorques_sym(f)
        thrust_vec = ca.vertcat(0, 0, (f[0] + f[1] + f[2] + f[3]) / self.mass)

        # rotation matrix for quaternions from body to world frame
        Rquat = quaternion_to_rotation_matrix(quat)

        # Define the dynamics
        dpos = vel
        if self.useDrags:
            dvel = self.gv + Rquat @ thrust_vec - ca.mtimes([Rquat, self.DragMat, Rquat.T, vel])
        else:
            dvel = self.gv + Rquat @ thrust_vec
        dquat = 0.5 * quaternion_product(quat, ca.vertcat(w, 0))
        dw = self.J_inv @ (torques - ca.cross(w, self.J @ w))

        df = df
        dprogress = dprogress
        ddprogress = ddprogress

        dx = ca.vertcat(dpos, dvel, dquat, dw, df, dprogress, ddprogress)
        self.x = x
        self.nx = x.size()[0]
        self.dx = dx
        self.u = u
        self.nu = u.size()[0]
        self.ny = self.nx + self.nu
        # Equilibrium state and control
        f_eq = 0.25 * self.mass * self.g * np.ones((4,))
        self.x_eq = np.concatenate([np.zeros((13,)), f_eq, np.zeros((2,))])  # Equilibrium state
        self.u_eq = np.zeros((5,))  # Equilibrium control

    def transformState(self, x):
        # Extract the position, velocity, euler angles, and angular velocities
        if self.last_u is None:
            self.last_u = np.zeros((4,))
        if self.ukf_info["useUKF"]:
            # x1 = x
            x = self.runUKF(z=x, u=self.last_u)
        pos = x[:3]
        vel = x[3:6]
        rpy = x[6:9]
        drpy = x[9:12]
        w = W1(rpy) @ drpy
        quat = Rot.from_euler("xyz", rpy).as_quat()
        progress, dprogress = self.pathPlanner.computeProgress(pos, vel, self.last_theta)
        self.last_theta = progress
        x = np.concatenate(
            [pos, vel, quat, w, self.last_u[self.control_indices["df"]], [progress], [dprogress]]
        )
        # Predict the state into the future if self.usePredict is True
        if self.usePredict and self.last_u is not None:
            # fd_predict is a discrete dynamics function (RK4) with the time step t_predict
            x = self.fd_predict(x, self.last_u)
        return x

    def transformAction(self, x_sol: np.ndarray, u_sol: np.ndarray) -> np.ndarray:
        """Transforms optimizer solutions to controller inferfaces (Mellinger or Thrust)."""
        self.last_u = x_sol[self.state_indices["f"], 1]

        if self.interface == "Thrust":
            thrusts = x_sol[self.state_indices["f"], 1]
            tot_thrust = np.sum(thrusts)
            quat = x_sol[self.state_indices["quat"], 1]
            rpy = Rot.from_quat(quat).as_euler("xyz")

            action = np.concatenate([[tot_thrust], rpy])
        elif self.interface == "Mellinger":
            action = x_sol[:, 1]
            pos = action[self.state_indices["pos"]]
            vel = action[self.state_indices["vel"]]
            w = action[self.state_indices["w"]]
            quat = action[self.state_indices["quat"]]
            yaw = Rot.from_quat(quat).as_euler("xyz")[-1]

            acc_world = (vel - x_sol[self.state_indices["vel"], 0]) / self.ts
            drpy = W2(rpy) @ w
            action = np.concatenate([pos, vel, acc_world, [yaw], w])
        return action.flatten()

    def setupNLConstraints(self):
        """Setup the nonlinear constraints for the drone/environment controller."""
        super().setupQuatConstraints()
        super().setupObstacleConstraints()

        self.setupTunnelConstraints()

        self.updateParameters(obs=self.initial_obs, init=True)

    def updateParameters(self, obs: dict = None, init: bool = False) -> np.ndarray:
        """Update the parameters of the drone/environment controller."""
        # Checks whether gate observation has been updated, replans if needed, and updates the path, dpath, and gate progresses parameters
        updated = False
        if init and self.p is not None:
            self.param_values = np.zeros((self.p.size()[0],))
            self.param_values[self.param_indices["obstacles_pos"]] = self.obstacle_pos[
                :, :-1
            ].flatten()
            self.param_values[self.param_indices["gate_progresses"]] = (
                self.pathPlanner.theta_switch[1:]
            )
        elif self.p is not None:
            if np.any(np.not_equal(self.obstacles_visited, obs["obstacles_visited"])):
                self.obstacles_visited = obs["obstacles_visited"]
                self.obstacles_pos = obs["obstacles_pos"]
                self.param_values[self.param_indices["obstacles_pos"]] = self.obstacle_pos[
                    :, :-1
                ].flatten()
                updated = True
            if np.any(np.not_equal(self.gates_visited, obs["gates_visited"])):
                self.gates_visited = obs["gates_visited"]
                self.gates_pos = obs["gates_pos"]
                self.gates_rpy = obs["gates_rpy"]
                self.pathPlanner.updateGates(self.gates_pos, self.gates_rpy)
                self.param_values[self.param_indices["gate_progresses"]] = (
                    self.pathPlanner.theta_switch[1:]
                )
                updated = True
        return updated

    def setupTunnelConstraints(self):
        """Setup the tunnel constraints for the drone/environment controller."""
        # Progress along the path state
        progress = self.x[self.state_indices["progress"]]
        pos = self.x[self.state_indices["pos"]]
        # Parameter (all symbolic variables)
        gate_progresses = ca.MX.sym(
            "gate_progresses", len(self.gates_pos)
        )  # self.p[self.param_indices["gate_progresses"]]
        self.addToParameters(gate_progresses, "gate_progresses")
        # Nominal tunnel width = tunnel height
        Wn = self.Wn
        # Tunnel width = tunnel height at the gate
        Wgate = self.Wgate

        def getTunnelWidth(gate_progresses: ca.MX, progress: ca.MX) -> ca.MX:
            """Calculate the tunnel width at the current progress."""
            # Calculate the progress distance to the nearest gate
            d = ca.mmin(ca.fabs(gate_progresses - progress))
            k = 10  # Steepness of the transition
            x0 = 0.1  # Midpoint of the transition
            sigmoid = 1 / (1 + ca.exp(-k * (d - x0)))
            return Wn + (Wgate - Wn) * sigmoid

        W = getTunnelWidth(gate_progresses, progress)
        H = W  # Assuming W(θk) = H(θk)

        # Symbolic functions for the path and its derivative
        path = self.pathPlanner.path_function(
            progress
        )  # , start_pos, start_rpy, gates_pos, gates_rpy)
        dpath = self.pathPlanner.dpath_function(
            progress
        )  # , start_pos, start_rpy, gates_pos, gates_rpy)

        t = dpath / ca.norm_2(dpath)  # Normalized Tangent vector at the current progress
        # Compute the normal vector n (assuming the normal is in the xy-plane)
        n = ca.vertcat(-t[1], t[0], 0)
        # Compute the binormal vector b
        b = ca.cross(t, n)

        pd = path  # Position of the path at the current progress
        p0 = pd - W * n - H * b
        # Tunnel constraints
        tunnel_constraints = []
        tunnel_constraints.append((pos - p0).T @ n)
        tunnel_constraints.append(2 * H - (pos - p0).T @ n)
        tunnel_constraints.append((pos - p0).T @ b)
        tunnel_constraints.append(2 * W - (pos - p0).T @ b)

        # Add the tunnel constraints to the the constraints
        tunnel_constraints = ca.vertcat(*tunnel_constraints)
        tunnel_lh = np.zeros((4,))
        tunnel_uh = np.ones((4,)) * 1e9

        self.addToConstraints(tunnel_constraints, tunnel_lh, tunnel_uh, "tunnel")

    def setupBoundsAndScals(self):
        x_lb = np.concatenate(
            [self.pos_lb, self.vel_lb, self.quat_lb, self.w_lb, self.thrust_lb, [0, -0.1]]
        )
        x_ub = np.concatenate(
            [self.pos_ub, self.vel_ub, self.quat_ub, self.w_ub, self.thrust_ub, [1, 1]]
        )
        u_lb = np.concatenate([self.thrust_rate_lb, [-1]])
        u_ub = np.concatenate([self.thrust_rate_lb, [1]])

        self.slackStates = np.concatenate(
            [
                self.state_indices["pos"],
                self.state_indices["vel"],
                self.state_indices["w"],
                self.state_indices["progress"],
                self.state_indices["dprogress"],
            ]
        )

        self.slackControls = np.concatenate(
            [self.control_indices["df"], self.control_indices["ddprogress"]]
        )

        self.x_lb = x_lb
        self.x_ub = x_ub
        self.x_scal = self.x_ub - self.x_lb
        # Set x_scal values that are zero or close to zero to 0.1
        if any(x_ub - x_lb < 0):
            Warning("Some states have upper bounds lower than lower bounds")
        if any(self.x_scal < 1e-4):
            Warning("Some states have scales close to zero, setting them to 0.1")
            self.x_scal = np.where(np.abs(self.x_scal) < 1e-4, 0.1, self.x_scal)

        self.u_lb = u_lb
        self.u_ub = u_ub
        self.u_scal = self.u_ub - self.u_lb

    def setupMPCCCosts(self):
        """Setup the cost function for the MPCCpp controller.

        We are using the EXTERNAL interface of acados to define the cost function.
        The cost function has 6 components:
        1. Lag error: The error between the current position and the desired position
        2. Contour error: The error between the current position and the desired contour
        3. Body angular velocity: The angular velocity of the body
        5. Thrust rate: The rate of change of the thrust
        4. Progress rate: The L2 norm rate of progress along the path
        6. Progress rate: The negative L1 rate of progress along the path.
        """

        self.cost_type = "external"
        # Lag error weights
        Ql = self.cost_info.get("Ql", 1)
        self.Ql = ca.diag([Ql, Ql, Ql])
        # Contour error weights
        Qc = self.cost_info.get("Qc", 1)
        self.Qc = ca.diag([Qc, Qc, Qc])
        # Body Angular velocity weights
        Qw = self.cost_info.get("Qw", 1)
        self.Qw = ca.diag([Qw, Qw, Qw])
        # Progress rate weights
        self.Qmu = self.cost_info.get("Qmu", 1)
        # Thrust rate weights
        Rdf = self.cost_info.get("Rdf", 1)
        self.Rdf = ca.diag([Rdf, Rdf, Rdf, Rdf])
        # Progress rate weights
        self.Rdprogress = self.cost_info.get("Rdprogress", 1)
        # Define the cost function
        self.stageCostFunc = self.MPCC_stage_cost
        self.terminalCostFunc = self.MPCC_stage_cost  # use zero for u

    def MPCC_stage_cost(self, x, u, p, x_ref=None, u_ref=None):
        pos = x[self.state_indices["pos"]]
        w = x[self.state_indices["w"]]
        progress = x[self.state_indices["progress"]]
        dprogress = x[self.state_indices["dprogress"]]
        df = u[self.control_indices["df"]]

        # Desired position and tangent vector on the path
        path = self.pathPlanner.path_function  # Unpack the path function
        dpath = self.pathPlanner.dpath_function  # Unpack the path gradient function
        pd = path(progress)  # Desired position on the path
        tangent_line = dpath(progress)  # Tangent vector of the path at the current progress
        tangent_line = tangent_line / ca.norm_2(tangent_line)  # Normalize the tangent vector
        pos_err = pos - pd  # Error between the current position and the desired position

        # Lag error
        lag_err = ca.mtimes([ca.dot(pos_err, tangent_line), tangent_line])
        lag_cost = ca.mtimes([lag_err.T, self.Ql, lag_err])

        # Contour error
        contour_err = pos_err - lag_err
        # contour_err = pos_err - ca.mtimes([lag_err, tangent_line])
        contour_cost = contour_err.T @ self.Qc @ contour_err

        # Body angular velocity cost
        w_cost = w.T @ self.Qw @ w

        # Progress rate cost
        dprogress_cost_L2 = dprogress.T @ self.Rdprogress @ dprogress

        # Negative progress rate cost
        dprogress_cost = -dprogress * self.Qmu

        # Thrust rate cost
        thrust_rate_cost = df.T @ self.Rdf @ df  # ca.mtimes([df.T, self.Rdf, df])

        # Total stage cost
        stage_cost = (
            lag_cost + contour_cost + w_cost + dprogress_cost + dprogress_cost_L2 + thrust_rate_cost
        )

        # Debugging statements
        nan_elements = ca.fabs(stage_cost - stage_cost) > 0

        if ca.sum1(nan_elements) > 0:
            raise ValueError("NaN detected in stage cost computation.")

        return stage_cost
