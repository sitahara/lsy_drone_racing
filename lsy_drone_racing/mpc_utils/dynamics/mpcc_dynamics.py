import os

import casadi as ca
import numpy as np
import toml
from scipy.spatial.transform import Rotation as Rot

from lsy_drone_racing.mpc_utils.planners import HermiteSplinePathPlanner, PathPlanner
from lsy_drone_racing.mpc_utils.utils import (
    W1,
    Rbi,
    W1s,
    W2s,
    dW2s,
    quaternion_conjugate,
    quaternion_product,
    quaternion_rotation,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rungeKuttaExpr,
    rungeKuttaFcn,
    shuffleQuat,
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

        # Setup the Dynamics, returns expressions for the continuous dynamics
        self.setup_dynamics()
        # Defines the bounds and scaling factors for the states and controls, and which states/controls have slack variables
        self.setupBoundsAndScals()
        # Init the path Planner
        self.pathPlanner = HermiteSplinePathPlanner(
            self.p,  # Complete parameter vector
            self.param_indices,  # Indices of the parameters
            self.current_param_index,  # Current index of the parameters
            initial_obs["gates_pos"],
            initial_obs["gates_rpy"],
            initial_obs["pos"],
            initial_obs["rpy"],
            self.x[self.state_indices["progress"]],  # Pass the progress variable
        )
        self.pathPlanner.testPath()
        raise Exception("Test")
        self.p = self.pathPlanner.p
        self.param_indices = self.pathPlanner.param_indices
        self.current_param_index = self.pathPlanner.current_param_index
        # Setup nonlinear constraints
        self.setupNLConstraints()
        # Setup the cost function
        self.setupMPCCCosts()
        # Last step
        super().setupCasadiFunctions()

    def setup_dynamics(self):
        self.modelName = "MPCCpp"
        # States
        pos = ca.MX.sym("pos", 3)  # position in world frame
        vel = ca.MX.sym("vel", 3)  # velocity in world frame
        quat = ca.MX.sym("quat", 4)  # [qx, qy, qz,qw] quaternion rotation from body to world
        w = ca.MX.sym("w", 3)  # angular velocity in body frame
        f = ca.MX.sym("f", 4)  # individual rotor thrusts
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
        df = ca.MX.sym("df", 4)  # individual rotor thrust rates
        ddprogress = ca.MX.sym("ddprogress", 1)  # progress rate, virtual control
        u = ca.vertcat(df, ddprogress)  # control vector
        self.control_indices = {"df": np.arange(0, 4), "ddprogress": np.arange(4, 5)}

        # Helper variables
        beta = self.arm_length / ca.sqrt(2.0)
        # Motor Thrusts to torques
        torques = ca.vertcat(
            beta * (f[0] + f[1] - f[2] - f[3]),
            beta * (-f[0] + f[1] + f[2] - f[3]),
            self.gamma * (f[0] - f[1] + f[2] - f[3]),
        )  # tau_x, tau_y, tau_z
        # total thrust
        thrust_total = ca.vertcat(0, 0, (f[0] + f[1] + f[2] + f[3]) / self.mass)
        # rotation matrix for quaternions from body to world frame
        Rquat = quaternion_to_rotation_matrix(quat)

        # Define the dynamics
        d_pos = vel
        if self.useDrags:
            d_vel = (
                self.gv
                + quaternion_rotation(quat, thrust_total)
                - ca.mtimes([Rquat, self.DragMat, Rquat.T, vel])
            )
        else:
            d_vel = self.gv + quaternion_rotation(quat, thrust_total)

        d_quat = 0.5 * quaternion_product(quat, ca.vertcat(w, 0))
        d_w = self.J_inv @ (torques - (ca.skew(w) @ self.J @ w))
        d_f = df
        d_progress = dprogress
        d_dprogress = ddprogress

        dx = ca.vertcat(d_pos, d_vel, d_quat, d_w, d_f, d_progress, d_dprogress)
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
        pos = x[:3]
        vel = x[3:6]
        eul_ang = x[6:9]
        deul_ang = x[9:12]
        progress, dprogress = self.pathPlanner.computeProgress(pos, vel)

        # Convert to used states
        w = W1(eul_ang) @ deul_ang
        quat = Rot.from_euler("xyz", eul_ang).as_quat()

        if self.last_u is None:
            self.last_u = np.zeros((5,))

        x = np.concatenate(
            [pos, vel, quat, w, self.last_u[self.control_indices["df"]], [progress], dprogress]
        )
        # Predict the state into the future if self.usePredict is True
        if self.usePredict and self.last_u is not None:
            # fd_predict is a discrete dynamics function (RK4) with the time step t_predict
            x = self.fd_predict(x, self.last_u)
        return x

    def transformAction(self, x_sol: np.ndarray, u_sol: np.ndarray) -> np.ndarray:
        """Transforms optimizer solutions to controller inferfaces (Mellinger or Thrust)."""
        self.last_u = x_sol[
            np.concatenate([self.state_indices["f"], self.state_indices["dprogress"]])
        ]

        if self.interface == "Thrust":
            thrusts = self.last_u[self.control_indices["df"]]
            torques = np.array(
                [
                    self.beta * (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3]),
                    self.beta * (-thrusts[0] + thrusts[1] + thrusts[2] - thrusts[3]),
                    self.gamma * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3]),
                ]
            )
            tot_thrust = np.sum(thrusts)
            action = np.concatenate([[tot_thrust], torques])
        elif self.interface == "Mellinger":
            action = x_sol[:, 1]
            pos = action[self.state_indices["pos"]]
            vel = action[self.state_indices["vel"]]
            w = action[self.state_indices["w"]]
            quat = action[self.state_indices["quat"]]
            yaw = Rot.from_quat(quat).as_euler("xyz")[2]

            acc_world = (vel - x_sol[:, 0][self.state_indices["vel"]]) / self.ts
            yaw = action[8]
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
        if init:
            self.param_values = np.zeros((self.p.size()[0],))
            self.param_values[self.param_indices["obstacles_pos"]] = self.obstacle_pos[
                :, :-1
            ].flatten()
            self.param_values[self.param_indices["gate_progresses"]] = (
                self.pathPlanner.gate_progresses
            )
            self.param_values[self.param_indices["gates_pos"]] = self.gates_pos.flatten()
            self.param_values[self.param_indices["gates_rpy"]] = self.gates_rpy.flatten()
            self.param_values[self.param_indices["start_pos"]] = self.initial_obs["pos"]
            self.param_values[self.param_indices["start_rpy"]] = self.initial_obs["rpy"]

        else:
            if np.any(np.not_equal(self.gates_visited, obs["gates_visited"])):
                self.gates_visited = obs["gates_visited"]
                self.gates_pos = obs["gates_pos"]
                self.gates_rpy = obs["gates_rpy"]
                self.param_values[self.param_indices["gates_pos"]] = self.gates_pos.flatten()
                self.param_values[self.param_indices["gates_rpy"]] = self.gates_rpy.flatten()
                self.pathPlanner.update_gates(self.gates_pos, self.gates_rpy)
                self.param_values[self.param_indices["gate_progresses"]] = (
                    self.pathPlanner.gate_progresses
                )
                updated = True
            # Checks whether obstacle observation has been updated, updates the obstacle positions
            if np.any(np.not_equal(self.obstacles_visited, obs["obstacles_visited"])):
                self.obstacles_visited = obs["obstacles_visited"]
                self.obstacles_pos = obs["obstacles_pos"]
                self.param_values[self.param_indices["obstacles_pos"]] = self.obstacle_pos[
                    :, :-1
                ].flatten()
                updated = True
            # Return the updated parameter values for the acados interface
        return updated

    def setupTunnelConstraints(self):
        """Setup the tunnel constraints for the drone/environment controller."""
        # Progress along the path state
        progress = self.x[self.state_indices["progress"]]
        pos = self.x[self.state_indices["pos"]]
        # Parameter (all symbolic variables)
        gate_progresses = self.p[self.param_indices["gate_progresses"]]
        gates_pos = self.p[self.param_indices["gates_pos"]].reshape((self.pathPlanner.num_gates, 3))
        gates_rpy = self.p[self.param_indices["gates_rpy"]].reshape((self.pathPlanner.num_gates, 3))
        start_pos = self.p[self.param_indices["start_pos"]]
        start_rpy = self.p[self.param_indices["start_rpy"]]
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
        path = self.pathPlanner.path_func(progress, start_pos, start_rpy, gates_pos, gates_rpy)
        dpath = self.pathPlanner.dpath_func(progress, start_pos, start_rpy, gates_pos, gates_rpy)

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
        if self.nl_constr is None:
            self.nl_constr = ca.vertcat(*tunnel_constraints)
            self.nl_constr_lh = np.zeros((4,))
            self.nl_constr_uh = np.ones((4,)) * 1e9
        else:
            self.nl_constr = ca.vertcat(self.nl_constr, *tunnel_constraints)
            self.nl_constr_lh = np.concatenate([self.nl_constr_lh, np.zeros((4,))])
            self.nl_constr_uh = np.concatenate([self.nl_constr_uh, np.ones((4,)) * 1e9])

        self.nl_constr_indices["tunnel"] = np.arange(
            self.current_nl_constr_index, len(tunnel_constraints) + self.current_nl_constr_index
        )
        self.current_nl_constr_index += len(tunnel_constraints)

    def setupBoundsAndScals(self):
        x_lb = np.concatenate(
            [self.pos_lb, self.vel_lb, self.quat_lb, self.w_lb, self.thrust_lb, [0, 0]]
        )
        x_ub = np.concatenate(
            [self.pos_ub, self.vel_ub, self.quat_ub, self.w_ub, self.thrust_ub, [1, 1]]
        )
        u_lb = np.concatenate([self.thrust_rate_lb, [-1]])
        u_ub = np.concatenate([self.thrust_rate_lb, [1]])

        self.slackStates = np.concatenate(
            [
                self.state_indices["pos"],
                self.state_indices["progress"],
                self.state_indices["dprogress"],
            ]
        )
        self.nsx = len(self.slackStates)

        self.slackControls = np.concatenate(
            [self.control_indices["df"], self.control_indices["ddprogress"]]
        )
        self.nsu = len(self.slackControls)

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
        gates_pos = p[self.param_indices["gates_pos"]].reshape((self.pathPlanner.num_gates, 3))
        gates_rpy = p[self.param_indices["gates_rpy"]].reshape((self.pathPlanner.num_gates, 3))
        start_pos = p[self.param_indices["start_pos"]].reshape((1, 3))
        start_rpy = p[self.param_indices["start_rpy"]].reshape((1, 3))

        # Desired position and tangent vector on the path
        path = self.pathPlanner.path_func  # Unpack the path function
        dpath = self.pathPlanner.dpath_func  # Unpack the path gradient function
        pd = path(
            progress, start_pos, start_rpy, gates_pos, gates_rpy
        )  # Desired position on the path
        tangent_line = dpath(
            progress, start_pos, start_rpy, gates_pos, gates_rpy
        )  # Tangent vector of the path at the current progress
        tangent_line = tangent_line / ca.norm_2(tangent_line)  # Normalize the tangent vector
        pos_err = pos - pd  # Error between the current position and the desired position

        # Lag error
        lag_err = ca.mtimes([ca.dot(pos_err, tangent_line), tangent_line])
        lag_cost = ca.mtimes([lag_err.T, self.Ql, lag_err])

        # Contour error
        contour_err = pos_err - lag_err
        contour_cost = ca.mtimes([contour_err.T, self.Qc, contour_err])

        # Body angular velocity cost
        w_cost = ca.mtimes([w.T, self.Qw, w])

        # Progress rate cost
        dprogress_cost_L2 = ca.mtimes([dprogress.T, self.Rdprogress, dprogress])

        # Progress rate cost
        dprogress_cost = -dprogress * self.Qmu

        # Thrust rate cost
        thrust_rate_cost = df.T @ self.Rdf @ df  # ca.mtimes([df.T, self.Rdf, df])

        # Total stage cost
        stage_cost = (
            lag_cost + contour_cost + w_cost + dprogress_cost + dprogress_cost_L2 + thrust_rate_cost
        )

        return stage_cost
