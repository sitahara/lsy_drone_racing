from __future__ import annotations
import os

import casadi as ca
import numpy as np
import toml
import enum
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as Rot
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from lsy_drone_racing.mpc_utils.planners import HermiteSpline
from lsy_drone_racing.mpc_utils.utils import (
    W1,
    W2,
    Rbi,
    W1s,
    W2s,
    dW2s,
    quaternion_conjugate,
    quaternion_product,
    quaternion_norm,
    quaternion_rotation,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rungeKuttaExpr,
    rungeKuttaFcn,
    shuffleQuat,
)

from .dynamics import BaseDynamics


class DroneDynamics(BaseDynamics):
    """This class implements the base dynamics for a drone.

    Features:
        - Euler and Quaternion Dynamics
        - Thrust, Torque, and Motor RPM Control
        - Nonlinear Obstacle Constraints
        - Control Rates as control inputs
        - Linear Least-Squares cost function
        - Compatibility with IPOPT and acados optimizers

    General Usage of Dynamics classes:
    The dynamic classes implement dynamics, bounds, constraints, cost functions, and state/action mappings.
    The dynamics classes are used by the optimizer classes to solve the optimization problem.
    The general usage of the dynamics classes is as follows:
        - Setup the nominal parameters of the system
        - Setup the base bounds of the system (environment and drone bounds)
        - Define the symbolical dynamics, state, control expressions
        - Define the bounds and scaling factors for the dynamics
        - Define nonlinear constraints (e.g., obstacle constraints)
        - Define a cost function (e.g., linear least-squares cost function)
        - Define mapping functions to transform the state and action
        - Define a function to update the parameters of the system
    """

    def __init__(
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        initial_info: dict,
        dynamics_info: dict,
        constraints_info: dict,
        cost_info: dict,
        pathPlanner: HermiteSpline = None,
    ):
        """Initialization of the dynamics constraints.

        Args:
            initial_obs: The initial observation of the environment's state.
            initial_info: Additional environment information from the reset.
            dynamics_info: Additional information for the dynamics setup.
            constraints_info: Constraints information for the dynamics.
            cost_info: Cost information for the dynamics.
            pathPlanner: Instance of HermiteSpline for path planning.

        Methods:
            setup_nominal_parameters: Setup the unchanging parameters of the drone/environment controller.
            setup_base_bounds: Setup the unchanging bounds of the drone/environment controller.
            setup_optimization_parameters: Setup the optimization parameters for the drone/environment controller.
            setup_dynamics: Setup the dynamics for the controller.
            setupBoundsAndScals: Setup the constraints and scaling factors for the controller.
            setupCasadiFunctions: Setup the casadi functions for the controller.
            setupObstacleConstraints: Setup the obstacle constraints for the controller.
            transformState: Transform the state from the observation to the states used in the dynamics.
            transformAction: Transform the solution of the MPC controller to the format required for the Mellinger interface.

        Attributes:
            modelName: The name of the dynamics model.
            x: The state variables.
            x_all: Dict of Base definitions of all the state variables.
            nx: The number of state variables.
            state_indices: A dict that includes the indices of the individual state variables.
            u: The control variables.
            u_all: Dict of base definitions of all the control variables.
            nu: The number of control variables.
            dx: The time derivative of the state variables.
            dx_all: Dict of base definitions of all the time derivatives of the state variables.
            fc: The continuous dynamic function.
            fd: The discrete dynamic function.
            dx_d: The discrete dynamic expression.
            xdot: The derivative of the state variables.
            p: The parameters of the optimization problem.
            param_indices: A dict that includes the indices of the individual parameters.
            x_eq: The equilibrium state vector.
            x_lb: The lower bounds of the state variables.
            x_ub: The upper bounds of the state variables.
            x_scal: The scaling of the state variables.
            u_eq: The equilibrium control vector.
            u_lb: The lower bounds of the control variables.
            u_ub: The upper bounds of the control variables.
            u_scal: The scaling of the control variables.
            ts: The time step for the discrete dynamics.
        """
        super().__init__()
        self.initial_obs = initial_obs
        self.initial_info = initial_info
        self.cost_info = cost_info
        self.dynamics_info = dynamics_info
        self.constraints_info = constraints_info

        # Unpack the dynamics info
        self.ukf_info = dynamics_info.get(
            "ukf_info",
            {
                "useUKF": False,
                "measurement_noise": 0.01,
                "process_noise": 0.01,
                "action_disturbance": 0.001,
            },
        )
        self.n_horizon = dynamics_info.get("n_horizon", 60)
        self.ts = dynamics_info.get("ts", 1 / 60)
        self.t_predict = dynamics_info.get("t_predict", 0)
        if self.t_predict == 0:
            self.usePredict = False
        else:
            self.usePredict = True

        self.interface = dynamics_info.get("interface", "Mellinger")
        if self.interface not in ["Mellinger", "Thrust"]:
            raise ValueError("Currently only Mellinger or Thrust interfaces are supported.")

        self.baseDynamics = dynamics_info.get("dynamicsType", "Euler")
        if self.baseDynamics not in ["Euler", "Quaternion", "system_id"]:
            raise ValueError(
                "Currently only Euler, Quaternion, and system_id formulations are supported."
            )

        self.controlType = dynamics_info.get("controlType", "Thrusts")
        if self.controlType not in ["Thrusts", "Torques", "RPMs"]:
            raise ValueError(
                "Currently only Thrusts, Torques, or RPMs are supported. MPCC only uses Thrusts."
            )

        self.useControlRates = dynamics_info.get("useControlRates", False)

        self.useDrags = dynamics_info.get("useDrags", False)

        # Initial values of the obstacles and gates
        self.obstacle_pos = self.initial_obs.get("obstacles_pos", np.zeros((4, 3)))
        self.obstacle_diameter = self.constraints_info.get("obstacle_diameter", 0.1)
        self.obstacles_visited = self.initial_obs.get("obstacles_visited", np.zeros((4,)))
        self.gates_pos = self.initial_obs.get("gates_pos", np.zeros((4, 3)))
        self.gates_rpy = self.initial_obs.get("gates_rpy", np.zeros((4, 3)))
        self.gates_visited = self.initial_obs.get("gates_visited", np.zeros((4,)))

        # Setup basic parameters, dynamics, bounds, scaling, and constraints
        self.setupNominalParameters()
        self.setupBaseBounds()

        self.last_thrusts = np.zeros((4,))

        # Casadi parameter vector for the optimization problem
        self.p = None
        # Dict that maps the parameter names to their indices
        self.param_indices = {}
        # Current index of the parameters
        self.current_param_index = 0

        # Init the path Planner
        if pathPlanner is not None:
            self.pathPlanner = pathPlanner
            if self.pathPlanner.parametric:
                self.addToParameters(self.pathPlanner.path_params, "path")
        else:
            self.pathPlanner = None
            self.useTunnelConstraints = False

        if self.baseDynamics == "system_id":
            self.baseSystemIDDynamics()
            self.setupSystemIDBounds()
        else:
            if self.baseDynamics == "Euler":
                self.baseEulerDynamics()
            elif self.baseDynamics == "Quaternion":
                self.baseQuaternionDynamics()
            self.setupBoundsAndScals()
        # Setup the nonlinear constraints
        self.setupNLConstraints()
        # Setup the Cost function
        if cost_info.get("cost_type", "linear") == "linear":
            self.setupLinearCosts()
        else:
            self.setupMPCCCosts()
        # Initialize Parameters
        self.updateParameters(init=True)
        # Setup the casadi functions
        self.setupCasadiFunctions()
        # Setup the UKF (Unscented Kalman Filter)
        if self.ukf_info["useUKF"]:
            self.initUKF()

    def setupCasadiFunctions(self):
        """Setup explicit, implicit, and discrete dynamics functions."""
        # Continuous dynamic function
        # if self.useControlRates:
        #     self.fc = ca.Function("fc", [self.x, self.u], [self.dx], ["x", "du"], ["dx"])
        # else:
        self.fc = ca.Function("fc", [self.x, self.u], [self.dx], ["x", "u"], ["dx"])
        self.fd = rungeKuttaFcn(self.nx, self.nu, self.ts, self.fc)
        if self.usePredict:
            self.fd_predict = rungeKuttaFcn(self.nx, self.nu, self.t_predict, self.fc)
        # Discrete dynamic expression and dynamic function
        self.dx_d = rungeKuttaExpr(self.x, self.u, self.ts, self.fc)

        self.xdot = ca.MX.sym("xdot", self.nx)
        # Continuous implicit dynamic expression
        self.f_impl = self.xdot - self.dx

    def transformState(self, x: np.ndarray) -> np.ndarray:
        """Transforms observations from the environment to the respective states used in the dynamics."""
        # Extract the position, velocity, euler angles, and angular velocities

        if hasattr(self, "last_thrusts") and (self.ukf_info["useUKF"]):
            # x1 = x
            x = self.runUKF(z=x, u=self.last_thrusts)
            # print("UKF State Changes", x - x1)
        pos = x[:3]
        vel = x[3:6]
        rpy = x[6:9]
        drpy = x[9:12]
        w = W1(rpy) @ drpy

        if not hasattr(self, "next_predicted_states") or self.next_predicted_states is None:
            self.next_predicted_states = np.zeros(self.nx)

        if self.baseDynamics == "Euler":
            x = self.next_predicted_states
            x[:12] = np.concatenate([pos, vel, rpy, w])
        elif self.baseDynamics == "Quaternion":
            x = self.next_predicted_states
            quat = Rot.from_euler("xyz", rpy).as_quat()
            x[:13] = np.concatenate([pos, vel, quat, w])
        elif self.baseDynamics == "system_id":
            # States for the system identification model (pos, vel, rpy, ftot, theta, dtheta, ftot_cmd, rpy_cmd)
            # Controls for the system identification model (dftot_cmd, drpy_cmd, ddtheta_cmd)

            # Theta, dtheta, ddtheta are currently not used
            x = self.next_predicted_states
            x[:9] = np.concatenate([pos, vel, rpy])
            # print("System ID theta", x[self.state_indices["theta"]])
        # Predict the state into the future if self.usePredict is True
        if self.usePredict:
            # fd_predict is a discrete dynamics function (RK4) with the time step t_predict
            x = self.fd_predict(x, self.last_thrusts)
        return x

    def transformAction(self, x_sol: np.ndarray, u_sol: np.ndarray) -> np.ndarray:
        """Transforms optimizer solutions to controller interfaces (Mellinger or Thrust)."""
        self.next_predicted_states = x_sol[:, 1]
        self.current_controls = u_sol[:, 0]

        if self.baseDynamics == "system_id":
            tot_thrust = x_sol[self.state_indices["ftot_cmd"], 1]
            desired_rpy = x_sol[self.state_indices["rpy_cmd"], 1]
            action = np.concatenate([[tot_thrust], desired_rpy])
            # print("Thrust/Euler desired", action)
            self.last_action = action
        elif self.interface == "Thrust":
            if self.useControlRates:
                controls = x_sol[self.state_indices["u"], 1]
            else:
                controls = u_sol[:, 0]

            if self.baseDynamics == "Euler":
                rpy = x_sol[self.state_indices["rpy"], 1]
            elif self.baseDynamics == "Quaternion":
                quat = x_sol[self.state_indices["quat"], 1]
                rpy = Rot.from_quat(quat).as_euler("xyz")

            if self.controlType == "Torques":
                tot_thrust = controls[0]
            elif self.controlType == "RPMs":
                tot_thrust = self.ct * np.sum(controls**2)
            elif self.controlType == "Thrusts":
                tot_thrust = np.sum(controls)
            action = np.concatenate([[tot_thrust], rpy])
            # print("Thrust/Euler desired", action)
        elif self.interface == "Mellinger":
            action = x_sol[:, 1]
            pos = action[self.state_indices["pos"]]
            vel = action[self.state_indices["vel"]]
            w = action[self.state_indices["w"]]

            if self.baseDynamics == "Euler":
                rpy = action[self.state_indices["rpy"]]
            elif self.baseDynamics == "Quaternion":
                quat = action[self.state_indices["quat"]]
                rpy = Rot.from_quat(quat).as_euler("xyz")
            yaw = rpy[-1]
            acc_world = (vel - x_sol[self.state_indices["vel"], 0]) / self.ts

            drpy = W2(rpy) @ w
            action = np.concatenate([pos, vel, acc_world, [yaw], drpy])
        return action.flatten()

    def setupNominalParameters(self):
        """Setup the nominal parameters of the drone/environment/controller."""
        self.mass = self.initial_info.get("drone_mass", 0.027)
        self.g = 9.81
        self.gv = ca.vertcat(0, 0, -self.g)
        Ixx = 1.395e-5
        Iyy = 1.436e-5
        Izz = 2.173e-5

        self.J = ca.diag([Ixx, Iyy, Izz])
        self.J_inv = ca.diag([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz])

        self.ct = 3.1582e-10
        self.cd = 7.9379e-12
        self.gamma = self.cd / self.ct

        self.arm_length = 0.046
        self.beta = self.arm_length / ca.sqrt(2.0)
        self.c_tau_xy = self.arm_length * self.ct / ca.sqrt(2)
        self.rpmToTorqueMat = np.array(
            [
                [self.c_tau_xy, self.c_tau_xy, -self.c_tau_xy, -self.c_tau_xy],
                [-self.c_tau_xy, self.c_tau_xy, self.c_tau_xy, -self.c_tau_xy],
                [self.cd, -self.cd, self.cd, -self.cd],
            ]
        )
        self.DragMat = np.diag([self.cd, self.cd, self.cd])

    def baseSystemIDDynamics(self):
        self.modelName = "ThrustSystemID"
        self.system_id_info = {
            "params_pitch_rate": [[-6.003842038081178], [6.213752925707588]],
            "params_roll_rate": [[-3.960889336015948], [4.078293254657104]],
            "params_yaw_rate": [[-0.005347588299390372], [0.0]],
            "params_acc": [20.907574256269616, 3.653687545690674],
        }
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        rpy = ca.MX.sym("rpy", 3)
        ftot = ca.MX.sym("ftot")
        ftot_cmd = ca.MX.sym("ftot_cmd")
        rpy_cmd = ca.MX.sym("rpy_cmd", 3)
        # NEW
        theta = ca.MX.sym("theta")
        dtheta = ca.MX.sym("dtheta")
        ddtheta = ca.MX.sym("ddtheta")
        dftot_cmd = ca.MX.sym("dftot_cmd")
        drpy_cmd = ca.MX.sym("drpy_cmd", 3)
        self.state_indices = {
            "pos": np.arange(0, 3),
            "vel": np.arange(3, 6),
            "rpy": np.arange(6, 9),
            "ftot": 9,
            "theta": 10,
            "dtheta": 11,
            "ftot_cmd": 12,
            "rpy_cmd": np.arange(13, 16),
        }
        self.control_indices = {"dftot_cmd": 0, "drpy_cmd": np.arange(1, 4), "ddtheta_cmd": 4}

        # define state and input vector
        x = ca.vertcat(pos, vel, rpy, ftot, theta, dtheta, ftot_cmd, rpy_cmd)
        u = ca.vertcat(dftot_cmd, drpy_cmd, ddtheta)
        # Define nonlinear system dynamics
        dx = ca.vertcat(
            vel,
            (self.system_id_info["params_acc"][0] * ftot + self.system_id_info["params_acc"][1])
            * (ca.cos(rpy[0]) * ca.sin(rpy[1]) * ca.cos(rpy[2]) + ca.sin(rpy[0]) * ca.sin(rpy[2])),
            (self.system_id_info["params_acc"][0] * ftot + self.system_id_info["params_acc"][1])
            * (ca.cos(rpy[0]) * ca.sin(rpy[1]) * ca.sin(rpy[2]) - ca.sin(rpy[0]) * ca.cos(rpy[2])),
            (self.system_id_info["params_acc"][0] * ftot + self.system_id_info["params_acc"][1])
            * ca.cos(rpy[0])  # Changed from `ca.cos(r)`
            * ca.cos(rpy[1])  # Changed from `ca.cos(p)`
            - self.g,
            (
                self.system_id_info["params_roll_rate"][0] * rpy[0]
                + self.system_id_info["params_roll_rate"][1] * rpy_cmd[0]
            ),
            (
                self.system_id_info["params_pitch_rate"][0] * rpy[1]
                + self.system_id_info["params_pitch_rate"][1] * rpy_cmd[1]
            ),
            (
                self.system_id_info["params_yaw_rate"][0] * rpy[2]
                + self.system_id_info["params_yaw_rate"][1] * rpy_cmd[2]
            ),
            10.0 * (ftot_cmd - ftot),
            dtheta,
            ddtheta,
            dftot_cmd,
            drpy_cmd,
        )
        self.x = x
        self.dx = dx
        self.u = u
        self.nu = self.u.size()[0]
        self.nx = x.size()[0]
        self.x_eq = np.zeros((self.nx,))
        self.u_eq = np.zeros((self.nu,))

        self.ny = self.nx + self.nu

    def baseEulerDynamics(self):
        """Setup the base Euler dynamics for the drone/environment controller."""
        self.modelName = self.controlType + self.baseDynamics + self.interface
        # States
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        rpy = ca.MX.sym("rpy", 3)
        w = ca.MX.sym("w", 3)
        self.state_indices = {
            "pos": np.arange(0, 3),
            "vel": np.arange(3, 6),
            "rpy": np.arange(6, 9),
            "w": np.arange(9, 12),
        }
        try:
            x_eq = np.concatenate(
                [self.initial_obs["pos"], np.zeros(3), self.initial_obs["rpy"], np.zeros(3)]
            )
        except KeyError:
            x_eq = np.zeros((12,))
        # Controls
        if self.controlType == "Thrusts":
            u = ca.MX.sym("f", 4)
            torques = self.thrustsToTorques_sym(u)
            thrust_total = ca.sum1(u)
            u_eq = 0.25 * self.mass * self.g * np.ones((4,))
        elif self.controlType == "Torques":
            thrust_total = ca.MX.sym("thrust", 1)
            torques = ca.MX.sym("torques", 3)

            u_eq = np.array([self.mass * self.g, 0, 0, 0]).T
            u = ca.vertcat(thrust_total, torques)
        elif self.controlType == "RPMs":
            u = ca.MX.sym("rpm", 4)
            u_eq = np.sqrt((self.mass * self.g / 4) / self.ct) * np.ones((4,))
            f = self.ct * (u**2)
            torques = self.thrustsToTorques_sym(f)
            thrust_total = ca.sum1(f)

        thrust_vec = ca.vertcat(0, 0, (thrust_total) / self.mass)

        dpos = vel
        dvel = self.gv + Rbi(rpy) @ thrust_vec
        drpy = W2s(rpy) @ w
        dw = self.J_inv @ (torques - ca.cross(w, self.J @ w))  #   (ca.skew(w) @ self.J @ w))

        if self.useControlRates:
            du = ca.MX.sym("du", 4)
            self.state_indices["u"] = np.arange(12, 16)
            self.control_indices = {"du": np.arange(0, 4)}
            x = ca.vertcat(pos, vel, rpy, w, u)
            dx = ca.vertcat(dpos, dvel, drpy, dw, du)
            self.u = du
            self.x_eq = np.concatenate([x_eq, u_eq])
            self.u_eq = np.zeros((4,))
        else:
            self.control_indices = {"u": np.arange(0, 4)}
            x = ca.vertcat(pos, vel, rpy, w)
            dx = ca.vertcat(dpos, dvel, drpy, dw)
            self.u = u
            self.x_eq = x_eq
            self.u_eq = u_eq

        self.x = x
        self.nx = x.size()[0]
        self.dx = dx
        self.nu = self.u.size()[0]
        self.ny = self.nx + self.nu

    def baseQuaternionDynamics(self):
        """Setup the base quaternion dynamics for the drone/environment controller."""
        self.modelName = self.controlType + self.baseDynamics + self.interface
        # States
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        quat = ca.MX.sym("quat", 4)
        w = ca.MX.sym("w", 3)
        self.state_indices = {
            "pos": np.arange(0, 3),
            "vel": np.arange(3, 6),
            "quat": np.arange(6, 10),
            "w": np.arange(10, 13),
        }
        try:
            rpy_i = self.initial_obs["rpy"]
            quat_i = Rot.from_euler("xyz", rpy_i).as_quat()
            x_eq = np.concatenate(
                [self.initial_obs["pos"], self.initial_obs["vel"], quat_i, np.zeros(3)]
            )
            x_eq = np.zeros((13,))
        except KeyError:
            x_eq = np.zeros((13,))
        # Controls
        if self.controlType == "Thrusts":
            u = ca.MX.sym("f", 4)
            torques = self.thrustsToTorques_sym(u)
            thrust_total = ca.sum1(u)

            u_eq = 0.25 * self.mass * self.g * np.ones((4,))
        elif self.controlType == "Torques":
            torques = ca.MX.sym("torques", 3)
            thrust_total = ca.MX.sym("tot_thrust", 1)
            u_eq = np.array([self.mass * self.g, 0, 0, 0]).T
            u = ca.vertcat(thrust_total, torques)
        elif self.controlType == "RPMs":
            u = ca.MX.sym("rpm", 4)
            u_eq = np.sqrt((self.mass * self.g * 0.25) / self.ct) * np.ones((4,))
            f = self.ct * (u**2)
            torques = self.thrustsToTorques_sym(f)
            thrust_total = ca.sum1(f)

        thrust_vec = ca.vertcat(0, 0, thrust_total) / self.mass
        Rquat = quaternion_to_rotation_matrix(quat)

        dpos = vel
        if self.useDrags:
            dvel = self.gv + Rquat @ thrust_vec - ca.mtimes([Rquat, self.DragMat, Rquat.T, vel])
        else:
            dvel = self.gv + Rquat @ thrust_vec
        dquat = 0.5 * quaternion_product(quat, ca.vertcat(w, 0))
        dw = self.J_inv @ (torques - ca.cross(w, self.J @ w))

        if self.useControlRates:
            du = ca.MX.sym("du", 4)
            self.state_indices["u"] = np.arange(13, 17)
            self.control_indices = {"du": np.arange(0, 4)}
            x = ca.vertcat(pos, vel, quat, w, u)
            dx = ca.vertcat(dpos, dvel, dquat, dw, du)
            self.u = du
            x_eq = np.concatenate([x_eq, u_eq])
            u_eq = np.zeros((4,))
        else:
            self.control_indices = {"u": np.arange(0, 4)}
            x = ca.vertcat(pos, vel, quat, w)
            dx = ca.vertcat(dpos, dvel, dquat, dw)
            self.u = u

        self.x_eq = x_eq
        self.u_eq = u_eq

        self.x = x
        self.nx = x.size()[0]
        self.dx = dx
        self.nu = self.u.size()[0]
        self.ny = self.nx + self.nu

    def setupNLConstraints(self):
        """Setup the basic constraints for the drone/environment controller. Each constraint setup appends the constraints to the existing constraints and extends the parameter vector when needed."""
        # casadi vector of all non-linear constraints
        self.nl_constr = None
        # Dict of all non-linear constraints
        self.nl_constr_indices = {}
        # Current index of the non-linear constraints
        self.current_nl_constr_index = 0

        self.useObstacleConstraints = self.constraints_info.get("useObstacleConstraints", True)
        self.useTunnelConstraints = self.constraints_info.get("useTunnelConstraints", False)
        self.tunnelTransitionMargin = self.constraints_info.get("tunnelTransitionMargin", 0.1)
        self.tunnelTransitionSteepness = self.constraints_info.get("tunnelTransitionSteepness", 10)
        self.Wn = self.constraints_info.get("Wn", 0.3)
        self.Wgate = self.constraints_info.get("Wgate", 0.2)
        if self.baseDynamics == "Quaternion":
            # Norm and euler angle constraints
            self.setupQuatConstraints()
        if self.useObstacleConstraints:
            # Obstacle constraints
            self.setupObstacleConstraints()
        if self.useTunnelConstraints and self.pathPlanner.parametric:
            # Tunnel constraints
            self.setupTunnelConstraints()

    def setupLinearCosts(self):
        """Setup the linear (Quadratic) costs of form: (sum_i ||x_i-x_ref_i||_{Qs}^2 + ||u_i-u_ref_i||_{R}^2) + ||x_N-x_ref_N||_{Qt}^2."""
        self.cost_type = "linear"
        # print("Setting up linear costs", self.cost_info)
        Qs_pos = self.cost_info.get("Qs_pos", 1)
        Qt_pos = self.cost_info.get("Qt_pos", Qs_pos)

        Qs_vel = self.cost_info.get("Qs_vel", 0.1)
        Qt_vel = self.cost_info.get("Qt_vel", Qs_vel)

        Qs_rpy = self.cost_info.get("Qs_rpy", 0.1)
        Qt_rpy = self.cost_info.get("Qt_rpy", Qs_rpy)

        Qs_drpy = self.cost_info.get("Qs_drpy", 0.1)
        Qt_drpy = self.cost_info.get("Qt_drpy", Qs_drpy)

        Qs_quat = self.cost_info.get("Qs_quat", 0.01)
        Qt_quat = self.cost_info.get("Qt_quat", Qs_quat)

        Qs_pos = np.array([Qs_pos, Qs_pos, Qs_pos])
        Qt_pos = np.array([Qt_pos, Qt_pos, Qt_pos])

        Qs_vel = np.array([Qs_vel, Qs_vel, Qs_vel])
        Qt_vel = np.array([Qt_vel, Qt_vel, Qt_vel])

        Qs_rpy = np.array([Qs_rpy, Qs_rpy, Qs_rpy])
        Qt_rpy = np.array([Qt_rpy, Qt_rpy, Qt_rpy])

        Qs_drpy = np.array([Qs_drpy, Qs_drpy, Qs_drpy])
        Qt_drpy = np.array([Qt_drpy, Qt_drpy, Qt_drpy])

        Qs_quat = np.array([Qs_quat, Qs_quat, Qs_quat, Qs_quat])
        Qt_quat = np.array([Qt_quat, Qt_quat, Qt_quat, Qt_quat])

        R_f = self.cost_info.get("R_f", 0.1)
        R_f = np.array([R_f, R_f, R_f, R_f])
        R_df = self.cost_info.get("R_df", 0.01)
        R_df = np.array([R_df, R_df, R_df, R_df])

        if self.baseDynamics == "Euler":
            Qs = np.concatenate([Qs_pos, Qs_vel, Qs_rpy, Qs_drpy])
            Qt = np.concatenate([Qt_pos, Qt_vel, Qt_rpy, Qt_drpy])
            if self.useControlRates:
                Qs = np.concatenate([Qs, R_f])
                Qt = np.concatenate([Qt, R_f])
                R = R_df
            else:
                R = R_f
        elif self.baseDynamics == "Quaternion":
            Qs = np.concatenate([Qs_pos, Qs_vel, Qs_quat, Qs_drpy])
            Qt = np.concatenate([Qt_pos, Qt_vel, Qt_quat, Qt_drpy])
            if self.useControlRates:
                Qs = np.concatenate([Qs, R_f])
                Qt = np.concatenate([Qt, R_f])
                R = R_df
            else:
                R = R_f
        elif self.baseDynamics == "system_id":
            # State: pos, vel, rpy, ftot, theta, dtheta, ftot_cmd, rpy_cmd
            Qs = np.concatenate([Qs_pos, Qs_vel, Qs_rpy, [R_f[0]], np.zeros(2), [R_f[0]], Qs_rpy])
            Qt = np.concatenate([Qt_pos, Qt_vel, Qt_rpy, [R_f[0]], np.zeros(2), [R_f[0]], Qt_rpy])
            # Control: dftot_cmd, drpy_cmd, ddtheta_cmd
            R = np.concatenate([[R_df[0]], Qs_drpy, np.zeros(1)])
        else:
            raise ValueError("Base dynamics not recognized.")

        self.Qs = np.diag(Qs)
        self.Qt = np.diag(Qt)
        self.R = np.diag(R)
        # print(self.u_ref.shape)
        self.stageCostFunc = self.LQ_stageCost
        self.terminalCostFunc = self.LQ_terminalCost
        # print("Linear costs set up.")

    def LQ_stageCost(self, x, u, p, x_ref=None, u_ref=None):
        """Compute the LQR cost."""
        return ca.mtimes([(x - x_ref).T, self.Qs, (x - x_ref)]) + ca.mtimes(
            [(u - u_ref).T, self.R, u - u_ref]
        )

    def LQ_terminalCost(self, x, u, p, x_ref=None, u_ref=None):
        """Compute the LQR cost."""
        return ca.mtimes([(x - x_ref).T, self.Qt, (x - x_ref)])

    def setupMPCCCosts(self):
        """Setup the MPCC costs."""
        self.cost_type = "external"
        # Lag error weights
        Qs_l = self.cost_info.get("Qs_l", 1)
        self.Qs_l = ca.diag([Qs_l, Qs_l, Qs_l])
        # Contour error weights
        Qs_c = self.cost_info.get("Qs_c", 1)
        self.Qs_c = ca.diag([Qs_c, Qs_c, Qs_c])
        # RPY weights
        Qs_rpy = self.cost_info.get("Qs_rpy", 0.5)
        self.Qs_rpy = ca.diag([Qs_rpy, Qs_rpy, Qs_rpy])
        # Body Angular velocity weights
        Qs_drpy = self.cost_info.get("Qs_drpy", 0.1)
        self.Qs_drpy = ca.diag([Qs_drpy, Qs_drpy, Qs_drpy])
        # L1 Progress rate weights
        self.Q_dtheta = self.cost_info.get("Q_dtheta", 1)
        # L2 Progress rate weights
        self.R_dtheta = self.cost_info.get("R_dtheta", 0.01)

        # Thrust weights
        self.R_ftot = self.cost_info.get("R_f", 0.1)
        self.R_dftot = self.cost_info.get("R_df", 0.01)
        # Define the cost function
        self.stageCostFunc = self.MPCC_stage_cost
        self.terminalCostFunc = self.MPCC_stage_cost

    def MPCC_stage_cost(self, x, u, p, x_ref=None, u_ref=None):
        """Compute the MPCC cost."""
        pos = x[self.state_indices["pos"]]
        theta = x[self.state_indices["theta"]]
        dtheta = x[self.state_indices["dtheta"]]
        rpy = x[self.state_indices["rpy"]]
        rpy_cmd = x[self.state_indices["rpy_cmd"]]
        ftot = x[self.state_indices["ftot"]]
        ftot_cmd = x[self.state_indices["ftot_cmd"]]

        dftot_cmd = u[self.control_indices["dftot_cmd"]]
        drpy_cmd = u[self.control_indices["drpy_cmd"]]
        ddtheta_cmd = u[self.control_indices["ddtheta_cmd"]]

        path_params = p[self.param_indices["path"]]
        path_pos = self.pathPlanner.path_function(theta, path_params)
        path_tangent = self.pathPlanner.dpath_function(theta, path_params)

        pos_error = pos - path_pos
        # Lag Cost
        lag_error = ca.mtimes([ca.dot(path_tangent, pos_error), path_tangent])
        lag_cost = ca.mtimes([lag_error.T, self.Qs_l, lag_error])
        # Contour Cost
        contour_error = pos_error - lag_error
        contour_cost = ca.mtimes([contour_error.T, self.Qs_c, contour_error])
        # (d)RPY Cost
        rpy_cost = ca.mtimes([(rpy).T, self.Qs_rpy, rpy])
        drpy_cost = ca.mtimes([drpy_cmd.T, self.Qs_drpy, drpy_cmd])
        # Progress rate cost
        dtheta_cost = -dtheta * self.Q_dtheta
        ddtheta_cost = ddtheta_cmd * self.R_dtheta * ddtheta_cmd

        # Total Thrust cost
        # ftot_cost = ftot_cmd * self.R_ftot * ftot_cmd
        dftot_cost = dftot_cmd * self.R_dftot * dftot_cmd
        stage_cost = (
            lag_cost + contour_cost + rpy_cost + drpy_cost + dtheta_cost + ddtheta_cost + dftot_cost
        )
        return stage_cost

    def updateParameters(self, obs: dict = None, init: bool = False) -> bool:
        """Update the parameters of the drone/environment controller."""
        # Checks whether gate observation has been updated, replans if needed, and updates the path, dpath, and gate progresses parameters
        updated = False
        if init and self.p is not None:
            self.param_values = np.zeros((self.p.size()[0],))
            if self.useObstacleConstraints:
                self.param_values[self.param_indices["obstacles_pos"]] = self.obstacle_pos[
                    :, :-1
                ].flatten()
            if self.pathPlanner.parametric:
                self.param_values[self.param_indices["path"]] = self.pathPlanner.path_params_values
            if self.useTunnelConstraints:
                self.param_values[self.param_indices["gate_thetas"]] = (
                    self.pathPlanner.theta_switch[1 : self.gates_visited.size + 1]
                )
                self.param_values[self.param_indices["gates_pos"]] = self.gates_pos.flatten()
            updated = True
        elif obs is not None and self.p is not None:
            if self.useObstacleConstraints and np.any(
                np.not_equal(self.obstacles_visited, obs["obstacles_visited"])
            ):
                self.obstacles_visited = obs["obstacles_visited"]
                self.obstacles_pos = obs["obstacles_pos"]
                self.param_values[self.param_indices["obstacles_pos"]] = self.obstacle_pos[
                    :, :-1
                ].flatten()
                print("Obstacle positions updated")
                updated = True
            if self.pathPlanner.parametric and np.any(
                np.not_equal(self.gates_visited, obs["gates_visited"])
            ):
                self.gates_visited = obs["gates_visited"]
                self.gates_pos = obs["gates_pos"]
                self.gates_rpy = obs["gates_rpy"]
                self.pathPlanner.updateGates(self.gates_pos, self.gates_rpy)
                self.param_values[self.param_indices["path"]] = self.pathPlanner.path_params_values
                self.param_values[self.param_indices["gates_pos"]] = self.gates_pos.flatten()
                updated = True
                print("Gates positions updated")

        return updated

    def setupQuatConstraints(self):
        """Setup the quaternion constraints (normalization property and euler angle bounds)."""
        # Quaternion constraints (norm = 1)
        quat = self.x[self.state_indices["quat"]]
        quat_norm = quaternion_norm(quat) - 1  # Ensure quaternion is normalized
        quat_norm_lh = np.zeros((1,))  # Lower bound for normalization constraint
        quat_norm_uh = np.zeros((1,))  # Upper bound for normalization constraint
        # Quaternion constraints (euler angle bounds)
        quat_eul = quaternion_to_euler(quat)
        quat_eul_lh = self.rpy_lb
        quat_eul_uh = self.rpy_ub

        quat_constraints_lh = np.concatenate([quat_norm_lh, quat_eul_lh])
        quat_constraints_uh = np.concatenate([quat_norm_uh, quat_eul_uh])
        quat_constraints = ca.vertcat(quat_norm, quat_eul)
        self.addToConstraints(quat_constraints, quat_constraints_lh, quat_constraints_uh, "quat")

    def addToConstraints(
        self, nl_constr: ca.MX, lh: np.ndarray, uh: np.ndarray, nl_const_name: str
    ):
        """Add the non-linear constraints to the existing constraints."""
        if self.nl_constr is None:
            self.nl_constr = nl_constr
            self.nl_constr_lh = lh
            self.nl_constr_uh = uh
        else:
            self.nl_constr = ca.vertcat(self.nl_constr, nl_constr)
            self.nl_constr_lh = np.concatenate([self.nl_constr_lh, lh])
            self.nl_constr_uh = np.concatenate([self.nl_constr_uh, uh])
        self.nl_constr_indices[nl_const_name] = np.arange(
            self.current_nl_constr_index, self.current_nl_constr_index + len(lh)
        )
        self.current_nl_constr_index += len(lh)

    def addToParameters(self, param: ca.MX, param_name: str):
        """Add the parameters to the existing parameters."""
        if self.p is None:
            self.p = param
        else:
            self.p = ca.vertcat(self.p, param)

        self.param_indices[param_name] = np.arange(
            self.current_param_index, self.current_param_index + param.size()[0]
        )
        self.current_param_index += param.size()[0]

    def setupObstacleConstraints(self):
        """Setup the obstacle constraints for the drone/environment controller."""
        num_obstacles = self.obstacle_pos.shape[0]
        num_params_per_obstacle = 2
        num_params_obstacle = num_obstacles * num_params_per_obstacle
        # Parameters for the obstacle positions
        obstacles_pos_sym = ca.MX.sym("obstacles_pos", num_params_obstacle)
        # Extract the position of the drone
        pos = self.x[self.state_indices["pos"]]
        obstacle_constraints = []
        for k in range(num_obstacles):
            obstacle_constraints.append(
                ca.norm_2(
                    pos[:num_params_per_obstacle]
                    - obstacles_pos_sym[
                        k * num_params_per_obstacle : (k + 1) * num_params_per_obstacle
                    ]
                )
                - self.obstacle_diameter
            )

        obstacle_constraints_lh = np.zeros((num_obstacles,))
        obstacle_constraints_uh = np.ones((num_obstacles,)) * 1e9
        obstacle_constraints = ca.vertcat(*obstacle_constraints)
        # Add the obstacle constraints to the the constraints
        self.addToConstraints(
            obstacle_constraints, obstacle_constraints_lh, obstacle_constraints_uh, "obstacles"
        )
        self.addToParameters(obstacles_pos_sym, "obstacles_pos")
        print("Obstacle constraints set up.")
        return None

    def setupTunnelConstraints(self):
        theta = self.x[self.state_indices["theta"]]  # Progress state
        pos = self.x[self.state_indices["pos"]]  # Position state

        gate_thetas = ca.MX.sym("gate_thetas", self.gates_pos.shape[0])  # Progress of the gates
        gates_pos_sym = ca.MX.sym("gates_pos", self.gates_pos.size)
        self.addToParameters(gates_pos_sym, "gates_pos")
        self.addToParameters(gate_thetas, "gate_thetas")

        path = self.pathPlanner.path_function(
            theta=theta, path_params=self.p[self.param_indices["path"]]
        )["path"]
        dpath = self.pathPlanner.dpath_function(
            theta=theta, path_params=self.p[self.param_indices["path"]]
        )["dpath"]

        t = dpath / ca.norm_2(dpath)  # Normalized Tangent vector at the current progress
        n = ca.vertcat(-t[1], t[0], 0)  # Normal vector at the current progress
        b = ca.cross(t, n)  # Binormal vector at the current progress

        def getTunnelWidth2(gates_pos_sym: ca.MX, pos: ca.MX) -> ca.MX:
            """Calculate the tunnel width at the current progress."""
            # Calculate the progress distance to the nearest gate
            d = ca.MX.zeros(self.gates_pos.shape[0])
            for k in range(self.gates_pos.shape[0]):
                d[k] = ca.norm_2(gates_pos_sym[k * 3 : (k + 1) * 3] - pos)
            dmin = ca.mmin(d)
            sigmoid = 1 / (
                1 + ca.exp(-self.tunnelTransitionSteepness * (dmin - self.tunnelTransitionMargin))
            )
            return self.Wn + (self.Wgate - self.Wn) * sigmoid

        def getTunnelWidth(gate_thetas: ca.MX, theta: ca.MX) -> ca.MX:
            """Calculate the tunnel width at the current progress."""
            # Calculate the progress distance to the nearest gate
            d = ca.mmin(ca.fabs(gate_thetas - theta))
            sigmoid = 1 / (
                1 + ca.exp(-self.tunnelTransitionSteepness * (d - self.tunnelTransitionMargin))
            )
            return self.Wn + (self.Wgate - self.Wn) * sigmoid

        W = getTunnelWidth(gate_thetas, theta)
        # W = getTunnelWidth2(gates_pos_sym, pos)
        H = W
        # Left-lower corner of the tunnel around the path
        p0 = path - W * n - H * b
        # Tunnel constraints
        tunnel_constraints = []
        tunnel_constraints.append((pos - p0).T @ n)
        tunnel_constraints.append(2 * H - (pos - p0).T @ n)
        tunnel_constraints.append((pos - p0).T @ b)
        tunnel_constraints.append(2 * W - (pos - p0).T @ b)

        tunnel_constraints_lh = np.zeros((len(tunnel_constraints),))
        tunnel_constraints_uh = np.ones((len(tunnel_constraints))) * 1e9
        tunnel_constraints = ca.vertcat(*tunnel_constraints)
        self.addToConstraints(
            tunnel_constraints, tunnel_constraints_lh, tunnel_constraints_uh, "tunnel"
        )
        print("Tunnel constraints set up.")

    def setupBaseBounds(self):
        """Setup the nominal and unchanging bounds of the drone/environment controller."""
        # Rotor limits
        rpm_lb = 4070.3 + 0.2685 * 0
        rpm_ub = 4070.3 + 0.2685 * 65535
        self.rpm_lb = rpm_lb * np.ones((4,))
        self.rpm_ub = rpm_ub * np.ones((4,))
        # Individual rotor thrust limits
        self.thrust_lb = self.ct * (rpm_lb**2) * np.ones((4,))
        self.thrust_ub = self.ct * (rpm_ub**2) * np.ones((4,))
        self.tot_thrust_lb = np.sum(self.thrust_lb)
        self.tot_thrust_ub = np.sum(self.thrust_ub)
        # Individual rotor thrust rate limits
        rate_max = (
            (self.thrust_ub[0] - self.thrust_lb[0]) / 0.2
        )  # Assuming the motor can go from 0 to max thrust in 0.2s, loose bounds, only for control rate
        self.thrust_rate_lb = -rate_max * np.ones((4,))
        self.thrust_rate_ub = rate_max * np.ones((4,))
        self.tot_thrust_rate_lb = np.sum(self.thrust_rate_lb)
        self.tot_thrust_rate_ub = np.sum(self.thrust_rate_ub)
        # State limits (from the observation space, )
        x_y_max = 3.0
        z_max = 2.5
        z_min = 0.10  # Drone starts ~5 cm above the ground + 5cm margin to avoid hitting the ground
        rpy_max = 75 / 180 * np.pi  # 85 degrees in radians, reduced to 75 for safety
        large_val = 1e2
        self.pos_lb = np.array([-x_y_max, -x_y_max, z_min]).T
        self.pos_ub = np.array([x_y_max, x_y_max, z_max]).T
        self.vel_lb = np.array([-large_val, -large_val, -large_val]).T
        self.vel_ub = np.array([large_val, large_val, large_val]).T
        self.rpy_lb = np.array([-rpy_max, -rpy_max, -rpy_max]).T
        self.rpy_ub = np.array([rpy_max, rpy_max, rpy_max]).T
        self.drpy_lb = np.array([-large_val, -large_val, -large_val]).T
        self.drpy_ub = np.array([large_val, large_val, large_val]).T
        self.w_lb = np.array([-large_val, -large_val, -large_val]).T
        self.w_ub = np.array([large_val, large_val, large_val]).T
        # Quaternion bounds are implemented as non-linear constraints. Define loose placeholder bounds here.
        self.quat_lb = -1e3 * np.ones((4,))
        self.quat_ub = 1e3 * np.ones((4,))

    def setupSystemIDBounds(self):
        """Setup the bounds for the system identification model."""
        self.x_lb = np.zeros((self.nx,))
        self.x_ub = np.zeros((self.nx,))
        self.u_lb = np.zeros((self.nu,))
        self.u_ub = np.zeros((self.nu,))
        self.x_lb[self.state_indices["pos"]] = self.pos_lb
        self.x_ub[self.state_indices["pos"]] = self.pos_ub
        self.x_lb[self.state_indices["vel"]] = self.vel_lb
        self.x_ub[self.state_indices["vel"]] = self.vel_ub

        self.x_lb[self.state_indices["rpy"]] = self.rpy_lb
        self.x_ub[self.state_indices["rpy"]] = self.rpy_ub
        self.x_lb[self.state_indices["rpy_cmd"]] = self.rpy_lb
        self.x_ub[self.state_indices["rpy_cmd"]] = self.rpy_ub

        self.x_lb[self.state_indices["ftot"]] = self.tot_thrust_lb
        self.x_ub[self.state_indices["ftot"]] = self.tot_thrust_ub
        self.x_lb[self.state_indices["ftot_cmd"]] = self.tot_thrust_lb
        self.x_ub[self.state_indices["ftot_cmd"]] = self.tot_thrust_ub

        self.x_lb[self.state_indices["theta"]] = 0
        self.x_ub[self.state_indices["theta"]] = 1
        self.x_lb[self.state_indices["dtheta"]] = -0.01
        self.x_ub[self.state_indices["dtheta"]] = 1

        self.u_lb[self.control_indices["dftot_cmd"]] = self.tot_thrust_rate_lb
        self.u_ub[self.control_indices["dftot_cmd"]] = self.tot_thrust_rate_ub
        self.u_lb[self.control_indices["drpy_cmd"]] = self.drpy_lb
        self.u_ub[self.control_indices["drpy_cmd"]] = self.drpy_ub

        self.u_lb[self.control_indices["ddtheta_cmd"]] = -1
        self.u_ub[self.control_indices["ddtheta_cmd"]] = 1

        self.slackStates = np.array(
            [self.state_indices["pos"][-1]]
        )  # Only z_position has slack variables
        self.slackControls = np.array(
            [self.control_indices["dftot_cmd"], self.control_indices["ddtheta_cmd"]]
        )

        self.x_scal = self.x_ub - self.x_lb
        self.u_scal = self.u_ub - self.u_lb

    def setupBoundsAndScals(self):
        """Setup the bounds and scaling factors for states and controls."""
        if self.controlType == "Thrusts":
            u_lb = self.thrust_lb
            u_ub = self.thrust_ub
            u_lb_rate = self.thrust_rate_lb
            u_ub_rate = self.thrust_rate_ub
        elif self.controlType == "Torques":
            u_lb = np.array([self.tot_thrust_lb, -1, -1, -1])
            u_ub = np.array([self.tot_thrust_ub, 1, 1, 1])
            u_lb_rate = np.array([self.tot_thrust_rate_lb, -5, -5, -5])
            u_ub_rate = np.array([self.tot_thrust_rate_ub, 5, 5, 5])
        elif self.controlType == "RPMs":
            u_lb = self.rpm_lb
            u_ub = self.rpm_ub
            u_lb_rate = -self.rpm_lb / 0.2
            u_ub_rate = self.rpm_ub / 0.2

        if self.baseDynamics == "Euler":
            x_lb = np.concatenate([self.pos_lb, self.vel_lb, self.rpy_lb, self.w_lb])
            x_ub = np.concatenate([self.pos_ub, self.vel_ub, self.rpy_ub, self.w_ub])
        elif self.baseDynamics == "Quaternion":
            x_lb = np.concatenate([self.pos_lb, self.vel_lb, self.quat_lb, self.w_lb])
            x_ub = np.concatenate([self.pos_ub, self.vel_ub, self.quat_ub, self.w_ub])

        if self.useControlRates:
            x_lb = np.concatenate([x_lb, u_lb])
            x_ub = np.concatenate([x_ub, u_ub])
            u_lb = u_lb_rate
            u_ub = u_ub_rate

        self.x_lb = x_lb
        self.x_ub = x_ub
        self.x_scal = self.x_ub - self.x_lb
        self.slackStates = np.concatenate(
            [self.state_indices["pos"], self.state_indices["w"]]
        )  # Slack variables on pos and w
        self.slackControls = np.array(
            []
        )  # np.arange(0, self.nu)  # No control bounds have slack variables

        if any(x_ub - x_lb < 0):
            Warning("Some states have upper bounds lower than lower bounds")
        if any(self.x_scal < 1e-4):
            Warning("Some states have scalings close to zero, setting them to 0.1")
            self.x_scal = np.where(np.abs(self.x_scal) < 1e-4, 0.1, self.x_scal)

        self.u_lb = u_lb
        self.u_ub = u_ub
        self.u_scal = self.u_ub - self.u_lb

        if any(u_ub - u_lb < 0):
            Warning("Some controls have upper bounds lower than lower bounds")
        if any(self.u_scal < 1e-4):
            Warning("Some controls have scalings close to zero, setting them to 0.1")
            self.u_scal = np.where(np.abs(self.u_scal) < 1e-4, 0.1, self.u_scal)

    def thrustsToTorques(self, u: np.ndarray) -> np.ndarray:
        """Convert thrusts to torques."""
        torques = np.vstack(
            [
                self.beta * (u[0] + u[1] - u[2] - u[3]),
                self.beta * (-u[0] + u[1] + u[2] - u[3]),
                self.gamma * (u[0] - u[1] + u[2] - u[3]),
            ]
        )
        return torques

    def thrustsToTorques_sym(self, u: ca.MX) -> ca.MX:
        """Convert thrusts to torques."""
        torques = ca.vertcat(
            self.beta * (u[0] + u[1] - u[2] - u[3]),
            self.beta * (-u[0] + u[1] + u[2] - u[3]),
            self.gamma * (u[0] - u[1] + u[2] - u[3]),
        )
        return torques

    def initUKF(self):
        """Initialize the ukf for the drone/environment controller."""
        # The UKF is called right after we get new observations from the environment
        # , i.e., at the start of the transformState function. It uses the euler dynamics
        nx = 12
        nz = 12
        nu = 4

        self.last_u = np.zeros((nu,))
        process_noise = self.ukf_info.get("process_noise", 1e-3)
        measurement_noise = self.ukf_info.get("measurement_noise", 1e-3)
        action_disturbance = self.ukf_info.get("action_disturbance", 1e-4)

        # Create sigma points
        points = MerweScaledSigmaPoints(n=nx, alpha=1e-3, beta=2.0, kappa=0.0)
        # Create the discrete dynamics function and the Jacobian with respect to the control input
        fd_eul, Ju = self.createfd_eul()

        # Sample multiple state and action pairs
        num_samples = 1000
        x_lb = np.concatenate([self.pos_lb, -30 * np.ones(3), self.rpy_lb, -3 * np.pi * np.ones(3)])
        x_ub = np.concatenate([self.pos_ub, 30 * np.ones(3), self.rpy_ub, 3 * np.pi * np.ones(3)])
        u_lb = self.thrust_lb
        u_ub = self.thrust_ub
        state_samples = np.random.uniform(x_lb, x_ub, (num_samples, nx))
        action_samples = np.random.uniform(u_lb, u_ub, (num_samples, nu))

        # Define reasonable action disturbances
        action_disturbances = np.random.randn(num_samples, nu) * action_disturbance

        # Quantify the influence on the states
        influence = np.zeros((nx, nx))
        for i in range(num_samples):
            x_sample = state_samples[i, :]
            u_sample = action_samples[i, :]
            Ju_eval = np.array(Ju(x_sample, u_sample)).reshape(nx, nu)
            disturbance_impact = Ju_eval @ action_disturbances[i, :].reshape(nu, 1)
            influence += disturbance_impact @ disturbance_impact.T

        influence /= num_samples

        # Create state transition function
        def fstate(x, dt, u):
            # u += action_disturbance * np.random.randn(nu)
            x_next = np.array(fd_eul(x, u)).flatten()
            # Add action disturbance
            return x_next

        # Create measurement function
        def hmeas(x):
            return x

        # Create UKF
        self.ukf = UKF(dim_x=nx, dim_z=nz, fx=fstate, hx=hmeas, dt=self.ts, points=points)
        # Set the initial state
        initial_state = np.concatenate(
            [
                self.initial_obs["pos"],
                self.initial_obs["vel"],
                self.initial_obs["rpy"],
                self.initial_obs["ang_vel"],
            ]
        )
        self.ukf.x = initial_state
        # Set the initial covariance
        self.ukf.Q = np.eye(nx) * (process_noise**2) + influence / 10  # Process noise
        self.ukf.P = np.eye(nx) * 0.01  # Initial uncertainty
        self.ukf.R = np.eye(nz) * measurement_noise**2  # Measurement noise

    def runUKF(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Run the UKF for the drone/environment controller.

        Args:
            z: np.ndarray: The measurement
            u: np.ndarray: The control input
        Returns:
            np.ndarray: The updated state estimate
        """
        self.ukf.predict(u=u)
        self.ukf.update(z)
        return self.ukf.x

    def createfd_eul(self):
        """Create the discrete dynamics function for the Euler dynamics."""
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        rpy = ca.MX.sym("rpy", 3)
        w = ca.MX.sym("w", 3)

        u = ca.MX.sym("f", 4)
        torques = self.thrustsToTorques_sym(u)
        thrust_total = ca.sum1(u)
        thrust_vec = ca.vertcat(0, 0, (thrust_total) / self.mass)

        dpos = vel
        dvel = self.gv + Rbi(rpy[0], rpy[1], rpy[2]) @ thrust_vec
        drpy = W2s(rpy) @ w
        dw = self.J_inv @ (torques - ca.cross(w, self.J @ w))
        x = ca.vertcat(pos, vel, rpy, w)
        dx = ca.vertcat(dpos, dvel, drpy, dw)
        fc = ca.Function("fc", [x, u], [dx], ["x", "u"], ["dx"])
        fd = rungeKuttaFcn(x.size()[0], u.size()[0], self.ts, fc)

        fd_eval = fd(x, u)  # Evaluate the function
        J_u = ca.jacobian(fd_eval, u)  # Jacobian with respect to u
        Ju = ca.Function("B", [x, u], [J_u], ["x", "u"], ["Ju"])  # Create the function
        return fd, Ju
