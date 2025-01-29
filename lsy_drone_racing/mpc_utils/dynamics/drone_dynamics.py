import os

import casadi as ca
import numpy as np
import toml
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as Rot

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
    ):
        """Initialization of the dynamics constraints.

        Args:
            initial_obs: The initial observation of the environment's state.
            initial_info: Additional environment information from the reset.
            dynamics_info: Additional information for the dynamics setup.
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
        if self.baseDynamics not in ["Euler", "Quaternion", "MPCC"]:
            raise ValueError(
                "Currently only Euler, Quaternion, and MPCC formulations are supported."
            )

        if self.baseDynamics == "MPCC":
            onlyBaseInit = True
        else:
            onlyBaseInit = False

        self.controlType = dynamics_info.get("controlType", "Thrusts")
        if self.controlType not in ["Thrusts", "Torques", "MotorRPMs"]:
            raise ValueError(
                "Currently only Thrusts, Torques, or MotorRPMs are supported. MPCC only uses Thrusts."
            )

        self.useControlRates = dynamics_info.get("useControlRates", False)
        self.useAngVel = dynamics_info.get("useAngVel", True)
        if not self.useAngVel:
            self.useAngVel = True
            raise Warning("Currently only useAngVel=True is supported.")

        self.useDrags = dynamics_info.get("useDrags", False)

        # Constraints
        self.useObstacleConstraints = constraints_info.get("useObstacleConstraints", True)
        # Initial values of the obstacles and gates
        self.obstacle_pos = self.initial_obs.get("obstacles_pos", np.zeros((4, 3)))
        self.obstacle_diameter = self.constraints_info.get("obstacle_diameter", 0.1)
        self.obstacles_visited = self.initial_obs.get("obstacles_visited", np.zeros((4,)))
        self.gates_pos = self.initial_obs.get("gates_pos", np.zeros((4, 3)))
        self.gates_rpy = self.initial_obs.get("gates_rpy", np.zeros((4, 3)))
        self.gates_visited = self.initial_obs.get("gates_visited", np.zeros((4,)))
        # Used for the control rates and the prediction
        self.last_u = None

        # Setup basic parameters, dynamics, bounds, scaling, and constraints
        self.setupNominalParameters()
        self.setupBaseBounds()
        # Casadi parameter vector for the optimization problem
        self.param_values = None
        self.p = None
        # Dict that maps the parameter names to their indices
        self.param_indices = {}
        # Current index of the parameters
        self.current_param_index = 0
        # casadi vector of all non-linear constraints
        self.nl_constr = None
        # Dict of all non-linear constraints
        self.nl_constr_indices = {}
        # Current index of the non-linear constraints
        self.current_nl_constr_index = 0

        if not onlyBaseInit:
            if self.baseDynamics == "Euler":
                self.baseEulerDynamics()
            elif self.baseDynamics == "Quaternion":
                self.baseQuaternionDynamics()
            self.setupBoundsAndScals()
            self.setupNLConstraints()
            self.setupLinearCosts()
            self.updateParameters(init=True)
            self.setupCasadiFunctions()

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
        pos = x[:3]
        vel = x[3:6]
        eul_ang = x[6:9]
        deul_ang = x[9:12]
        # Convert to used states
        w = W1(eul_ang) @ deul_ang
        if self.baseDynamics == "Euler":
            x = np.concatenate([pos, vel, eul_ang, w])
        elif self.baseDynamics == "Quaternion":
            quat = Rot.from_euler("xyz", eul_ang).as_quat()
            x = np.concatenate([pos, vel, quat, w])

        if self.useControlRates:
            if self.last_u is None:
                self.last_u = self.u_eq
            x = np.concatenate([x, self.last_u])
        # Predict the state into the future if self.usePredict is True
        if self.usePredict and self.last_u is not None:
            # fd_predict is a discrete dynamics function (RK4) with the time step t_predict
            x = self.fd_predict(x, self.last_u)
        # print(x)
        return x

    def transformAction(self, x_sol: np.ndarray, u_sol: np.ndarray) -> np.ndarray:
        """Transforms optimizer solutions to controller inferfaces (Mellinger or Thrust)."""
        if self.useControlRates:
            self.last_u = x_sol[self.state_indices["u"], 1]
        if self.interface == "Thrust":
            if self.useControlRates:
                action = x_sol[self.state_indices["u"], 1]
            else:
                action = u_sol[:, 0]

            if self.controlType == "Torques":
                action = action
            elif self.controlType == "MotorRPMs":
                torques = self.rpmToTorqueMat @ (action**2)
                action = np.concatenate([self.ct * np.sum(action**2), torques])
            elif self.controlType == "Thrusts":
                torques = np.array(
                    [
                        self.beta * (action[0] + action[1] - action[2] - action[3]),
                        self.beta * (-action[0] + action[1] + action[2] - action[3]),
                        self.gamma * (action[0] - action[1] + action[2] - action[3]),
                    ]
                )
                tot_thrust = action[0] + action[1] + action[2] + action[3]
                action = np.concatenate([[tot_thrust], torques])
        elif self.interface == "Mellinger":
            action = x_sol[:, 1]
            pos = action[self.state_indices["pos"]]
            vel = action[self.state_indices["vel"]]
            w = action[self.state_indices["w"]]
            if self.baseDynamics == "Euler":
                yaw = action[self.state_indices["eul_ang"][2]]
            elif self.baseDynamics == "Quaternion":
                quat = action[self.state_indices["quat"]]
                yaw = Rot.from_quat(quat).as_euler("xyz")[2]

            acc_world = (vel - x_sol[self.state_indices["vel"], 0]) / self.ts
            yaw = action[8]
            action = np.concatenate([pos, vel, acc_world, [yaw], w])
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

    def baseEulerDynamics(self):
        """Setup the base Euler dynamics for the drone/environment controller."""
        self.modelName = self.controlType + self.baseDynamics + self.interface
        # States
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        eul_ang = ca.MX.sym("eul_ang", 3)
        w = ca.MX.sym("w", 3)
        self.state_indices = {
            "pos": np.arange(0, 3),
            "vel": np.arange(3, 6),
            "eul_ang": np.arange(6, 9),
            "w": np.arange(9, 12),
        }
        try:
            x_eq = np.concatenate(
                [
                    self.initial_obs["pos"],
                    self.initial_obs["vel"],
                    self.initial_obs["eul_ang"],
                    np.zeros(3),
                ]
            )
        except KeyError:
            x_eq = np.zeros((12,))
        # Controls
        if self.controlType == "Thrusts":
            u = ca.MX.sym("f", 4)
            torques = ca.vertcat(
                self.beta * (u[0] + u[1] - u[2] - u[3]),
                self.beta * (-u[0] + u[1] + u[2] - u[3]),
                self.gamma * (u[0] - u[1] + u[2] - u[3]),
            )  # tau_x, tau_y, tau_z
            thrust_total = ca.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) / self.mass)
            u_eq = 0.25 * self.mass * self.g * np.ones((4,))
        elif self.controlType == "Torques":
            torques = ca.MX.sym("torques", 3)
            thrust_total = ca.MX.sym("thrust", 1)
            u_eq = np.array([self.mass * self.g, 0, 0, 0]).T
            u = ca.vertcat(thrust_total, torques)
        elif self.controlType == "MotorRPMs":
            u = ca.MX.sym("rpm", 4)
            u_eq = np.sqrt((self.mass * self.g / 4) / self.ct) * np.ones((4,))
            f = self.ct * (u**2)
            torques = torques = ca.vertcat(
                self.beta * (f[0] + f[1] - f[2] - f[3]),
                self.beta * (-f[0] + f[1] + f[2] - f[3]),
                self.gamma * (f[0] - f[1] + f[2] - f[3]),
            )  # tau_x, tau_y, tau_z
            thrust_total = ca.vertcat(0, 0, ca.sum1(f) / self.mass)

        dpos = vel
        dvel = self.gv + Rbi(eul_ang[0], eul_ang[1], eul_ang[2]) @ thrust_total
        deul_ang = W2s(eul_ang) @ w
        dw = self.J_inv @ (torques - (ca.skew(w) @ self.J @ w))

        if self.useControlRates:
            du = ca.MX.sym("du", 4)
            self.state_indices["u"] = np.arange(12, 16)
            self.control_indices = {"du": np.arange(0, 4)}
            x = ca.vertcat(pos, vel, eul_ang, w, u)
            dx = ca.vertcat(dpos, dvel, deul_ang, dw, du)
            self.u = du
            self.x_eq = np.concatenate([x_eq, u_eq])
            self.u_eq = np.zeros((4,))
        else:
            self.control_indices = {"u": np.arange(0, 4)}
            x = ca.vertcat(pos, vel, eul_ang, w)
            dx = ca.vertcat(dpos, dvel, deul_ang, dw)
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
            x_eq = np.concatenate(
                [self.initial_obs["pos"], self.initial_obs["vel"], np.zeros(4), np.zeros(3)]
            )
        except KeyError:
            x_eq = np.zeros((13,))
        # Controls
        if self.controlType == "Thrusts":
            u = ca.MX.sym("f", 4)
            torques = ca.vertcat(
                self.beta * (u[0] + u[1] - u[2] - u[3]),
                self.beta * (-u[0] + u[1] + u[2] - u[3]),
                self.gamma * (u[0] - u[1] + u[2] - u[3]),
            )  # tau_x, tau_y, tau_z
            thrust_total = ca.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) / self.mass)
            u_eq = 0.25 * self.mass * self.g * np.ones((4,))
        elif self.controlType == "Torques":
            torques = ca.MX.sym("torques", 3)
            thrust_tot = ca.MX.sym("tot_thrust", 1)
            thrust_total = ca.vertcat(0, 0, thrust_tot / self.mass)
            u_eq = np.array([self.mass * self.g, 0, 0, 0]).T
            u = ca.vertcat(thrust_tot, torques)
        elif self.controlType == "MotorRPMs":
            u = ca.MX.sym("rpm", 4)
            u_eq = np.sqrt((self.mass * self.g / 4) / self.ct) * np.ones((4,))
            f = self.ct * (u**2)
            torques = torques = ca.vertcat(
                self.beta * (f[0] + f[1] - f[2] - f[3]),
                self.beta * (-f[0] + f[1] + f[2] - f[3]),
                self.gamma * (f[0] - f[1] + f[2] - f[3]),
            )  # tau_x, tau_y, tau_z
            thrust_total = ca.vertcat(0, 0, ca.sum1(f) / self.mass)

        Rquat = quaternion_to_rotation_matrix(quat)

        dpos = vel
        if self.useDrags:
            dvel = (
                self.gv
                + quaternion_rotation(quat, thrust_total)
                - ca.mtimes([Rquat, self.DragMat, Rquat.T, vel])
            )
        else:
            dvel = self.gv + quaternion_rotation(quat, thrust_total)
        dquat = 0.5 * quaternion_product(quat, ca.vertcat(w, 0))
        dw = self.J_inv @ (torques - (ca.skew(w) @ self.J @ w))

        if self.useControlRates:
            du = ca.MX.sym("du", 4)
            self.state_indices["u"] = np.arange(13, 17)
            self.control_indices = {"du": np.arange(0, 4)}
            x = ca.vertcat(pos, vel, quat, w, u)
            dx = ca.vertcat(dpos, dvel, dquat, dw, du)
            self.u = du
            self.x_eq = np.concatenate([x_eq, u_eq])
            self.u_eq = np.zeros((4,))
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
        if self.baseDynamics == "Quaternion":
            # Norm and euler angle constraints
            self.setupQuatConstraints()
        if self.useObstacleConstraints:
            # Obstacle constraints
            self.setupObstacleConstraints()

    def setupLinearCosts(self):
        """Setup the linear (Quadratic) costs of form: (sum_i ||x_i-x_ref_i||_{Qs}^2 + ||u_i-u_ref_i||_{R}^2) + ||x_N-x_ref_N||_{Qt}^2."""
        self.cost_type = "linear"
        Qs_pos = self.cost_info.get("Qs_pos", 1)
        Qs_pos = np.array([Qs_pos, Qs_pos, Qs_pos])
        Qs_vel = self.cost_info.get("Qs_vel", 0.1)
        Qs_vel = np.array([Qs_vel, Qs_vel, Qs_vel])
        Qs_ang = self.cost_info.get("Qs_ang", 0.1)
        Qs_ang = np.array([Qs_ang, Qs_ang, Qs_ang])
        Qs_dang = self.cost_info.get("Qs_dang", 0.1)
        Qs_dang = np.array([Qs_dang, Qs_dang, Qs_dang])
        Qs_quat = self.cost_info.get("Qs_quat", 0.01)
        Qs_quat = np.array([Qs_quat, Qs_quat, Qs_quat, Qs_quat])

        Qt_pos = self.cost_info.get("Qt_pos", Qs_pos)
        Qt_vel = self.cost_info.get("Qt_vel", Qs_vel)
        Qt_ang = self.cost_info.get("Qt_ang", Qs_ang)
        Qt_dang = self.cost_info.get("Qt_dang", Qs_dang)
        Qt_quat = self.cost_info.get("Qt_quat", Qs_quat)

        Rs = self.cost_info.get("Rs", 0.01)
        Rs = np.array([Rs, Rs, Rs, Rs])
        Rd = self.cost_info.get("Rd", 0.01)
        Rd = np.array([Rd, Rd, Rd, Rd])

        if self.baseDynamics == "Euler":
            Qs = np.concatenate([Qs_pos, Qs_vel, Qs_ang, Qs_dang])
            Qt = np.concatenate([Qt_pos, Qt_vel, Qt_ang, Qt_dang])
        elif self.baseDynamics == "Quaternion":
            Qs = np.concatenate([Qs_pos, Qs_vel, Qs_quat, Qs_dang])
            Qt = np.concatenate([Qt_pos, Qt_vel, Qt_quat, Qt_dang])
        else:
            raise ValueError("Base dynamics not recognized.")

        if self.useControlRates:
            Qs = np.concatenate([Qs, Rs])
            Qt = np.concatenate([Qt, Rs])
            R = Rd
        else:
            R = Rs

        self.Qs = np.diag(Qs)
        self.Qt = np.diag(Qt)
        self.R = np.diag(R)
        # print(self.u_ref.shape)
        self.stageCostFunc = self.LQ_stageCost
        self.terminalCostFunc = self.LQ_terminalCost

    def LQ_stageCost(self, x, u, p, x_ref=None, u_ref=None):
        """Compute the LQR cost."""
        return ca.mtimes([(x - x_ref).T, self.Qs, (x - x_ref)]) + ca.mtimes(
            [(u - u_ref).T, self.R, u - u_ref]
        )

    def LQ_terminalCost(self, x, u, p, x_ref=None, u_ref=None):
        """Compute the LQR cost."""
        return ca.mtimes([(x - x_ref).T, self.Qt, (x - x_ref)])

    def updateParameters(self, obs: dict = None, init: bool = False) -> np.ndarray:
        """Update the parameters of the drone/environment controller."""
        # Checks whether gate observation has been updated, replans if needed, and updates the path, dpath, and gate progresses parameters
        updated = False
        if init and self.p is not None:
            self.param_values = np.zeros((self.p.size()[0],))
            self.param_values[self.param_indices["obstacles_pos"]] = self.obstacle_pos[
                :, :-1
            ].flatten()
        elif self.p is not None:
            if np.any(np.not_equal(self.obstacles_visited, obs["obstacles_visited"])):
                self.obstacles_visited = obs["obstacles_visited"]
                self.obstacles_pos = obs["obstacles_pos"]
                self.param_values[self.param_indices["obstacles_pos"]] = self.obstacle_pos[
                    :, :-1
                ].flatten()
                updated = True
        return updated

    def setupQuatConstraints(self):
        """Setup the quaternion constraints (normalization property and euler angle bounds)."""
        # Quaternion constraints (norm = 1)
        quat = self.x[self.state_indices["quat"]]
        quat_norm = ca.norm_2(quat) - 1  # Ensure quaternion is normalized
        quat_norm_lh = np.zeros((1,))  # Lower bound for normalization constraint
        quat_norm_uh = np.zeros((1,))  # Upper bound for normalization constraint
        # Quaternion constraints (euler angle bounds)
        quat_eul = quaternion_to_euler(quat)
        quat_eul_lh = self.eul_ang_lb
        quat_eul_uh = self.eul_ang_ub

        quat_constraints_lh = np.concatenate([quat_norm_lh, quat_eul_lh])
        quat_constraints_uh = np.concatenate([quat_norm_uh, quat_eul_uh])
        # Add the quat constraints to the the constraints
        if self.nl_constr is None:
            self.nl_constr = ca.vertcat(quat_norm, quat_eul)
            self.nl_constr_lh = quat_constraints_lh
            self.nl_constr_uh = quat_constraints_uh
        else:
            self.nl_constr = ca.vertcat(self.nl_constr, quat_norm, quat_eul)
            self.nl_constr_lh = np.concatenate([self.nl_constr_lh, quat_constraints_lh])
            self.nl_constr_uh = np.concatenate([self.nl_constr_uh, quat_constraints_uh])

        self.nl_constr_indices["quat"] = np.arange(
            self.current_nl_constr_index, quat_constraints_lh.__len__()
        )
        self.current_nl_constr_index += quat_constraints_lh.__len__()
        # No parameters required for the quaternion constraints

    def setupObstacleConstraints(self):
        """Setup the obstacle constraints for the drone/environment controller."""
        # Obstacle constraints
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
        # Add the obstacle constraints to the the constraints
        if self.nl_constr is None:
            self.nl_constr = ca.vertcat(*obstacle_constraints)
            self.nl_constr_lh = obstacle_constraints_lh
            self.nl_constr_uh = obstacle_constraints_uh
        else:
            self.nl_constr = ca.vertcat(self.nl_constr, *obstacle_constraints)
            self.nl_constr_lh = np.concatenate([self.nl_constr_lh, obstacle_constraints_lh])
            self.nl_constr_uh = np.concatenate([self.nl_constr_uh, obstacle_constraints_uh])

        self.nl_constr_indices["obstacles"] = np.arange(self.current_nl_constr_index, num_obstacles)
        self.current_nl_constr_index += num_obstacles
        # Add the obstacle parameters to the parameter vector
        self.p = ca.vertcat(self.p, obstacles_pos_sym) if self.p is not None else obstacles_pos_sym
        self.param_indices["obstacles_pos"] = np.arange(
            self.current_param_index, self.current_param_index + num_params_obstacle
        )
        self.current_param_index += num_params_obstacle

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
        self.tot_thrust_lb = -1  # 4 * self.thrust_lb[0]
        self.tot_thrust_ub = +1  # 4 * self.thrust_ub[0]
        # Individual rotor thrust rate limits
        rate_max = (
            (self.thrust_ub[0] - self.thrust_lb[0]) / 0.5
        )  # Assuming the motor can go from 0 to max thrust in 0.1s, loose bounds, only for control rate
        self.thrust_rate_lb = -rate_max * np.ones((4,))
        self.thrust_rate_ub = rate_max * np.ones((4,))
        # State limits (from the observation space, )
        x_y_max = 3.0
        z_max = 2.5
        z_min = 0.07  # 5 cm above the ground, to avoid hitting the ground
        eul_ang_max = 75 / 180 * np.pi  # 85 degrees in radians, reduced to 75 for safety
        large_val = 1e4
        self.pos_lb = np.array([-x_y_max, -x_y_max, z_min]).T
        self.pos_ub = np.array([x_y_max, x_y_max, z_max]).T
        self.vel_lb = np.array([-large_val, -large_val, -large_val]).T
        self.vel_ub = np.array([large_val, large_val, large_val]).T
        self.eul_ang_lb = np.array([-eul_ang_max, -eul_ang_max, -eul_ang_max]).T
        self.eul_ang_ub = np.array([eul_ang_max, eul_ang_max, eul_ang_max]).T
        self.eul_rate_lb = np.array([-large_val, -large_val, -large_val]).T
        self.eul_rate_ub = np.array([large_val, large_val, large_val]).T
        self.w_lb = np.array([-large_val, -large_val, -large_val]).T
        self.w_ub = np.array([large_val, large_val, large_val]).T
        # What are the bounds for the quaternions? Implemented as non-linear constraints
        self.quat_lb = -1e3 * np.ones((4,))
        self.quat_ub = 1e3 * np.ones((4,))

    def setupBoundsAndScals(self):
        """Setup the constraints and scaling factors for the controller."""
        if self.controlType == "Thrusts":
            u_lb = self.thrust_lb
            u_ub = self.thrust_ub
            u_lb_rate = self.thrust_rate_lb
            u_ub_rate = self.thrust_rate_ub
        elif self.controlType == "Torques":
            u_lb = np.array([self.tot_thrust_lb, -1, -1, -1])
            u_ub = np.array([self.tot_thrust_ub, 1, 1, 1])
            u_lb_rate = np.array([np.sum(self.thrust_rate_lb), -20, -20, -20])
            u_ub_rate = np.array([np.sum(self.thrust_rate_ub) * 4, 20, 20, 20])
        elif self.controlType == "MotorRPMs":
            u_lb = self.rpm_lb
            u_ub = self.rpm_ub
            u_lb_rate = np.sqrt(self.thrust_rate_lb / self.ct)
            u_ub_rate = np.sqrt(self.thrust_rate_ub / self.ct)

        if self.baseDynamics == "Euler":
            x_lb = np.concatenate([self.pos_lb, self.vel_lb, self.eul_ang_lb, self.w_lb])
            x_ub = np.concatenate([self.pos_ub, self.vel_ub, self.eul_ang_ub, self.w_ub])
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
        )  # Slack variables on z_pos, w, and quat

        if any(x_ub - x_lb < 0):
            Warning("Some states have upper bounds lower than lower bounds")
        if any(self.x_scal < 1e-4):
            Warning("Some states have scalings close to zero, setting them to 0.1")
            self.x_scal = np.where(np.abs(self.x_scal) < 1e-4, 0.1, self.x_scal)

        self.u_lb = u_lb
        self.u_ub = u_ub
        self.u_scal = self.u_ub - self.u_lb

        self.slackControls = np.arange(0, self.nu)  # All control bounds have slack variables

        if any(u_ub - u_lb < 0):
            Warning("Some controls have upper bounds lower than lower bounds")
        if any(self.u_scal < 1e-4):
            Warning("Some controls have scalings close to zero, setting them to 0.1")
            self.u_scal = np.where(np.abs(self.u_scal) < 1e-4, 0.1, self.u_scal)
