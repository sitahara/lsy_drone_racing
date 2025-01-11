"""Dynamic classes implementations.

This file implements multiple classes that define the dynamics, bounds, nonlinear constraints, and cost functions  of the drone.
The base class defines the general used methods and attributes, while the subclasses implement the specific dynamics and constraints.
Shared utility functions are defined in the utils.py file.
"""

from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
from acados_template.utils import ACADOS_INFTY
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as Rot

from lsy_drone_racing.mpc_utils.utils import (
    W1,
    Rbi,
    W1s,
    W2s,
    dW2s,
    rungeKuttaExpr,
    rungeKuttaFcn,
    shuffleQuat,
    quaternion_conjugate,
    quaternion_product,
    quaternion_rotation,
    quaternion_to_euler,
)


class BaseDynamics(ABC):
    """Abstract base class for dynamics implementations including bounds, nonlinear constraints, and costs for states and controls."""

    def __init__(
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        initial_info: dict,
        dynamics_info: dict = {
            "Interface": "Mellinger",  # Mellinger or Thrust
            "ts": 1 / 60,  # Time step for the discrete dynamics, Used also for MPC
            "n_horizon": 60,  # Number of steps in the MPC horizon, used for external cost funtion: parameter definition
            "ControlType": "Thrusts",  # Thrusts, Torques, or MotorRPMs
            "BaseDynamics": "Euler",  # Euler or Quaternion
            "useAngVel": True,  # Currently only supports True, False if deul_ang is used instead of w
            "useControlRates": False,  # True if u is used as state and du as control
            "Constraints": {"Obstacles": True, "Gates": False},
            "usePredict": True,  # True if the dynamics are predicted into the future
            "t_predict": 0.05,  # in [s] To compensate control delay, the optimization problem is solved for the current state shifted by t_predict into the future with the current controls
        },
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
        self.initial_obs = initial_obs
        self.initial_info = initial_info
        # Unpack the dynamics info
        self.n_horizon = dynamics_info.get("n_horizon", 60)
        self.ts = dynamics_info.get("ts", 1 / 60)
        self.t_predict = dynamics_info.get("t_predict", 0.05)
        self.usePredict = dynamics_info.get("usePredict", True)
        self.interface = dynamics_info.get("Interface", "Mellinger")
        if self.interface not in ["Mellinger", "Thrust"]:
            raise ValueError("Currently only Mellinger or Thrust interfaces are supported.")
        self.baseDynamics = dynamics_info.get("BaseDynamics", "Euler")
        if self.baseDynamics not in ["Euler", "Quaternion"]:
            raise ValueError("Currently only Euler or Quaternion formulations are supported.")
        self.controlType = dynamics_info.get("ControlType", "Thrusts")
        if self.controlType not in ["Thrusts", "Torques", "MotorRPMs"]:
            raise ValueError("Currently only Thrusts, Torques, or MotorRPMs are supported.")
        self.useControlRates = dynamics_info.get("useControlRates", False)
        self.useAngVel = dynamics_info.get("useAngVel", True)
        if not self.useAngVel:
            self.useAngVel = True
            raise Warning("Currently only useAngVel=True is supported.")
        # Constraints
        self.useObstacleConstraints = dynamics_info.get("Constraints", {}).get("Obstacles", True)
        self.useGateConstraints = dynamics_info.get("Constraints", {}).get("Gates", False)

        # Setup basic parameters, dynamics, bounds, scaling, and constraints
        self.setupNominalParameters()
        self.setupBaseBounds()
        if self.baseDynamics == "Euler":
            self.baseEulerDynamics()
        elif self.baseDynamics == "Quaternion":
            self.baseQuaternionDynamics()
        self.setupBoundsAndScals()

        self.current_param_index = 0
        self.param_indices = {}
        self.current_nl_constr_index = 0
        self.nl_constr_indices = {}
        self.nl_constr = None
        self.setupBaseConstraints()
        # self.setupBaseCosts()
        self.initParamValues()
        self.setupCasadiFunctions()
        self.last_u = None  # Used for the control rates and the prediction

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
                self.last_u = np.zeros((4,))
            x = np.concatenate([x, self.last_u])
        # Predict the state into the future if self.usePredict is True
        if self.usePredict and self.last_u is not None:
            # fd_predict is a discrete dynamics function (RK4) with the time step t_predict
            x = self.fd_predict(x, self.last_u)
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
                tot_thrust = np.sum(action)
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

            acc_world = (vel - x_sol[:, 0][self.state_indices["vel"]]) / self.ts
            yaw = action[8]
            action = np.concatenate([pos, vel, acc_world, [yaw], w])
        return action.flatten()

    def setupCasadiFunctions(self):
        """Setup explicit, implicit, and discrete dynamics functions."""
        # Continuous dynamic function
        if self.useControlRates:
            self.fc = ca.Function("fc", [self.x, self.u], [self.dx], ["x", "du"], ["dx"])
        else:
            self.fc = ca.Function("fc", [self.x, self.u], [self.dx], ["x", "u"], ["dx"])
        self.fd = rungeKuttaFcn(self.nx, self.nu, self.ts, self.fc)
        self.fd_predict = rungeKuttaFcn(self.nx, self.nu, self.t_predict, self.fc)
        # Discrete dynamic expression and dynamic function
        self.dx_d = rungeKuttaExpr(self.x, self.u, self.ts, self.fc)

        self.xdot = ca.MX.sym("xdot", self.nx)
        # Continuous implicit dynamic expression
        self.f_impl = self.xdot - self.dx

    def setupNominalParameters(self):
        """Setup the nominal parameters of the drone/environment/controller."""
        self.mass = self.initial_info.get("drone_mass", 0.027)
        self.g = 9.0
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

        # self.obstacles_pos = self.initial_obs.get("obstacles_pos", np.zeros((6, 3)))
        # self.obstacles_in_range = self.initial_obs.get("obstacles_in_range", np.zeros((6,)))
        # self.gates_pos = self.initial_obs.get("gates_pos", np.zeros((6, 3)))
        # self.gates_rpy = self.initial_obs.get("gates_rpy", np.zeros((6, 3)))
        # self.gates_in_range = self.initial_obs.get("gates_in_range", np.zeros((6,)))

    def baseEulerDynamics(self):
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
            torques = ca.MX.sym("u", 3)
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
            torques = ca.MX.sym("u", 3)
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

    def setupBaseConstraints(self):
        """Setup the basic constraints for the drone/environment controller."""
        # self.p = None
        # self.param_indices = {}

        if self.baseDynamics == "Quaternion":
            self.setupQuatConstraints()
            # No parameters required for the quaternion constraints
            if self.current_nl_constr_index == 0:
                self.nl_constr = self.quat_constraints
                self.nl_constr_lh = self.quat_constraints_lh
                self.nl_constr_uh = self.quat_constraints_uh
            else:
                self.nl_constr = ca.vertcat(self.nl_constr, self.quat_constraints)
                self.nl_constr_lh = np.concatenate([self.nl_constr_lh, self.quat_constraints_lh])
                self.nl_constr_uh = np.concatenate([self.nl_constr_uh, self.quat_constraints_uh])
            self.nl_constr_indices["quat"] = np.arange(
                self.current_nl_constr_index, self.quat_constraints.size()[0]
            )
            self.current_nl_constr_index += self.quat_constraints.size()[0]

        if self.useObstacleConstraints:
            self.setupObstacleConstraints()
            if self.current_param_index == 0:
                self.p = self.p_obst
            else:
                self.p = ca.vertcat(self.p, self.p_obst)
            self.param_indices["p_obst"] = np.arange(
                self.current_param_index, self.p_obst.size()[0]
            )
            self.current_param_index += self.p_obst.size()[0]
            if self.nl_constr is None:
                self.nl_constr = self.obstacle_constraints
                self.nl_constr_lh = self.obstacle_constraints_lh
                self.nl_constr_uh = self.obstacle_constraints_uh
            else:
                self.nl_constr = ca.vertcat(self.nl_constr, self.obstacle_constraints)
                self.nl_constr_lh = np.concatenate(
                    [self.nl_constr_lh, self.obstacle_constraints_lh]
                )
                self.nl_constr_uh = np.concatenate(
                    [self.nl_constr_uh, self.obstacle_constraints_uh]
                )
            self.nl_constr_indices["obstacles"] = np.arange(
                self.current_nl_constr_index, self.obstacle_constraints.size()[0]
            )
            self.current_nl_constr_index += self.obstacle_constraints.size()[0]

    def setupBaseCosts(self):
        raise NotImplementedError

    def initParamValues(self):
        """Initialize the parameter values for the drone/environment controller."""
        self.param_values = np.zeros((self.p.size()[0],))
        self.param_values[self.param_indices["p_obst"]] = self.obstacle_pos.flatten()

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

        # Combine constraints into a single vector
        self.quat_constraints = ca.vertcat(quat_norm, quat_eul)
        self.quat_constraints_lh = ca.vertcat(quat_norm_lh, quat_eul_lh)
        self.quat_constraints_uh = ca.vertcat(quat_norm_uh, quat_eul_uh)

    def setupObstacleConstraints(self):
        """Setup the obstacle constraints for the drone/environment controller."""
        # Obstacle constraints
        self.obstacle_pos = self.initial_obs.get("obstacles_pos", np.zeros((4, 3)))
        self.obstacle_diameter = self.initial_info.get("obstacle_diameter", 0.1)
        self.obstacle_visited = self.initial_obs.get("obstacle_visited", np.zeros((4,)))
        num_obstacles = self.obstacle_pos.shape[0]
        num_params_per_obstacle = 3
        num_params_obstacle = num_obstacles * num_params_per_obstacle
        self.p_obst = ca.MX.sym("p_obst", num_params_obstacle)
        obstacle_constraints = []
        for k in range(num_obstacles):
            obstacle_constraints.append(
                ca.norm_2(
                    self.x[self.state_indices["pos"][:num_params_per_obstacle]]
                    - self.p_obst[k * num_params_per_obstacle : (k + 1) * num_params_per_obstacle]
                )
                - self.obstacle_diameter
            )
        self.obstacle_constraints = ca.vertcat(*obstacle_constraints)
        self.obstacle_constraints_lh = np.zeros((num_obstacles,))
        self.obstacle_constraints_uh = np.ones((num_obstacles,)) * 1e9

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
        self.tot_thrust_lb = 4 * self.thrust_lb[0]
        self.tot_thrust_ub = 4 * self.thrust_ub[0]
        # Individual rotor thrust rate limits
        rate_max = (
            (self.thrust_ub[0] - self.thrust_lb[0]) / 0.1
        )  # Assuming the motor can go from 0 to max thrust in 0.1s, loose bounds, only for control rate
        self.thrust_rate_lb = -rate_max * np.ones((4,))
        self.thrust_rate_ub = rate_max * np.ones((4,))
        # State limits (from the observation space, )
        x_y_max = 3.0
        z_max = 2.5
        z_min = 0.05  # 5 cm above the ground, to avoid hitting the ground
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
        # What are the bounds for the quaternions?
        # TODO: Review quaternion bounds or maybe implement them as non-linear constraints
        # Convert Euler angle bounds to quaternions
        # Note: Implementing quaternion bounds as non-linear constraints
        quat_max = Rot.from_euler("xyz", self.eul_ang_ub).as_quat()
        quat_min = Rot.from_euler("xyz", self.eul_ang_lb).as_quat()
        self.quat_lb = np.array([quat_min[0], quat_min[1], quat_min[2], quat_min[3]])
        self.quat_ub = np.array([quat_max[0], quat_max[1], quat_max[2], quat_max[3]])
        self.quat_lb = -1e3 * np.ones((4,))  # remove bounds (enforced by constraints)
        self.quat_ub = 1e3 * np.ones((4,))  # remove bounds (enforced by constraints)

    def setupBoundsAndScals(self):
        """Setup the constraints and scaling factors for the controller."""
        if self.controlType == "Thrusts":
            u_lb = self.thrust_lb
            u_ub = self.thrust_ub
            u_lb_rate = self.thrust_rate_lb
            u_ub_rate = self.thrust_rate_ub
        elif self.controlType == "Torques":
            u_lb = np.array([self.tot_thrust_lb, -0.2, -0.2, -0.2])
            u_ub = np.array([self.tot_thrust_ub, 0.2, 0.2, 0.2])
            u_lb_rate = np.array([self.thrust_rate_lb * 4, -2, -2, -2])
            u_lb_rate = np.array([self.thrust_rate_ub * 4, 2, 2, 2])
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
        self.noSlackStates = np.concatenate(
            [self.state_indices["pos"][:-1], self.state_indices["vel"]]
        )  # Slack variables on z_pos, w, and quat

        if any(x_ub - x_lb < 0):
            Warning("Some states have upper bounds lower than lower bounds")
        if any(self.x_scal < 1e-4):
            Warning("Some states have scalings close to zero, setting them to 0.1")
            self.x_scal = np.where(np.abs(self.x_scal) < 1e-4, 0.1, self.x_scal)

        self.u_lb = u_lb
        self.u_ub = u_ub
        self.u_scal = self.u_ub - self.u_lb

        self.noSlackControls = []  # All control bounds have slack variables

        if any(u_ub - u_lb < 0):
            Warning("Some controls have upper bounds lower than lower bounds")
        if any(self.u_scal < 1e-4):
            Warning("Some controls have scalings close to zero, setting them to 0.1")
            self.u_scal = np.where(np.abs(self.u_scal) < 1e-4, 0.1, self.u_scal)


class ThrustTimeDynamics(BaseDynamics):
    def __init__(self, initial_obs, initial_info, dynamics_info=None):
        super().__init__(initial_obs, initial_info, dynamics_info)
        if not self.useAngVel:
            Warning("Time Dynamics uses angular velocity")
            self.useAngVel = True
        if self.useControlRates:
            Warning("Time Dynamics does not use control rates")
            self.useControlRates = False
        self.setup_dynamics()
        self.nx = self.x.size()[0]
        self.nu = self.u.size()[0]
        self.ny = self.nx + self.nu
        self.setupBoundsAndScals()
        super().setupCasadiFunctions()
        super().setupObstacleConstraints()

    def setup_dynamics(self):
        self.modelName = "ThrustEulerTime"
        # Define the state variables
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        eul_ang = ca.MX.sym("eul_ang", 3)
        w = ca.MX.sym("w", 3)
        t = ca.MX.sym("t", 1)
        gate_passed = ca.MX.sym("gate_passed", 1)
        # Combine state variables into a single vector
        x = ca.vertcat(pos, vel, eul_ang, w, t)
        self.x_eq = np.zeros((14,))
        # Create a dictionary for indexing
        self.state_indices = {
            "pos": np.arange(0, 3),
            "vel": np.arange(3, 6),
            "eul_ang": np.arange(6, 9),
            "w": np.arange(9, 12),
            "time": 12,
            "gate_passed": 13,
        }

        # Define the control variables
        u = ca.MX.sym("f", 4)  # [f1, f2, f3, f4]
        eq = 0.25 * self.mass * self.g
        self.u_eq = eq * np.ones((4,))

        # Define helper variables
        thrust_total = (u[0] + u[1] + u[2] + u[3]) / self.mass
        beta = self.arm_length / ca.sqrt(2.0)
        torques = ca.vertcat(
            beta * (u[0] + u[1] - u[2] - u[3]),
            beta * (-u[0] + u[1] + u[2] - u[3]),
            self.gamma * (u[0] - u[1] + u[2] - u[3]),
        )
        # Define the distance to the next gate
        distance_to_next_gate = ca.norm_2(
            ca.vertcat(pos, eul_ang) - self.p[self.param_indices["next_gate"]]
        )
        # Once the distance is less than 0.05, the gate is considered passed and the gate passed variable increases
        gate_passed_update = ca.if_else(distance_to_next_gate < 0.05, 1 / self.ts, 0)
        # In cost function, it is checked if the gate passed variable is greater than 0 to determine which gate is the next one
        # The gate passed variable is reset to 0 in each iteration
        # Define the dynamics
        dx = ca.vertcat(
            vel,  # Linear velocity
            ca.vertcat(0, 0, -self.g)
            + Rbi(eul_ang[0], eul_ang[1], eul_ang[2])
            @ ca.vertcat(0, 0, thrust_total),  # Linear acceleration
            W2s(eul_ang) @ w,  # Angular velocity
            self.J_inv @ (torques - (ca.skew(w) @ self.J @ w)),  # Angular acceleration
            1,  # Time derivative
            gate_passed_update,  # Gate passed update
        )

        self.x = x
        self.dx = dx
        self.u = u

    def setupBoundsAndScals(self):
        x_lb = np.concatenate([self.pos_lb, self.vel_lb, self.eul_ang_lb, self.eul_rate_lb, [0]])
        x_ub = np.concatenate([self.pos_ub, self.vel_ub, self.eul_ang_ub, self.eul_rate_ub, [1e9]])
        u_lb = np.array([self.thrust_lb, self.thrust_lb, self.thrust_lb, self.thrust_lb])
        u_ub = np.array([self.thrust_ub, self.thrust_ub, self.thrust_ub, self.thrust_ub])

        self.x_lb = x_lb
        self.x_ub = x_ub
        self.x_scal = self.x_ub - self.x_lb
        self.u_lb = u_lb
        self.u_ub = u_ub
        self.u_scal = self.u_ub - self.u_lb

    def transformState(self, x, n_step):
        w = W1(x[6:9]) @ x[9:12]
        x = np.concatenate([x[:9], w.flatten(), [n_step * self.ts]])
        self.current_state = x
        return x

    def transformAction(self, action):
        """Transforms the solution of the MPC controller to the format required for the Mellinger interface."""
        pos = action[:3]
        vel = action[3:6]
        eul_ang = action[6:9]
        w = action[9:12]

        # if self.useControlRates:
        #     action = action[: -self.nu]
        acc_world = (vel - self.current_state[3:6]) / self.ts
        yaw = action[8]
        return np.concatenate([pos, vel, acc_world, [yaw], w])


class MPCCppDynamics(BaseDynamics):
    def __init__(
        self,
        initial_obs,
        initial_info,
        dynamics_info: dict = {
            "Interface": "Mellinger",  # Mellinger or Thrust
            "ts": 1 / 60,  # Time step for the discrete dynamics, Used also for MPc
            "ControlType": "Thrusts",  # Thrusts, Torques, or MotorRPMs
            "BaseDynamics": "Quaternion",  # Euler or Quaternion
            "useAngVel": True,  # Currently only supports True, False if deul_ang is used instead of w
            "useControlRates": True,  # True if u is used as state and du as control
        },
    ):
        super().__init__(initial_obs, initial_info, dynamics_info)
        super().setupCasadiFunctions()

    def setup_dynamics(self):
        self.modelName = "MPCCpp"
        pos = ca.MX.sym("pos", 3)  # position in world frame
        vel = ca.MX.sym("vel", 3)  # velocity in world frame
        quat = ca.MX.sym("quat", 4)  # [qx, qy, qz,qw] quaternion rotation from body to world
        w = ca.MX.sym("w", 3)  # angular velocity in body frame

        u = ca.MX.sym("f", 4)  # individual rotor thrusts
        du = ca.MX.sym("du", 4)  # individual rotor thrust rates

        beta = self.arm_length / ca.sqrt(2.0)
        torques = ca.vertcat(
            beta * (u[0] + u[1] - u[2] - u[3]),
            beta * (-u[0] + u[1] + u[2] - u[3]),
            self.gamma * (u[0] - u[1] + u[2] - u[3]),
        )  # tau_x, tau_y, tau_z
        thrust_total = ca.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) / self.mass)

        # Define the dynamics
        dpos = vel
        dvel = self.gv + Rot.from_quat(quat).apply(thrust_total)
        dquat = 0.5 * quaternion_product(quat, ca.vertcat(w, 0))
        dw = self.J_inv @ (torques - (ca.skew(w) @ self.J @ w))
        du = du
        x = ca.vertcat(pos, vel, quat, w, u)
        self.state_indices = {
            "pos": np.arange(0, 3),
            "vel": np.arange(3, 6),
            "quat": np.arange(6, 10),
            "w": np.arange(10, 13),
            "u": np.arange(13, 17),
        }
        dx = ca.vertcat(dpos, dvel, dquat, dw, du)
        self.x = x
        self.dx = dx
        self.u = du
        self.control_indices = {"du": np.arange(0, 4)}

        u_eq = 0.25 * self.mass * self.g * np.ones((4,))
        self.x_eq = np.concatenate([np.zeros(13), u_eq])  # Equilibrium state
        self.u_eq = np.zeros((4,))  # Equilibrium control

    def setupBoundsAndScals(self):
        x_lb = np.concatenate([self.pos_lb, self.vel_lb, self.quat_lb, self.w_lb, self.thrust_lb])
        x_ub = np.concatenate([self.pos_ub, self.vel_ub, self.quat_ub, self.w_ub, self.thrust_ub])
        u_lb = self.thrust_rate_lb
        u_ub = self.thrust_rate_ub
        self.noSlackStates = np.diff1d(
            np.arange(0, len(x_lb), 1),
            np.concatenate(
                self.state_indices["u"],
                self.state_indices["w"],
                self.state_indices["vel"],
                self.state_indices["pos"][:-1],
            ),  # no slack for thurst, vel, w, pos_x, pos_y
        )
        self.noSlackControls = np.diff1d(
            np.arange(0, len(u_lb), 1), []
        )  # All controls have slack variables

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
