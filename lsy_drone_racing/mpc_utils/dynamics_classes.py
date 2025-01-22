"""Dynamic classes implementations.

This file implements multiple classes that define the dynamics, bounds, nonlinear constraints, and cost functions  of the drone.
The base class defines the general used methods and attributes, while the subclasses implement the specific dynamics and constraints.
Shared utility functions are defined in the utils.py file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
from acados_template.utils import ACADOS_INFTY
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as Rot
from lsy_drone_racing.mpc_utils.Planner.PathPlanner import PathPlanner, HermiteSplinePathPlanner
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
    quaternion_to_rotation_matrix,
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
            "usePredict": False,  # True if the dynamics are predicted into the future
            "t_predict": 0.05,  # in [s] To compensate control delay, the optimization problem is solved for the current state shifted by t_predict into the future with the current controls
            "OnlyBaseInit": False,  # True if only the basic initialization is required
            "useDrags": False,  # True if the drag forces are included in the dynamics
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

        self.useDrags = dynamics_info.get("useDrags", False)

        # Constraints
        self.useObstacleConstraints = dynamics_info.get("Constraints", {}).get("Obstacles", True)
        # if self.useObstacleConstraints:
        self.obstacle_pos = self.initial_obs.get("obstacles_pos", np.zeros((4, 3)))
        self.obstacle_diameter = self.initial_info.get("obstacle_diameter", 0.1)
        self.obstacles_visited = self.initial_obs.get("obstacles_visited", np.zeros((4,)))
        self.gates_pos = self.initial_obs.get("gates_pos", np.zeros((4, 3)))
        self.gates_rpy = self.initial_obs.get("gates_rpy", np.zeros((4, 3)))
        self.gates_visited = self.initial_obs.get("gates_visited", np.zeros((4,)))
        self.useGateConstraints = dynamics_info.get("Constraints", {}).get("Gates", False)
        self.last_u = None  # Used for the control rates and the prediction

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

        if not dynamics_info.get("OnlyBaseInit", False):
            if self.baseDynamics == "Euler":
                self.baseEulerDynamics()
            elif self.baseDynamics == "Quaternion":
                self.baseQuaternionDynamics()
            self.setupBoundsAndScals()
            self.setupNLConstraints()
            self.setupCosts()
            self.updateParameters(init=True)
            self.setupCasadiFunctions()

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
        Rquat = quaternion_to_rotation_matrix(quat)

        dpos = vel
        if self.useDrags:
            dvel = (
                self.gv
                + quaternion_rotation(quat, thrust_total)
                - ca.mtimes(Rquat, self.DragMat, Rquat.T, vel)
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

    def updateParameters(self, obs: dict = None, init: bool = False) -> np.ndarray:
        """Update the parameters of the drone/environment controller."""
        # Checks whether gate observation has been updated, replans if needed, and updates the path, dpath, and gate progresses parameters
        if init:
            self.param_values = np.zeros((self.p.size()[0],))
            self.param_values[self.param_indices["p_obst"]] = self.obstacle_pos.flatten()
        else:
            if np.any(np.not_equal(self.obstacles_visited, obs["obstacles_visited"])):
                self.obstacles_visited = obs["obstacles_visited"]
                self.obstacles_pos = obs["obstacles_pos"]
                self.param_values[self.param_indices["p_obst"]] = self.obstacle_pos.flatten()
            # Return the updated parameter values for the acados interface
            return self.param_values

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
            self.current_nl_constr_index, quat_constraints_lh.size()[0]
        )
        self.current_nl_constr_index += quat_constraints_lh.size()[0]
        # No parameters required for the quaternion constraints

    def setupObstacleConstraints(self):
        """Setup the obstacle constraints for the drone/environment controller."""
        # Obstacle constraints
        num_obstacles = self.obstacle_pos.shape[0]
        num_params_per_obstacle = 2
        num_params_obstacle = num_obstacles * num_params_per_obstacle
        # Parameters for the obstacle positions
        p_obst = ca.MX.sym("p_obst", num_params_obstacle)
        # Extract the position of the drone
        pos = self.x[self.state_indices["pos"]]
        obstacle_constraints = []
        for k in range(num_obstacles):
            obstacle_constraints.append(
                ca.norm_2(
                    pos[:num_params_per_obstacle]
                    - p_obst[k * num_params_per_obstacle : (k + 1) * num_params_per_obstacle]
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

        self.nl_constr_indices["obstacles"] = np.arange(
            self.current_nl_constr_index, obstacle_constraints.size()[0]
        )
        self.current_nl_constr_index += obstacle_constraints.size()[0]
        # Add the obstacle parameters to the parameter vector
        self.p = ca.vertcat(self.p, p_obst) if self.p is not None else p_obst
        self.param_indices["p_obst"] = np.arange(
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
        self.slackStates = np.concatenate(
            self.state_indices["pos"], self.state_indices["w"]
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
            "Constraints": {"Obstacles": True, "Gates": False},  # Obstacle and gate constraints
            "OnlyBaseInit": True,  # Only setup the base dynamics and not the full dynamics
            "IncludeDrags": False,  # Include drag forces in the dynamics
        },
        cost_info: dict = {
            "Ql": 1,  # lag error weights,
            "Qc": 1,  # contour error weights,
            "Qw": 1,  # angular velocity weights,
            "Qmu": 1,  # progress rate weights,
            "Rdf": 0.1,  # rotor rate weights,
            "Rdprogress": 0.1,  # progress rate weights,
        },
    ):
        self.cost_info = cost_info
        super().__init__(initial_obs, initial_info, dynamics_info)
        # Setup the Dynamics, returns expressions for the continuous dynamics
        self.setup_dynamics()
        # Defines the bounds and scaling factors for the states and controls, and which states/controls have slack variables
        self.setupBoundsAndScals()
        # Init the path Planner
        self.pathPlanner = HermiteSplinePathPlanner(
            initial_obs["gates_pos"],
            initial_obs["gates_rpy"],
            initial_obs["pos"],
            initial_obs["rpy"],
            self.x[self.state_indices["progress"]],  # Pass the progress variable
        )
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
                - ca.mtimes(Rquat, self.DragMat, Rquat.T, vel)
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
        self.dx = dx
        self.u = u
        # Equilibrium state and control
        f_eq = 0.25 * self.mass * self.g * np.ones((4,))
        self.x_eq = np.concatenate([np.zeros((13,)), f_eq, np.zeros((2,))])  # Equilibrium state
        self.u_eq = np.zeros((5,))  # Equilibrium control

    def setupNLConstraints(self):
        """Setup the nonlinear constraints for the drone/environment controller."""
        super().setupQuatConstraints()
        super().setupObstacleConstraints()

        self.setupTunnelConstraints()

        self.updateParameters(obs=self.initial_obs, init=True)

    def updateParameters(self, obs: dict = None, init: bool = False) -> np.ndarray:
        """Update the parameters of the drone/environment controller."""
        # Checks whether gate observation has been updated, replans if needed, and updates the path, dpath, and gate progresses parameters
        if init:
            self.param_values = np.zeros((self.p.size()[0],))
            self.param_values[self.param_indices["p_obst"]] = self.obstacle_pos.flatten()
            self.param_values[self.param_indices["p_gate_progress"]] = (
                self.pathPlanner.gate_progresses
            )
        else:
            if np.any(np.not_equal(self.gates_visited, obs["gates_visited"])):
                self.gates_visited = obs["gates_visited"]
                self.gates_pos = obs["gates_pos"]
                self.gates_rpy = obs["gates_rpy"]
                self.pathPlanner.update_gates(self.gates_pos, self.gates_rpy)
                self.param_values[self.param_indices["p_gate_progress"]] = (
                    self.pathPlanner.gate_progresses
                )
            # Checks whether obstacle observation has been updated, updates the obstacle positions
            if np.any(np.not_equal(self.obstacles_visited, obs["obstacles_visited"])):
                self.obstacles_visited = obs["obstacles_visited"]
                self.obstacles_pos = obs["obstacles_pos"]
                self.param_values[self.param_indices["p_obst"]] = self.obstacle_pos.flatten()
            # Return the updated parameter values for the acados interface
            return self.param_values

    def setupTunnelConstraints(self):
        """Setup the tunnel constraints for the drone/environment controller."""
        # Progress along the path state
        progress = self.x[self.state_indices["progress"]]
        pos = self.x[self.state_indices["pos"]]
        # Nominal tunnel width = tunnel height
        Wn = self.Wn
        # Tunnel width = tunnel height at the gate
        Wgate = self.Wgate
        num_gates = self.gates_pos.size()[0]
        # Parameter for the gate progresses, e.g., at which progress the respective is gate is reached. Exaple: [0.1, 0.3, 0.5, 0.7]
        p_gate_progress = ca.MX.sym("gate_progress", num_gates)

        def getTunnelWidth(gate_progress: ca.MX, progress: ca.MX) -> ca.MX:
            """Calculate the tunnel width at the current progress."""
            # Calculate the progress distance to the nearest gate
            d = ca.fmin(ca.fabs(gate_progress - progress))
            k = 10  # Steepness of the transition
            x0 = 0.1  # Midpoint of the transition
            sigmoid = 1 / (1 + ca.exp(-k * (d - x0)))
            return Wn + (Wgate - Wn) * sigmoid

        W = getTunnelWidth(p_gate_progress, progress)
        H = W  # Assuming W(θk) = H(θk)

        # Symbolic functions for the path and its derivative
        path_func = self.pathPlanner.path_func(progress)
        dpath_func = self.pathPlanner.dpath_func(progress)

        t = dpath_func / ca.norm_2(dpath_func)  # Normalized Tangent vector at the current progress
        # Compute the normal vector n (assuming the normal is in the xy-plane)
        n = ca.vertcat(-t[1], t[0], 0)
        # Compute the binormal vector b
        b = ca.cross(t, n)

        pd = path_func  # Position of the path at the current progress
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
            self.current_nl_constr_index, tunnel_constraints.size()[0]
        )
        self.current_nl_constr_index += tunnel_constraints.size()[0]

        # Add the path, dpath, and gate_progress parameters to the parameter vector
        if self.p is None:
            self.p = p_gate_progress
        else:
            self.p = ca.vertcat(self.p, p_gate_progress)
        self.param_indices["p_gate_progress"] = np.arange(
            self.current_param_index, p_gate_progress.size()[0]
        )
        self.current_param_index += p_gate_progress.size()[0]

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
        self.nsx = self.slackStates.size()[0]

        self.slackControls = np.concatenate(
            [self.control_indices["df"], self.control_indices["ddprogress"]]
        )
        self.nsu = self.slackControls.size()[0]

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
        self.cost_dict = {}
        cost_info = self.cost_info

        self.cost_dict["cost_type"] = "external"
        self.cost_dict["cost_type_e"] = "external"
        # Lag error weights
        Ql = cost_info.get("Ql", 1)
        self.cost_dict["Ql"] = ca.diag([Ql, Ql, Ql])
        # Contour error weights
        Qc = cost_info.get("Qc", 1)
        self.cost_dict["Qc"] = ca.diag([Qc, Qc, Qc])
        # Body Angular velocity weights
        Qw = cost_info.get("Qw", 1)
        self.cost_dict["Qw"] = ca.diag([Qw, Qw, Qw])
        # Progress rate weights
        self.cost_dict["Qmu"] = cost_info.get("Qmu", 1)
        # Thrust rate weights
        Rdf = cost_info.get("Rdf", 1)
        self.cost_dict["Rdf"] = ca.diag([Rdf, Rdf, Rdf, Rdf])
        # Progress rate weights
        self.cost_dict["Rdprogress"] = cost_info.get("Rdprogress", 1)
        # Define the cost function
        self.cost_dict["stage_cost"] = self.MPCC_stage_cost
        self.cost_dict["terminal_cost"] = self.MPCC_stage_cost  # use zero for u

    def MPCC_stage_cost(self, x, u, p):
        pos = x[self.state_indices["pos"]]
        w = x[self.state_indices["w"]]
        progress = x[self.state_indices["progress"]]
        dprogress = x[self.state_indices["dprogress"]]
        df = u[self.control_indices["df"]]

        # Desired position and tangent vector on the path
        path = self.pathPlanner.path_func  # Unpack the path function
        dpath = self.pathPlanner.dpath_func  # Unpack the path gradient function
        pd = path(progress)  # Desired position on the path
        tangent_line = dpath(progress)  # Tangent vector of the path at the current progress
        tangent_line = tangent_line / ca.norm_2(tangent_line)  # Normalize the tangent vector
        pos_err = pos - pd  # Error between the current position and the desired position

        # Lag error
        lag_err = ca.mtimes([ca.dot(pos_err, tangent_line), tangent_line])
        lag_cost = ca.mtimes([lag_err.T, self.cost_dict["Ql"], lag_err])

        # Contour error
        contour_err = pos_err - lag_err
        contour_cost = ca.mtimes([contour_err.T, self.cost_dict["Qc"], contour_err])

        # Body angular velocity cost
        w_cost = ca.mtimes([w.T, self.cost_dict["Qw"], w])

        # Progress rate cost
        dprogress_cost_L2 = ca.mtimes([dprogress.T, self.cost_dict["Rdprogress"], dprogress])

        # Progress rate cost
        dprogress_cost = -dprogress * self.cost_dict["Qmu"]

        # Thrust rate cost
        thrust_rate_cost = ca.mtimes([df.T, self.cost_dict["Rdf"], df])

        # Total stage cost
        stage_cost = (
            lag_cost + contour_cost + w_cost + dprogress_cost + dprogress_cost_L2 + thrust_rate_cost
        )

        return stage_cost
