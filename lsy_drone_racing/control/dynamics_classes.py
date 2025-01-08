from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
from acados_template.utils import ACADOS_INFTY
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as Rot

from lsy_drone_racing.control.utils import W1, Rbi, W1s, W2s, dW2s, rungeKuttaExpr, rungeKuttaFcn


class BaseDynamics(ABC):
    """Abstract base class for dynamics implementations including basic bounds for states and controls."""

    def __init__(
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        initial_info: dict,
        dynamics_info: dict = {"useAngVel": False, "useControlRates": False},
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
        self.setupNominalParameters()
        self.setupBaseBounds()
        self.useAngVel = dynamics_info.get("useAngVel", False)
        self.ts = dynamics_info.get("ts", 1 / 60)
        self.useControlRates = dynamics_info.get("useControlRates", False)
        self.setupOptimizationParameters()

    @abstractmethod
    def setup_dynamics(self):
        """Setup the dynamics for the controller."""
        return NotImplementedError

    @abstractmethod
    def setupBoundsAndScals(self):
        """Setup the constraints for the controller."""
        return NotImplementedError

    @abstractmethod
    def transformState(self, x: np.ndarray, last_u: np.ndarray) -> np.ndarray:
        """Transforms observations from the environment to the respective states used in the dynamics."""
        return NotImplementedError

    @abstractmethod
    def transformAction(self, action: np.ndarray) -> np.ndarray:
        """Transforms optimizer solutions to controller inferfaces (Mellinger or Thrust)."""
        return NotImplementedError

    def setupCasadiFunctions(self):
        """Setup explicit, implicit, and discrete dynamics functions."""
        # Continuous dynamic function
        if self.useControlRates:
            self.fc = ca.Function("fc", [self.x, self.u], [self.dx], ["x", "du"], ["dx"])
        else:
            self.fc = ca.Function("fc", [self.x, self.u], [self.dx], ["x", "u"], ["dx"])
        self.fd = rungeKuttaFcn(self.nx, self.nu, self.ts, self.fc)
        # Discrete dynamic expression and dynamic function
        self.dx_d = rungeKuttaExpr(self.x, self.u, self.ts, self.fc)

        self.xdot = ca.MX.sym("xdot", self.nx)
        # Continuous implicit dynamic expression
        self.f_impl = self.xdot - self.dx

    def setupBaseOptimizationVariables(self):
        """Setup the base optimization variables for the drone/environment controller."""
        # TODO: NOT IMPLEMENTED YET
        # All state variables
        # Use always
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        eul_ang = ca.MX.sym("eul_ang", 3)
        # Use if useAngVel is True
        w = ca.MX.sym("w", 3)
        # Use if useAngVel is False
        deul_ang = ca.MX.sym("deul_ang", 3)
        # Use if quaternion dynamics
        quat = ca.MX.sym("quat", 4)
        # Use if time is used
        t = ca.MX.sym("time", 1)
        # Use if gate costs are used
        current_gate = ca.MX.sym("current_gate", 1)

        # All control variables
        # Use if thrust dynamics are used
        if self.useThrusts:
            u = ca.MX.sym("f", 4)
            torques = ca.vertcat(
                self.beta * (u[0] + u[1] - u[2] - u[3]),
                self.beta * (-u[0] + u[1] + u[2] - u[3]),
                self.gamma * (u[0] - u[1] + u[2] - u[3]),
            )
            tot_thrust = ca.sum1(u) / self.mass
        else:
            tot_thrust = ca.MX.sym("thrust", 1)
            torques = ca.MX.sym("torques", 3)
            u = ca.vertcat(tot_thrust, torques)
        # Use if control rates are used
        du = ca.MX.sym("du", 4)

        # All state derivatives
        # Use always
        dpos = vel
        dvel = ca.vertcat(0, 0, -self.g) + Rbi(eul_ang[0], eul_ang[1], eul_ang[2]) @ ca.vertcat(
            0, 0, tot_thrust / self.mass
        )
        # Use if useAngVel is True
        if self.useAngVel:
            deul_ang = W2s(eul_ang) @ w
        else:
            w = W1s(eul_ang) @ deul_ang

    def setupNominalParameters(self):
        """Setup the unchanging parameters of the drone/environment controller."""
        self.mass = self.initial_info.get("drone_mass", 0.027)
        self.g = 9.81
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

    def setupOptimizationParameters(self):
        """Setup the optimization parameters for the drone/environment controller."""
        # Obstacle parameters
        p_obst = ca.MX.sym("p_obst", self.initial_obs["obstacles_pos"].size)
        # Gate parameters
        next_gate = ca.MX.sym("next_gate", 6)  # [x, y, z, roll, pitch, yaw]
        subsequent_gate = ca.MX.sym("subsequent_gate", 6)  # [x, y, z, roll, pitch, yaw]

        self.p = ca.vertcat(p_obst, next_gate, subsequent_gate)
        self.param_indices = {
            "obstacle": slice(0, self.initial_obs["obstacles_pos"].size),
            "next_gate": slice(
                self.initial_obs["obstacles_pos"].size, self.initial_obs["obstacles_pos"].size + 6
            ),
            "subsequent_gate": slice(
                self.initial_obs["obstacles_pos"].size + 6,
                self.initial_obs["obstacles_pos"].size + 12,
            ),
        }

    def setupObstacleConstraints(self):
        """Setup the obstacle constraints for the drone/environment controller."""
        # Obstacle constraints
        self.obstacle_pos = self.initial_obs.get("obstacles_pos", np.zeros((6, 3)))
        self.obstacle_diameter = self.initial_info.get("obstacle_diameter", 0.1)
        self.obstacle_in_range = self.initial_obs.get("obstacle_in_range", np.zeros(6))
        obstacle_constraints = []
        for k in range(self.initial_obs["obstacles_pos"].size):
            obstacle_constraints.append(
                ca.norm_2(self.x[self.state_indices["pos"]] - self.p[k * 3 : (k + 1) * 3])
                - self.obstacle_diameter
            )
        self.obstacle_constraints = ca.vertcat(*obstacle_constraints)
        self.obstacle_constraints_lh = np.zeros((self.initial_obs["obstacles_pos"].size,))
        self.obstacle_constraints_uh = np.ones((self.initial_obs["obstacles_pos"].size,)) * 1e9

    def setupBaseBounds(self):
        """Setup the nominal and unchanging bounds of the drone/environment controller."""
        # Rotor limits
        rpm_lb = 4070.3 + 0.2685 * 0
        rpm_ub = 4070.3 + 0.2685 * 65535
        # Individual rotor thrust limits
        self.thrust_lb = self.ct * (rpm_lb**2)
        self.thrust_ub = self.ct * (rpm_ub**2)
        # Individual rotor thrust rate limits
        rate_max = (
            (self.thrust_ub - self.thrust_lb) / 0.2
        )  # Assuming the motor can go from 0 to max thrust in 0.1s, loose bounds, only for control rate
        self.thrust_rate_lb = -rate_max
        self.thrust_rate_ub = rate_max
        # State limits (from the observation space, )
        x_y_max = 3.0
        z_max = 2.5
        z_min = 0.05  # 5 cm above the ground, to avoid hitting the ground
        eul_ang_max = 85 / 180 * np.pi
        large_val = 1e4
        self.pos_lb = np.array([-x_y_max, -x_y_max, z_min]).T
        self.pos_ub = np.array([x_y_max, x_y_max, z_max]).T
        self.vel_lb = np.array([-large_val, -large_val, -large_val]).T
        self.vel_ub = np.array([large_val, large_val, large_val]).T
        self.eul_ang_lb = np.array([-eul_ang_max, -eul_ang_max, -eul_ang_max]).T
        self.eul_ang_ub = np.array([eul_ang_max, eul_ang_max, eul_ang_max]).T
        self.eul_rate_lb = np.array([-large_val, -large_val, -large_val]).T
        self.eul_rate_ub = np.array([large_val, large_val, large_val]).T
        # What are the bounds for the quaternions?
        # TODO: Review quaternion bounds or maybe implement them as non-linear constraints
        # Convert Euler angle bounds to quaternions
        quat_max = Rot.from_euler("xyz", self.eul_ang_ub).as_quat()
        quat_min = Rot.from_euler("xyz", self.eul_ang_lb).as_quat()
        self.quat_lb = np.array([quat_min[3], quat_min[0], quat_min[1], quat_min[2]])
        self.quat_ub = np.array([quat_max[3], quat_max[0], quat_max[1], quat_max[2]])


class ThrustEulerDynamics(BaseDynamics):
    def __init__(self, initial_obs, initial_info, dynamics_info=None):
        super().__init__(initial_obs, initial_info, dynamics_info)
        self.setup_dynamics()
        self.setupBoundsAndScals()
        super().setupCasadiFunctions()

    def setup_dynamics(self):
        self.modelName = "ThrustEuler"
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        eul_ang = ca.MX.sym("eul_ang", 3)
        u = ca.MX.sym("f", 4)
        eq = 0.25 * self.mass * self.g
        self.u_eq = eq * np.ones((4,))
        self.x_eq = np.zeros((12,))
        beta = self.arm_length / ca.sqrt(2.0)
        torques = ca.vertcat(
            beta * (u[0] + u[1] - u[2] - u[3]),
            beta * (-u[0] + u[1] + u[2] - u[3]),
            self.gamma * (u[0] - u[1] + u[2] - u[3]),
        )
        self.state_indices = {
            "pos": slice(0, 3),
            "vel": slice(3, 6),
            "eul_ang": slice(6, 9),
            "w": slice(9, 12),
        }
        if self.useAngVel:
            w = ca.MX.sym("w", 3)
            dx = ca.vertcat(
                vel,
                ca.vertcat(0, 0, -self.g)
                + Rbi(eul_ang[0], eul_ang[1], eul_ang[2])
                @ ca.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) / self.mass),
                W2s(eul_ang) @ w,
                self.J_inv @ (torques - (ca.skew(w) @ self.J @ w)),
            )
            x = ca.vertcat(pos, vel, eul_ang, w)
        else:
            deul_ang = ca.MX.sym("deul_ang", 3)
            w = W1s(eul_ang) @ deul_ang
            dx = ca.vertcat(
                vel,
                ca.vertcat(0, 0, -self.g)
                + Rbi(eul_ang[0], eul_ang[1], eul_ang[2])
                @ ca.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) / self.mass),
                deul_ang,
                dW2s(eul_ang, deul_ang) @ w
                + W2s(eul_ang) @ (self.J_inv @ (ca.cross(self.J @ w, w) + torques)),
            )
            x = ca.vertcat(pos, vel, eul_ang, deul_ang)
        if self.useControlRates:
            x = ca.vertcat(x, u)
            u = ca.MX.sym("du", 4)
            dx = ca.vertcat(dx, u)
            self.x_eq = np.concatenate([self.x_eq, self.u_eq])
            self.u_eq = np.zeros((4,))
        self.x = x
        self.nx = x.size()[0]
        self.dx = dx
        self.u = u
        self.nu = u.size()[0]
        self.ny = self.nx + self.nu

    def setupBoundsAndScals(self):
        x_lb = np.concatenate([self.pos_lb, self.vel_lb, self.eul_ang_lb, self.eul_rate_lb])
        x_ub = np.concatenate([self.pos_ub, self.vel_ub, self.eul_ang_ub, self.eul_rate_ub])
        u_lb = np.array([self.thrust_lb, self.thrust_lb, self.thrust_lb, self.thrust_lb])
        u_ub = np.array([self.thrust_ub, self.thrust_ub, self.thrust_ub, self.thrust_ub])
        if self.useControlRates:
            x_lb = np.concatenate([x_lb, u_lb])
            x_ub = np.concatenate([x_ub, u_ub])
            u_lb = np.array(
                [self.thrust_rate_lb, self.thrust_rate_lb, self.thrust_rate_lb, self.thrust_rate_lb]
            )
            u_ub = np.array(
                [self.thrust_rate_ub, self.thrust_rate_ub, self.thrust_rate_ub, self.thrust_rate_ub]
            )

        self.x_lb = x_lb
        self.x_ub = x_ub
        self.x_scal = self.x_ub - self.x_lb
        self.u_lb = u_lb
        self.u_ub = u_ub
        self.u_scal = self.u_ub - self.u_lb

    def transformState(self, x, last_u):
        if self.useAngVel:
            w = W1(x[6:9]) @ x[9:12]
            x = np.concatenate([x[:9], w.flatten()])
        if self.useControlRates:
            x = np.concatenate([x, last_u])
        return x

    def transformAction(self, action):
        """Transforms the solution of the MPC controller to the format required for the Mellinger interface."""
        pos = action[:3]
        vel = action[3:6]
        eul_ang = action[6:9]
        if self.useAngVel:
            w = action[9:12]
        else:
            w = W1(eul_ang) @ action[9:12]
        # if self.useControlRates:
        #     action = action[: -self.nu]
        acc_world = np.zeros(3)  # Replace with actual acceleration computation if available
        yaw = action[8]
        return np.concatenate([pos, vel, acc_world, [yaw], w])


class TorqueEulerDynamics(BaseDynamics):
    def __init__(self, initial_obs, initial_info, dynamics_info=None):
        super().__init__(initial_obs, initial_info, dynamics_info)
        self.setup_dynamics()
        self.nx = self.x.size()[0]
        self.nu = self.u.size()[0]
        self.ny = self.nx + self.nu
        self.setupBoundsAndScals()
        super().setupCasadiFunctions()

    def setup_dynamics(self):
        self.modelName = "MPCTorqueEuler"
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        eul_ang = ca.MX.sym("eul_ang", 3)

        tot_thrust = ca.MX.sym("thrust", 1)
        torques = ca.MX.sym("torques", 3)
        u = ca.vertcat(tot_thrust, torques)
        self.x_eq = np.zeros((12,))
        self.u_eq = np.array([self.mass * self.g, 0, 0, 0]).T
        if self.useAngVel:
            w = ca.MX.sym("w", 3)
            dx = ca.vertcat(
                vel,
                ca.vertcat(0, 0, -self.g)
                + Rbi(eul_ang[0], eul_ang[1], eul_ang[2])
                @ ca.vertcat(0, 0, (tot_thrust) / self.mass),
                W2s(eul_ang) @ w,
                self.J_inv @ (torques - (ca.skew(w) @ self.J @ w)),
            )
            x = ca.vertcat(pos, vel, eul_ang, w)
        else:
            deul_ang = ca.MX.sym("deul_ang", 3)
            w = W1s(eul_ang) @ deul_ang
            dx = ca.vertcat(
                vel,
                ca.vertcat(0, 0, -self.g)
                + Rbi(eul_ang[0], eul_ang[1], eul_ang[2])
                @ ca.vertcat(0, 0, tot_thrust / self.mass),
                deul_ang,
                dW2s(eul_ang, deul_ang) @ w
                + W2s(eul_ang) @ (self.J_inv @ (ca.cross(self.J @ w, w) + torques)),
            )
            x = ca.vertcat(pos, vel, eul_ang, deul_ang)
        if self.useControlRates:
            x = ca.vertcat(x, u)
            u = ca.MX.sym("du", 4)
            dx = ca.vertcat(dx, u)
            self.x_eq = np.concatenate([self.x_eq, self.u_eq])
            self.u_eq = np.zeros((4,))
        self.x = x
        self.dx = dx
        self.u = u

    def setupBoundsAndScals(self):
        tot_thrust_ub = 4 * self.thrust_lb
        tot_thrust_lb = 4 * self.thrust_ub
        torque_lb = -0.2
        torque_ub = 0.2

        x_lb = np.concatenate([self.pos_lb, self.vel_lb, self.eul_ang_lb, self.eul_rate_lb])
        x_ub = np.concatenate([self.pos_ub, self.vel_ub, self.eul_ang_ub, self.eul_rate_ub])
        u_lb = np.array([tot_thrust_lb, torque_lb, torque_lb, torque_lb])
        u_ub = np.array([tot_thrust_ub, torque_ub, torque_ub, torque_ub])
        if self.useControlRates:
            x_lb = np.concatenate([x_lb, u_lb])
            x_ub = np.concatenate([x_ub, u_ub])
            u_lb = np.array(
                [self.thrust_rate_lb, self.thrust_rate_lb, self.thrust_rate_lb, self.thrust_rate_lb]
            )
            u_ub = np.array(
                [self.thrust_rate_ub, self.thrust_rate_ub, self.thrust_rate_ub, self.thrust_rate_ub]
            )
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.x_scal = self.x_ub - self.x_lb
        self.u_lb = u_lb
        self.u_ub = u_ub
        self.u_scal = self.u_ub - self.u_lb

    def transformState(self, x, last_u):
        if self.useAngVel:
            w = W1(x[6:9]) @ x[9:12]
            x = np.concatenate([x[:9], w.flatten()])
        if self.useControlRates:
            x = np.concatenate([x, last_u])
        return x

    def transformAction(self, action):
        """Transforms the solution of the MPC controller to the format required for the Mellinger interface."""
        pos = action[:3]
        vel = action[3:6]
        eul_ang = action[6:9]
        if self.useAngVel:
            w = action[9:12]
        else:
            w = W1(eul_ang) @ action[9:12]
        # if self.useControlRates:
        #     action = action[: -self.nu]
        acc_world = np.zeros(3)  # Replace with actual acceleration computation if available
        yaw = action[8]
        return np.concatenate([pos, vel, acc_world, [yaw], w])


class ThrustQuaternionDynamics(BaseDynamics):
    # Created according to https://www.researchgate.net/publication/279448184_Quadrotor_Quaternion_Control
    def __init__(self, initial_obs, initial_info, dynamics_info=None):
        super().__init__(initial_obs, initial_info, dynamics_info)
        if not self.useAngVel:
            self.useAngVel = True
            Warning("Quaternion Formulation uses angular velocity")
        self.setup_dynamics()
        self.nx = self.x.size()[0]
        self.nu = self.u.size()[0]
        self.ny = self.nx + self.nu
        self.setupBoundsAndScals()
        super().setupCasadiFunctions()

    def setup_dynamics(self):
        self.modelName = "MPCThrustQuaternion"
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        quat = ca.MX.sym("quat", 4)  # [qw, qx, qy, qz], Quaternion, scalar first
        w = ca.MX.sym("w", 3)

        u = ca.MX.sym("f", 4)
        eq = 0.25 * self.mass * self.g
        self.u_eq = eq * np.ones((4,))
        self.x_eq = np.zeros((13,))
        beta = self.arm_length / ca.sqrt(2.0)
        torques = ca.vertcat(
            beta * (u[0] + u[1] - u[2] - u[3]),
            beta * (-u[0] + u[1] + u[2] - u[3]),
            self.gamma * (u[0] - u[1] + u[2] - u[3]),
        )

        # Thrust vector in the body frame
        thrust_body = ca.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) / self.mass)
        # Transform thrust vector to the inertial frame using quaternion rotation
        thrust_inertial = self.quaternion_rotation(quat, thrust_body)
        dpos = vel
        dvel = ca.vertcat(0, 0, -self.g) + thrust_inertial
        dquat = 0.5 * self.quaternion_product(quat, ca.vertcat(0, w))
        dw = self.J_inv @ (torques - (ca.skew(w) @ self.J @ w))

        dx = ca.vertcat(dpos, dvel, dquat, dw)
        x = ca.vertcat(pos, vel, quat, w)

        if self.useControlRates:
            x = ca.vertcat(x, u)
            u = ca.MX.sym("du", 4)
            dx = ca.vertcat(dx, u)
            self.x_eq = np.concatenate([self.x_eq, self.u_eq])
            self.u_eq = np.zeros((4,))
        self.x = x
        self.dx = dx
        self.u = u

    def setupBoundsAndScals(self):
        # quat_lb = np.array([-0.6, -0.15, -0.15, -0.15])
        # quat_ub = np.array([0.6, 0.15, 0.15, 0.15])
        print(self.quat_lb, self.quat_ub)
        x_lb = np.concatenate([self.pos_lb, self.vel_lb, self.quat_lb, self.eul_rate_lb])
        x_ub = np.concatenate([self.pos_ub, self.vel_ub, self.quat_ub, self.eul_rate_ub])
        u_lb = np.array([self.thrust_lb, self.thrust_lb, self.thrust_lb, self.thrust_lb])
        u_ub = np.array([self.thrust_ub, self.thrust_ub, self.thrust_ub, self.thrust_ub])
        if self.useControlRates:
            x_lb = np.concatenate([x_lb, u_lb])
            x_ub = np.concatenate([x_ub, u_ub])
            u_lb = np.array(
                [self.thrust_rate_lb, self.thrust_rate_lb, self.thrust_rate_lb, self.thrust_rate_lb]
            )
            u_ub = np.array(
                [self.thrust_rate_ub, self.thrust_rate_ub, self.thrust_rate_ub, self.thrust_rate_ub]
            )

        self.x_lb = x_lb
        self.x_ub = x_ub
        self.x_scal = self.x_ub - self.x_lb
        self.u_lb = u_lb
        self.u_ub = u_ub
        self.u_scal = self.u_ub - self.u_lb

    def quaternion_conjugate(self, q: ca.MX) -> ca.MX:
        """Compute the conjugate of a quaternion."""
        return ca.vertcat(q[0], -q[1], -q[2], -q[3])

    def quaternion_rotation(self, q: ca.MX, v: ca.MX) -> ca.MX:
        """Rotate a vector by a quaternion."""
        t = 2 * ca.cross(q[1:], v)
        return v + q[0] * t + ca.cross(q[1:], t)

    def quaternion_product(self, q: ca.MX, r: ca.MX) -> ca.MX:
        """Compute the product of two quaternions."""
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        rw, rx, ry, rz = r[0], r[1], r[2], r[3]
        return ca.vertcat(
            qw * rw - qx * rx - qy * ry - qz * rz,
            qw * rx + qx * rw + qy * rz - qz * ry,
            qw * ry - qx * rz + qy * rw + qz * rx,
            qw * rz + qx * ry - qy * rx + qz * rw,
        )

    def transformState(self, x, last_u):
        """Transforms the state from the observation to the states used in the dynamics."""
        # Convert Euler angles to quaternion

        quat = Rot.from_euler("xyz", x[6:9]).as_quat()
        # Reorder quaternion to [qw, qx, qy, qz]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        # Transform angular velocities
        w = W1(x[6:9]) @ x[9:12]
        # Concatenate position, velocity, quaternion, and angular velocities
        x = np.concatenate([x[:6], quat, w])
        # Include control rates if used
        if self.useControlRates:
            x = np.concatenate([x, last_u])
        self.current_state = x
        return x

    def transformAction(self, action):
        """Transforms the solution of the MPC controller to the format required for the Mellinger interface."""
        # Extract position, velocity, quaternion, and angular velocities from the action
        pos = action[:3]
        vel = action[3:6]
        quat = action[6:10]
        w = action[10:13]

        # Compute the acceleration in the world frame
        acc_world = (vel - self.current_state[3:6]) / self.ts

        # Extract yaw from the quaternion # Convert to [qx, qy, qz, qw] for scipy
        yaw = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")[-1]

        return np.concatenate([pos, vel, acc_world, [yaw], w])


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
            "pos": slice(0, 3),
            "vel": slice(3, 6),
            "eul_ang": slice(6, 9),
            "w": slice(9, 12),
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
