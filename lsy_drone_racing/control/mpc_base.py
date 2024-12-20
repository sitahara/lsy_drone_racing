"""Module for Model Predictive Controller implementation using do_mpc."""

import casadi as ca
import numpy as np
import pybullet as p
import scipy as sp
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as Rmat

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.control.utils import (
    W1,
    W2,
    Rbi,
    W1s,
    W2s,
    dW1s,
    dW2s,
    rpm_to_torques_mat,
    rungeKutta4,
)


class MPC_BASE(BaseController):
    """Model Predictive Controller implementation using do_mpc."""

    def __init__(self, initial_obs: NDArray[np.floating], initial_info: dict):  # noqa: D107
        super().__init__(initial_obs, initial_info)

        # Inital parameters
        self.useMellinger = True  # whether to use the thrust or Mellinger interface
        self.useTorqueModel = (
            False  # whether to use the total thrust & torque or individual thrust model
        )
        self.useEulerDynamics = True  # whether to use euler or quaternion dynamics
        self.ts = 1 / 60  # Time step, 60Hz
        self.n_step = 0  # current step for the target trajectory
        self.n_horizon = 60  # Prediction horizon, 1s
        self.nx = 12  # number of states
        self.nu = 4  # number of control inputs
        self.ny = self.nx + self.nu  # number of
        # Constraint parameters
        self.soft_constraints = False  # whether to use soft constraints
        self.soft_penalty = 1e3  # penalty for soft constraints
        # Cost parameters
        self.Qs = np.diag(
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01]
        )  # stage cost for states (position, velocity, euler angles, angular velocity)
        self.Qt = self.Qs  # terminal cost
        self.Rs = np.diag([0.01, 0.01, 0.01, 0.01])  # control cost
        self.Rsdelta = np.diag([0.01, 0.01, 0.01, 0.01])  # control rate cost
        # MPC parameters
        self.tick = 0  # running tick that indicates when optimization problem should be reexecuted
        self.cycles_to_update = (
            3  # after how many control calls the optimization problem is reexecuted
        )
        # Define the (nominal) system parameters
        self.mass = initial_info["drone_mass"]  # kg
        self.g = 9.81
        # Ixx, Iyy, Izz = drone.nominal_params.J.diagonal()  # kg*m^2
        Ixx = 1.395e-5  # kg*m^2, moment of inertia
        Iyy = 1.436e-5  # kg*m^2, moment of inertia
        Izz = 2.173e-5  # kg*m^2, moment of inertia

        self.J = ca.diag([Ixx, Iyy, Izz])  # moment of inertia
        self.J_inv = ca.diag([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz])  # inverse of moment of inertia

        self.ct = 3.1582e-10  # N/RPM^2, lift coefficient drone.nominal_params.kf  #
        self.cd = 7.9379e-12  # N/RPM^2, drag coefficient drone.nominal_params.km  #
        self.gamma = self.cd / self.ct

        self.arm_length = 0.046  # m, arm length drone.nominal_params.arm_len  #
        self.c_tau_xy = self.arm_length * self.ct / ca.sqrt(2)  # torque coefficient
        self.rpmToTorqueMat = rpm_to_torques_mat(self.c_tau_xy, self.cd)  # Torque matrix

        rpm_lb = 4070.3
        rpm_ub = 4070.3 + 0.2685 * 65535
        self.thrust_lb = self.ct * rpm_lb**2
        self.thrust_ub = self.ct * rpm_ub**2

        self.x = None  # symbolical state vector (casadi used for dynamics)
        self.u = None  # symbolical control input vector (casadi used for dynamics)
        self.dx = None  # Continuous dynamics expression (casadi)
        self.dx_d = None  # Discrete dynamics expression (casadi)
        self.fc = None  # CasADi function for the dynamics: dx = f(x, u)
        self.fd = None  # Discrete CasADi function for the dynamics: x_{k+1} = f(x_k, u_k)
        self.f_impl = None  # Implicit dynamics: 0 = x_dot - f(x, u)
        self.modelName = None  # Model name (AcadosModel, Code generation)
        self.x_guess = None  # holds the last solution for the state trajectory (already shifted)
        self.u_guess = None  # holds the last solution for the controls trajectory (already shifted)
        self.x_ref = np.zeros(
            (self.nx, self.n_horizon + 1)
        )  # reference trajectory (from current time to horizon)

        self.x0 = np.concatenate(
            [initial_obs["pos"], initial_obs["vel"], initial_obs["rpy"], initial_obs["ang_vel"]]
        )
        if not self.useTorqueModel:
            w0 = (
                W1(initial_obs["rpy"]) @ initial_obs["ang_vel"]
            )  # convert euler angle rate to body frame
            self.x0 = np.concatenate(
                [initial_obs["pos"], initial_obs["vel"], initial_obs["rpy"], w0]
            )

    def compute_control(
        self, obs: NDArray[np.floating], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [thrust, tau_des] as a numpy array.
        """
        # Get current observations
        pos = obs["pos"]  # position in world frame
        vel = obs["vel"]  # velocity in world frame
        eul_ang = obs["rpy"]  # euler angles roll, pitch, yaw
        deul_ang = obs["ang_vel"]  # angular velocity in body frame
        current_state = np.concatenate([pos, vel, eul_ang, deul_ang])

        # Update MPC controls and data. Only executed every self.cycles_to_update calls
        if self.tick == 0:
            # Note: initial guess is automatically updated with last solution
            u = self.mpc.make_step(current_state)  # noqa: F841

        print("current time:", self.mpc.t0)
        # Plotting the predicted states of prediction horizon for debuging
        next_poss = self.mpc.data.prediction(("_x", "pos"))[:, :, 0]
        try:
            for k in range(self.n_horizon):
                p.addUserDebugLine(
                    next_poss[:, k],
                    next_poss[:, k + 1],
                    lineColorRGB=[0, 0, 1],  # Blue color
                    lineWidth=2,
                    lifeTime=self.t_step * 4,  # 0 means the line persists indefinitely
                    physicsClientId=0,
                )
        except p.error:
            ...

        # Extract the next predicted states from the MPC
        next_pos = self.mpc.data.prediction(("_x", "pos"))[:, self.tick + 1, 0]
        print("next_pos: ", next_pos)
        next_vel = self.mpc.data.prediction(("_x", "vel"))[:, self.tick + 1, 0]
        acc = (
            self.mpc.data.prediction(("_x", "vel"))[:, self.tick + 1, 0]
            - self.mpc.data.prediction(("_x", "vel"))[:, self.tick, 0]
        ) / self.t_step
        next_eul_ang = self.mpc.data.prediction(("_x", "eul_ang"))[:, self.tick + 1, 0]
        next_deul_ang = self.mpc.data.prediction(("_x", "deul_ang"))[:, self.tick + 1, 0]
        # u = np.array([self.ct * np.sum(rpm), self.rpmToTorqueMat @ rpm])

        # action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] to follow.
        # where ax, ay, az are the acceleration in world frame, rrate, prate, yrate are the roll, pitch, yaw rate in body frame

        rpy_rate = W1(next_eul_ang) @ next_deul_ang  # convert euler angle rate to body frame
        action = np.concatenate([next_pos, next_vel, acc, [next_eul_ang[2]], rpy_rate.flatten()])
        self.tick = (self.tick + 1) % self.cycles_to_update
        return action.flatten()

    def setupIPOPTOptimizer(self):
        """Setup the IPOPT optimizer for the MPC controller."""
        opti = ca.Opti()

        # Define optimization variables
        X = opti.variable(self.nx, self.n_horizon + 1)  # State trajectory
        U = opti.variable(self.nu, self.n_horizon)  # Control trajectory
        X_ref = opti.parameter(self.nx, self.n_horizon + 1)  # Reference trajectory
        X0 = opti.parameter(self.nx)  # Initial state
        # Initial state constraint
        opti.subject_to(X[:, 0] == X0)

        # Dynamics constraints
        for k in range(self.n_horizon):
            x_next = X[:, k] + self.fd(X[:, k], U[:, k])
            opti.subject_to(X[:, k + 1] == x_next)

        # Cost function

        cost = 0
        for k in range(self.n_horizon):
            cost += ca.mtimes(
                [(X[:, k] - X_ref[:, k]).T, self.Qs, (X[:, k] - X_ref[:, k])]
            )  # State cost
            cost += ca.mtimes(
                [(U[:, k] - self.u_eq).T, self.Rs, (U[:, k] - self.u_eq)]
            )  # Control cost
        cost += ca.mtimes(
            [
                (X[:, self.n_horizon] - X_ref[:, -1]).T,
                self.Qt,
                (X[:, self.n_horizon] - X_ref[:, -1]),
            ]
        )  # Terminal cost
        opti.minimize(cost)

        # Control constraints
        opti.subject_to(opti.bounded(self.u_lb, U, self.u_ub))
        opti.subject_to(opti.bounded(self.x_lb, X, self.x_ub))

        # Solver options
        opts = {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-4,
            "ipopt.max_iter": 25,
            "ipopt.linear_solver": "mumps",
        }
        opti.solver("ipopt", opts)

        # Store the optimizer
        self.IPOPT_var = {"opti": opti, "X0": X0, "X_ref": X_ref, "X": X, "U": U, "cost": cost}
        return None

    def stepIPOPT(self) -> np.ndarray:
        """Perfroms one optimization step using the IPOPT solver. Also works without warmstart. Used as first step for the acados model. Returns action and updates/instantiates guess for next iteration."""
        # Unpack IPOPT variables
        opti = self.IPOPT_var["opti"]
        X = self.IPOPT_var["X"]
        U = self.IPOPT_var["U"]
        X_ref = self.IPOPT_var["X_ref"]
        X0 = self.IPOPT_var["X0"]
        cost = self.IPOPT_var["cost"]

        # Set initial state
        opti.set_value(X0, self.current_state)

        # Set reference trajectory
        opti.set_value(X_ref, self.x_ref)
        if self.x_guess is None:
            opti.set_initial(X, np.hstack((self.current_state, self.x_ref[:, 1:])))
        else:
            opti.set_initial(X, np.hstack((self.current_state, self.x_guess)))

        # Solve the optimization problem
        sol = opti.solve()

        # Extract the solution
        x_sol = opti.value(X)
        u_sol = opti.value(U)
        if self.useMellinger:
            action = x_sol[:, 1]
        else:
            action = u_sol[:, 0]
        self.last_action = action

        # Update/Instantiate guess for next iteration
        self.x_guess = np.hstack((x_sol[:, 2:], x_sol[:, -1]))
        self.u_guess = np.hstack((u_sol[:, 1:], u_sol[:, -1]))

        self.x_last = x_sol
        self.u_last = u_sol
        # reset the solver after the initial guess
        self.setupIPOPTOptimizer()
        return action
        # Debugging
        # debug_x = np.zeros((self.nx, self.n_horizon + 1))
        # debug_x[:, 0] = current_state
        # for k in range(self.n_horizon):
        #     debug_x[:, k + 1] = (
        #         self.disc_dyn(self.x_guess[:, k], self.u_guess[:, k]).full().flatten()
        #     )
        # print("Initial guess", debug_x[0:3, :20])
        # try:
        #     # Plot the spline as a line in PyBullet
        #     for k in range(self.n_horizon):
        #         p.addUserDebugLine(
        #             debug_x[:3, k],
        #             debug_x[:3, k + 1],
        #             lineColorRGB=[k, 0, 1],  # Red color
        #             lineWidth=2,
        #             lifeTime=0,  # 0 means the line persists indefinitely
        #             physicsClientId=0,
        #         )
        # except p.error:
        #     ...  # Ignore errors if PyBullet is not available
        return None

    def setupDynamics(self):
        """Setup the dynamics for the MPC controller."""
        if self.useTorqueModel:
            if self.useEulerDynamics:
                self.TorqueEulerDynamics()  # Uses euler angle rates
            if not self.useEulerDynamics:
                raise NotImplementedError
                self.TorqueQuaternionDynamics()  # Not implemented yet
        elif not self.useTorqueModel:
            if self.useEulerDynamics:
                self.ThrustEulerDynamics()  # Uses body angular velocity
            if not self.useEulerDynamics:
                raise NotImplementedError
                self.ThrustQuaternionDynamics()  # Not implemented yet
        # Continuous dynamic function
        self.fc = ca.Function("fc", [self.x, self.u], [self.dx], ["x", "u"], ["dx"])
        # Discrete dynamic expression and dynamic function
        self.dx_d, self.fd = rungeKutta4(self.x, self.u, self.ts, self.fc)
        xdot = ca.MX.sym("xdot", self.nx)
        # Continuous implicit dynamic expression
        self.f_impl = xdot - self.dx
        return None

    def TorqueEulerDynamics(self):
        """Setup the dynamics model using the torque model."""
        # Define state variables
        pos = ca.MX.sym("pos", 3)  # position in world frame
        vel = ca.MX.sym("vel", 3)  # velocity in world frame
        eul_ang = ca.MX.sym("eul_ang", 3)  # euler angles roll, pitch, yaw
        deul_ang = ca.MX.sym("deul_ang", 3)  # euler angle rates in world frame

        # Define Control variables
        thrust = ca.MX.sym("thrust", 1)  # total thrust
        torques = ca.MX.sym("torques", 3)  # body frame torques
        self.u_eq = np.array([self.mass * self.g, 0, 0, 0]).T  # equilibrium control input

        # Define Dynamics in world frame as euler angles
        w = W1s(eul_ang) @ deul_ang  # Body angular velocity

        dx = ca.vertcat(
            vel,
            ca.vertcat(0, 0, -self.g)
            + Rbi(eul_ang[0], eul_ang[1], eul_ang[2]) @ ca.vertcat(0, 0, thrust / self.mass),
            deul_ang,
            dW2s(eul_ang, deul_ang) @ w
            + W2s(eul_ang) @ (self.J_inv @ (ca.cross(self.J @ w, w) + torques)),
        )
        self.x = ca.vertcat(pos, vel, eul_ang, deul_ang)  # state vector
        self.dx = dx
        self.u = ca.vertcat(thrust, torques)  # control input vector
        self.modelName = "MPCTorqueEuler"
        return None

    def ThrustEulerDynamics(self):
        """Setup the dynamics model using the thrust model."""
        # Define state variables
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        eul_ang = ca.MX.sym("eul_ang", 3)
        w = ca.MX.sym("w", 3)

        # Define Control variables
        u = ca.MX.sym("f", 4)  # thrust of each rotor
        eq = 0.25 * self.mass * self.g  # equilibrium control input per motor
        self.u_eq = np.array([eq, eq, eq, eq]).T  # equilibrium control input, drone hovers
        beta = self.arm_length / ca.sqrt(2.0)  # beta = l/sqrt(2)
        torques = ca.vertcat(
            beta * (u[0] + u[1] - u[2] - u[3]),
            beta * (-u[0] + u[1] + u[2] - u[3]),
            self.gamma * (u[0] - u[1] + u[2] - u[3]),
        )  # torques in the body frame

        dx = ca.vertcat(
            vel,
            ca.vertcat(0, 0, -self.g)
            + Rbi(eul_ang[0], eul_ang[1], eul_ang[2]) @ ca.vertcat(0, 0, ca.sum(u) / self.mass),
            W2s(eul_ang) @ w,
            self.J_inv @ (torques - (ca.skew(w) @ self.J @ w)),
        )
        self.x = ca.vertcat(pos, vel, eul_ang, w)  # state vector
        self.dx = dx
        self.u = u
        self.modelName = "MPCThrustEuler"

        # Setup base constraints and scaling factors
        self.u_lb = np.array([self.thrust_lb, self.thrust_lb, self.thrust_lb, self.thrust_lb]).T
        self.u_ub = np.array([self.thrust_ub, self.thrust_ub, self.thrust_ub, self.thrust_ub]).T
        self.u_scal = self.u_ub - self.u_lb

        x_y_max = 3.0
        z_max = 2.5
        z_min = 0.00
        eul_ang_max = 85 / 180 * np.pi
        large_val = 1e6
        self.x_lb = np.array(
            [
                -x_y_max,
                -x_y_max,
                z_min,
                -large_val,
                -large_val,
                -large_val,
                -eul_ang_max,
                -eul_ang_max,
                -eul_ang_max,
                -large_val,
                -large_val,
                -large_val,
            ]
        ).T
        self.x_ub = np.array(
            [
                x_y_max,
                x_y_max,
                z_max,
                large_val,
                large_val,
                large_val,
                eul_ang_max,
                eul_ang_max,
                eul_ang_max,
                large_val,
                large_val,
                large_val,
            ]
        ).T
        self.x_scal = self.x_ub - self.x_lb
        return None

    def set_target_trajectory(self, t_total: float = 11) -> None:
        """Set the target trajectory for the MPC controller."""
        self.n_step = 0  # current step for the target trajectory
        waypoints = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.8, 0.5, 0.2],
                [0.55, -0.8, 0.4],
                [0.2, -1.8, 0.65],
                [1.1, -1.35, 1.0],
                [0.2, 0.0, 0.65],
                [0.0, 0.75, 0.525],
                [0.0, 0.75, 1.1],
                [-0.5, -0.5, 1.1],
                [-0.5, -1.0, 1.1],
            ]
        )
        self.t_total = t_total
        t = np.linspace(0, self.t_total, len(waypoints))
        self.target_trajectory = CubicSpline(t, waypoints)
        # Generate points along the spline for visualization

        t_vis = np.linspace(0, self.t_total - 1, 100)
        spline_points = self.target_trajectory(t_vis)
        try:
            # Plot the spline as a line in PyBullet
            for i in range(len(spline_points) - 1):
                p.addUserDebugLine(
                    spline_points[i],
                    spline_points[i + 1],
                    lineColorRGB=[1, 0, 0],  # Red color
                    lineWidth=2,
                    lifeTime=0,  # 0 means the line persists indefinitely
                    physicsClientId=0,
                )
        except p.error:
            ...  # Ignore errors if PyBullet is not available
        return None

    def updateTargetTrajectory(self):
        """Update the target trajectory for the MPC controller."""
        current_time = self.n_step * self.ts
        t_horizon = np.linspace(
            current_time, current_time + self.n_horizon * self.ts, self.n_horizon + 1
        )

        # Evaluate the spline at the time points
        pos_des = self.target_trajectory(t_horizon)

        # Handle the case where the end time exceeds the total time
        if t_horizon[-1] > self.t_total:
            last_value = self.target_trajectory(self.t_total).reshape(3, 1)
            n_repeat = np.sum(t_horizon > self.t_total)
            pos_des[:, -n_repeat] = np.tile(last_value, (1, n_repeat))
        # print(reference_trajectory_horizon)
        self.x_ref[:3, :] = pos_des
        self.n_step += 1
        return None
