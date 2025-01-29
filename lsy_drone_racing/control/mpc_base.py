"""Module for Model Predictive Controller implementation using do_mpc."""

from __future__ import annotations

import casadi as ca
import numpy as np
import pybullet as p
import scipy
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
import scipy.linalg
from scipy.spatial.transform import Rotation as Rmat

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.mpc_utils import (
    W1,
    W2,
    Rbi,
    W1s,
    W2s,
    dW1s,
    dW2s,
    rpm_to_torques_mat,
    rungeKuttaExpr,
    rungeKuttaFcn,
)


class MPC_BASE(BaseController):
    """Model Predictive Controller implementation using do_mpc."""

    def __init__(
        self,
        initial_obs: NDArray[np.floating],
        initial_info: dict,
        useMellinger: bool = True,
        useAngVel: bool = False,
        useTorqueModel: bool = False,
        useEulerDynamics: bool = True,
        t_total: float = 7,
        ts: float = 1 / 60,
        n_horizon: int = 60,
        cycles_to_update: int = 3,
        useSoftConstraints: bool = True,
        soft_penalty: float = 1e3,
        useObstacleConstraints: bool = True,
        useGateConstraints: bool = False,
        useControlRates: bool = False,
        out_dir: str = "generated_code/mpc_base",
        **kwargs,
    ):
        super().__init__(initial_obs, initial_info)
        self.initial_info = initial_info
        self.initial_obs = initial_obs
        self.useControlRates = useControlRates
        # Adjustable parameters
        self.useMellinger = useMellinger  # whether to use the thrust or Mellinger interface
        self.useAngVel = (
            useAngVel  # whether to body angular velocity or euler angle rates in the dynamics
        )

        self.useTorqueModel = (
            useTorqueModel  # whether to use the torque model or individual thrust model
        )
        self.useEulerDynamics = useEulerDynamics  # whether to use euler or quaternion dynamics#
        self.t_total = t_total  # total time for the trajectory
        self.ts = ts  # Time step, 60Hz
        self.n_horizon = n_horizon  # Prediction horizon, 1s

        self.cycles_to_update = (
            cycles_to_update  # after how many control calls the optimization problem is reexecuted
        )
        self.useSoftConstraints = useSoftConstraints  # whether to use soft constraints
        self.useObstacleConstraints = useObstacleConstraints  # whether to use obstacle constraints
        self.useGateConstraints = useGateConstraints  # whether to use gate constraints
        self.soft_penalty = soft_penalty  # penalty for soft constraints
        self.tick = 0  # running tick that indicates when optimization problem should be reexecuted
        # Setup various variables
        self.setupNominalParameters()
        self.setupBaseConstraints()

        # Cost function parameters
        self.setupCostFunction()

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
        self.last_action = None  # last action computed by the controller
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
        self.set_target_trajectory()
        self.setupDynamics()
        self.setupIPOPTOptimizer()

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
        if self.useAngVel:
            w = W1(eul_ang) @ deul_ang  # convert euler angle rate to body frame
            self.current_state = np.concatenate([pos, vel, eul_ang, w.flatten()])
        else:
            self.current_state = np.concatenate([pos, vel, eul_ang, deul_ang])
        # Updates x_ref, the current target trajectory and upcounts the trajectory tick
        self.updateTargetTrajectory()
        action = self.stepIPOPT()

        if self.useMellinger:
            # action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] to follow.
            # where ax, ay, az are the acceleration in world frame, rrate, prate, yrate are the roll, pitch, yaw rate in body frame
            acc = (self.x_guess[3:6, 0] - action[3:6]) / self.ts
            action = np.concatenate([action[:6], acc, [action[8]], action[9:]])
        else:
            # action: [thrust, tau_des]
            action = np.array(action)

        print(f"Current position: {self.current_state[:3]}")
        print(f"Desired position: {self.x_ref[:3, 1]}")
        print(f"Next position: {action[:3]}")

        # self.tick = (self.tick + 1) % self.cycles_to_update
        return action.flatten()

        # # Update MPC controls and data. Only executed every self.cycles_to_update calls
        # if self.tick == 0:
        #     # Note: initial guess is automatically updated with last solution
        #     u = self.mpc.make_step(current_state)  # noqa: F841

        # print("current time:", self.mpc.t0)
        # # Plotting the predicted states of prediction horizon for debuging
        # next_poss = self.mpc.data.prediction(("_x", "pos"))[:, :, 0]
        # try:
        #     for k in range(self.n_horizon):
        #         p.addUserDebugLine(
        #             next_poss[:, k],
        #             next_poss[:, k + 1],
        #             lineColorRGB=[0, 0, 1],  # Blue color
        #             lineWidth=2,
        #             lifeTime=self.t_step * 4,  # 0 means the line persists indefinitely
        #             physicsClientId=0,
        #         )
        # except p.error:
        #     ...

        # # Extract the next predicted states from the MPC
        # next_pos = self.mpc.data.prediction(("_x", "pos"))[:, self.tick + 1, 0]
        # print("next_pos: ", next_pos)
        # next_vel = self.mpc.data.prediction(("_x", "vel"))[:, self.tick + 1, 0]
        # acc = (
        #     self.mpc.data.prediction(("_x", "vel"))[:, self.tick + 1, 0]
        #     - self.mpc.data.prediction(("_x", "vel"))[:, self.tick, 0]
        # ) / self.t_step
        # next_eul_ang = self.mpc.data.prediction(("_x", "eul_ang"))[:, self.tick + 1, 0]
        # next_deul_ang = self.mpc.data.prediction(("_x", "deul_ang"))[:, self.tick + 1, 0]
        # # u = np.array([self.ct * np.sum(rpm), self.rpmToTorqueMat @ rpm])

        # # action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] to follow.
        # # where ax, ay, az are the acceleration in world frame, rrate, prate, yrate are the roll, pitch, yaw rate in body frame

        # rpy_rate = W1(next_eul_ang) @ next_deul_ang  # convert euler angle rate to body frame
        # action = np.concatenate([next_pos, next_vel, acc, [next_eul_ang[2]], rpy_rate.flatten()])
        # self.tick = (self.tick + 1) % self.cycles_to_update
        # return action.flatten()

    def setupIPOPTOptimizer(self):
        """Setup the IPOPT optimizer for the MPC controller."""
        opti = ca.Opti()

        # Define optimization variables
        X = opti.variable(self.nx, self.n_horizon + 1)  # State trajectory
        U = opti.variable(self.nu, self.n_horizon)  # Control trajectory
        X_ref = opti.parameter(self.nx, self.n_horizon + 1)  # Reference trajectory
        X0 = opti.parameter(self.nx, 1)  # Initial state
        X_lb = opti.parameter(self.nx, 1)  # State lower bound
        X_ub = opti.parameter(self.nx, 1)  # State upper bound
        U_lb = opti.parameter(self.nu, 1)  # Control lower bound
        U_ub = opti.parameter(self.nu, 1)  # Control upper bound

        # Define slack variables for soft constraints
        if self.useSoftConstraints:
            s_x = opti.variable(self.nx, self.n_horizon + 1)  # Slack for state constraints
            s_u = opti.variable(self.nu, self.n_horizon)  # Slack for control constraints
            slack_penalty = self.soft_penalty
        else:
            s_x = np.zeros((self.nx, self.n_horizon + 1))
            s_u = np.zeros((self.nu, self.n_horizon))
            slack_penalty = 0
        no_slack_states = [6, 7, 8]  # euler angles

        ### Constraints

        # Initial state constraint
        opti.subject_to(X[:, 0] == X0)

        # Dynamics constraints
        for k in range(self.n_horizon):
            xn = self.fd(x=X[:, k], u=U[:, k])["xn"]
            opti.subject_to(X[:, k + 1] == xn)
        # State/Control constraints with slack variables (no slack for z position)
        for i in range(self.n_horizon + 1):
            for k in range(self.nx):
                if k in no_slack_states:
                    opti.subject_to(opti.bounded(X_lb[k], X[k, i], X_ub[k]))
                else:
                    opti.subject_to(opti.bounded(X_lb[k] - s_x[k, i], X[k, i], X_ub[k] + s_x[k, i]))
        for i in range(self.n_horizon):
            opti.subject_to(opti.bounded(U_lb - s_u[:, i], U[:, i], U_ub + s_u[:, i]))

        ### Costs
        cost = 0
        cost_func = self.cost_function
        for k in range(self.n_horizon):
            cost += cost_func(X[:, k], X_ref[:, k], self.Qs, U[:, k], self.u_eq, self.Rs)
        cost += cost_func(
            X[:, -1], X_ref[:, -1], self.Qt, np.zeros((self.nu, 1)), self.u_eq, self.Rs
        )  # Terminal cost

        # Add slack penalty to the cost function
        cost += slack_penalty * (ca.sumsqr(s_x) + ca.sumsqr(s_u))
        opti.minimize(cost)

        # Solver options
        opts = {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-4,
            "ipopt.max_iter": 25,
            "ipopt.linear_solver": "mumps",
        }
        opti.solver("ipopt", opts)

        # Store the optimizer
        self.IPOPT_var = {
            "opti": opti,
            "X0": X0,
            "X_ref": X_ref,
            "X": X,
            "U": U,
            "cost": cost,
            "opts": opts,
            "X_lb": X_lb,
            "X_ub": X_ub,
            "U_lb": U_lb,
            "U_ub": U_ub,
            "s_x": s_x,
            "s_u": s_u,
        }
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
        X_lb = self.IPOPT_var["X_lb"]
        X_ub = self.IPOPT_var["X_ub"]
        U_lb = self.IPOPT_var["U_lb"]
        U_ub = self.IPOPT_var["U_ub"]

        # Set initial state
        opti.set_value(X0, self.current_state)
        # Set reference trajectory
        opti.set_value(X_ref, self.x_ref)
        # Set state and control bounds
        opti.set_value(X_lb, self.x_lb)
        opti.set_value(X_ub, self.x_ub)
        opti.set_value(U_lb, self.u_lb)
        opti.set_value(U_ub, self.u_ub)
        # Set initial guess
        if self.x_guess is None or self.u_guess is None:
            opti.set_initial(
                X, np.hstack((self.current_state.reshape(self.nx, 1), self.x_ref[:, 1:]))
            )
            opti.set_initial(U, self.u_ref)
        else:
            opti.set_initial(X, self.x_guess)
            opti.set_initial(U, self.u_guess)

        # Solve the optimization problem
        try:
            sol = opti.solve()
            # Extract the solution
            x_sol = opti.value(X)
            u_sol = opti.value(U)
        except Exception as e:
            print("IPOPT solver failed: ", e)
            x_sol = opti.debug.value(X)
            u_sol = opti.debug.value(U)

        if self.useMellinger:
            action = x_sol[:, 1]
        else:
            action = u_sol[:, 0]
        self.last_action = action

        # Update/Instantiate guess for next iteration
        if self.x_guess is None:
            self.x_guess = np.hstack((x_sol[:, 1:], x_sol[:, -1].reshape(self.nx, 1)))
            self.u_guess = np.hstack((u_sol[:, 1:], u_sol[:, -1].reshape(self.nu, 1)))
        else:
            self.x_guess[:, :-1] = x_sol[:, 1:]
            self.u_guess[:, :-1] = u_sol[:, 1:]

        self.x_last = x_sol
        # print("x_sol: ", x_sol.shape)
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
        if self.useEulerDynamics:  # Use euler dynamic formulation
            if self.useTorqueModel:  # Use total thrust and torque model
                self.TorqueEulerDynamics()
            else:  # Use individual thrust model
                self.ThrustEulerDynamics()
        else:
            raise NotImplementedError  # Not implemented yet
        if self.useControlRates:
            self.setupControlRates()
        # Continuous dynamic function
        self.fc = ca.Function("fc", [self.x, self.u], [self.dx], ["x", "u"], ["dx"])
        # Discrete dynamic expression and dynamic function
        self.dx_d = rungeKuttaExpr(self.x, self.u, self.ts, self.fc)
        self.fd = rungeKuttaFcn(self.nx, self.nu, self.ts, self.fc)
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
        self.u_eq = np.array([self.mass * self.g, 0, 0, 0]).reshape(
            self.nu, 1
        )  # equilibrium control input
        self.u_ref = np.tile(self.u_eq, (1, self.n_horizon))  # reference control input

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
        """Setup the dynamics model using the thrust model with angular velocity as a state variable."""
        # Define state variables
        pos = ca.MX.sym("pos", 3)
        vel = ca.MX.sym("vel", 3)
        eul_ang = ca.MX.sym("eul_ang", 3)

        # Define Control variables
        u = ca.MX.sym("f", 4)  # thrust of each rotor
        eq = 0.25 * self.mass * self.g  # equilibrium control input per motor
        self.u_eq = np.array([eq, eq, eq, eq]).reshape(
            self.nu, 1
        )  # equilibrium control input, drone hovers
        self.u_ref = np.tile(self.u_eq, (1, self.n_horizon))  # reference control input
        beta = self.arm_length / ca.sqrt(2.0)  # beta = l/sqrt(2)
        torques = ca.vertcat(
            beta * (u[0] + u[1] - u[2] - u[3]),
            beta * (-u[0] + u[1] + u[2] - u[3]),
            self.gamma * (u[0] - u[1] + u[2] - u[3]),
        )  # torques in the body frame
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
            self.x = ca.vertcat(pos, vel, eul_ang, w)  # state vector
        else:
            deul_ang = ca.MX.sym("deul_ang", 3)
            w = W1s(eul_ang) @ deul_ang  # convert euler angle rate to body frame

            dx = ca.vertcat(
                vel,
                ca.vertcat(0, 0, -self.g)
                + Rbi(eul_ang[0], eul_ang[1], eul_ang[2])
                @ ca.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) / self.mass),
                deul_ang,
                dW2s(eul_ang, deul_ang) @ w
                + W2s(eul_ang) @ (self.J_inv @ (ca.cross(self.J @ w, w) + torques)),
            )
            self.x = ca.vertcat(pos, vel, eul_ang, deul_ang)  # state vector
        self.dx = dx
        self.u = u
        self.modelName = "MPCThrustEuler"
        return None

    # def UncertainThrustEulerDynamics(self):
    #     """Setup the dynamics model using the thrust model with angular velocity as a state variable."""
    #     # Define uncertain parameters
    #     mass = ca.MX.sym("mass", 1)  # mass of the drone
    #     Jd = ca.MX.sym("Jd", 3)  # diagonal elements of the moment of inertia matrix

    #     # Define state variables
    #     pos = ca.MX.sym("pos", 3)
    #     vel = ca.MX.sym("vel", 3)
    #     eul_ang = ca.MX.sym("eul_ang", 3)

    #     # Define Control variables
    #     u = ca.MX.sym("f", 4)  # thrust of each rotor
    #     eq = 0.25 * self.mass * self.g  # equilibrium control input per motor
    #     self.u_eq = np.array([eq, eq, eq, eq]).reshape(
    #         self.nu, 1
    #     )  # equilibrium control input, drone hovers
    #     self.u_ref = np.tile(self.u_eq, (1, self.n_horizon))  # reference control input
    #     beta = self.arm_length / ca.sqrt(2.0)  # beta = l/sqrt(2)
    #     torques = ca.vertcat(
    #         beta * (u[0] + u[1] - u[2] - u[3]),
    #         beta * (-u[0] + u[1] + u[2] - u[3]),
    #         self.gamma * (u[0] - u[1] + u[2] - u[3]),
    #     )  # torques in the body frame
    #     if self.useAngVel:
    #         w = ca.MX.sym("w", 3)
    #         dx = ca.vertcat(
    #             vel,
    #             ca.vertcat(0, 0, -self.g)
    #             + Rbi(eul_ang[0], eul_ang[1], eul_ang[2])
    #             @ ca.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) / mass),
    #             W2s(eul_ang) @ w,
    #             ca.diag(1 / Jd) @ (torques - (ca.skew(w) @ ca.diag(Jd) @ w)),
    #         )
    #         self.x = ca.vertcat(pos, vel, eul_ang, w)  # state vector
    #     else:
    #         deul_ang = ca.MX.sym("deul_ang", 3)
    #         w = W1s(eul_ang) @ deul_ang  # convert euler angle rate to body frame

    #         dx = ca.vertcat(
    #             vel,
    #             ca.vertcat(0, 0, -self.g)
    #             + Rbi(eul_ang[0], eul_ang[1], eul_ang[2])
    #             @ ca.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) / mass),
    #             deul_ang,
    #             dW2s(eul_ang, deul_ang) @ w
    #             + W2s(eul_ang) @ (ca.diag(1 / Jd) @ (ca.cross(ca.diag(Jd) @ w, w) + torques)),
    #         )
    #         self.x = ca.vertcat(pos, vel, eul_ang, deul_ang)  # state vector
    #     self.dx = dx
    #     self.p = ca.vertcat(mass, Jd)
    #     self.u = u
    #     self.modelName = "MPCThrustEuler"
    #     return None

    def setupBaseConstraints(self):
        # Setup base constraints and scaling factors
        rpm_lb = 4070.3 + 0.2685 * 0
        rpm_ub = 4070.3 + 0.2685 * 65535
        self.thrust_lb = self.ct * (rpm_lb**2)
        self.thrust_ub = self.ct * (rpm_ub**2)
        if self.useTorqueModel:
            tthrust_lb = 0.3 * self.mass * self.g
            tthrust_ub = 1.5 * self.mass * self.g
            self.u_lb = np.array([tthrust_lb, -0.1, -0.1, -0.1]).T
            self.u_ub = np.array([tthrust_ub, 0.1, 0.1, 0.1]).T
            self.u_scal = self.u_ub - self.u_lb
        else:
            self.u_lb = np.array([self.thrust_lb, self.thrust_lb, self.thrust_lb, self.thrust_lb]).T
            self.u_ub = np.array([self.thrust_ub, self.thrust_ub, self.thrust_ub, self.thrust_ub]).T
            self.u_scal = self.u_ub - self.u_lb

        x_y_max = 3.0
        z_max = 2.5
        z_min = 0.05  # minimum height, safety margin, soft constraints required
        eul_ang_max = 85 / 180 * np.pi
        large_val = 1e3
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
        self.u_rate_ub = (self.u_ub - self.u_lb) * 0.3
        self.u_rate_lb = -(self.u_ub - self.u_lb) * 0.3
        self.u_rate_scal = self.u_rate_ub - self.u_rate_lb
        self.du_ref = np.zeros((self.nu, self.n_horizon))
        return None

    def set_target_trajectory(self, t_total: float = 8) -> None:
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
        # self.t_total = t_total
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
        pos_des = self.target_trajectory(t_horizon).T
        # print("pos_des shape: ", pos_des.shape)
        # Handle the case where the end time exceeds the total time
        if t_horizon[-1] > self.t_total:
            last_value = self.target_trajectory(self.t_total).reshape(3, 1)
            n_repeat = np.sum(t_horizon > self.t_total)
            pos_des[:, -n_repeat:] = np.tile(last_value, (1, n_repeat))
        # print(reference_trajectory_horizon)

        # print("x_ref shape: ", self.x_ref.shape)
        self.x_ref[:3, :] = pos_des
        self.n_step += 1
        return None

    def setupNominalParameters(self):
        """Setup the unchanging parameters of the drone/environment controller."""
        # Define the (nominal) system parameters
        self.mass = self.initial_info["drone_mass"]  # kg
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

        self.nx = 12  # number of states
        self.nu = 4  # number of control inputs
        self.ny = self.nx + self.nu  # number of

        self.obstacles_pos = self.initial_obs["obstacles_pos"]  # obstacles positions in
        self.obstacles_visited = self.initial_obs["obstacles_visited"]
        self.gates_pos = self.initial_obs["gates_pos"]
        self.gates_rpy = self.initial_obs["gates_rpy"]
        self.gates_visited = self.initial_obs["gates_visited"]

    def setupCostFunction(self):
        """Setup the cost function for the MPC controller."""
        # Define the cost function parameters
        Qs = (
            np.array([1, 1, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1]) / self.x_scal
        )  # stage cost for states (position, velocity, euler angles, angular velocity)
        Qt = np.array([1, 1, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1]) / self.x_scal
        R = np.array([1e-2, 1e-2, 1e-2, 1e-2]) / self.u_scal
        dR = np.array([1e-2, 1e-2, 1e-2, 1e-2]) / self.u_rate_scal
        self.Qs = np.diag(Qs)
        self.Qt = np.diag(Qt)  # terminal cost
        self.Rs = np.diag(R)
        self.Rsdelta = np.diag(dR)

        self.cost_function = self.baseCost  # cost function for the MPC

        # Constraint parameters

        # Cost parameters

    def baseCost(self, x, x_ref, Q, u, u_ref, R):
        """Base Cost function for the MPC controller."""
        return ca.mtimes([(x - x_ref).T, Q, (x - x_ref)]) + ca.mtimes(
            [(u - u_ref).T, R, (u - u_ref)]
        )

    def setupControlRates(self):
        """Setup the augmented dynamics for control rates as control inputs."""
        self.nx += self.nu
        self.ny = self.nx + self.nu
        self.x = ca.vertcat(self.x, self.u)
        self.u = ca.MX.sym("du", self.nu)
        self.x_ref = np.vstack(
            (self.x_ref, np.hstack((self.u_ref, self.u_ref[:, -1].reshape(self.nu, 1))))
        )
        self.u_ref = np.zeros((self.nu, self.n_horizon))

        self.u_eq = np.zeros((self.nu, 1))
        self.dx = ca.vertcat(self.dx, self.u)
        self.Qs = scipy.linalg.block_diag(self.Qs, self.Rs)
        self.Qt = scipy.linalg.block_diag(self.Qt, self.Rs)
        self.Rs = self.Rsdelta
        self.x_lb = np.hstack((self.x_lb, self.u_lb))
        self.x_ub = np.hstack((self.x_ub, self.u_ub))
        self.x_scal = np.hstack((self.x_scal, self.u_scal))
        self.u_lb = self.u_rate_lb
        self.u_ub = self.u_rate_ub
        self.u_scal = self.u_rate_scal
        return None
