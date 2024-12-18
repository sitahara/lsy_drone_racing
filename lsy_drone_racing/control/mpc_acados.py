"""MPC controller using acados and native casadi for the drone racing environment."""

import casadi as ca
import numpy as np
import pybullet as p
import scipy as sp
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.control.utils import (
    W1,
    R_body_to_inertial,
    Rbi,
    W2_dot_symb,
    W2s,
    rpm_to_torques_mat,
    rungeKutta4,
)

# from lsy_drone_racing.sim.drone import Drone
# from lsy_drone_racing.sim.physics import GRAVITY


class MPCController(BaseController):
    """Model Predictive Controller implementation using CasADi and acados."""

    def __init__(self, initial_obs: NDArray[np.floating], initial_info: dict):  # noqa: D107
        super().__init__(initial_obs, initial_info)

        # Inital parameters
        # whether to use the torque (total thrust + body torques) or thrust model (individual rotor thrusts)
        self.useTorqueModel = False
        self.t_step = 1 / 60  # Time step, 60Hz
        self.n_horizon = 60  # Prediction horizon, 1s
        self.nx = 12  # number of states
        self.nu = 4  # number of control inputs
        self.ny = self.nx + self.nu  # number of outputs
        self.soft_constraints = False  # whether to use soft constraints
        self.soft_penalty = 1e3  # penalty for soft constraints
        self.Qs = np.diag(
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01]
        )  # stage cost for states (position, velocity, euler angles, angular velocity)
        self.Qt = self.Qs  # terminal cost
        self.Rs = np.diag([0.01, 0.01, 0.01, 0.01]) * 0  # control cost
        self.Rsdelta = np.diag([0.1, 0.1, 0.1, 0.1]) * 0  # control rate cost
        # For warm start
        self.x_guess = None
        self.u_guess = None

        # self.cycles_to_update = 3  # after how many control calls the optimization problem is reexecuted)
        # self.tick = 0  # counter for the cycles to update

        # Define the (nominal) system parameters
        self.mass = 0.027  # kg, drone.nominal_params.mass
        self.g = 9.81  # GRAVITY  # m/s^2

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

        self.setupAcadosModel()
        x0 = np.concatenate(
            [initial_obs["pos"], initial_obs["vel"], initial_obs["rpy"], initial_obs["ang_vel"]]
        )

        self.setupAcadosOptimizer(x0=x0)
        # Create solver
        self.ocp_solver = AcadosOcpSolver(self.ocp)

        # Set the target trajectory
        self.set_target_trajectory()

    def reset(self):
        """Reset the MPC controller to its initial state."""
        self.x_guess = None
        self.u_guess = None

        self.model = None
        self.ocp = None
        self.ocp_solver = None

        self.set_target_trajectory()
        # Setup the acados model and optimizer
        self.setupAcadosModel()
        x0 = np.concatenate(
            [self.obs["pos"], self.obs["vel"], self.obs["rpy"], self.obs["ang_vel"]]
        )
        self.setupAcadosOptimizer(x0=x0)
        # Create solver
        self.ocp_solver = AcadosOcpSolver(self.ocp)

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
        self.obs = obs
        pos = obs["pos"]  # position in world frame
        vel = obs["vel"]  # velocity in world frame
        eul_ang = obs["rpy"]  # euler angles roll, pitch, yaw
        deul_ang = obs["ang_vel"]  # angular velocity in body frame

        if not self.useTorqueModel:
            deul_ang = (
                W1(eul_ang) @ deul_ang
            )  # convert euler angle rate to body frame angular velocity

        current_state = np.concatenate([pos, vel, eul_ang, deul_ang.flatten()])

        # Update solver with current state
        self.ocp_solver.set(0, "lbx", current_state)
        self.ocp_solver.set(0, "ubx", current_state)

        # Check whether initial guess is available
        if self.x_guess is None or self.u_guess is None:
            target_trajectory = self.updateTargetTrajectory()
            self.x_guess = np.vstack([target_trajectory, np.zeros((9, self.n_horizon + 1))])
            self.u_guess = np.ones((self.nu, self.n_horizon)) * self.u_eq[0]
            print("Initial guess", self.x_guess[:, 0:3], self.u_guess[:, 0:3])
            # self.u_guess = np.zeros((self.nu, self.n_horizon))

            for k in range(self.n_horizon):
                self.ocp_solver.set(k, "x", self.x_guess[:, k])
                self.ocp_solver.set(k, "u", self.u_guess[:, k])
            self.ocp_solver.set(self.n_horizon, "x", self.x_guess[:, self.n_horizon])

        # Update the target trajectory
        pos_des = self.updateTargetTrajectory()

        # Set the reference trajectory for stage and terminal states
        # Correct update checked
        for k in range(self.n_horizon):
            y_ref = np.concatenate([pos_des[:, k], np.zeros(9), self.u_eq])
            self.ocp_solver.set(k, "yref", y_ref)
        self.ocp_solver.set(
            self.n_horizon, "yref", np.concatenate([pos_des[:, self.n_horizon], np.zeros(9)])
        )

        # Solve the OCP
        status = self.ocp_solver.solve()
        if status not in [0, 2]:
            self.ocp_solver.print_statistics()
            raise Exception(f"acados failed with status {status}. Exiting.")
        if status == 2:
            # print(f"acados returned status {status}. ")
            pass

        # Extract the control input
        u = self.ocp_solver.get(0, "u")

        # Store the current solution as the initial guess for the next iteration
        self.x_guess = np.zeros((self.nx, self.n_horizon + 1))
        self.u_guess = np.zeros((self.nu, self.n_horizon))
        for k in range(self.n_horizon):
            self.x_guess[:, k] = self.ocp_solver.get(k, "x")
            self.u_guess[:, k] = self.ocp_solver.get(k, "u")
        self.x_guess[:, self.n_horizon] = self.ocp_solver.get(self.n_horizon, "x")
        print("Guessed position: ", self.x_guess[0:3, :10])
        # Extract the next predicted states from the solver
        next_pos = self.ocp_solver.get(1, "x")[0:3]
        next_vel = self.ocp_solver.get(1, "x")[3:6]
        acc = (self.ocp_solver.get(1, "x")[3:6] - self.ocp_solver.get(0, "x")[3:6]) / self.t_step
        next_eul_ang = self.ocp_solver.get(1, "x")[6:9]
        next_deul_ang = self.ocp_solver.get(1, "x")[9:12]

        # action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] to follow.
        # where ax, ay, az are the acceleration in world frame, rrate, prate, yrate are the roll, pitch, yaw rate in body frame
        if self.useTorqueModel:
            next_deul_ang = (
                W1(next_eul_ang) @ next_deul_ang
            )  # convert euler angle rate to body frame angular velocity
        action = np.concatenate(
            [next_pos, next_vel, acc, [next_eul_ang[2]], next_deul_ang.flatten()]
        )
        action = np.concatenate([next_pos, np.zeros(10)])
        print(f"Current position: {pos}")
        print(f"Desired position: {pos_des[:, 0]}")
        print(f"Next position: {next_pos}")

        # self.tick = (self.tick + 1) % self.cycles_to_update
        return action.flatten()

    def setupAcadosModel(self):
        """Setup the acados model including dynamics.

        This version (Ver2) uses
            State variables:
                - pos
                - vel
                - euler angles in the world frame
                - angular velocities in the body frame
            Control inputs:
                - forces in the body frame: f1, f2, f3, f4
        """
        model = AcadosModel()
        if self.useTorqueModel:
            model.name = "quadrotor_torque_mpc"
            # Define state variables and dynamics
            pos = ca.MX.sym("pos", 3)  # position in world frame
            vel = ca.MX.sym("vel", 3)  # velocity in world frame
            eul_ang = ca.MX.sym("eul_ang", 3)  # euler angles roll, pitch, yaw
            deul_ang = ca.MX.sym("deul_ang", 3)  # euler angle rates in world frame
            model.x = ca.vertcat(pos, vel, eul_ang, deul_ang)
            dx = ca.MX.sym("dx", 12)  # state derivative
            w = ca.MX.sym("w", 3)  # Body Angular velocities
            # Define Control variables
            thrust = ca.MX.sym("thrust", 1)  # total thrust
            torques = ca.MX.sym("torques", 3)  # body frame torques
            model.u = ca.vertcat(thrust, torques)  # control input
            self.u_eq = np.array([self.mass * self.g, 0, 0, 0]).T  # equilibrium control input

            # Define Dynamics in world frame as euler angles
            w = W1(eul_ang) @ deul_ang  # Body Angular velocity
            dx = ca.vertcat(
                vel,
                ca.vertcat(0, 0, -self.g)
                + R_body_to_inertial(eul_ang) @ ca.vertcat(0, 0, thrust / self.mass),
                deul_ang,
                W2_dot_symb(eul_ang, deul_ang) @ w
                + W2s(eul_ang) @ (self.J_inv @ (ca.cross(self.J @ w, w) + torques)),
            )
        else:
            model.name = "quadrotor_thrust_mpc"
            # Define state variables and dynamics
            pos = ca.MX.sym("pos", 3)  # position in world frame
            vel = ca.MX.sym("vel", 3)  # velocity in world frame

            phi = ca.MX.sym("phi")
            theta = ca.MX.sym("theta")
            psi = ca.MX.sym("psi")
            eul_ang = ca.vertcat(phi, theta, psi)  # euler angles roll, pitch, yaw
            # eul_ang = ca.MX.sym("eul_ang", 3)  # euler angles roll, pitch, yaw
            w = ca.MX.sym("w", 3)  # body angular velocities

            model.x = ca.vertcat(pos, vel, eul_ang, w)
            # dx = ca.MX.sym("dx", 12)  # state derivative

            # Define Control variables
            f1 = ca.MX.sym("f1")  # motor 1 thrust
            f2 = ca.MX.sym("f2")  # motor 2 thrust
            f3 = ca.MX.sym("f3")  # motor 3 thrust
            f4 = ca.MX.sym("f4")  # motor 4 thrust
            model.u = ca.vertcat(f1, f2, f3, f4)  # control

            eq = 0.25 * self.mass * self.g  # equilibrium control input per motor
            self.u_eq = np.array([eq, eq, eq, eq]).T  # equilibrium control input, drone hovers
            beta = self.arm_length / ca.sqrt(2.0)  # beta = l/sqrt(2)
            torques = ca.vertcat(
                beta * (f1 + f2 - f3 - f4),
                beta * (-f1 + f2 + f3 - f4),
                self.gamma * (f1 - f2 + f3 - f4),
            )  # torques in the body frame
            # Define Dynamics
            Rbdin = Rbi(phi, theta, psi)
            dx = ca.vertcat(
                vel,
                ca.vertcat(0, 0, -self.g)
                + Rbdin @ (ca.vertcat(0, 0, (f1 + f2 + f3 + f4)) / self.mass),
                W2s(phi, theta) @ w,
                self.J_inv @ (torques - (ca.skew(w) @ self.J @ w)),
            )
            # model.xdot = dx

        self.cont_dyn = ca.Function("cont_dyn", [model.x, model.u], [dx], ["x", "u"], ["dx"])
        model.f_expl_expr = dx  # continuous-time explicit dynamics
        model.disc_dyn_expr, self.disc_dyn = rungeKutta4(
            model.x, model.u, self.t_step, self.cont_dyn
        )  # discrete-time dynamics
        xdot = ca.MX.sym("xdot", dx.shape)
        model.f_impl_expr = xdot - dx  # continuous-time implicit dynamics

        self.model = model

    def setupAcadosOptimizer(self, x0: NDArray[np.floating]):
        """Setup the acados optimizer with the given initial state.

        Args:
            x0: The initial state of the system.
        """
        if not self.useTorqueModel:
            w = W1(x0[6:9]) @ x0[9:]  # convert euler angle rate to body frame angular velocity
            x0[9:] = ca.reshape(w, -1, 1).full().flatten()
        ocp = AcadosOcp()
        ocp.model = self.model
        # Set solver options
        ocp.solver_options.N_horizon = self.n_horizon  # number of control intervals
        ocp.solver_options.tf = self.n_horizon * self.t_step  # prediction horizon
        ocp.solver_options.integrator_type = "ERK"  # "ERK", "IRK", "GNSF", "DISCRETE"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # "EXACT", "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
        ocp.solver_options.nlp_solver_max_iter = 20  # TODO: optimize
        ocp.solver_options.globalization = (
            "MERIT_BACKTRACKING"  # "FIXED_STEP", "MERIT_BACKTRACKING"
        )

        # Set cost
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        # Define the reference values and weight matrices for the least squares cost

        ocp.cost.yref = np.zeros(
            (self.ny,)
        )  # dummy reference, states and controls are considered for stage cost
        ocp.cost.yref_e = np.zeros(
            (self.nx,)
        )  # dummy reference, only states are considered for terminal cost
        ocp.cost.W = sp.linalg.block_diag(self.Qs, self.Rs)  # Weight matrix for stage cost
        ocp.cost.W_e = self.Qt  # Weight matrix for terminal cost

        ocp.cost.Vx = np.zeros((self.ny, self.nx))
        ocp.cost.Vx[: self.nx, : self.nx] = np.eye(self.nx)
        ocp.cost.Vx_e = np.eye(self.nx)

        ocp.cost.Vu = np.zeros((self.ny, self.nu))
        ocp.cost.Vu[self.nx :, : self.nu] = np.eye(self.nu)

        # Set constraints
        rpm_min = 4070.3**2  # lower bound for rotor rates
        rpm_max = (4070.3 + 0.2685 * 65535) ** 2  # upper bound for rotor rates
        thrust_lb = 0.3 * 0.25 * self.mass * self.g  # lower bound for thrust to avoid tumbling
        thrust_ub = self.ct * rpm_max  # upper bound for thrust as maximum achievable
        torque_xy_lb = -self.c_tau_xy * 2 * (rpm_max - rpm_min)
        torque_xy_ub = self.c_tau_xy * 2 * (rpm_max - rpm_min)
        torque_z_lb = -self.cd * 2 * (rpm_max - rpm_min)
        torque_z_ub = self.cd * 2 * (rpm_max - rpm_min)
        # Define control constraints: thrust_model: individual rotor thrusts, torque_model: total thrust and torques
        if self.model.name == "quadrotor_thrust_mpc":
            ocp.constraints.lbu = np.array([thrust_lb, thrust_lb, thrust_lb, thrust_lb])
            ocp.constraints.ubu = np.array([thrust_ub, thrust_ub, thrust_ub, thrust_ub])
            ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        elif self.model.name == "quadrotor_torque_mpc":
            ocp.constraints.lbu = np.array([thrust_lb * 4, torque_xy_lb, torque_xy_lb, torque_z_lb])
            ocp.constraints.ubu = np.array([thrust_ub * 4, torque_xy_ub, torque_xy_ub, torque_z_ub])
            ocp.constraints.idxbu = np.array([1, 2, 3, 4])
        else:
            raise ValueError(f"Unknown model name {self.model.name}")

        x_y_max = 3
        z_max = 2.5
        z_min = 0.01
        rpy_max = 85 / 180 * np.pi
        large_val = 1e6
        ocp.constraints.lbx = np.array(
            [
                -x_y_max,
                -x_y_max,
                z_min,
                -large_val,
                -large_val,
                -large_val,
                -rpy_max,
                -rpy_max,
                -rpy_max,
                -large_val,
                -large_val,
                -large_val,
            ]
        )
        ocp.constraints.ubx = np.array(
            [
                x_y_max,
                x_y_max,
                z_max,
                large_val,
                large_val,
                large_val,
                rpy_max,
                rpy_max,
                rpy_max,
                large_val,
                large_val,
                large_val,
            ]
        )
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        if self.soft_constraints:
            # Define slack variables
            ocp.constraints.lsbx = np.zeros(self.nx)
            ocp.constraints.usbx = np.zeros(self.nx)
            ocp.constraints.idxsbx = np.arange(self.nx)

            ocp.constraints.lsbu = np.zeros(self.nu)
            ocp.constraints.usbu = np.zeros(self.nu)
            ocp.constraints.idxsbu = np.arange(self.nu)

            # Define the Jsh matrix
            ocp.cost.Jsh = np.eye(self.nx + self.nu)

            # Define the penalty matrices Zl and Zu
            ocp.cost.Zl = self.soft_penalty * np.eye(self.nx + self.nu)
            ocp.cost.Zu = self.soft_penalty * np.eye(self.nx + self.nu)

        # Set initial state
        ocp.constraints.x0 = x0

        # Code generation
        ocp.code_export_directory = "generated_code/mpc_acados"
        self.ocp = ocp

        return None

    def updateTargetTrajectory(self) -> NDArray[np.floating]:
        """Update the target trajectory for the MPC controller."""
        current_time = self.n_step * self.t_step
        t_horizon = np.linspace(
            current_time, current_time + self.n_horizon * self.t_step, self.n_horizon + 1
        )

        # Evaluate the spline at the time points
        reference_trajectory_horizon = self.target_trajectory(t_horizon)

        # Handle the case where the end time exceeds the total time
        if t_horizon[-1] > self.t_total:
            last_value = self.target_trajectory(self.t_total).reshape(3, 1)
            n_repeat = np.sum(t_horizon > self.t_total)
            repeated_values = np.tile(last_value, (1, n_repeat))
            reference_trajectory_horizon[-n_repeat:, :] = repeated_values.T
        # print(reference_trajectory_horizon)
        self.n_step += 1
        return reference_trajectory_horizon.T

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

    def setupIPOPToptimizer(self, current_state: NDArray[np.floating]):
        """Setup the IPOPT optimizer for the MPC controller."""
        # Define the optimization variables
        opti = ca.Opti()
        X = opti.variable(self.nx, self.n_horizon + 1)
        U = opti.variable(self.nu, self.n_horizon)
        # x_ref = opti.parameter(self.nx, self.n_horizon + 1)
        # u_ref = opti.parameter(self.nu, 1)
        # Slack variables for soft constraints
        slack_x = self.soft_penalty * opti.variable(len(self.ocp.constraints.idxbx))
        slack_u = self.soft_penalty * opti.variable(len(self.ocp.constraints.idxbu))

        self.n_step = 0
        target_trajectory = self.updateTargetTrajectory()
        target_trajectory = np.vstack([target_trajectory, np.zeros((9, self.n_horizon + 1))])
        u_ref = self.u_eq
        # Define the objective function
        obj = 0
        for k in range(self.n_horizon):
            x_k = X[:, k]
            u_k = U[:, k]
            ref_k = target_trajectory[:, k]
            obj += ca.mtimes([(x_k - ref_k).T, self.Qs, (x_k - ref_k)]) + ca.mtimes(
                [(u_k - u_ref).T, self.Rs, (u_k - u_ref)]
            )
        obj += ca.mtimes(
            [
                (X[:, -1] - target_trajectory[:, -1]).T,
                self.Qt,
                (X[:, -1] - target_trajectory[:, -1]),
            ]
        )

        # Define the constraints
        opti.subject_to(X[:, 0] == current_state)
        for k in range(self.n_horizon):
            # Dynamics constraints
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]
            f_k = self.disc_dyn(x_k, u_k)
            opti.subject_to(x_next == f_k)
            # State constraints
            opti.subject_to(self.ocp.constraints.lbx <= x_k <= self.ocp.constraints.ubx)
            # Control constraints
            opti.subject_to(self.ocp.constraints.lbu <= u_k <= self.ocp.constraints.ubu)
        # Final state constraints
        opti.subject_to(self.ocp.constraints.lbx <= X[:, -1] <= self.ocp.constraints.ubx)

        opti.minimize(obj)
        # Set the IPOPT options
        opts = {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-4,
            "ipopt.max_iter": 25,
            "ipopt.linear_solver": "mumps",
        }
        opti.solver("ipopt", opts)

        self.opti = opti

    def calculateInitialGuess(self, current_state: NDArray[np.floating]):
        """Calculate the initial guess for the optimization problem.

        Args:
            current_state: The current state of the system.
        """
        self.setupIPOPToptimizer(current_state=current_state)

        sol = self.opti.solve()

        # Extract the solution
        x_sol = np.reshape(
            sol["X"][: self.nx * (self.n_horizon + 1)], (self.nx, self.n_horizon + 1)
        )
        u_sol = np.reshape(sol["U"][self.nx * (self.n_horizon + 1) :], (self.nu, self.n_horizon))

        # Set the initial guess for the acados optimizer
        self.x_guess = x_sol
        self.u_guess = u_sol

        # Debugging information
        print("Initial guess for states (X):", x_sol)
        print("Initial guess for controls (U):", u_sol)
