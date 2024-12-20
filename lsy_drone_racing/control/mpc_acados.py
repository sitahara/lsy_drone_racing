"""MPC controller using acados and native casadi for the drone racing environment."""

import casadi as ca
import numpy as np
import pybullet as p
import scipy as sp
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.mpc_base import MPC_BASE
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


class MPC_ACADOS(MPC_BASE):
    """Model Predictive Controller implementation using CasADi and acados."""

    def __init__(self, initial_obs: NDArray[np.floating], initial_info: dict):
        # Initialize the base class
        super().__init__(initial_obs, initial_info)
        super().setupDynamics()
        # Setup the acados model, optimizer, and solver
        self.setupAcadosModel()
        self.setupAcadosOptimizer()

        # Setup the IPOPT optimizer for initial guess
        super().setupIPOPToptimizer()
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
            The next action either as 13-array (Mellinger Interface) or as 4-array (Thrust Interface).
        """
        # Get current observations
        self.obs = obs
        # Update the current state
        self.current_state = np.concatenate([obs["pos"], obs["vel"], obs["rpy"], obs["ang_vel"]])
        # Updates x_ref, the current target trajectory and upcounts the trajectory tick
        super().updateTargetTrajectory()
        if not self.useTorqueModel:
            # convert euler angle rate to body frame angular velocity
            ang_vel = W1(obs["rpy"]) @ obs["ang_vel"]
            self.current_state = np.concatenate(
                [obs["pos"], obs["vel"], obs["rpy"], ang_vel.flatten()]
            )
        if self.x_guess is None or self.u_guess is None:
            # Use IPOPT optimizer to get initial guess
            action = super().stepIPOPT()
        else:
            # Use acados to get the next action using the previous solution as initial guess
            action = self.stepAcados()

        if self.useMellinger:
            # action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] to follow.
            # where ax, ay, az are the acceleration in world frame, rrate, prate, yrate are the roll, pitch, yaw rate in body frame
            acc = (self.x_guess[3:6, 0] - action[3:6]) / self.ts
            action = np.concatenate([action[:6], acc, [action[9]], action[10:]])
        else:
            # action: [thrust, tau_des]
            action = np.array(action)

        print(f"Current position: {self.current_state[:3]}")
        print(f"Desired position: {self.x_ref[:3, 1]}")
        print(f"Next position: {action[:3]}")

        # self.tick = (self.tick + 1) % self.cycles_to_update
        return action.flatten()

    def setupAcadosModel(self):
        """Setup the acados model using a selected dynamics model."""
        model = AcadosModel()
        model.name = self.modelName
        model.x = self.x
        model.u = self.u
        model.xdot = self.dx

        model.f_expl_expr = self.dx  # continuous-time explicit dynamics
        model.dyn_disc_fun = self.dx_d  # discrete-time dynamics

        xdot = ca.MX.sym("xdot", self.nx)
        # Continuous implicit dynamic expression
        model.f_impl_expr = xdot - self.dx
        self.model = model

    def setupAcadosOptimizer(self):
        """Setup the acados optimizer (parameters, costs, constraints) given the class parameters set by the setupDynamics function.

        Args:
            x0: The initial state of the system.
        """
        ocp = AcadosOcp()
        ocp.model = self.model
        # Set solver options
        ocp.solver_options.N_horizon = self.n_horizon  # number of control intervals
        ocp.solver_options.tf = self.n_horizon * self.ts  # prediction horizon
        ocp.solver_options.integrator_type = "ERK"  # "ERK", "IRK", "GNSF", "DISCRETE"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # "EXACT", "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP, SQP_RTI
        # ocp.solver_options.nlp_solver_max_iter = 20  # TODO: optimize
        ocp.solver_options.globalization = (
            "MERIT_BACKTRACKING"  # "FIXED_STEP", "MERIT_BACKTRACKING"
        )
        ocp.solver_options.tol = 1e-4  # tolerance

        # Set cost
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        # Define the reference values and weight matrices for the least squares cost

        ocp.cost.yref = np.zeros(
            (self.ny, self.n_horizon)
        )  # dummy reference, states and controls are considered for stage cost
        ocp.cost.yref_e = np.zeros(
            (self.nx, 1)
        )  # dummy reference, only states are considered for terminal cost
        ocp.cost.W = sp.linalg.block_diag(self.Qs, self.Rs)  # Weight matrix for stage cost
        ocp.cost.W_e = self.Qt  # Weight matrix for terminal cost

        # ocp.cost.Vx = np.zeros((self.ny, self.nx))
        # ocp.cost.Vx[: self.nx, : self.nx] = np.eye(self.nx)
        # ocp.cost.Vx_e = np.eye(self.nx)

        # ocp.cost.Vu = np.zeros((self.ny, self.nu))
        # ocp.cost.Vu[self.nx :, : self.nu] = np.eye(self.nu)

        # Set constraints
        # rpm_min = 4070.3**2  # lower bound for rotor rates
        # rpm_max = (4070.3 + 0.2685 * 65535) ** 2  # upper bound for rotor rates
        # thrust_lb = 0.3 * 0.25 * self.mass * self.g  # lower bound for thrust to avoid tumbling
        # thrust_ub = self.ct * (rpm_max**2)  # upper bound for thrust as maximum achievable
        # torque_xy_lb = -self.c_tau_xy * 2 * (rpm_max - rpm_min)
        # torque_xy_ub = self.c_tau_xy * 2 * (rpm_max - rpm_min)
        # torque_z_lb = -self.cd * 2 * (rpm_max - rpm_min)
        # torque_z_ub = self.cd * 2 * (rpm_max - rpm_min)
        # Define control constraints: thrust_model: individual rotor thrusts, torque_model: total thrust and torques
        # Basic State and control constraints
        ocp.constraints.lbu = self.u_lb
        ocp.constraints.ubu = self.u_ub
        ocp.constraints.lbx = self.x_lb
        ocp.constraints.lbx_e = self.x_lb
        ocp.constraints.ubx = self.x_ub
        ocp.constraints.ubx_e = self.x_ub

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
        ocp.constraints.x0 = self.x0

        # Code generation
        ocp.code_export_directory = "generated_code/mpc_acados"
        self.ocp = ocp
        self.ocp_solver = AcadosOcpSolver(self.ocp)
        self.ocp_integrator = AcadosSimSolver(self.ocp)

        return None

    def stepAcados(self) -> NDArray[np.floating]:
        """Performs one optimization step using the current state, reference trajectory, and the previous solution for warmstarting. Updates the previous solution and returns the control input."""
        # Set initial state
        self.ocp_solver.set(0, "lbx", self.current_state)
        self.ocp_solver.set(0, "ubx", self.current_state)

        # Set reference trajectory
        y_ref = np.vstack([self.x_ref[:, :-1], self.u_eq * np.ones((1, self.n_horizon))])
        y_ref_e = self.x_ref[:, -1]
        for i in range(self.n_horizon + 1):
            self.ocp_solver.set(i, "yref", y_ref[:, i])
        self.ocp_solver.set(self.n_horizon, "yref", y_ref_e)

        # Set initial guess (u_guess/x_guess are the previous solution moved one step forward)
        self.ocp_solver.set(0, "x", self.current_state)
        for k in range(self.n_horizon):
            self.ocp_solver.set(k + 1, "x", self.x_guess[:, k])
            self.ocp_solver.set(k, "u", self.u_guess[:, k])

        # Solve the OCP
        if self.ocp.solver_options.nlp_solver_type == "SQP_RTI":
            # phase 1
            self.ocp_solver.options_set("rti_phase", 1)
            status = self.ocp_solver.solve()

            # phase 2
            self.ocp_solver.options_set("rti_phase", 2)
            status = self.ocp_solver.solve()
        else:
            status = self.ocp_solver.solve()

        if status not in [0, 2]:
            self.ocp_solver.print_statistics()
            raise Exception(f"acados failed with status {status}. Exiting.")

        # Update previous solution
        for k in range(self.n_horizon + 1):
            self.x_last[:, k] = self.ocp_solver.get(k, "x")
        for k in range(self.n_horizon):
            self.u_last[:, k] = self.ocp_solver.get(k, "u")

        # Update the initial guess
        self.x_guess = np.hstack((self.x_last[:, 2:], self.x_last[:, -1]))
        self.u_guess = np.hstack((self.x_last[:, 1:], self.x_last[:, -1]))

        # Extract the control input
        if self.useMellinger:
            action = self.ocp_solver.get(1, "x")
        else:
            action = self.ocp_solver.get(0, "u")
        self.last_action = action
        return action
