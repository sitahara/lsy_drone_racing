from abc import ABC, abstractmethod

import casadi as ca
import l4acados as l4a
import numpy as np
import scipy
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSimSolver,
    ZoroDescription,
)
from acados_template.utils import ACADOS_INFTY
from lsy_drone_racing.mpc_utils.dynamics_classes import BaseDynamics
from lsy_drone_racing.mpc_utils.costs_classes import BaseCost
from numpy.typing import NDArray


class BaseOptimizer(ABC):
    """Abstract base class for optimizer implementations."""

    def __init__(
        self,
        dynamics: BaseDynamics,
        costs: BaseCost,
        optimizer_info: dict = {
            "useSoftConstraints": True,
            "softPenalty": 1e3,
            "useGP": False,
            "useZoro": False,
            "export_dir": "generated_code/mpc",
        },
    ):
        self.useSoftConstraints = optimizer_info.get("useSoftConstraints", True)
        self.softPenalty = optimizer_info.get("softPenalty", 1e3)
        self.dynamics = dynamics
        self.n_horizon = dynamics.n_horizon
        self.ts = dynamics.ts
        self.costs = costs
        self.nx = dynamics.nx
        self.nu = dynamics.nu
        self.ny = dynamics.ny
        self.x_guess = None
        self.u_guess = None
        self.x_last = None
        self.u_last = None

    @abstractmethod
    def setup_optimizer(self):
        """Setup the optimizer."""
        return NotImplementedError

    @abstractmethod
    def step(
        self,
        current_state: NDArray[np.floating],
        x_ref: NDArray[np.floating],
        u_ref: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Perform one optimization step.

        Args:
            current_state: The current state of the system.
            x_ref: The reference state trajectory.
            u_ref: The reference control trajectory.

        Returns:
            The optimized control input.
        """
        pass


class IPOPTOptimizer(BaseOptimizer):
    def __init__(
        self,
        dynamics: BaseDynamics,
        costs: BaseCost,
        optimizer_info: dict = {
            "useSoftConstraints": True,
            "softPenalty": 1e3,
            "useGP": False,
            "useZoro": False,
            "export_dir": "generated_code/mpc",
        },
    ):
        super().__init__(dynamics, costs, optimizer_info)
        self.setup_optimizer()

    def setup_optimizer(self):
        """Setup the IPOPT optimizer for the MPC controller."""
        opti = ca.Opti()

        # Define the optimization variables
        X = opti.variable(self.nx, self.n_horizon + 1)  # State trajectory
        U = opti.variable(self.nu, self.n_horizon)  # Control trajectory
        X_ref = opti.parameter(self.nx, self.n_horizon + 1)  # Reference trajectory
        U_ref = opti.parameter(self.nu, self.n_horizon)  # Reference control
        X0 = opti.parameter(self.nx, 1)  # Initial state
        X_lb = opti.parameter(self.nx, 1)  # State lower bound
        X_ub = opti.parameter(self.nx, 1)  # State upper bound
        U_lb = opti.parameter(self.nu, 1)  # Control lower bound
        U_ub = opti.parameter(self.nu, 1)  # Control upper bound
        if self.useSoftConstraints:
            s_x = opti.variable(self.nx, self.n_horizon + 1)  # Slack for state constraints
            s_u = opti.variable(self.nu, self.n_horizon)  # Slack for control constraints
            slack_penalty = self.softPenalty
        else:
            s_x = np.zeros((self.nx, self.n_horizon + 1))
            s_u = np.zeros((self.nu, self.n_horizon))
            slack_penalty = 0

        ### Constraints

        # Initial state constraint
        opti.subject_to(X[:, 0] == X0)

        # Dynamics constraints
        for k in range(self.n_horizon):
            xn = self.dynamics.fd(x=X[:, k], u=U[:, k])["xn"]
            opti.subject_to(X[:, k + 1] == xn)
        # State/Control constraints with slack variables (no slack for certain states/controls)
        for i in range(self.n_horizon + 1):
            for k in range(self.nx):
                if k in self.dynamics.slackStates:
                    opti.subject_to(opti.bounded(X_lb[k] - s_x[k, i], X[k, i], X_ub[k] + s_x[k, i]))
                else:
                    opti.subject_to(opti.bounded(X_lb[k], X[k, i], X_ub[k]))

        for i in range(self.n_horizon):
            for k in range(self.nu):
                if k in self.dynamics.slackControls:
                    opti.subject_to(opti.bounded(U_lb - s_u[:, i], U[:, i], U_ub + s_u[:, i]))
                else:
                    opti.subject_to(opti.bounded(U_lb[k], U[k, i], U_ub[k]))

        ### Costs
        cost = 0
        stage_cost_function = self.costs.stageCostFunc
        terminal_cost_function = self.costs.terminalCostFunc

        for k in range(self.n_horizon):
            cost += stage_cost_function(X[:, k], U[:, k], X_ref[:, k], U_ref[:, k])

        cost += terminal_cost_function(X[:, -1], X_ref[:, -1])  # Terminal cost

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
        self.optiVars = {
            "opti": opti,
            "X0": X0,
            "X_ref": X_ref,
            "U_ref": U_ref,
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

    def step(
        self,
        current_state: NDArray[np.floating],
        x_ref: NDArray[np.floating],
        u_ref: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Performs one optimization step using the current state, reference trajectory, and the previous solution for warmstarting. Updates the previous solution and returns the control input."""
        # Unpack IPOPT variables
        opti = self.optiVars["opti"]
        X = self.optiVars["X"]
        U = self.optiVars["U"]
        X_ref = self.optiVars["X_ref"]
        U_ref = self.optiVars["U_ref"]
        X0 = self.optiVars["X0"]
        # cost = self.optiVars["cost"]
        X_lb = self.optiVars["X_lb"]
        X_ub = self.optiVars["X_ub"]
        U_lb = self.optiVars["U_lb"]
        U_ub = self.optiVars["U_ub"]

        # u_last is needed when using control rates as inputs

        current_state = self.dynamics.transformState(current_state)
        # Set initial state
        opti.set_value(X0, current_state)
        # Set reference trajectory
        opti.set_value(X_ref, x_ref)
        opti.set_value(U_ref, u_ref)
        # Set state and control bounds
        opti.set_value(X_lb, self.dynamics.x_lb)
        opti.set_value(X_ub, self.dynamics.x_ub)
        opti.set_value(U_lb, self.dynamics.u_lb)
        opti.set_value(U_ub, self.dynamics.u_ub)
        # Set initial guess
        if self.x_guess is None or self.u_guess is None:
            opti.set_initial(X, np.hstack((current_state.reshape(self.nx, 1), x_ref[:, 1:])))
            opti.set_initial(U, u_ref)
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

        # Extract the control input and transform it if needed
        action = self.dynamics.transformAction(x_sol, u_sol)
        self.last_action = action

        # Update/Instantiate guess for next iteration
        if self.x_guess is None:
            self.x_guess = np.hstack((x_sol[:, 1:], x_sol[:, -1].reshape(self.nx, 1)))
            self.u_guess = np.hstack((u_sol[:, 1:], u_sol[:, -1].reshape(self.nu, 1)))
        else:
            self.x_guess[:, :-1] = x_sol[:, 1:]
            self.u_guess[:, :-1] = u_sol[:, 1:]

        self.x_last = x_sol
        self.u_last = u_sol

        # reset the solver after the initial guess
        self.setup_optimizer()
        return action


class AcadosOptimizer(BaseOptimizer):
    def __init__(
        self,
        dynamics: BaseDynamics,
        costs: BaseCost,
        optimizer_info: dict = {
            "useSoftConstraints": True,
            "softPenalty": 1e3,
            "useGP": False,
            "useZoro": False,
            "export_dir": "generated_code/mpc",
            "json_file": "acados_ocp.json",
            "IntegratorType": "ERK",
        },
    ):
        super().__init__(dynamics, costs, optimizer_info)
        self.useGP = optimizer_info.get("useGP", False)
        self.useZoro = optimizer_info.get("useZoro", False)
        self.json_file = optimizer_info.get("json_file", "acados_ocp.json")
        self.export_dir = optimizer_info.get("export_dir", "generated_code/mpc_acados")
        self.setupAcadosModel()
        self.setup_optimizer()

    def setupAcadosModel(self):
        """Setup the acados model using a selected dynamics model."""
        model = AcadosModel()
        model.name = "acados_" + self.dynamics.modelName
        model.x = self.dynamics.x
        model.u = self.dynamics.u
        model.xdot = self.dynamics.xdot
        # Define the parameters

        model.p = self.dynamics.p
        # Dynamics
        model.f_expl_expr = self.dynamics.dx  # continuous-time explicit dynamics
        model.dyn_disc_fun = self.dynamics.dx_d  # discrete-time dynamics

        # Continuous implicit dynamic expression
        model.f_impl_expr = model.xdot - model.f_expl_expr
        self.model = model

    def setup_optimizer(self):
        """Setup the acados optimizer (parameters, costs, constraints) given the class parameters set by the setupDynamics function."""
        ocp = AcadosOcp()
        ocp.model = self.model
        # Set solver options
        ocp.solver_options.N_horizon = self.n_horizon  # number of control intervals
        ocp.solver_options.tf = self.n_horizon * self.ts  # prediction horizon
        ocp.solver_options.integrator_type = "ERK"  # "ERK", "IRK", "GNSF", "DISCRETE"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # "EXACT", "GAUSS_NEWTON"
        ocp.solver_options.cost_discretization = "EULER"  # "INTEGRATOR", "EULER"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP, SQP_RTI
        ocp.solver_options.globalization = (
            "MERIT_BACKTRACKING"  # "FIXED_STEP", "MERIT_BACKTRACKING"
        )
        ocp.solver_options.tol = 1e-3  # tolerance
        ocp.solver_options.qp_tol = 1e-3  # QP solver tolerance

        if self.costs.cost_type == "linear":
            # Linear LS: J = || Vx * (x - x_ref) ||_W^2 + || Vu * (u - u_ref) ||_W^2
            # J_e = || Vx_e * (x - x_ref) ||^2
            # Update y_ref and y_ref_e at each iteration, if needed
            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"
            ocp.cost.W = scipy.linalg.block_diag(self.costs.Qs, self.costs.R)
            ocp.cost.W_e = self.costs.Qt
            # Dummy reference trajectory
            ocp.cost.yref = np.zeros((self.ny,))
            ocp.cost.yref_e = np.zeros((self.nx,))

            Vx = np.zeros((self.ny, self.nx))
            Vx[: self.nx, : self.nx] = np.eye(self.nx)
            ocp.cost.Vx = Vx

            ocp.cost.Vx_e = np.eye(self.nx)

            Vu = np.zeros((self.ny, self.nu))
            Vu[self.nx :, :] = np.eye(self.nu)
            ocp.cost.Vu = Vu
        elif self.costs.cost_type == "nonlinear":
            raise NotImplementedError("Nonlinear cost functions are not implemented yet.")
            # ocp.cost.cost_type = "NONLINEAR_LS"
            # ocp.cost.cost_type_e = "NONLINEAR_LS"
            # ocp.model.cost_y_expr = self.costs["cost_fcn"]
            # ocp.cost_expr_e = self.costs["cost_fcn"]
        elif self.costs.cost_type == "external":
            ocp.cost.cost_type = "EXTERNAL"
            ocp.cost.cost_type_e = "EXTERNAL"
            ocp.model.cost_expr_ext_cost = self.costs.stageCostFunc(
                ocp.model.x, ocp.model.u, ocp.model.p[self.dynamics.param_indices["cost"]]
            )
            ocp.model.cost_expr_ext_cost_e = self.costs.terminalCostFunc(
                ocp.model.x, ocp.model.p[self.dynamics.param_indices["cost"]]
            )
        else:
            raise NotImplementedError("cost_type must be linear, nonlinear, or external.")

        # Set Basic bounds
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = self.dynamics.u_lb
        ocp.constraints.ubu = self.dynamics.u_ub

        ocp.constraints.idxbx = np.arange(self.nx)
        ocp.constraints.lbx = self.dynamics.x_lb
        ocp.constraints.ubx = self.dynamics.x_ub

        ocp.constraints.idxbx_0 = ocp.constraints.idxbx
        ocp.constraints.lbx_0 = ocp.constraints.lbx
        ocp.constraints.ubx_0 = ocp.constraints.ubx

        ocp.constraints.idxbx_e = ocp.constraints.idxbx
        ocp.constraints.lbx_e = ocp.constraints.lbx
        ocp.constraints.ubx_e = ocp.constraints.ubx
        # Set soft constraints
        if self.useSoftConstraints:
            # Define the penalty matrices Zl and Zu
            norm2_penalty = self.softPenalty
            norm1_penalty = self.softPenalty
            # upperSlackBounds are 1 by default, optimize if needed

            # Define slack variables
            ocp.constraints.idxsbx = np.setdiff1d(np.arange(self.nx), self.dynamics.noSlackStates)
            nsx = len(ocp.constraints.idxsbx)
            ocp.constraints.lsbx = np.zeros((nsx,))
            ocp.constraints.usbx = np.ones((nsx,))

            ocp.constraints.idxsbx_e = ocp.constraints.idxsbx
            ocp.constraints.lsbx_e = ocp.constraints.lsbx
            ocp.constraints.usbx_e = ocp.constraints.usbx

            ocp.constraints.idxsbu = np.setdiff1d(np.arange(self.nu), self.dynamics.noSlackControls)
            nsu = len(ocp.constraints.idxsbu)
            ocp.constraints.lsbu = np.zeros((nsu,))
            ocp.constraints.usbu = np.ones((nsu,))

            nsy = nsx + nsu
            ocp.cost.Zl = norm2_penalty * np.ones((nsy,))
            ocp.cost.Zu = norm2_penalty * np.ones((nsy,))
            ocp.cost.zl = norm1_penalty * np.ones((nsy,))
            ocp.cost.zu = norm1_penalty * np.ones((nsy,))

            ocp.cost.Zl_e = norm2_penalty * np.ones((nsx,))
            ocp.cost.Zu_e = norm2_penalty * np.ones((nsx,))
            ocp.cost.zl_e = norm1_penalty * np.ones((nsx,))
            ocp.cost.zu_e = norm1_penalty * np.ones((nsx,))

            ocp.cost.Zl_0 = norm2_penalty * np.ones((nsu,))
            ocp.cost.Zu_0 = norm2_penalty * np.ones((nsu,))
            ocp.cost.zl_0 = norm1_penalty * np.ones((nsu,))
            ocp.cost.zu_0 = norm1_penalty * np.ones((nsu,))
        # Set initial state (not required and should not be set for moving horizon estimation)
        # ocp.constraints.x0 = self.x0
        # Set nonlinear constraints
        ocp.model.con_h_expr = self.dynamics.nl_constr
        ocp.constraints.lh = self.dynamics.nl_constr_lh
        ocp.constraints.uh = self.dynamics.nl_constr_uh
        ocp.parameter_values = self.dynamics.param_values

        if self.useZoro:
            raise NotImplementedError("Zoro not implemented yet.")
        ocp.code_export_directory = self.export_dir
        self.ocp = ocp
        if self.useGP:
            raise NotImplementedError("Gaussian Process not implemented yet.")
            self.residual_model = l4a.PytorchResidualModel(self.pytorch_model)
            self.ocp_solver = l4a.ResidualLearningMPC(
                ocp=self.ocp, residual_model=self.residual_model, use_cython=True
            )
        else:
            self.ocp_solver = AcadosOcpSolver(self.ocp, self.json_file)
        # self.ocp_integrator = AcadosSimSolver(self.ocp)

    def step(
        self,
        current_state: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        x_ref: NDArray[np.floating],
        u_ref: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Performs one optimization step using the current state, reference trajectory, and the previous solution for warmstarting. Updates the previous solution and returns the control input."""
        # Transform the state and update the parameters
        current_state = self.dynamics.transformState(current_state)
        self.updateParameters(obs)

        # Set initial state
        self.ocp_solver.set(0, "lbx", current_state)
        self.ocp_solver.set(0, "ubx", current_state)

        # Set reference trajectory
        y_ref = np.vstack([x_ref[:, :-1], u_ref])
        y_ref_e = x_ref[:, -1]
        for i in range(self.n_horizon):
            self.ocp_solver.set(i, "yref", y_ref[:, i])
        self.ocp_solver.set(self.n_horizon, "yref", y_ref_e)

        # Set initial guess (u_guess/x_guess are the previous solution moved one step forward)
        self.ocp_solver.set(0, "x", current_state)
        for k in range(self.n_horizon):
            self.ocp_solver.set(k + 1, "x", self.x_guess[:, k + 1])
            self.ocp_solver.set(k, "u", self.u_guess[:, k])

        # Solve the OCP
        if self.ocp.solver_options.nlp_solver_type == "SQP_RTI":
            # phase 1
            self.ocp_solver.options_set("rti_phase", 1)
            status = self.ocp_solver.solve()

            if self.useZoro:
                self.ocp_solver.custom_update([])

            # phase 2
            self.ocp_solver.options_set("rti_phase", 2)
            status = self.ocp_solver.solve()
        else:
            status = self.ocp_solver.solve()

        if status not in [0, 2]:
            self.ocp_solver.print_statistics()
            raise Exception(f"acados failed with status {status}. Exiting.")
        else:
            print(f"acados succeded with status {status}.")

        # Update previous solution
        self.x_last = np.zeros((self.nx, self.n_horizon + 1))
        self.u_last = np.zeros((self.nu, self.n_horizon))
        for k in range(self.n_horizon + 1):
            self.x_last[:, k] = self.ocp_solver.get(k, "x")
        for k in range(self.n_horizon):
            self.u_last[:, k] = self.ocp_solver.get(k, "u")

        # Update the initial guess
        self.x_guess[:, :-1] = self.x_last[:, 1:]
        self.u_guess[:, :-1] = self.u_last[:, 1:]

        # Extract the control input
        action = self.dynamics.transformAction(self.x_last, self.u_last)
        self.last_action = action
        return action

    def updateParameters(self, obs):
        """Update the obstacle constraints based on the current obstacle positions."""
        params = self.dynamics.param_values

        # Update obstacle constraints
        obstacles_visited = obs.get("obstacles_visited", self.dynamics.obstacle_visited)
        if np.array_equal(obstacles_visited, self.dynamics.obstacle_visited):
            pass
        else:
            self.dynamics.obstacle_visited = obstacles_visited
            self.dynamics.obstacle_pos = obs.get("obstacles_pos", self.dynamics.obstacle_pos)

        params[self.dynamics.param_indices["p_obst"]] = self.dynamics.obstacle_pos.flatten()

        # Update the parameters in the solver
        for stage in range(self.n_horizon + 1):
            self.ocp_solver.set(stage, "p", params)
        # Update the parameter values in the dynamics model
        self.dynamics.param_values = params
        return None
