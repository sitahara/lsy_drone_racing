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
from lsy_drone_racing.control.dynamics_classes import (
    BaseDynamics,
    ThrustEulerDynamics,
    TorqueEulerDynamics,
    ThrustQuaternionDynamics,
)
from lsy_drone_racing.control.constraint_class import CONSTRAINTS
from numpy.typing import NDArray


class BaseOptimizer(ABC):
    """Abstract base class for optimizer implementations."""

    def __init__(
        self,
        dynamics: BaseDynamics,
        constraints: CONSTRAINTS,
        costs: dict,
        mpc_info: dict = {"ts": 1 / 60, "n_horizon": 60},
        optimizer_info: dict = {
            "useSoftConstraints": True,
            "softPenalty": 1e3,
            "noSlackStates": [],
            "noSlackControls": [],
            "useMellinger": True,
            "useGP": False,
            "useZoro": False,
            "export_dir": "generated_code/mpc",
        },
    ):
        self.ts = mpc_info.get("ts", 1 / 60)
        self.n_horizon = mpc_info.get("n_horizon", 60)
        self.useSoftConstraints = optimizer_info.get("useSoftConstraints", True)
        self.softPenalty = optimizer_info.get("softPenalty", 1e3)
        self.noSlackStates = optimizer_info.get("noSlackStates", [2])
        self.noSlackControls = optimizer_info.get("noSlackControls", 0)
        self.useMellinger = optimizer_info.get("useMellinger", True)
        self.dynamics = dynamics
        self.costs = costs
        self.constraints = constraints
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
        constraints: CONSTRAINTS,
        costs: dict,
        mpc_info: dict = {"ts": 1 / 60, "n_horizon": 60},
        optimizer_info: dict = {
            "useSoftConstraints": True,
            "softPenalty": 1e3,
            "noSlackStates": [],
            "noSlackControls": [],
            "useMellinger": True,
            "useGP": False,
            "useZoro": False,
            "export_dir": "generated_code/mpc",
        },
    ):
        super().__init__(dynamics, constraints, costs, mpc_info, optimizer_info)
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
        # State/Control constraints with slack variables (no slack for z position)
        for i in range(self.n_horizon + 1):
            for k in range(self.nx):
                if k in self.noSlackStates:
                    opti.subject_to(opti.bounded(X_lb[k], X[k, i], X_ub[k]))
                else:
                    opti.subject_to(opti.bounded(X_lb[k] - s_x[k, i], X[k, i], X_ub[k] + s_x[k, i]))
        for i in range(self.n_horizon):
            opti.subject_to(opti.bounded(U_lb - s_u[:, i], U[:, i], U_ub + s_u[:, i]))
        ### Costs
        cost = 0
        cost_func = self.costs["cost_function"]

        for k in range(self.n_horizon):
            cost += cost_func(X[:, k], X_ref[:, k], U[:, k], U_ref[:, k])
        cost += cost_func(
            X[:, -1], X_ref[:, -1], np.zeros((self.nu, 1)), np.zeros((self.nu, 1))
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
        # Unpack IPOPT variables
        opti = self.optiVars["opti"]
        X = self.optiVars["X"]
        U = self.optiVars["U"]
        X_ref = self.optiVars["X_ref"]
        U_ref = self.optiVars["U_ref"]
        X0 = self.optiVars["X0"]
        cost = self.optiVars["cost"]
        X_lb = self.optiVars["X_lb"]
        X_ub = self.optiVars["X_ub"]
        U_lb = self.optiVars["U_lb"]
        U_ub = self.optiVars["U_ub"]

        # u_last is needed when using control rates as inputs
        if self.x_last is None:
            u_last = self.dynamics.u_eq
        else:
            u_last = self.x_last[self.nx - self.nu : self.nx, 1]

        current_state = self.dynamics.transformState(current_state, last_u=u_last)
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
        if self.useMellinger:
            action = x_sol[:, 1]
            action = self.dynamics.transformAction(action)
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
        self.setup_optimizer()
        return action


class AcadosOptimizer(BaseOptimizer):
    def __init__(
        self,
        dynamics: BaseDynamics,
        constraints: CONSTRAINTS,
        costs: dict,
        mpc_info: dict = {"ts": 1 / 60, "n_horizon": 60},
        optimizer_info: dict = {
            "useSoftConstraints": True,
            "softPenalty": 1e3,
            "noSlackStates": [],
            "noSlackControls": [],
            "useMellinger": True,
            "useGP": False,
            "useZoro": False,
            "export_dir": "generated_code/mpc",
        },
    ):
        super().__init__(dynamics, constraints, costs, mpc_info, optimizer_info)
        self.useGP = mpc_info.get("useGP", False)
        self.useZoro = mpc_info.get("useZoro", False)
        self.json_file = mpc_info.get("json_file", "acados_ocp.json")
        self.export_dir = mpc_info.get("export_dir", "generated_code/mpc_acados")
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

        if self.costs["cost_type"] == "linear":
            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"
            ocp.cost.W = scipy.linalg.block_diag(self.costs["Qs"], self.costs["R"])
            ocp.cost.W_e = self.costs["Qt"]
            ocp.cost.yref = np.zeros((self.ny,))
            ocp.cost.yref_e = np.zeros((self.nx,))
            Vx = np.zeros((self.ny, self.nx))
            Vx[: self.nx, : self.nx] = np.eye(self.nx)
            Vu = np.zeros((self.ny, self.nu))
            Vu[self.nx :, :] = np.eye(self.nu)
            ocp.cost.Vx = Vx
            ocp.cost.Vx_e = np.eye(self.nx)
            ocp.cost.Vu = Vu
        elif self.costs["cost_type"] == "nonlinear":
            raise NotImplementedError("Nonlinear cost functions are not implemented yet.")
            # ocp.cost.cost_type = "NONLINEAR_LS"
            # ocp.cost.cost_type_e = "NONLINEAR_LS"
            # ocp.model.cost_y_expr = self.costs["cost_fcn"]
            # ocp.cost_expr_e = self.costs["cost_fcn"]
        elif self.costs["cost_type"] == "external":
            ocp.cost.cost_type = "EXTERNAL"
            ocp.cost.cost_type_e = "EXTERNAL"
            ocp.model.cost_expr_ext_cost = self.costs["cost_fcn"](
                ocp.model.x, ocp.model.u, ocp.model.p
            )
            ocp.model.cost_expr_ext_cost_e = self.costs["cost_fcn"](
                ocp.model.x, np.zeros((self.nu, 1)), ocp.model.p
            )
        else:
            raise NotImplementedError("cost_type must be linear, nonlinear, or external.")

        # Set Basic bounds
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = self.dynamics.u_lb.reshape(self.nu)
        ocp.constraints.ubu = self.dynamics.u_ub.reshape(self.nu)

        ocp.constraints.idxbx = np.arange(self.nx)
        ocp.constraints.lbx = self.dynamics.x_lb.reshape(self.nx)
        ocp.constraints.ubx = self.dynamics.x_ub.reshape(self.nx)

        ocp.constraints.idxbx_0 = ocp.constraints.idxbx
        ocp.constraints.lbx_0 = ocp.constraints.lbx
        ocp.constraints.ubx_0 = ocp.constraints.ubx

        ocp.constraints.idxbx_e = ocp.constraints.idxbx
        ocp.constraints.lbx_e = ocp.constraints.lbx
        ocp.constraints.ubx_e = ocp.constraints.ubx
        if self.useSoftConstraints:
            # Define the penalty matrices Zl and Zu
            norm2_penalty = self.softPenalty
            norm1_penalty = self.softPenalty

            # Define slack variables
            ocp.constraints.idxsbx = np.arange(self.nx)
            ocp.constraints.lsbx = np.zeros((self.nx,))
            ocp.constraints.usbx = np.ones((self.nx,))

            ocp.constraints.idxsbu = np.arange(self.nu)
            ocp.constraints.lsbu = np.zeros((self.nu,))
            ocp.constraints.usbu = np.ones((self.nu,))

            ocp.constraints.idxsbx_e = np.arange(self.nx)
            ocp.constraints.lsbx_e = np.zeros((self.nx,))
            ocp.constraints.usbx_e = np.ones((self.nx,))

            ocp.cost.Zl = norm2_penalty * np.ones(self.ny)
            ocp.cost.Zu = norm2_penalty * np.ones(self.ny)
            ocp.cost.zl = norm1_penalty * np.ones(self.ny)
            ocp.cost.zu = norm1_penalty * np.ones(self.ny)

            ocp.cost.Zl_e = norm2_penalty * np.ones(self.nx)
            ocp.cost.Zu_e = norm2_penalty * np.ones(self.nx)
            ocp.cost.zl_e = norm1_penalty * np.ones(self.nx)
            ocp.cost.zu_e = norm1_penalty * np.ones(self.nx)

            ocp.cost.Zl_0 = norm2_penalty * np.ones(self.nu)
            ocp.cost.Zu_0 = norm2_penalty * np.ones(self.nu)
            ocp.cost.zl_0 = norm1_penalty * np.ones(self.nu)
            ocp.cost.zu_0 = norm1_penalty * np.ones(self.nu)
        # Set initial state (not required and should not be set for moving horizon estimation)
        # ocp.constraints.x0 = self.x0
        ocp = self.initConstraints(ocp, self.constraints)

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
            self.ocp_solver = AcadosOcpSolver(self.ocp)
        self.ocp_integrator = AcadosSimSolver(self.ocp)

    def step(
        self,
        current_state: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        x_ref: NDArray[np.floating],
        u_ref: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Performs one optimization step using the current state, reference trajectory, and the previous solution for warmstarting. Updates the previous solution and returns the control input."""
        if self.x_last is None:
            u_last = self.dynamics.u_eq
        else:
            u_last = self.x_last[self.nx - self.nu : self.nx, 1]

        current_state = self.dynamics.transformState(current_state, last_u=u_last)
        if self.constraints.dict["obstacle"] is not None:
            self.updateObstacleConstraints(obs)
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
        if self.useMellinger:
            action = self.ocp_solver.get(1, "x")
            action = self.dynamics.transformAction(action)
        else:
            action = self.ocp_solver.get(0, "u")
        self.last_action = action
        return action

    def initConstraints(self, ocp: AcadosOcp, constraints) -> AcadosOcp:
        """Initialize the constraints."""
        ocp.model.con_h_expr = self.dynamics.obstacle_constraints
        ocp.constraints.lh = self.dynamics.obstacle_constraints_lh
        ocp.constraints.uh = self.dynamics.obstacle_constraints_uh
        # if constraints.dict["obstacle"] is not None:
        #     ocp.model.con_h_expr = constraints.dict["obstacle"]["expr"]
        #     ocp.constraints.lh = constraints.dict["obstacle"]["lh"]
        #     ocp.constraints.uh = constraints.dict["obstacle"]["uh"]
        #     ocp.parameter_values = np.zeros(constraints.dict["obstacle"]["param"].shape)
        # if constraints.dict["goal"] is not None:
        #     NotImplementedError("Goal constraints not implemented yet.")
        # if constraints.dict["gate"] is not None:
        #     NotImplementedError("Gate constraints not implemented yet.")
        return ocp

    def updateObstacleConstraints(self, obs):
        """Update the obstacle constraints based on the current obstacle positions."""
        if np.array_equal(
            obs["obstacles_in_range"], self.constraints.dict["obstacle"]["obstacle_in_range"]
        ):
            return None
        self.constraints.dict["obstacle"]["obstacle_in_range"] = obs["obstacles_in_range"]
        params = obs["obstacles_pos"].flatten()
        # print("Updating obstacle constraints: Param Shape: ", params.shape)
        for stage in range(self.n_horizon):
            self.ocp_solver.set(stage, "p", params)
        return None
