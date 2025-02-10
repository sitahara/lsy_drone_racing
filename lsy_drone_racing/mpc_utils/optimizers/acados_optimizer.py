from __future__ import annotations
import casadi as ca

# import l4acados as l4a
import numpy as np
import scipy
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSim,
    AcadosSimSolver,
    ZoroDescription,
)
from acados_template.utils import ACADOS_INFTY
from numpy.typing import NDArray

from lsy_drone_racing.mpc_utils.dynamics import DroneDynamics
# from lsy_drone_racing.mpc_utils.models import ResidualPytorchModel, GpPytorchModel

from .optimizer import BaseOptimizer


class AcadosOptimizer(BaseOptimizer):
    def __init__(self, dynamics: DroneDynamics, solver_options, optimizer_info):
        super().__init__(dynamics, solver_options, optimizer_info)
        self.useGP = self.optimizer_info.get("useGP", False)
        self.useGPPytorch = self.optimizer_info.get("useGPPytorch", True)
        self.useZoro = self.optimizer_info.get("useZoro", False)
        self.json_file = self.optimizer_info.get("json_file", "acados_ocp.json")
        self.export_dir = self.optimizer_info.get("export_dir", "generated_code/mpc_acados")
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
        ocp.solver_options.integrator_type = self.solver_options.get(
            "integrator", "ERK"
        )  # "ERK", "IRK", "GNSF", "DISCRETE"
        ocp.solver_options.qp_solver = self.solver_options.get(
            "qp_solver", "PARTIAL_CONDENSING_HPIPM"
        )
        # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # "EXACT", "GAUSS_NEWTON"
        ocp.solver_options.cost_discretization = self.solver_options.get(
            "cost_discretization", "EULER"
        )  # "INTEGRATOR", "EULER"
        ocp.solver_options.nlp_solver_type = self.solver_options.get(
            "nlp_solver", "SQP_RTI"
        )  # SQP, SQP_RTI
        ocp.solver_options.globalization = self.solver_options.get(
            "globalization", "MERIT_BACKTRACKING"
        )  # "FIXED_STEP", "MERIT_BACKTRACKING"

        ocp.solver_options.tol = self.solver_options.get("tol", 1e-3)  # NLP error tolerance
        ocp.solver_options.qp_tol = self.solver_options.get("qp_tol", 1e-3)  # QP error tolerance

        if self.dynamics.cost_type == "linear":
            # Linear LS: J = || Vx * (x - x_ref) ||_W^2 + || Vu * (u - u_ref) ||_W^2
            # J_e = || Vx_e * (x - x_ref) ||^2
            # Update y_ref and y_ref_e at each iteration, if needed
            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"
            ocp.cost.W = scipy.linalg.block_diag(self.dynamics.Qs, self.dynamics.R)
            ocp.cost.W_e = self.dynamics.Qt
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
        elif self.dynamics.cost_type == "nonlinear":
            raise NotImplementedError("Nonlinear cost functions are not implemented yet.")
            # ocp.cost.cost_type = "NONLINEAR_LS"
            # ocp.cost.cost_type_e = "NONLINEAR_LS"
            # ocp.model.cost_y_expr = self.costs["cost_fcn"]
            # ocp.cost_expr_e = self.costs["cost_fcn"]
        elif self.dynamics.cost_type == "external":
            ocp.cost.cost_type = "EXTERNAL"
            ocp.cost.cost_type_e = "EXTERNAL"
            ocp.model.cost_expr_ext_cost = self.dynamics.stageCostFunc(
                ocp.model.x, ocp.model.u, ocp.model.p
            )
            ocp.model.cost_expr_ext_cost_e = self.dynamics.terminalCostFunc(
                ocp.model.x, ca.MX.zeros(self.nu), ocp.model.p
            )
        else:
            raise NotImplementedError("cost_type must be linear, nonlinear, or external.")

        # Set Basic bounds
        # Set control bounds
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = self.dynamics.u_lb
        ocp.constraints.ubu = self.dynamics.u_ub
        # Set state bounds
        ocp.constraints.idxbx = np.arange(self.nx)
        ocp.constraints.lbx = self.dynamics.x_lb
        ocp.constraints.ubx = self.dynamics.x_ub
        # Set start state bounds
        ocp.constraints.idxbx_0 = ocp.constraints.idxbx
        ocp.constraints.lbx_0 = ocp.constraints.lbx
        ocp.constraints.ubx_0 = ocp.constraints.ubx
        # Set terminal state bounds
        ocp.constraints.idxbx_e = ocp.constraints.idxbx
        ocp.constraints.lbx_e = ocp.constraints.lbx
        ocp.constraints.ubx_e = ocp.constraints.ubx

        # Set soft constraints
        if self.useSoftConstraints:
            # Define the penalty matrices Zl and Zu (we use the same for all states and controls)
            norm2_penalty = self.softPenalty
            norm1_penalty = self.softPenalty
            # upperSlackBounds are 1 by default, optimize if needed

            # Define slack variables and limits
            ocp.constraints.idxsbx = self.dynamics.slackStates
            nsx = len(ocp.constraints.idxsbx)
            ocp.constraints.lsbx = np.zeros((nsx,))
            ocp.constraints.usbx = np.ones((nsx,))

            ocp.constraints.idxsbx_e = ocp.constraints.idxsbx
            ocp.constraints.lsbx_e = ocp.constraints.lsbx
            ocp.constraints.usbx_e = ocp.constraints.usbx

            ocp.constraints.idxsbu = self.dynamics.slackControls
            nsu = len(ocp.constraints.idxsbu)
            ocp.constraints.lsbu = np.zeros((nsu,))
            ocp.constraints.usbu = np.ones((nsu,))
            # Define the soft constraints weights
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

        # Set nonlinear constraints (taken from the dynamics)
        if self.dynamics.nl_constr is not None:
            ocp.model.con_h_expr = self.dynamics.nl_constr
            ocp.constraints.lh = self.dynamics.nl_constr_lh
            ocp.constraints.uh = self.dynamics.nl_constr_uh

        # Initialize the parameters (if any)
        if self.dynamics.param_values is not None:
            ocp.parameter_values = self.dynamics.param_values

        if self.useZoro:
            raise NotImplementedError("Zoro not implemented yet.")

        ocp.code_export_directory = self.export_dir
        self.ocp = ocp
        if self.useGP:
            raise NotImplementedError("GP not implemented yet.")
            # if self.useGPPytorch:
            #     self.pytorch_model = l4a.models.GPyTorchResidualModel
            #     self.pytorch_model = GpPytorchModel()
            #     self.residual_model = l4a.models.GPyTorchResidualModel(self.pytorch_model,)
            #     self.ocp_solver = l4a.controllers.ZeroOrderGPMPC(
            #         ocp=self.ocp, residual_model=self.residual_model
            #     )
            # else:
            #     self.pytorch_model = ResidualPytorchModel(
            #         state_dim=self.nx, control_dim=self.nu, hidden_dim=128
            #     )
            #     self.residual_model = l4a.models.PyTorchResidualModel(self.pytorch_model)
            #     self.ocp_solver = l4a.controllers.ResidualLearningMPC(
            #         ocp=self.ocp, residual_model=self.residual_model, use_cython=True
            #     )
        else:
            self.ocp_solver = AcadosOcpSolver(
                self.ocp,
                self.json_file,
                build=self.solver_options.get("build", True),
                generate=self.solver_options.get("generate", True),
                verbose=False,
            )
            # self.ocp_sim = AcadosSim(self.ocp)
            # self.ocp_integrator = AcadosSimSolver(
            #     self.ocp,
            #     build=self.solver_options.get("build", True),
            #     generate=self.solver_options.get("generate", True),
            # )

    def step(
        self,
        current_state: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        x_ref: NDArray[np.floating] = None,
        u_ref: NDArray[np.floating] = None,
    ) -> NDArray[np.floating]:
        """Performs one optimization step using the current state, reference trajectory, and the previous solution for warmstarting. Updates the previous solution and returns the control input."""
        # Map the observations onto dynamic states (includes Unscented Kalman Filter) the state
        current_state = self.dynamics.transformState(current_state)
        # Update the solver parameters
        self.updateParameters(obs)

        # Set initial state
        self.ocp_solver.set(0, "lbx", current_state)
        self.ocp_solver.set(0, "ubx", current_state)

        # Set reference trajectory for linear and nonlinear LS cost functions
        if self.dynamics.cost_type != "external":
            y_ref = np.vstack([x_ref[:, :-1], u_ref])
            y_ref_e = x_ref[:, -1]
            for i in range(self.n_horizon):
                self.ocp_solver.set(i, "yref", y_ref[:, i])
            self.ocp_solver.set(self.n_horizon, "yref", y_ref_e)

        # Set initial guess (u_guess/x_guess are the previous solution moved one step forward)
        self.ocp_solver.set(0, "x", current_state)
        if self.x_guess is None:
            Warning("No initial state guess provided. Using the default.")
            self.x_guess = np.tile(current_state, (self.n_horizon + 1, 1)).T
            if x_ref is not None:
                self.x_guess[: x_ref.shape[0], : x_ref.shape[1]] = x_ref
        if self.u_guess is None:
            Warning("No initial control guess provided. Using the default.")
            self.u_guess = np.tile(self.dynamics.u_eq, (self.n_horizon, 1)).T
            if u_ref is not None:
                self.u_guess[: u_ref.shape[0], :] = u_ref

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
            # raise Warning(f"acados failed with status {status}. Exiting.")
        else:
            pass
            # print(f"acados succeded with status {status}.")

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

        # Extract the control input (depends on the dynamics and interface)
        action = self.dynamics.transformAction(self.x_last, self.u_last)
        self.last_action = action
        return action

    def updateParameters(self, obs):
        """Update the parameters of the acados solver. Here we call the dynamics.updateParameters function to get the new values."""
        update = self.dynamics.updateParameters(obs)

        if update:
            params = self.dynamics.param_values
            # Update the parameters in the solver
            for stage in range(self.n_horizon + 1):
                self.ocp_solver.set(stage, "p", params)
        return None
