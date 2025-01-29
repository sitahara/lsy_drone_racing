from __future__ import annotations

import casadi as ca
import numpy as np
import scipy
from numpy.typing import NDArray

from .optimizer import BaseOptimizer


class IPOPTOptimizer(BaseOptimizer):
    def __init__(self, dynamics, solver_options, optimizer_info):
        super().__init__(dynamics, solver_options, optimizer_info)
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
                    opti.subject_to(opti.bounded(U_lb - s_u[k, i], U[:, i], U_ub + s_u[k, i]))
                else:
                    opti.subject_to(opti.bounded(U_lb[k], U[k, i], U_ub[k]))

        ### Costs (All cost functions have args: x,u,p,x_ref,u_ref)
        cost = 0
        stage_cost_function = self.dynamics.stageCostFunc
        terminal_cost_function = self.dynamics.terminalCostFunc

        for k in range(self.n_horizon):
            cost += stage_cost_function(
                X[:, k], U[:, k], self.dynamics.param_values, X_ref[:, k], U_ref[:, k]
            )

        cost += terminal_cost_function(
            X[:, -1],
            np.zeros((self.nu,)),
            self.dynamics.param_values,
            X_ref[:, -1],
            np.zeros((self.nu,)),
        )  # Terminal cost

        # Add slack penalty to the cost function
        cost += slack_penalty * (ca.sumsqr(s_x) + ca.sumsqr(s_u))
        opti.minimize(cost)

        # Solver options
        opts = {
            "ipopt.print_level": self.solver_options.get("ipopt.print_level", 0),
            "ipopt.tol": self.solver_options.get("ipopt.tol", 1e-4),
            "ipopt.max_iter": self.solver_options.get("ipopt.max_iter", 25),
            "ipopt.linear_solver": self.solver_options.get("ipopt.linear_solver", "mumps"),
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
        x_ref: NDArray[np.floating] = None,
        u_ref: NDArray[np.floating] = None,
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
        if x_ref is not None:
            opti.set_value(X_ref, x_ref)
        if u_ref is not None:
            opti.set_value(U_ref, u_ref)
        # Set state and control bounds
        opti.set_value(X_lb, self.dynamics.x_lb)
        opti.set_value(X_ub, self.dynamics.x_ub)
        opti.set_value(U_lb, self.dynamics.u_lb)
        opti.set_value(U_ub, self.dynamics.u_ub)
        # print("X_lb: ", self.dynamics.x_lb)
        # print("X_ub: ", self.dynamics.x_ub)
        # print("U_lb: ", self.dynamics.u_lb)
        # print("U_ub: ", self.dynamics.u_ub)
        # raise Exception("Stop here")
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
