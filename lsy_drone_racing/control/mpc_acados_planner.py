"""MPC controller w/ acados for the racing environment.

Uses trajectory generated by the planner as the reference trajectory.
"""

from __future__ import annotations

import casadi as ca
import threading
import l4acados as l4a
import numpy as np
import pybullet as p
import scipy as sp
import queue
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSimSolver,
    ZoroDescription,
)
from acados_template.utils import ACADOS_INFTY
from numpy.typing import NDArray

from lsy_drone_racing.control.mpc_base import MPC_BASE
from lsy_drone_racing.control.utils import W1
from lsy_drone_racing.planner import ObservationManager, Planner

# from lsy_drone_racing.sim.drone import Drone
# from lsy_drone_racing.sim.physics import GRAVITY


class MPC_ACADOS_PLANNER(MPC_BASE):
    """Model Predictive Controller implementation using CasADi and acados."""

    def __init__(
        self,
        initial_obs: NDArray[np.floating],
        initial_info: dict,
        export_dir: str = "generated_code/mpc_acados",
        cost_type: str = "Linear",
        useGP: bool = False,
        useZoro: bool = False,
        json_file: str = "acados_ocp",
    ):
        """Initialize the MPC_ACADOS controller.

        Args:
            initial_obs: The initial observation of the environment.
            initial_info: Additional information as a dictionary.
        """
        # Initialize the base class
        super().__init__(initial_obs, initial_info)
        self.cost_type = cost_type
        self.export_dir = export_dir
        self.useGP = useGP
        self.useZoro = useZoro
        self.json_file = json_file

        # Define the dynamics model, constraints, and cost matrices
        # super().setupDynamics() # This is already done in the base class

        # Setup the acados model, optimizer, and solver
        self.setupAcadosModel()
        self.setupAcadosOptimizer()

        # Setup the IPOPT optimizer for initial guess
        # super().setupIPOPTOptimizer() # This is already done in the base class

        # Set the target trajectory
        # super().set_target_trajectory() # This is already done in the base class

        # Set up observation variable here
        self.obs = initial_obs

        # Use ObservationManager to record true coordinates
        self.ObsMgr = ObservationManager()

        # Set up the planner
        # Since the planner is slow, run the planner on a separate thread
        self.queue_state = queue.Queue()
        self.queue_trajectory = queue.Queue()
        self.planner = Planner(MAX_ROAD_WIDTH=0.4, D_ROAD_W=0.1)

        self.planner_thread = threading.Thread(target=self.do_planning, daemon=True)
        
        ## Since the controller requires the existence of x_ref, we will plan once in advance
        ## using initial observation
        gate_x, gate_y, gate_z = (
            initial_obs["gates_pos"][:, 0],
            initial_obs["gates_pos"][:, 1],
            initial_obs["gates_pos"][:, 2],
        )
        gate_yaw = initial_obs["gates_rpy"][:, 2]
        obs_x, obs_y = initial_obs["obstacles_pos"][:, 0], initial_obs["obstacles_pos"][:, 1]
        next_gate = initial_obs["target_gate"] + 1
        drone_x, drone_y = initial_obs["pos"][0], initial_obs["pos"][1]
        result_path, _, _ = self.planner.plan_path_from_observation(
            gate_x, gate_y, gate_z, gate_yaw, obs_x, obs_y, drone_x, drone_y, next_gate
        )
        num_points = len(result_path.x)
        self.x_ref[0, :num_points] = np.array(result_path.x)
        self.x_ref[1, :num_points] = np.array(result_path.y)
        self.x_ref[2, :num_points] = np.array(result_path.z)

    def reset(self):
        """Reset the MPC controller to its initial state."""
        self.__init__(self.obs, self.initial_info)

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
        self.obs = self.ObsMgr.update(obs)
        if self.useObstacleConstraints:
            self.updateObstacleConstraints()
        # Update the current state
        if self.useAngVel:
            w = W1(obs["rpy"]) @ obs["ang_vel"]
            self.current_state = np.concatenate([obs["pos"], obs["vel"], obs["rpy"], w.flatten()])
        else:
            self.current_state = np.concatenate(
                [obs["pos"], obs["vel"], obs["rpy"], obs["ang_vel"]]
            )
        # Updates x_ref, the current target trajectory and upcounts the trajectory tick
        # super().updateTargetTrajectory()
        self.updateTargetTrajectory()

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
            action = np.concatenate([action[:6], acc, [action[8]], action[9:]])
        else:
            # action: [thrust, tau_des]
            action = np.array(action)
        print(
            f"Curren position error: {np.linalg.norm(self.current_state[:3] - self.x_ref[:3, 0])}, Next position: {action[:3]}"
        )
        # print(f"Current position: {self.current_state[:3]}")
        # print(f"Desired position: {self.x_ref[:3, 1]}")
        # print(f"Next position: {action[:3]}")

        # self.tick = (self.tick + 1) % self.cycles_to_update
        return action.flatten()

    def setupAcadosModel(self):
        """Setup the acados model using a selected dynamics model."""
        model = AcadosModel()
        model.name = self.modelName
        model.x = self.x
        model.u = self.u
        xdot = ca.MX.sym("xdot", self.nx)
        model.xdot = xdot
        # Parameters
        num_obstacle_param = self.obstacles_pos.size
        num_gate_param = self.gates_pos.size + self.gates_rpy.size
        self.param_sym = ca.MX.sym("params", num_obstacle_param + num_gate_param)
        model.p = self.param_sym
        # Dynamics
        model.f_expl_expr = self.dx  # continuous-time explicit dynamics
        model.dyn_disc_fun = self.dx_d  # discrete-time dynamics

        # Continuous implicit dynamic expression
        model.f_impl_expr = xdot - model.f_expl_expr
        self.model = model

    def setupAcadosOptimizer(self):
        """Setup the acados optimizer (parameters, costs, constraints) given the class parameters set by the setupDynamics function."""
        ocp = AcadosOcp()
        ocp.model = self.model
        # Set solver options
        ocp.solver_options.N_horizon = self.n_horizon  # number of control intervals
        ocp.solver_options.tf = self.n_horizon * self.ts  # prediction horizon
        # Whether to use the eplicit, implicit, or discrete dynamics
        ocp.solver_options.integrator_type = "ERK"  # "ERK", "IRK", "GNSF", "DISCRETE"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # "EXACT", "GAUSS_NEWTON"
        ocp.solver_options.cost_discretization = "EULER"  # "INTEGRATOR", "EULER"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP, SQP_RTI
        # ocp.solver_options.nlp_solver_max_iter = 20  # TODO: optimize
        ocp.solver_options.globalization = (
            "MERIT_BACKTRACKING"  # "FIXED_STEP", "MERIT_BACKTRACKING"
        )
        ocp.solver_options.tol = 1e-3  # tolerance
        ocp.solver_options.qp_tol = 1e-3  # QP solver tolerance

        if self.cost_type == "Linear":
            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"
            # Set weighting matrices
            ocp.cost.W = sp.linalg.block_diag(self.Qs, self.Rs)
            ocp.cost.W_e = self.Qt
            # Set reference trajectory
            ocp.cost.yref = np.zeros((self.ny,))
            ocp.cost.yref_e = np.zeros((self.nx,))
            # Set mapping matrices
            Vx = np.zeros((self.ny, self.nx))
            Vx[: self.nx, : self.nx] = np.eye(self.nx)
            Vu = np.zeros((self.ny, self.nu))
            Vu[self.nx :, :] = np.eye(self.nu)
            ocp.cost.Vx = Vx
            ocp.cost.Vx_e = np.eye(self.nx)
            ocp.cost.Vu = Vu
        elif self.cost_type == "Nonlinear":
            ocp.cost.cost_type = "NONLINEAR_LS"  # "NONLINEAR_LS", "LINEAR_LS", "EXTERNAL"
            ocp.cost.cost_type_e = "NONLINEAR_LS"  # "NONLINEAR_LS", "LINEAR_LS", "EXTERNAL"
            # ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)  # output model
            # ocp.cost.yref = np.zeros((self.ny,))  # dummy reference, states and controls are considered for stage cost
            ocp.cost.W = sp.linalg.block_diag(self.Qs, self.Rs)  # Weight matrix for stage cost

            # ocp.model.cost_y_expr_e = ocp.model.x  # output model
            # ocp.cost.yref_e = np.zeros((self.nx,))  # dummy reference, only states are considered for terminal
            ocp.cost.W_e = self.Qt  # Weight matrix for terminal cost

            ocp.cost.Vx = np.zeros((self.ny, self.nx))  # linear output model

        # Set Basic constraints
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu = self.u_lb.reshape(self.nu)
        ocp.constraints.ubu = self.u_ub.reshape(self.nu)

        ocp.constraints.idxbx = np.arange(self.nx)
        ocp.constraints.lbx = self.x_lb.reshape(self.nx)
        ocp.constraints.ubx = self.x_ub.reshape(self.nx)

        ocp.constraints.idxbx_0 = np.arange(self.nx)
        ocp.constraints.lbx_0 = self.x_lb.reshape(self.nx)
        ocp.constraints.ubx_0 = self.x_ub.reshape(self.nx)

        ocp.constraints.idxbx_e = np.arange(self.nx)
        ocp.constraints.lbx_e = self.x_lb.reshape(self.nx)
        ocp.constraints.ubx_e = self.x_ub.reshape(self.nx)

        # ocp.constraints.Jbx_0 = np.eye(self.nx)
        # ocp.constraints.ubx_0 = self.x_ub.reshape(self.nx, 1)
        # ocp.constraints.lbx_0 = self.x_lb.reshape(self.nx, 1)

        if self.useSoftConstraints:
            # Define the penalty matrices Zl and Zu
            norm2_penalty = self.soft_penalty
            norm1_penalty = self.soft_penalty

            # Define slack variables
            # ocp.constraints.idxsbx = np.arange(self.nx)
            # ocp.constraints.lsbx = np.zeros((self.nx,))
            # ocp.constraints.usbx = np.ones((self.nx,))
            # Define slack variables
            idxsbx = np.setdiff1d(
                np.arange(self.nx), [5, 6, 7, 8, 9]
            )  # [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            idxsbx = np.array([2])
            ocp.constraints.idxsbx = idxsbx
            ocp.constraints.lsbx = np.zeros((len(idxsbx),))
            ocp.constraints.usbx = np.ones((len(idxsbx),))

            ocp.constraints.idxsbu = np.arange(self.nu)
            ocp.constraints.lsbu = np.zeros((self.nu,))
            ocp.constraints.usbu = np.ones((self.nu,))

            ocp.constraints.idxsbx_e = idxsbx
            ocp.constraints.lsbx_e = np.zeros((len(idxsbx),))
            ocp.constraints.usbx_e = np.ones((len(idxsbx),))

            # Adjust the size of the penalty matrices to match the number of slack variables
            ns = len(idxsbx) + self.nu  # Total number of slack variables
            ns_e = len(idxsbx)  # Total number of slack variables at the end of the horizon

            ocp.cost.Zl = norm2_penalty * np.ones(ns)
            ocp.cost.Zu = norm2_penalty * np.ones(ns)
            ocp.cost.zl = norm1_penalty * np.ones(ns)
            ocp.cost.zu = norm1_penalty * np.ones(ns)

            ocp.cost.Zl_e = norm2_penalty * np.ones(ns_e)
            ocp.cost.Zu_e = norm2_penalty * np.ones(ns_e)
            ocp.cost.zl_e = norm1_penalty * np.ones(ns_e)
            ocp.cost.zu_e = norm1_penalty * np.ones(ns_e)

            ocp.cost.Zl_0 = norm2_penalty * np.ones(self.nu)
            ocp.cost.Zu_0 = norm2_penalty * np.ones(self.nu)
            ocp.cost.zl_0 = norm1_penalty * np.ones(self.nu)
            ocp.cost.zu_0 = norm1_penalty * np.ones(self.nu)

        # Set initial state (not required and should not be set for moving horizon estimation)
        # ocp.constraints.x0 = self.x0
        # Add obstacle constraints
        if self.useObstacleConstraints:
            ocp = self.initObstacleConstraints(ocp)
        if self.useGateConstraints:
            NotImplementedError("Gate constraints not implemented yet.")
            # ocp = self.initGateConstraints(ocp)
        ocp = self.initZoro(ocp)
        # Code generation
        ocp.code_export_directory = self.export_dir
        self.ocp = ocp
        if self.useGP:
            self.residual_model = l4a.PytorchResidualModel(self.pytorch_model)
            self.ocp_solver = l4a.ResidualLearningMPC(
                ocp=self.ocp, residual_model=self.residual_model, use_cython=True
            )
            # self.ocp_solver =
        else:
            self.ocp_solver = AcadosOcpSolver(self.ocp)
        self.ocp_integrator = AcadosSimSolver(self.ocp)

        return None

    def stepAcados(self) -> NDArray[np.floating]:
        """Performs one optimization step using the current state, reference trajectory, and the previous solution for warmstarting. Updates the previous solution and returns the control input."""
        # Set initial state
        self.ocp_solver.set(0, "lbx", self.current_state)
        self.ocp_solver.set(0, "ubx", self.current_state)

        # Set reference trajectory
        y_ref = np.vstack([self.x_ref[:, :-1], self.u_ref])
        y_ref_e = self.x_ref[:, -1]
        for i in range(self.n_horizon):
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
        else:
            action = self.ocp_solver.get(0, "u")
        self.last_action = action
        return action

    def initObstacleConstraints(self, ocp: AcadosOcp) -> AcadosOcp:
        """Initialize the obstacle constraints."""
        self.obstacles_pos = self.initial_obs["obstacles_pos"]
        self.obstacles_in_range = self.initial_obs["obstacles_in_range"]
        self.obstacle_radius = 0.1
        self.obstacle_constraints = []
        num_obstacles = self.obstacles_pos.shape[0]
        nc_obst = self.obstacles_pos.shape[1]
        for k in range(num_obstacles):
            self.obstacle_constraints.append(
                ca.norm_2(self.x[:2] - self.param_sym[nc_obst * k : nc_obst * (k + 1) - 1])
                - self.obstacle_radius
            )
        ocp.model.con_h_expr = ca.vertcat(*self.obstacle_constraints)
        ocp.constraints.lh = np.zeros(len(self.obstacle_constraints))
        ocp.constraints.uh = ACADOS_INFTY * np.ones(len(self.obstacle_constraints))
        ocp.parameter_values = np.zeros(self.param_sym.shape[0])
        return ocp

    def updateObstacleConstraints(self):
        """Update the obstacle constraints based on the current obstacle positions."""
        if np.array_equal(
            self.obs["obstacles_in_range"], self.obstacles_in_range
        ) and np.array_equal(self.obs["gates_in_range"], self.gates_in_range):
            return None
        self.obstacles_in_range = self.obs["obstacles_in_range"]
        self.gates_in_range = self.obs["gates_in_range"]
        self.obstacle_pos = self.obs["obstacles_pos"]
        self.gate_pos = self.obs["gates_pos"]
        params = np.concatenate(
            [self.obstacle_pos.flatten(), self.gate_pos.flatten(), self.gates_rpy.flatten()]
        )
        # print("Updating obstacle constraints: Param Shape: ", params.shape)
        for stage in range(self.n_horizon):
            self.ocp_solver.set(stage, "p", params)
        return None
        # for stage in range(self.n_horizon + 1):
        #     self.ocp_solver.constraints_set(stage, "lh", self.ocp.constraints.lh)
        #     self.ocp_solver.constraints_set(stage, "uh", self.ocp.constraints.uh)
        return

    def initZoro(self, ocp: AcadosOcp):
        if self.useZoro:
            # custom update: disturbance propagation
            ocp.solver_options.custom_update_filename = "custom_update_function.c"
            ocp.solver_options.custom_update_header_filename = "custom_update_function.h"

            ocp.solver_options.custom_update_copy = False
            ocp.solver_options.custom_templates = [
                ("custom_update_function_zoro_template.in.c", "custom_update_function.c"),
                ("custom_update_function_zoro_template.in.h", "custom_update_function.h"),
            ]
            # zoro stuff
            zoro_description = ZoroDescription()
            zoro_description.backoff_scaling_gamma = 3.0
            # uncertainty propagation: P_{k+1} = (A_k+B_k K) @ P_k @ (A_k+B_kK)^T + G @ W @ G^T
            # G.shape = (nx, nw), W.shape = (nw, nw)
            # Noisy dynamics: x_{k+1} = A_k x_k + B_k u_k + G w_k
            # w_k ~ N(0, W)

            # Noise matrix W
            W = np.eye(self.nx) * 1e-3
            zoro_description.fdbk_K_mat = np.zeros((self.nu, self.nx))
            zoro_description.unc_jac_G_mat = 0.1 * np.diag(np.ones(self.nx))
            zoro_description.P0_mat = W
            zoro_description.W_mat = W
            zoro_description.idx_lbx_t = list(range(self.nx))
            zoro_description.idx_ubx_t = list(range(self.nx))
            zoro_description.idx_lbx_e_t = list(range(self.nx))
            zoro_description.idx_ubx_e_t = list(range(self.nx))
            ocp.zoro_description = zoro_description
        return ocp

    def updateTargetTrajectory(self):
        """Overriding base class' target trajectory update."""
        # Send latest observaton to the planner
        self.queue_state.put_nowait(self.obs)

        # Fetch latest plan
        try:
            result_path = self.queue_trajectory.get_nowait()
            num_points = len(result_path.x)
            self.x_ref[0, :num_points] = np.array(result_path.x)
            self.x_ref[1, :num_points] = np.array(result_path.y)
            self.x_ref[2, :num_points] = np.array(result_path.z)
        except queue.Empty: # Trajectory queue is empty
            pass # It's empty because we've fetched the latest trajectory - wait until we get the newest trajectory
        # if self.n_step % 10 == 0:
        #     gate_x, gate_y, gate_z = (
        #         self.obs["gates_pos"][:, 0],
        #         self.obs["gates_pos"][:, 1],
        #         self.obs["gates_pos"][:, 2],
        #     )
        #     gate_yaw = self.obs["gates_rpy"][:, 2]
        #     obs_x, obs_y = self.obs["obstacles_pos"][:, 0], self.obs["obstacles_pos"][:, 1]
        #     next_gate = self.obs["target_gate"] + 1
        #     drone_x, drone_y = self.obs["pos"][0], self.obs["pos"][1]
        #     result_path, ref_path, _ = self.planner.plan_path_from_observation(
        #         gate_x, gate_y, gate_z, gate_yaw, obs_x, obs_y, drone_x, drone_y, next_gate
        #     )
        #     l = len(result_path.x)
        #     self.x_ref[0, :l] = np.array(result_path.x)
        #     self.x_ref[1, :l] = np.array(result_path.y)
        #     self.x_ref[2, :l] = np.array(result_path.z)
        #     for i in range(len(result_path.x) - 1):
        #         p.addUserDebugLine(
        #             [result_path.x[i], result_path.y[i], result_path.z[i]],
        #             [result_path.x[i + 1], result_path.y[i + 1], result_path.z[i + 1]],
        #             lineColorRGB=[0, 0, 1],  # Red color
        #             lineWidth=2,
        #             lifeTime=0,  # 0 means the line persists indefinitely
        #             physicsClientId=0,
        #         )

    def do_planning(self):
        """The planning function to be run on a separate thread."""
        while True:
            # Fetch recent observation
            try:
                obs = self.queue_state.get_nowait()
            except queue.Empty: # State Queue is empty
                continue
            
            # Plan based on the fetched observation
            gate_x, gate_y, gate_z = (
                obs["gates_pos"][:, 0],
                obs["gates_pos"][:, 1],
                obs["gates_pos"][:, 2],
            )
            gate_yaw = obs["gates_rpy"][:, 2]
            obs_x, obs_y = obs["obstacles_pos"][:, 0], obs["obstacles_pos"][:, 1]
            next_gate = obs["target_gate"] + 1
            drone_x, drone_y = obs["pos"][0], obs["pos"][1]
            result_path, ref_path, _ = self.planner.plan_path_from_observation(
                gate_x, gate_y, gate_z, gate_yaw, obs_x, obs_y, drone_x, drone_y, next_gate
            )

            # Put the result to queue for commumication
            self.queue_trajectory.put_nowait(result_path)
