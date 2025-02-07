import queue
import threading
import time

import casadi as ca
import numpy as np
import pybullet as p
import toml
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as Rmat

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.mpc_utils import (
    AcadosOptimizer,
    DroneDynamics,
    IPOPTOptimizer,
    MPCCppDynamics,
)
from lsy_drone_racing.planner import Planner


class MPC(BaseController):
    """Model Predictive Controller implementation using do_mpc."""

    def __init__(
        self,
        initial_obs: NDArray[np.floating],
        initial_info: dict,
        config_path: str = "lsy_drone_racing/mpc_utils/config.toml",
        visualize: bool = False,
        separate_thread: bool = False,
        hyperparams: dict = None,
        print_info: bool = False,
    ):
        super().__init__(initial_obs, initial_info)
        self._tick = 0
        self._freq = initial_info["env_freq"]
        self.initial_info = initial_info
        self.initial_obs = initial_obs
        self.print_info = print_info
        config = toml.load(config_path)

        dynamics_info = config["dynamics_info"]
        self.ts = dynamics_info["ts"]
        self.n_horizon = dynamics_info["n_horizon"]

        optimizer_info = config["optimizer_info"]
        solver_options = config["solver_options"]
        constraints_info = config["constraints_info"]

        cost_info = config["cost_info"]
        mpcc_cost_info = config["cost_info_mpcc"]

        # For hyperparameter tuning
        self.tot_error = 0
        self.collided = False
        self.target_gate = initial_obs["target_gate"]
        self.gate_times = np.zeros(self.initial_obs["gates_visited"].shape[0])
        self.number_gates_passed = 0

        self.hyperparams = hyperparams
        if hyperparams is not None:
            cost_info["Qs_pos"] = hyperparams.get("Qs_pos", cost_info["Qs_pos"])
            cost_info["Qs_vel"] = hyperparams.get("Qs_vel", cost_info["Qs_vel"])
            cost_info["Qs_rpy"] = hyperparams.get("Qs_rpy", cost_info["Qs_rpy"])
            cost_info["Qs_drpy"] = hyperparams.get("Qs_drpy", cost_info["Qs_drpy"])
            cost_info["Qs_quat"] = hyperparams.get("Qs_quat", cost_info["Qs_quat"])

            cost_info["Qt_pos"] = hyperparams.get("Qt_pos", cost_info["Qs_pos"])
            cost_info["Qt_vel"] = hyperparams.get("Qt_vel", cost_info["Qs_vel"])
            cost_info["Qt_rpy"] = hyperparams.get("Qt_rpy", cost_info["Qs_rpy"])
            cost_info["Qt_drpy"] = hyperparams.get("Qt_drpy", cost_info["Qs_drpy"])
            cost_info["Qt_quat"] = hyperparams.get("Qt_quat", cost_info["Qs_quat"])

            cost_info["Rs"] = hyperparams.get("Rs", cost_info["Rs"])
            cost_info["Rd"] = hyperparams.get("Rd", cost_info["Rd"])

            optimizer_info["softPenalty"] = hyperparams.get(
                "softPenalty", optimizer_info["softPenalty"]
            )

        # Init Dynamics including control bounds
        if dynamics_info["dynamicsType"] == "MPCC":
            self.dynamics = MPCCppDynamics(
                initial_obs, initial_info, dynamics_info, constraints_info, cost_info=mpcc_cost_info
            )
        else:
            self.dynamics = DroneDynamics(
                initial_obs, initial_info, dynamics_info, constraints_info, cost_info=cost_info
            )

        # Init reference trajectory
        # self.dynamics.x_eq = np.zeros((self.dynamics.nx,))
        # self.dynamics.u_eq = np.zeros((self.dynamics.nu,))
        self.x_ref = np.tile(
            self.dynamics.x_eq.reshape(self.dynamics.nx, 1), (1, self.n_horizon + 1)
        )
        self.u_ref = np.tile(self.dynamics.u_eq.reshape(self.dynamics.nu, 1), (1, self.n_horizon))
        # print("u_ref", self.u_ref[:, :5], "x_ref", self.x_ref[:, :5])
        # Init Optimizer (acados needs also ipopt for initial guess, ipopt can be used standalone)
        self.ipopt = IPOPTOptimizer(
            dynamics=self.dynamics, solver_options=solver_options, optimizer_info=optimizer_info
        )
        self.opt = AcadosOptimizer(
            dynamics=self.dynamics, solver_options=solver_options, optimizer_info=optimizer_info
        )

        self.visualize = visualize
        self.separate_thread = separate_thread

        # Set up the planner
        # Since the planner is slow, run the planner on a separate thread
        self.obs = initial_obs
        self.queue_state = queue.Queue()
        self.queue_trajectory = queue.Queue()
        self.planner = Planner(
            DEBUG=True,
            USE_QUINTIC_SPLINE=False,
            MAX_ROAD_WIDTH=0.4,
            SAFETY_MARGIN=0.1,
            K_J=0.1,
            MAX_CURVATURE=100.0,
        )
        # self.planner = Planner(
        #     MAX_ROAD_WIDTH=0.5,
        #     D_ROAD_W=0.1,
        #     DT=0.015,
        #     NUM_POINTS=self.n_horizon,
        #     DEBUG=(visualize and not separate_thread),
        # )
        # if separate_thread:
        #     ## Since the controller requires the existence of x_ref, we will plan once in advance
        #     ## using initial observation
        #     gate_x, gate_y, gate_z = (
        #         initial_obs["gates_pos"][:, 0],
        #         initial_obs["gates_pos"][:, 1],
        #         initial_obs["gates_pos"][:, 2],
        #     )
        #     gate_yaw = initial_obs["gates_rpy"][:, 2]
        #     obs_x, obs_y = initial_obs["obstacles_pos"][:, 0], initial_obs["obstacles_pos"][:, 1]
        #     next_gate = initial_obs["target_gate"] + 1
        #     drone_x, drone_y = initial_obs["pos"][0], initial_obs["pos"][1]
        #     result_path, _, _ = self.planner.plan_path_from_observation(
        #         gate_x, gate_y, gate_z, gate_yaw, obs_x, obs_y, drone_x, drone_y, 0, 0, next_gate
        #     )
        #     num_points = len(result_path.x)
        #     self.x_ref[0, :num_points] = np.array(result_path.x)
        #     self.x_ref[1, :num_points] = np.array(result_path.y)
        #     self.x_ref[2, :num_points] = np.array(result_path.z)
        #     if num_points < self.n_horizon:
        #         last_value = np.array([result_path.x[-1], result_path.y[-1], result_path.z[-1]])
        #         self.x_ref[:3, num_points:] = np.tile(
        #             last_value.reshape(3, 1), (1, self.n_horizon - num_points + 1)
        #         )

        #     self.planner_thread = threading.Thread(target=self.do_planning, daemon=True)
        #     # self.planner_thread.start()

        self.calculate_initial_guess()

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
            The action either for the thrust or mellinger interface.
        """
        self.obs = obs  # Update internal observation
        self.current_state = np.concatenate([obs["pos"], obs["vel"], obs["rpy"], obs["ang_vel"]])
        # Updates x_ref, the current target trajectory and upcounts the trajectory tick
        self.updateTargetTrajectory()
        start_time = time.time()
        if self.opt is None:
            action = self.ipopt.step(self.current_state, self.x_ref, self.u_ref)
        else:
            action = self.opt.step(self.current_state, obs, self.x_ref, self.u_ref)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.print_info:
            print(f"Control signal update time: {elapsed_time:.5f} seconds")
            print(f"Desired position: {self.x_ref[:3, 1]}")
            print(f"Current position: {self.current_state[:3]}")
            if self.dynamics.interface == "Mellinger":
                print(f"Next position: {action[:3]}")
                action[6:9] = np.zeros(3)
            elif self.dynamics.interface == "Thrust":
                print(f"Total Thrust:", action[0], "Desired RPY:", action[1:])

        return action.flatten()

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        """Callback function called once after the control step.

        You can use this function to update your controller's internal state, save training data,
        update your models, etc.

        Instructions:
            Use any collected information to learn, adapt, and/or re-plan.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.
        """
        self._tick += 1
        self.tot_error += np.linalg.norm(obs["pos"] - self.x_ref[:3, 0])
        if obs["target_gate"] != self.target_gate:
            if self.target_gate != 0:
                self.gate_times[self.target_gate] = (
                    self.n_step * self.ts - self.gate_times[self.target_gate - 1]
                )
            else:
                self.gate_times[self.target_gate] = self.n_step * self.ts
            self.number_gates_passed += 1
            self.target_gate = obs["target_gate"]
        # print(f"Error: {self.tot_error * self.ts}")
        if len(info["collisions"]) > 0:
            self.collided = True
            self.gate_times[self.target_gate :] = 30

        # planned_pos = self.opt.x_last[self.dynamics.state_indices["pos"], :]
        # self.plotInPyBullet(planned_pos.T, lifetime=1, color=[0, 1, 0])

    def episode_callback(self):
        # print("Episode finished")
        # print(f"Reported error: {self.tot_error * self.ts}")
        if self.hyperparams is not None:
            return (
                self.tot_error * 1000 * self.ts / self.n_step,
                self.collided,
                self.gate_times,
                self.number_gates_passed,
            )

    def calculate_initial_guess(self):
        self.obs = self.initial_obs
        self.current_state = np.concatenate(
            [self.obs["pos"], self.obs["vel"], self.obs["rpy"], self.obs["ang_vel"]]
        )
        self.set_target_trajectory()
        self.updateTargetTrajectory()
        self.ipopt.step(self.current_state, self.x_ref, self.u_ref)
        self.opt.x_guess = self.ipopt.x_guess
        self.opt.u_guess = self.ipopt.u_guess
        self.n_step = 0

    def set_target_trajectory(self, t_total: float = 9) -> None:
        """Set the target trajectory for the MPC controller."""
        self.n_step = 0  # current step for the target trajectory
        self.t_total = t_total
        waypoints = np.array(
            [
                [1.0, 1.0, 0.05],
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
        t = np.linspace(0, t_total, len(waypoints))
        self.target_trajectory = CubicSpline(t, waypoints)
        # Generate points along the spline for visualization

        t_vis = np.linspace(0, t_total - 1, 100)
        spline_points = self.target_trajectory(t_vis)
        self.plotInPyBullet(spline_points, lifetime=0)
        return None

    # def updateTargetTrajectory(self):
    #     """Update the target trajectory for the MPC controller."""
    #     current_time = self.n_step * self.ts
    #     t_horizon = np.linspace(
    #         current_time, current_time + self.n_horizon * self.ts, self.n_horizon + 1
    #     )

    #     # Evaluate the spline at the time points
    #     pos_des = self.target_trajectory(t_horizon).T
    #     # Handle the case where the end time exceeds the total time
    #     if t_horizon[-1] > self.t_total:
    #         last_value = self.target_trajectory(self.t_total).reshape(3, 1)
    #         n_repeat = np.sum(t_horizon > self.t_total)
    #         pos_des[:, -n_repeat:] = np.tile(last_value, (1, n_repeat))
    #     # print(reference_trajectory_horizon)
    #     self.x_ref[:3, :] = (
    #         pos_des  # np.tile(np.array([1, 1, 0.5]).reshape(3, 1), (1, self.n_horizon + 1))  # pos_des
    #     )
    #     self.n_step += 1
    #     return None

    def updateTargetTrajectory(self):
        """Overriding base class' target trajectory update."""
        obs = self.obs
        gate_x, gate_y, gate_z = (
            obs["gates_pos"][:, 0],
            obs["gates_pos"][:, 1],
            obs["gates_pos"][:, 2],
        )
        gate_yaw = obs["gates_rpy"][:, 2]
        obs_x, obs_y = obs["obstacles_pos"][:, 0], obs["obstacles_pos"][:, 1]
        next_gate = obs["target_gate"] + 1
        drone_x, drone_y = obs["pos"][0], obs["pos"][1]
        drone_vx, drone_vy = obs["vel"][0], obs["vel"][1]
        result_path, ref_path, _ = self.planner.plan_path_from_observation(
            gate_x,
            gate_y,
            gate_z,
            gate_yaw,
            obs_x,
            obs_y,
            drone_x,
            drone_y,
            drone_vx,
            drone_vy,
            next_gate,
        )

        num_points = len(result_path.x)
        self.x_ref[0, :num_points] = np.array(result_path.x)
        self.x_ref[1, :num_points] = np.array(result_path.y)
        self.x_ref[2, :num_points] = np.array(result_path.z)
        if num_points < self.n_horizon:
            last_value = np.array([result_path.x[-1], result_path.y[-1], result_path.z[-1]])
            self.x_ref[:3, num_points:] = np.tile(
                last_value.reshape(3, 1), (1, self.n_horizon - num_points + 1)
            )
        if self.visualize:
            self.plotInPyBullet(self.x_ref[:3, :].T, lifetime=0, color=[0, 1, 0])
        # if self.separate_thread:
        #     # Send latest observaton to the planner
        #     self.queue_state.put_nowait(self.obs)

        #     # Fetch latest plan
        #     try:
        #         result_path = self.queue_trajectory.get_nowait()
        #         num_points = len(result_path.x)
        #         self.x_ref[0, :num_points] = np.array(result_path.x)
        #         self.x_ref[1, :num_points] = np.array(result_path.y)
        #         self.x_ref[2, :num_points] = np.array(result_path.z)
        #     except queue.Empty:  # Trajectory queue is empty
        #         pass  # It's empty because we've fetched the latest trajectory - wait until we get the newest trajectory
        # else:
        #     if self.n_step % 10 == 0:
        #         gate_x, gate_y, gate_z = (
        #             self.obs["gates_pos"][:, 0],
        #             self.obs["gates_pos"][:, 1],
        #             self.obs["gates_pos"][:, 2],
        #         )
        #         gate_yaw = self.obs["gates_rpy"][:, 2]
        #         obs_x, obs_y = self.obs["obstacles_pos"][:, 0], self.obs["obstacles_pos"][:, 1]
        #         next_gate = self.obs["target_gate"] + 1
        #         drone_x, drone_y = self.obs["pos"][0], self.obs["pos"][1]
        #         drone_vx, drone_vy = self.obs["vel"][0], self.obs["vel"][1]
        #         result_path, ref_path, _ = self.planner.plan_path_from_observation(
        #             gate_x,
        #             gate_y,
        #             gate_z,
        #             gate_yaw,
        #             obs_x,
        #             obs_y,
        #             drone_x,
        #             drone_y,
        #             drone_vx,
        #             drone_vy,
        #             next_gate,
        #         )
        #         num_points = len(result_path.x)
        #         self.x_ref[0, :num_points] = np.array(result_path.x)
        #         self.x_ref[1, :num_points] = np.array(result_path.y)
        #         self.x_ref[2, :num_points] = np.array(result_path.z)
        #         if self.visualize:
        #             for i in range(len(result_path.x) - 1):
        #                 p.addUserDebugLine(
        #                     [result_path.x[i], result_path.y[i], result_path.z[i]],
        #                     [result_path.x[i + 1], result_path.y[i + 1], result_path.z[i + 1]],
        #                     lineColorRGB=[0, 0, 1],  # Red color
        #                     lineWidth=2,
        #                     lifeTime=0,  # 0 means the line persists indefinitely
        #                     physicsClientId=0,
        #                 )
        self.n_step += 1

    def do_planning(self):
        """The planning function to be run on a separate thread."""
        while True:
            # Fetch recent observation
            try:
                obs = self.queue_state.get_nowait()
            except queue.Empty:  # State Queue is empty
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
                gate_x, gate_y, gate_z, gate_yaw, obs_x, obs_y, drone_x, drone_y, 0, 0, next_gate
            )

            # Put the result to queue for commumication
            self.queue_trajectory.put_nowait(result_path)

    def plotInPyBullet(self, points, lifetime=0, color=[1, 0, 0]):
        """Plot the desired state in PyBullet."""
        try:
            # Plot the spline as a line in PyBullet
            for i in range(len(points) - 1):
                p.addUserDebugLine(
                    points[i],
                    points[i + 1],
                    lineColorRGB=color,  # Red color
                    lineWidth=2,
                    lifeTime=lifetime,  # 0 means the line persists indefinitely
                    physicsClientId=0,
                )
        except p.error:
            ...  # Ignore errors if PyBullet is not available
