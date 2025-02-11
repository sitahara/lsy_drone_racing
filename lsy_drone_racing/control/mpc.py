"""Module for Model Predictive Controller implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import pybullet as p
import toml
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as Rmat

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.mpc_utils import AcadosOptimizer, DroneDynamics, HermiteSpline, IPOPTOptimizer
from lsy_drone_racing.planner import Planner

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPC(BaseController):
    """Model Predictive Controller implementation."""

    def __init__(
        self,
        initial_obs: NDArray[np.floating],
        initial_info: dict,
        config_path: str = "lsy_drone_racing/mpc_utils/config.toml",
        hyperparams: dict = None,
        print_info: bool = True,
        visualize: bool = True,
    ):
        super().__init__(initial_obs, initial_info)
        self.initial_info = initial_info
        self.initial_obs = initial_obs
        self.print_info = print_info
        self.visualize = visualize

        # Load config file and update with hyperparameters if provided
        config = self.load_config_from_toml(config_path)
        self.hyperparameter_tuning = False
        if hyperparams is not None:
            self.hyperparameter_tuning = True
            config = self.update_config_with_hyperparameters(config, hyperparams)

        dynamics_info = config["dynamics_info"]
        self.ts = dynamics_info.get("ts", 1 / 60)
        self.n_horizon = dynamics_info.get("n_horizon", 60)
        self.referenceType = dynamics_info.get("referenceType", "hermite_spline")
        self.referenceTracking = dynamics_info.get("referenceTracking", True)
        self.useStartController = dynamics_info.get("start_controller", False)
        self.firstGuess = dynamics_info.get("firstGuess", "ipopt")
        self.print_info = dynamics_info.get("print_info", False)
        self.visualize = dynamics_info.get("visualize", False)

        optimizer_info = config["optimizer_info"]
        solver_options = config["solver_options"]
        constraints_info = config["constraints_info"]
        hermite_spline_info = config["hermite_spline_info"]

        cost_info = config["cost_info"]
        if self.referenceTracking:
            cost_info = cost_info["linear"]
        else:
            cost_info = cost_info["mpcc"]

        # For hyperparameter tuning
        self.tot_error = 0
        self.collided = False
        self.target_gate = initial_obs["target_gate"]
        self.gate_times = np.ones(self.initial_obs["gates_visited"].shape[0]) * 10
        self.number_gates_passed = 0

        pathPlanner = None

        if self.referenceType == "frenet":
            # Sho's path planner
            self._tick = 0
            self._freq = initial_info["env_freq"]

            self.DEBUG = False  # Toggles the debug display
            self.SAMPLE_IDX = 13  # Controls how much farther the desired position will be

            self.planner = Planner(
                DEBUG=self.DEBUG,
                USE_QUINTIC_SPLINE=False,
                MAX_ROAD_WIDTH=0.4,
                SAFETY_MARGIN=0.1,
                NUM_POINTS=20,
                K_J=0.1,
                MAX_CURVATURE=100.0,
            )
            constraints_info["tunnel"] = {"use": False}
        elif not self.referenceTracking or constraints_info.get("tunnel", {"use": False})["use"]:
            # Spline is parametric in gate positions and orientation
            pathPlanner = HermiteSpline(
                start_pos=self.initial_obs["pos"],
                start_rpy=self.initial_obs["rpy"],
                gates_pos=self.initial_obs["gates_pos"],
                gates_rpy=self.initial_obs["gates_rpy"],
                parametric=True,
                debug=hermite_spline_info["debug"],
                end_at_start=hermite_spline_info["end_at_start"],
                tangent_scaling=hermite_spline_info["tangent_scaling"],
                reverse_start_orientation=hermite_spline_info["reverse_start_orientation"],
            )
        elif self.referenceType == "hermite_spline":
            # Spline is recalculated when gates are updated
            pathPlanner = HermiteSpline(
                start_pos=self.initial_obs["pos"],
                start_rpy=self.initial_obs["rpy"],
                gates_pos=self.initial_obs["gates_pos"],
                gates_rpy=self.initial_obs["gates_rpy"],
                parametric=False,
                debug=hermite_spline_info["debug"],
                end_at_start=hermite_spline_info["end_at_start"],
                tangent_scaling=hermite_spline_info["tangent_scaling"],
                reverse_start_orientation=hermite_spline_info["reverse_start_orientation"],
            )

        # Initialize the dynamics model
        self.dynamics = DroneDynamics(
            initial_obs,
            initial_info,
            dynamics_info,
            constraints_info,
            cost_info=cost_info,
            pathPlanner=pathPlanner,
        )

        # Init reference arrays
        self.x_ref = np.tile(
            self.dynamics.x_eq.reshape(self.dynamics.nx, 1), (1, self.n_horizon + 1)
        )
        self.u_ref = np.tile(self.dynamics.u_eq.reshape(self.dynamics.nu, 1), (1, self.n_horizon))

        # Init optimizers
        self.ipopt = IPOPTOptimizer(
            dynamics=self.dynamics,
            solver_options=solver_options["ipopt"],
            optimizer_info=optimizer_info,
        )
        self.opt = AcadosOptimizer(
            dynamics=self.dynamics,
            solver_options=solver_options["acados"],
            optimizer_info=optimizer_info,
        )
        # Init reference trajectory
        self.set_target_trajectory()
        # Calculate initial guess (with IPOPt)
        self.calculate_initial_guess()

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
        self.obs = obs
        self.current_state = np.concatenate([obs["pos"], obs["vel"], obs["rpy"], obs["ang_vel"]])
        # Updates x_ref, the current target trajectory and upcounts the trajectory tick
        if self.useStartController and self.obs["pos"][-1] >= 0.3:
            self.useStartController = False
        if self.useStartController:
            if self.dynamics.interface == "Mellinger":
                action = np.zeros((13,))
                action[:3] = self.initial_obs["pos"]
                action[2] = 0.3
            else:
                action = np.zeros((4, 1))
                action[0] = self.dynamics.mass * self.dynamics.g * 1.3
            if obs["pos"][2] >= 0.2:
                self.control_state = 1
        else:
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
                print(f"Current position: {self.current_state[:3]}")
                print(f"Desired position: {self.x_ref[:3, 1]}")
                if self.dynamics.interface == "Mellinger":
                    print(f"Commanded position: {action[:3]}")
                    action[6:9] = np.zeros(3)
                elif self.dynamics.interface == "Thrust":
                    print("Commanded Total Thrust:", action[0], "Commanded RPY:", action[1:])
            # action = np.zeros((13,))
            # action[:3] = self.x_ref[:3, 1]

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
        if self.hyperparameter_tuning:
            if self.referenceTracking:
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
            if self.print_info:
                print(f"Error: {self.tot_error * self.ts}")
        if hasattr(self, "_tick"):
            self._tick += 1
        if not hasattr(self, "plot_counter"):
            self.plot_counter = 0
        if self.plot_counter % 10 == 0:
            planned_pos = self.opt.x_last[self.dynamics.state_indices["pos"], :]
            self.plotInPyBullet(planned_pos.T, lifetime=1, point_color=[0, 1, 0])
        self.plot_counter += 1

    def episode_callback(self):
        """Callback function called once after the episode is finished.

        Returns:
            The total error, whether the drone collided, the time spent on each segment, and the number of gates passed.
        """
        if self.hyperparameter_tuning:
            return (
                self.tot_error * 1000 * self.ts / self.n_step,
                self.collided,
                np.sum(self.gate_times),
                self.number_gates_passed,
            )
        if hasattr(self, "_tick"):
            self._tick += 0
        return None

    def calculate_initial_guess(self):
        self.obs = self.initial_obs
        self.current_state = np.concatenate(
            [self.obs["pos"], self.obs["vel"], self.obs["rpy"], self.obs["ang_vel"]]
        )
        self.updateTargetTrajectory()
        if self.firstGuess == "ipopt":
            self.ipopt.step(self.current_state, self.x_ref, self.u_ref)
            self.opt.x_guess = self.ipopt.x_guess
            self.opt.u_guess = self.ipopt.u_guess
        else:
            self.opt.step(self.current_state, self.obs, self.x_ref, self.u_ref)
        self.n_step = 0

    def set_target_trajectory(self, t_total: float = 9) -> None:
        """Set the target trajectory for the MPC controller."""
        self.n_step = 0  # current step for the target trajectory
        self.t_total = t_total
        num_points = int(t_total / self.ts)
        if self.referenceType == "hermite_spline":

            def target_trajectory(t):
                if isinstance(t, np.ndarray):
                    theta_0 = t[0] / self.t_total
                    theta_end = t[-1] / self.t_total
                    if theta_end > 1:
                        theta_end = 1
                    arr, _ = self.dynamics.pathPlanner.getPathPointsForPlotting(
                        theta_0=theta_0, theta_end=theta_end, num_points=len(t), only_path=True
                    )
                else:
                    theta_0 = t / self.t_total
                    theta_end = t / self.t_total
                    arr, _ = self.dynamics.pathPlanner.getPathPointsForPlotting(
                        theta_0=theta_0, theta_end=theta_end, num_points=1, only_path=True
                    )
                # if self.visualize:
                #     self.plotInPyBullet(arr, point_color=[0, 1, 0])
                return arr

            self.target_trajectory = target_trajectory
        elif self.referenceType == "frenet":
            pass
            # raise NotImplementedError("Frenet reference tracking is not implemented yet.")
        else:
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
            t = np.linspace(0, t_total, len(waypoints))
            self.target_trajectory = CubicSpline(t, waypoints)
        if self.visualize and self.referenceType != "frenet":
            t_vis = np.linspace(0, t_total, num_points)
            spline_points = self.target_trajectory(t_vis)
            self.plotInPyBullet(spline_points, lifetime=0)

        return None

    def updateTargetTrajectory(self):
        """Update the target trajectory for the MPC controller."""
        if self.referenceTracking:
            if self.referenceType == "frenet":
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
                num_points = min(len(result_path.x), self.n_horizon)
                # print(f"Number of points: {num_points}")
                self.x_ref[:3, :num_points] = np.array(
                    [
                        result_path.x[:num_points],
                        result_path.y[:num_points],
                        result_path.z[:num_points],
                    ]
                )
                self.x_ref[:3, num_points:] = np.tile(
                    np.array([result_path.x[-1], result_path.y[-1], result_path.z[-1]]).reshape(
                        3, 1
                    ),
                    (1, self.n_horizon - num_points + 1),
                )
                if self.visualize:
                    self.plotInPyBullet(
                        np.array([result_path.x, result_path.y, result_path.z]).T,
                        lifetime=0,
                        point_color=[1, 0, 0],
                    )
            else:
                current_time = self.n_step * self.ts
                t_horizon = np.linspace(
                    current_time, current_time + self.n_horizon * self.ts, self.n_horizon + 1
                )

                # Evaluate the target trajectory at the time points
                pos_des = self.target_trajectory(t_horizon).T
                # Handle the case where the end time exceeds the total time
                if t_horizon[-1] > self.t_total:
                    last_value = self.target_trajectory(self.t_total).reshape(3, 1)
                    n_repeat = np.sum(t_horizon > self.t_total)
                    pos_des[:, -n_repeat:] = np.tile(last_value, (1, n_repeat))
                # Update the reference trajectory
                # np.tile( np.array([1, 1, 0.5]).reshape(3, 1), (1, self.n_horizon + 1) )
                self.x_ref[:3, :] = pos_des
            self.n_step += 1
        return None

    def plotInPyBullet(
        self, points, lifetime=0, point_color=[1, 0, 0], tangents=None, tangent_color=[0, 1, 1]
    ):
        """Plot a series of points and optional tangents at those points in PyBullet."""
        try:
            # Plot the spline as a line in PyBullet
            for i in range(len(points) - 1):
                p.addUserDebugLine(
                    points[i],
                    points[i + 1],
                    lineColorRGB=point_color,  # Red color
                    lineWidth=2,
                    lifeTime=lifetime,  # 0 means the line persists indefinitely
                    physicsClientId=0,
                )
            if tangents is not None:
                for i in range(len(points)):
                    p.addUserDebugLine(
                        points[i],
                        points[i] + 0.1 * tangents[i],
                        lineColorRGB=tangent_color,  # Green color
                        lineWidth=2,
                        lifeTime=lifetime,  # 0 means the line persists indefinitely
                        physicsClientId=0,
                    )
        except p.error:
            ...  # Ignore errors if PyBullet is not available

    def load_config_from_toml(self, file_path):
        with open(file_path, "r") as file:
            config = toml.load(file)
        return config

    def update_config_with_hyperparameters(self, config, hyperparameter_dict):
        updated_config = config.copy()
        for section, params in hyperparameter_dict.items():
            if section in updated_config:
                updated_config[section].update(params)
            else:
                updated_config[section] = params
        return updated_config

    def episode_reset(self):
        super().episode_reset()
        # hyperparams = {"solver_options": {"acados": {"generate": False, "build": False}}}
        # self.__init__(self.initial_obs, self.initial_info, hyperparams=hyperparams)
