"""Module for Model Predictive Controller implementation."""

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
    PolynomialPlanner,
    HermiteSpline,
)


class MPC(BaseController):
    """Model Predictive Controller implementation using do_mpc."""

    def __init__(
        self,
        initial_obs: NDArray[np.floating],
        initial_info: dict,
        config_path: str = "lsy_drone_racing/mpc_utils/config.toml",
        hyperparams: dict = None,
        print_info: bool = True,
    ):
        super().__init__(initial_obs, initial_info)
        self.initial_info = initial_info
        self.initial_obs = initial_obs
        self.print_info = print_info
        config = toml.load(config_path)

        dynamics_info = config["dynamics_info"]
        self.ts = dynamics_info["ts"]
        self.n_horizon = dynamics_info["n_horizon"]
        self.reference = dynamics_info["reference"]
        self.useStartController = dynamics_info["start_controller"]

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
            # path_points, path_tangents = self.dynamics.pathPlanner.getPathPointsForPlotting()
            # self.plotInPyBullet(path_points, lifetime=0, color=[0, 0, 1], tangents=path_tangents)

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
        self.set_target_trajectory()

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
        if self.useStartController and obs["pos"][-1] <= 0.2:
            if self.dynamics.interface == "Mellinger":
                action = np.zeros((13,))
                action[:3] = self.initial_obs["pos"]
                action[2] = 0.2
            else:
                action = np.zeros((4, 1))
                action[0] = self.dynamics.mass * self.dynamics.g * 1.5
        else:
            self.updateTargetTrajectory()
            start_time = time.time()
            if self.opt is None:
                action = self.ipopt.step(self.current_state, self.x_ref, self.u_ref)
            else:
                action = self.opt.step(self.current_state, obs, self.x_ref, self.u_ref)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Control signal update time: {elapsed_time:.5f} seconds")
            # print(f"Current position: {self.current_state[:3]}")
            if self.print_info:
                print(f"Desired position: {self.x_ref[:3, 1]}")
                if self.dynamics.interface == "Mellinger":
                    print(f"Next position: {action[:3]}")
                    action[6:9] = np.zeros(3)
                elif self.dynamics.interface == "Thrust":
                    print("Total Thrust:", action[0], "Desired RPY:", action[1:])

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
        self.updateTargetTrajectory()
        self.ipopt.step(self.current_state, self.x_ref, self.u_ref)
        self.opt.x_guess = self.ipopt.x_guess
        self.opt.u_guess = self.ipopt.u_guess
        self.n_step = 0

    def set_target_trajectory(self, t_total: float = 15) -> None:
        """Set the target trajectory for the MPC controller."""
        self.n_step = 0  # current step for the target trajectory
        self.t_total = t_total
        if self.reference != "spline":
            self.pathPlanner = HermiteSpline(
                start_pos=self.initial_obs["pos"],
                start_rpy=self.initial_obs["rpy"],
                gates_pos=self.initial_obs["gates_pos"],
                gates_rpy=self.initial_obs["gates_rpy"],
            )
            num_points = int(t_total / self.ts)
            if self.reference == "hermite_spline":

                def target_trajectory(t):
                    if isinstance(t, np.ndarray):
                        theta_0 = t[0] / self.t_total
                        theta_end = t[-1] / self.t_total
                        if theta_end > 1:
                            theta_end = 1
                        arr, _ = self.pathPlanner.getPathPointsForPlotting(
                            theta_0=theta_0, theta_end=theta_end, num_points=len(t)
                        )
                    else:
                        theta_0 = t / self.t_total
                        theta_end = t / self.t_total
                        arr, _ = self.pathPlanner.getPathPointsForPlotting(
                            theta_0=theta_0, theta_end=theta_end, num_points=1
                        )

                    return arr

            if self.reference == "polynomial":
                self.pathPlanner.fitPolynomial(t_total=t_total)

                def target_trajectory(t):
                    arr = np.zeros((len(t), 3))
                    arr[:, 0] = self.pathPlanner.poly_x(t)
                    arr[:, 1] = self.pathPlanner.poly_y(t)
                    arr[:, 2] = self.pathPlanner.poly_z(t)
                    return arr

            self.target_trajectory = target_trajectory
            t = np.linspace(0, t_total, num_points)
            spline_points = self.target_trajectory(t)
            self.plotInPyBullet(spline_points, lifetime=0)
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
            # self.t_total = t_total
            t = np.linspace(0, t_total, len(waypoints))
            self.target_trajectory = CubicSpline(t, waypoints)
            # Generate points along the spline for visualization

            t_vis = np.linspace(0, t_total - 1, 100)
            spline_points = self.target_trajectory(t_vis)
            self.plotInPyBullet(spline_points, lifetime=0)

        return None

    def updateTargetTrajectory(self):
        """Update the target trajectory for the MPC controller."""
        current_time = self.n_step * self.ts
        t_horizon = np.linspace(
            current_time, current_time + self.n_horizon * self.ts, self.n_horizon + 1
        )

        # Evaluate the spline at the time points
        pos_des = self.target_trajectory(t_horizon).T
        # Handle the case where the end time exceeds the total time
        if t_horizon[-1] > self.t_total:
            last_value = self.target_trajectory(self.t_total).reshape(3, 1)
            n_repeat = np.sum(t_horizon > self.t_total)
            pos_des[:, -n_repeat:] = np.tile(last_value, (1, n_repeat))
        # print(reference_trajectory_horizon)
        # pos_des = self.planner.output_xref(current_time)
        self.x_ref[:3, :] = (
            pos_des  # np.tile( np.array([1, 1, 0.5]).reshape(3, 1), (1, self.n_horizon + 1) )  # pos_des
        )
        self.n_step += 1
        return None

    def plotInPyBullet(self, points, lifetime=0, color=[1, 0, 0], tangents=None):
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
            if tangents is not None:
                for i in range(len(points)):
                    p.addUserDebugLine(
                        points[i],
                        points[i] + 0.1 * tangents[i],
                        lineColorRGB=[0, 1, 1],  # Green color
                        lineWidth=2,
                        lifeTime=lifetime,  # 0 means the line persists indefinitely
                        physicsClientId=0,
                    )
        except p.error:
            ...  # Ignore errors if PyBullet is not available
