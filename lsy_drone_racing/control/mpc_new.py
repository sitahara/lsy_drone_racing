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
)


class MPC(BaseController):
    """Model Predictive Controller implementation using do_mpc."""

    def __init__(
        self,
        initial_obs: NDArray[np.floating],
        initial_info: dict,
        config_path: str = "lsy_drone_racing/mpc_utils/config.toml",
    ):
        super().__init__(initial_obs, initial_info)
        self.initial_info = initial_info
        self.initial_obs = initial_obs
        config = toml.load(config_path)

        dynamics_info = config["dynamics_info"]
        self.ts = dynamics_info["ts"]
        self.n_horizon = dynamics_info["n_horizon"]

        optimizer_info = config["optimizer_info"]
        solver_options = config["solver_options"]
        constraints_info = config["constraints_info"]

        # Init Dynamics including control bounds
        if dynamics_info["dynamicsType"] == "MPCC":
            self.dynamics = MPCCppDynamics(
                initial_obs,
                initial_info,
                dynamics_info,
                constraints_info,
                cost_info=config["cost_info_mpcc"],
            )
        else:
            self.dynamics = DroneDynamics(
                initial_obs,
                initial_info,
                dynamics_info,
                constraints_info,
                cost_info=config["cost_info"],
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

        self.obs = initial_obs
        self.current_state = np.concatenate(
            [initial_obs["pos"], initial_obs["vel"], initial_obs["rpy"], initial_obs["ang_vel"]]
        )
        self.updateTargetTrajectory()
        self.ipopt.step(self.current_state, self.x_ref, self.u_ref)
        self.opt.x_guess = self.ipopt.x_guess
        self.opt.u_guess = self.ipopt.u_guess
        self.n_step = 0

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
        self.updateTargetTrajectory()
        start_time = time.time()
        if self.opt is None:
            action = self.ipopt.step(self.current_state, self.x_ref, self.u_ref)
        else:
            action = self.opt.step(self.current_state, obs, self.x_ref, self.u_ref)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Control signal update time: {elapsed_time:.5f} seconds")
        print(f"Current position: {self.current_state[:3]}")
        print(f"Desired position: {self.x_ref[:3, 1]}")
        if self.dynamics.interface == "Mellinger":
            print(f"Next position: {action[:3]}")
        else:
            print(f"Total Thrust:", action[0], "Torques:", action[1:])

        return action.flatten()

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
        self.x_ref[:3, :] = pos_des
        self.n_step += 1
        return None

    # def setupCostFunction(self, cost_info: dict) -> None:
    #     """Setup the cost function for the MPC controller."""
    #     # Define the cost function parameters
    #     costs = {}
    #     costs["cost_type"] = cost_info["cost_type"]
    #     if self.dynamics.nx == 13:
    #         cost_info["Qs"] = np.concatenate(
    #             [
    #                 cost_info["Qs_pos"],
    #                 cost_info["Qs_vel"],
    #                 cost_info["Qs_quat"],
    #                 cost_info["Qs_dang"],
    #             ]
    #         )
    #         cost_info["Qt"] = np.concatenate(
    #             [
    #                 cost_info["Qt_pos"],
    #                 cost_info["Qt_vel"],
    #                 cost_info["Qt_quat"],
    #                 cost_info["Qt_dang"],
    #             ]
    #         )
    #     else:
    #         cost_info["Qs"] = np.concatenate(
    #             [
    #                 cost_info["Qs_pos"],
    #                 cost_info["Qs_vel"],
    #                 cost_info["Qs_ang"],
    #                 cost_info["Qs_dang"],
    #             ]
    #         )
    #         cost_info["Qt"] = np.concatenate(
    #             [
    #                 cost_info["Qt_pos"],
    #                 cost_info["Qt_vel"],
    #                 cost_info["Qt_ang"],
    #                 cost_info["Qt_dang"],
    #             ]
    #         )
    #     if self.dynamics.useControlRates:
    #         R = np.diag(cost_info["Rd"] / self.dynamics.u_scal)
    #         Qs = np.diag(np.concatenate([cost_info["Qs"], cost_info["Rs"]]) / self.dynamics.x_scal)
    #         Qt = np.diag(np.concatenate([cost_info["Qt"], cost_info["Rs"]]) / self.dynamics.x_scal)
    #     else:
    #         R = np.diag(cost_info["Rs"] / self.dynamics.u_scal)
    #         Qs = np.diag(cost_info["Qs"] / self.dynamics.x_scal)
    #         Qt = np.diag(cost_info["Qt"] / self.dynamics.x_scal)

    #     costs["R"] = R
    #     costs["Qs"] = Qs
    #     costs["Qt"] = Qt
    #     self.costs = costs
    #     self.costs["cost_function"] = self.baseCost  # cost function for the MPC

    # def baseCost(self, x, x_ref, u, u_ref):
    #     """Base Cost function for the MPC controller."""
    #     return ca.mtimes([(x - x_ref).T, self.costs["Qs"], (x - x_ref)]) + ca.mtimes(
    #         [(u - u_ref).T, self.costs["R"], (u - u_ref)]
    #     )

    # def setupCostTimeFunction(self, cost_info: dict) -> None:
    #     """Setup the cost function for the MPC controller."""
    #     # Define the cost function parameters
    #     costs = {}
    #     costs["cost_type"] = "External"
    #     costs["Qs_pos"] = np.diag(cost_info["Qs_pos"])
    #     costs["Qs_vel"] = np.diag(cost_info["Qs_vel"])
    #     costs["Qs_ang"] = np.diag(cost_info["Qs_ang"])
    #     costs["Qs_dang"] = np.diag(cost_info["Qs_dang"])
    #     costs["R"] = np.diag(cost_info["Rs"])
    #     costs["Q_time"] = cost_info["Q_time"]
    #     costs["Q_gate"] = cost_info["Q_gate"]
    #     self.costs = costs
    #     self.costs["cost_function"] = self.timeCost

    # def timeCost(self, x, u, p):
    #     vel = x[3:6]
    #     cost_vel = ca.mtimes([vel.T, self.costs["Qs_vel"], vel])

    #     ang = x[6:9]
    #     cost_ang = ca.mtimes([ang.T, self.costs["Qs_ang"], ang])

    #     dang = x[9:12]
    #     cost_dang = ca.mtimes([dang.T, self.costs["Qs_dang"], dang])

    #     t = x[self.dynamics.state_indices["time"]]
    #     cost_time = t * self.costs["Q_time"] * t

    #     cost_u = ca.mtimes([u.T, self.costs["R"], u])

    #     pos_rpy = ca.vertcat(x[:3], x[6:9])
    #     err = ca.if_else(
    #         x[self.dynamics.state_indices["gate_passed"]],
    #         ca.norm_2(pos_rpy - p[self.dynamics.param_indices["subsequent_gate"]]),
    #         ca.norm_2(pos_rpy - p[self.dynamics.param_indices["next_gate"]]),
    #     )
    #     cost_gate = ca.mtimes([err.T, self.costs["Q_gate"], err])

    #     return cost_gate + cost_vel + cost_ang + cost_dang + cost_u + cost_time
