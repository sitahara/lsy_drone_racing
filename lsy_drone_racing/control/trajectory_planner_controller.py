"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pybullet as p

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.planner import Planner

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryController(BaseController):
    """Controller that follows a pre-defined trajectory."""

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], initial_info: dict):
        """Initialization of the controller.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        super().__init__(initial_obs, initial_info)
        self.planner = Planner(DEBUG=True)
        self._tick = 0
        self._freq = initial_info["env_freq"]

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
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
        if self._tick % 10 == 0:
            for i in range(len(result_path.x) - 1):
                p.addUserDebugLine(
                    [result_path.x[i], result_path.y[i], result_path.z[i]],
                    [result_path.x[i + 1], result_path.y[i + 1], result_path.z[i + 1]],
                    lineColorRGB=[1, 0, 0],  # Red color
                    lineWidth=2,
                    lifeTime=0,  # 0 means the line persists indefinitely
                    physicsClientId=0,
                )
        if self._tick % 100 == 0:
            for i in range(len(ref_path.x_sampled) - 1):
                p.addUserDebugLine(
                    [ref_path.x_sampled[i], ref_path.y_sampled[i], 0.0],
                    [ref_path.x_sampled[i + 1], ref_path.y_sampled[i + 1], 0.0],
                    lineColorRGB=[0, 1, 1],  # Red color
                    lineWidth=2,
                    lifeTime=0,  # 0 means the line persists indefinitely
                    physicsClientId=0,
                )
        return np.concatenate(
            (np.array([result_path.x[-1], result_path.y[-1], result_path.z[-1]]), np.zeros(10))
        )

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """Increment the time step counter."""
        self._tick += 1

    def episode_reset(self):
        """Reset the time step counter."""
        self._tick = 0
