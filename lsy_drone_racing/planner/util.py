"""Utilities for remembering true positions of stuff from the observation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from np.typing import NDArray


class ObservationManager:
    """A class that records true position of items from the observation."""

    def __init__(self):
        """Prepares containers for storing coordinates."""
        # Gates
        self.gate_x = None
        self.gate_y = None
        self.gate_z = None
        self.gate_yaw = None

        # Obstacles
        self.obstacle_x = None
        self.obstacle_y = None

        # Nominal coordinates, to detect update ourselves witout relying on the observation
        self.gate_x_nominal = None
        self.obstacle_x_nominal = None

        # Misc
        self.first_update = True  # We use the first observation to initialize the variables

    def update(self, obs: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
        """Intercepts the observation and returns a better version of it."""
        # Positional data
        gate_x, gate_y, gate_z = (
            obs["gates_pos"][:, 0],
            obs["gates_pos"][:, 1],
            obs["gates_pos"][:, 2],
        )
        gate_yaw = obs["gates_rpy"][:, 2]
        obstacle_x, obstacle_y = obs["obstacles_pos"][:, 0], obs["obstacles_pos"][:, 1]

        if self.first_update is True:  # populate the variable with the initial value
            self.first_update = False
            self.gate_x = gate_x
            self.gate_y = gate_y
            self.gate_z = gate_z
            self.gate_yaw = gate_yaw
            self.obstacle_x = obstacle_x
            self.obstacle_y = obstacle_y

            #   Since it's not guaranteed that the coordinates' values
            # reflect the ground truth even if is_*_truth is True, we
            # record the initial value, and compare the observation against
            # the recorded initial nominal value.
            # ToDo for developers: fix the observation please
            self.gate_x_nominal = gate_x
            self.obstacle_x_nominal = obstacle_x

        else:
            # Update if there's any truth to it
            ## Detect update by comparing observation to nominal values
            is_gate_truth = gate_x != self.gate_x_nominal
            is_obstacle_truth = obstacle_x != self.obstacle_x_nominal

            ## Update
            self.gate_x[is_gate_truth] = gate_x[is_gate_truth]
            self.gate_y[is_gate_truth] = gate_y[is_gate_truth]
            self.gate_z[is_gate_truth] = gate_z[is_gate_truth]
            self.gate_yaw[is_gate_truth] = gate_yaw[is_gate_truth]
            self.obstacle_x[is_obstacle_truth] = obstacle_x[is_obstacle_truth]
            self.obstacle_y[is_obstacle_truth] = obstacle_y[is_obstacle_truth]

        # Putting back updated values
        obs["gates_pos"][:, 0] = self.gate_x
        obs["gates_pos"][:, 1] = self.gate_y
        obs["gates_pos"][:, 2] = self.gate_z
        obs["gates_rpy"][:, 2] = self.gate_yaw
        obs["obstacles_pos"][:, 0] = self.obstacle_x
        obs["obstacles_pos"][:, 1] = self.obstacle_y
        print(gate_x, self.gate_x)
        return obs
