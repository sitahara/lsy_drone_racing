from abc import ABC, abstractmethod

import numpy as np
import toml
from numpy.typing import NDArray


class BaseOptimizer(ABC):
    """Abstract base class for optimizer implementations."""

    def __init__(self, dynamics, solver_options, optimizer_info):
        self.solver_options = solver_options
        self.optimizer_info = optimizer_info
        self.useSoftConstraints = self.optimizer_info.get("useSoftConstraints", True)
        self.softPenalty = self.optimizer_info.get("softPenalty", 1e3)
        self.dynamics = dynamics
        self.n_horizon = dynamics.n_horizon
        self.ts = dynamics.ts
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
        pass

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
