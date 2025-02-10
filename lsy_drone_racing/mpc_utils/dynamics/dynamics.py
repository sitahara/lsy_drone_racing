from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class BaseDynamics(ABC):
    """Abstract base class for dynamics implementations including bounds, nonlinear constraints, and costs for states and controls."""

    def __init__(self):
        """Initialization of the key dictionaries."""

    @abstractmethod
    def transformState(self, x: np.ndarray) -> np.ndarray:
        """Transforms observations from the environment to the respective states used in the dynamics."""
        pass

    @abstractmethod
    def transformAction(self, x_sol: np.ndarray, u_sol: np.ndarray) -> np.ndarray:
        """Transforms optimizer solutions to controller interfaces (Mellinger or Thrust)."""
        pass

    @abstractmethod
    def setupCasadiFunctions(self):
        """Setup explicit, implicit, and discrete dynamics functions."""
        pass

    @abstractmethod
    def setupNominalParameters(self):
        """Setup the nominal parameters of the drone/environment/controller."""
        pass

    @abstractmethod
    def setupNLConstraints(self):
        """Setup the basic constraints for the drone/environment controller."""
        pass

    @abstractmethod
    def updateParameters(self, obs: dict = None, init: bool = False) -> np.ndarray:
        """Update the parameters of the drone/environment controller."""
        pass

    @abstractmethod
    def setupBaseBounds(self):
        """Setup the nominal and unchanging bounds of the drone/environment controller."""
        pass
