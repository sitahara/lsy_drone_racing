import casadi as ca
import numpy as np

from lsy_drone_racing.control.dynamics_classes import BaseDynamics
from numpy.typing import NDArray


class CONSTRAINTS:
    """A class to represent constraints for drone racing.

    Attributes:
    ----------
    initial_obs : dict
        Initial observations including obstacles positions.
    dynamics : BaseDynamics
        Dynamics of the drone.
    constraints_info : dict
        Collection if information about type of constraints to use, the respective constraints, and parameters.

    Methods:
    -------
    initObstacleConstraints():
        Initializes the obstacle constraints.
    initGateConstraints():
        Initializes the gate constraints.
    initGoalConstraints():
        Initializes the goal constraints.
    """

    def __init__(
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        dynamics: BaseDynamics,
        constraints_info: dict = {
            "useObstacleConstraints": True,
            "useGateConstraints": False,
            "useGoalConstraints": False,
            "obstacle_diameter": 0.1,
        },
    ):
        useObstacleConstraints = constraints_info.get("useObstacleConstraints", True)
        useGateConstraints = constraints_info.get("useGateConstraints", False)
        useGoalConstraints = constraints_info.get("useGoalConstraints", False)
        obstacle_diameter = constraints_info.get("obstacle_diameter", 0.1)
        x = dynamics.x
        # u = dynamics.u
        # dx = dynamics.dx
        self.dict = {"obstacle": None, "goal": None, "gate": None}

        if useObstacleConstraints:
            self.dict["obstacle"] = {"obstacle_diameter": obstacle_diameter}
            self.initObstacleConstraints(initial_obs=initial_obs, x=x)
        if useGateConstraints:
            raise NotImplementedError("Gate constraints not implemented yet.")
            self.initGateConstraints()
        if useGoalConstraints:
            raise NotImplementedError("Goal constraints not implemented yet.")
            self.initGoalConstraints()

    def initObstacleConstraints(self, initial_obs: dict[str, NDArray[np.floating]], x: ca.MX):
        """Initialize the obstacle constraints."""
        obstacles_pos = initial_obs.get("obstacles_pos", np.zeros((6, 3)))
        obstacle_in_range = initial_obs.get("obstacle_in_range", np.zeros(6))
        obstacle_constraints = []
        num_obstacles = obstacles_pos.shape[0]
        num_param_per_obst = obstacles_pos.shape[1]
        p_obst = ca.MX.sym("p_obst", num_obstacles * num_param_per_obst)
        for k in range(num_obstacles):
            obstacle_constraints.append(
                ca.norm_2(
                    x[:num_param_per_obst]
                    - p_obst[num_param_per_obst * k : num_param_per_obst * (k + 1)]
                )
                - self.dict["obstacle"]["obstacle_diameter"]
            )
        self.dict["obstacle"]["param"] = p_obst
        self.dict["obstacle"]["lh"] = np.zeros(len(obstacle_constraints))
        self.dict["obstacle"]["uh"] = 1e9 * np.ones(len(obstacle_constraints))
        self.dict["obstacle"]["expr"] = ca.vertcat(*obstacle_constraints)
        # self.dict["obstacle_constraints"]["obstacle_pos"] = obstacles_pos
        self.dict["obstacle"]["obstacle_in_range"] = obstacle_in_range

    def initGateConstraints(self):
        """Initialize the gate constraints."""
        raise NotImplementedError("Gate constraints not implemented yet.")

    def initGoalConstraints(self):
        """Initialize the goal constraints."""
        raise NotImplementedError("Goal constraints not implemented yet.")
