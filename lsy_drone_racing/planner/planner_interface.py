from .planner_core import PlannerCore
from .spline import CSP_2D
import numpy as np


class Planner:
    def __init__(self):
        self.planner_core = PlannerCore()

    def create_obstacles(self, ob_x, ob_y, gate_x, gate_y, gate_yaw):
        """
        Creates a 2D map of obstacles. Just uses supplied values without adding any noise
        - Input
        None.
        - Output
        A list with positions and radius of all obstacles.
        Gate bars are treated as obstacles.
        [(x,y,radius)]
        """
        obs_center_x = []
        obs_center_y = []
        obs_radius = []
        # add gate obstacles
        expand_rate = 1.5

        # Fixed obstacles
        for i in range(len(ob_x)):
            obs_center_x.append(ob_x[i])
            obs_center_y.append(ob_y[i])
            obs_radius.append(0.1 * expand_rate)

        # Gate bars
        line_length = 0.58 / 2
        for i in range(len(gate_x)):
            # Calculate the line's endpoint
            dx = line_length * np.cos(gate_yaw[i])  # X direction offset
            dy = line_length * np.sin(gate_yaw[i])  # Y direction offset
            obs_center_x.append(gate_x[i] + dx)
            obs_center_y.append(gate_y[i] + dy)
            obs_radius.append(0.09 * expand_rate)
            obs_center_x.append(gate_x[i] - dx)
            obs_center_y.append(gate_y[i] - dy)
            obs_radius.append(0.09 * expand_rate)
        return list(zip(obs_center_x, obs_center_y, obs_radius))

    def plan_path_from_observation(
        self, gate_x, gate_y, gate_z, gate_yaw, obs_x, obs_y, drone_x, drone_y, next_gate
    ):
        """
        From gate, obstacle and position observations, returns an '''''''optimal''''''' path.
        Parameters
        ----------
        ##### Gate information
        gate_x : float
            X coordinates of gates, in correct order.
        gate_y : float
            Y coordinates of gates, in correct order.
        gate_yaw : float
            Yaw angle of gates, in correct order.

        * You should put 3 positions in these arguments, so that the generated spline isn't crazy.
        If you're in the first or the last segment, just put 2 gates but enable corresponding flags at the end
        ##### Obstacle informaion
        obs_x : float
            X coordinates of obstacles. This should include gate bars.
        obs_y : float
            Y coordinates of obstacles. This should include gate bars.
        ##### Drone informaion
        drone_x : float
            Drone's X coordinate.
        drone_y : float
            Drone's Y coordinate.
        drone_y : float
            Drone's S coordinate (Frenet frame).
            With this clearly defined, conversion from cartesian to frenet becomes singular.
        ##### Specific configurations
        next_gate : int
            Index of the next gate.
        """
        # Create reference trajectory
        reference_csp = None
        real_wp_x = None
        real_wp_y = None
        real_wp_z = None
        if next_gate == 1:
            real_wp_x = [1, 0.975, gate_x[0]]
            real_wp_y = [1, 0.9, gate_y[0]]
            real_wp_x.append(gate_x[0] + 0.5 * np.cos(gate_yaw[0] + np.pi / 2))
            real_wp_y.append(gate_y[0] + 0.5 * np.sin(gate_yaw[0] + np.pi / 2))
            real_wp_x.append(gate_x[1])
            real_wp_y.append(gate_y[1])
            real_wp_x.append(gate_x[1] + 0.05 * np.cos(gate_yaw[1] + np.pi / 2))
            real_wp_y.append(gate_y[1] + 0.05 * np.sin(gate_yaw[1] + np.pi / 2))
            real_wp_z = [0.1, 0.1, gate_z[0], gate_z[0], gate_z[1], gate_z[1]]
        elif next_gate == 2:
            real_wp_x = [gate_x[0]]
            real_wp_y = [gate_y[0]]
            real_wp_x.append(gate_x[0] + 0.5 * np.cos(gate_yaw[0] + np.pi / 2))
            real_wp_y.append(gate_y[0] + 0.5 * np.sin(gate_yaw[0] + np.pi / 2))
            real_wp_x.append(gate_x[1])
            real_wp_y.append(gate_y[1])
            real_wp_x.append(gate_x[1] + 0.05 * np.cos(gate_yaw[1] + np.pi / 2))
            real_wp_y.append(gate_y[1] + 0.05 * np.sin(gate_yaw[1] + np.pi / 2))
            real_wp_x.append(gate_x[2])
            real_wp_y.append(gate_y[2])
            real_wp_x.append(gate_x[2] + 0.05 * np.cos(gate_yaw[2] + np.pi / 2))
            real_wp_y.append(gate_y[2] + 0.05 * np.sin(gate_yaw[2] + np.pi / 2))
            real_wp_z = [gate_z[0], gate_z[0], gate_z[1], gate_z[1], gate_z[2], gate_z[2]]
        elif next_gate == 3:
            real_wp_x = [gate_x[1]]
            real_wp_y = [gate_y[1]]
            real_wp_x.append(gate_x[1] + 0.05 * np.cos(gate_yaw[1] + np.pi / 2))
            real_wp_y.append(gate_y[1] + 0.05 * np.sin(gate_yaw[1] + np.pi / 2))
            real_wp_x.append(gate_x[2])
            real_wp_y.append(gate_y[2])
            real_wp_x.append(gate_x[2] + 0.15 * np.cos(gate_yaw[2] + np.pi / 2))
            real_wp_y.append(gate_y[2] + 0.15 * np.sin(gate_yaw[2] + np.pi / 2))
            real_wp_x.append(gate_x[2] - 0.5)
            real_wp_y.append(gate_y[2] + 0.5)
            real_wp_x.append(gate_x[3])
            real_wp_y.append(gate_y[3])
            real_wp_x.append(gate_x[3] + 0.05 * np.cos(gate_yaw[3] + np.pi / 2))
            real_wp_y.append(gate_y[3] + 0.05 * np.sin(gate_yaw[3] + np.pi / 2))
            real_wp_z = [
                gate_z[1],
                gate_z[1],
                gate_z[2],
                gate_z[2],
                (gate_z[2] + gate_z[3]) / 2,
                gate_z[3],
                gate_z[3],
            ]
        elif next_gate == 4:
            real_wp_x = [gate_x[2]]
            real_wp_y = [gate_y[2]]
            real_wp_x.append(gate_x[2] + 0.15 * np.cos(gate_yaw[2] + np.pi / 2))
            real_wp_y.append(gate_y[2] + 0.15 * np.sin(gate_yaw[2] + np.pi / 2))
            # spaxxxxxx
            real_wp_x.append(gate_x[2] - 0.5)
            real_wp_y.append(gate_y[2] + 0.5)
            real_wp_x.append(gate_x[3])
            real_wp_y.append(gate_y[3])
            real_wp_x.append(gate_x[3] + 0.05 * np.cos(gate_yaw[3] + np.pi / 2))
            real_wp_y.append(gate_y[3] + 0.05 * np.sin(gate_yaw[3] + np.pi / 2))
            real_wp_x.append(gate_x[3])
            real_wp_y.append(gate_y[3] - 0.5)
            real_wp_z = [
                gate_z[2],
                gate_z[2],
                (gate_z[2] + gate_z[3]) / 2,
                gate_z[3],
                gate_z[3],
                gate_z[3],
            ]
        reference_csp = CSP_2D(real_wp_x, real_wp_y, real_wp_z)
        # Create obstacles
        ob = self.create_obstacles(obs_x, obs_y, gate_x, gate_y, gate_yaw)

        # find out where I am wrt reference spline
        s, d = reference_csp.cartesian_to_frenet(drone_x, drone_y)

        # Generate path
        fplist, best_idx, path = self.planner_core.frenet_optimal_planning(
            reference_csp, s, d, 0.0, 0.0, ob
        )

        if path is None:
            return None, reference_csp, fplist
        else:
            return path, reference_csp, fplist
