"""Interface for the planner class."""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .planner_core import PlannerCore
from .spline import CSP_2D, FrenetPath


class Planner:
    """A helper interface class for the Planner."""

    def __init__(
        self,
        MAX_CURVATURE: float = 50.0,
        MAX_ROAD_WIDTH: float = 0.5,
        D_ROAD_W: float = 0.1,
        DT: float = 0.03,
        NUM_POINTS: int = 20,
        K_J: float = 0.5,
        K_D: float = 8.0,
        DEBUG: bool = False,
    ):
        """Initializes the planner core.

        Parameters
        ----------
        MAX_CURVATURE : float
            Maximum allowed curvature of the trajectory on the cartesian frame.
            Trajectory with more curvature (tighter curve) will get a penalty in the cost.
        MAX_ROAD_WIDTH : float
            Maximum value of lateral offset when creating trajetory candidates in frenet frame.
        D_ROAD_W : float
            Determines the width to sample the lateral offset of the trajectory in frenet frame.
        DT : float
            Sampling interval of the planned trajectory.
        NUM_POINTS : int
            Number of points included in the planned trajectory.
        K_J : float
            Weight constant for the trajectory's jerk.
        K_D : float
            Weight constant for the terminal deviation from the desired trajectory.
        DEBUG : bool
            Enables or disables display of planning information on a separate matplotlib window.
        """
        # calculate prediction horizon based on the number of points
        T_PRED = (NUM_POINTS) * DT

        self.planner_core = PlannerCore(
            MAX_CURVATURE=MAX_CURVATURE,
            MAX_ROAD_WIDTH=MAX_ROAD_WIDTH,
            D_ROAD_W=D_ROAD_W,
            DT=DT,
            T_PRED=T_PRED,
            K_J=K_J,
            K_D=K_D,
            DEBUG=DEBUG,
        )
        self.DEBUG = DEBUG
        print(f"Debugging is {'Enabled' if self.DEBUG else 'Disabled'}")

        if self.DEBUG is True:
            self.fig, self.ax = plt.subplots()
            self.ax.grid(True)
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-2, 2)
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            plt.ion()
            plt.show(block=False)

    def create_obstacles(
        self,
        ob_x: List[float],
        ob_y: List[float],
        gate_x: List[float],
        gate_y: List[float],
        gate_yaw: List[float],
        expand_rate: float = 1.1,
    ) -> List[Tuple[float, float, float]]:
        """Creates a 2D map of obstacles.

        This function assumes that the supplied coordinates already have noises included.

        - Input
        ob_x: List of Float
            X coordinates of the obstacles.
        ob_y: List of Float
            Y coordinates of the obstacles.
        gate_x: List of Float
            X coordinates of the gates.
        gate_y: List of Float
            Y coordinates of the gates.
        gate_yaw: List of Float
            Yaw angles of the gates.
        expand_rate: Float
            The size of the obstacles are multiplied with this value, so that
            inherently there is some robustness in the planner.
        - Output
        A list with positions and radius of all obstacles.
        Vertical gate frames are also treated as obstacles.
        The output is structured in this way: [(x,y,radius), ...]
        """
        obs_center_x = []
        obs_center_y = []
        obs_radius = []

        # Creating fixed obstacles
        for i in range(len(ob_x)):
            obs_center_x.append(ob_x[i])
            obs_center_y.append(ob_y[i])
            obs_radius.append(0.1 * expand_rate)

        # Creating horizontal gate frames
        ## Length of the vertex of the frame is 0.58m
        line_length = 0.58 / 2
        for i in range(len(gate_x)):
            ## Calculate the line's endpoint
            dx = line_length * np.cos(gate_yaw[i])  # X direction offset
            dy = line_length * np.sin(gate_yaw[i])  # Y direction offset
            ## Calculate the 2D position of two vertical gate frames
            obs_center_x.append(gate_x[i] + dx)
            obs_center_y.append(gate_y[i] + dy)
            obs_radius.append(0.09 * expand_rate)

            obs_center_x.append(gate_x[i] - dx)
            obs_center_y.append(gate_y[i] - dy)
            obs_radius.append(0.09 * expand_rate)

        # Turns three independent lists into one list of tuples
        return list(zip(obs_center_x, obs_center_y, obs_radius))

    def plan_path_from_observation(
        self,
        gate_x: List[float],
        gate_y: List[float],
        gate_z: List[float],
        gate_yaw: List[float],
        obs_x: List[float],
        obs_y: List[float],
        drone_x: float,
        drone_y: float,
        next_gate: int,
    ) -> Tuple[FrenetPath, CSP_2D, List[FrenetPath]]:
        """From gate, obstacle and position observations, returns an optimal path.

        Parameters
        ----------
        ##### Gate information
        gate_x : List[float]
            X coordinates of gates, in correct order.
        gate_y : float
            Y coordinates of gates, in correct order.
        gate_yaw : float
            Yaw angle of gates, in correct order.

        ##### Obstacle informaion
        obs_x : float
            X coordinates of obstacles. This must include gate frames.
        obs_y : float
            Y coordinates of obstacles. This must include gate frames.
        ##### Drone informaion
        drone_x : float
            Drone's X coordinate.
        drone_y : float
            Drone's Y coordinate.
        drone_y : float
            Drone's S coordinate (Frenet frame).
        ##### Additional information
        next_gate : int
            Index of the next gate.
        """
        # Create reference trajectory
        reference_csp = None
        real_wp_x = None
        real_wp_y = None
        real_wp_z = None
        ## Heurestically adjust points to use when creating reference spline
        if next_gate == 1:
            real_wp_x = [1, 0.975, gate_x[0]]
            real_wp_y = [1, 0.9, gate_y[0]]
            real_wp_x.append(gate_x[0] + 0.5 * np.cos(gate_yaw[0] + np.pi / 2 + 0.7))
            real_wp_y.append(gate_y[0] + 0.5 * np.sin(gate_yaw[0] + np.pi / 2 + 0.7))
            real_wp_x.append(gate_x[1])
            real_wp_y.append(gate_y[1])
            real_wp_x.append(gate_x[1] + 0.05 * np.cos(gate_yaw[1] + np.pi / 2))
            real_wp_y.append(gate_y[1] + 0.05 * np.sin(gate_yaw[1] + np.pi / 2))
            real_wp_z = [0.1, 0.1, gate_z[0], gate_z[0], gate_z[1], gate_z[1]]
        elif next_gate == 2:
            real_wp_x = [gate_x[0]]
            real_wp_y = [gate_y[0]]
            real_wp_x.append(gate_x[0] + 0.5 * np.cos(gate_yaw[0] + np.pi / 2 + 0.7))
            real_wp_y.append(gate_y[0] + 0.5 * np.sin(gate_yaw[0] + np.pi / 2 + 0.7))
            real_wp_x.append(gate_x[1])
            real_wp_y.append(gate_y[1])
            real_wp_x.append(gate_x[1] + 0.05 * np.cos(gate_yaw[1] + np.pi / 2))
            real_wp_y.append(gate_y[1] + 0.05 * np.sin(gate_yaw[1] + np.pi / 2))
            real_wp_x.append(gate_x[2])
            real_wp_y.append(gate_y[2])
            real_wp_x.append(gate_x[2] + 0.05 * np.cos(gate_yaw[2] + np.pi / 2))
            real_wp_y.append(gate_y[2] + 0.05 * np.sin(gate_yaw[2] + np.pi / 2))
            real_wp_z = [gate_z[0], gate_z[0], gate_z[1], gate_z[1], gate_z[2], gate_z[2]]
        elif next_gate == 3 or next_gate == 4:
            real_wp_x = [gate_x[1]]
            real_wp_y = [gate_y[1]]

            real_wp_x.append(gate_x[1] + 0.05 * np.cos(gate_yaw[1] + np.pi / 2))
            real_wp_y.append(gate_y[1] + 0.05 * np.sin(gate_yaw[1] + np.pi / 2))

            real_wp_x.append(gate_x[2])
            real_wp_y.append(gate_y[2])

            real_wp_x.append(gate_x[2] + 0.3 * np.cos(gate_yaw[2] + np.pi / 2 + 0.4))
            real_wp_y.append(gate_y[2] + 0.3 * np.sin(gate_yaw[2] + np.pi / 2 + 0.4))

            real_wp_x.append(gate_x[2] - 0.3)
            real_wp_y.append(gate_y[2] + 0.3)

            real_wp_x.append(gate_x[3])
            real_wp_y.append(gate_y[3])

            real_wp_x.append(gate_x[3] + 0.5 * np.cos(gate_yaw[3] + np.pi / 2))
            real_wp_y.append(gate_y[3] + 0.5 * np.sin(gate_yaw[3] + np.pi / 2))
            real_wp_z = [
                gate_z[1],
                gate_z[1],
                gate_z[2],
                gate_z[2],
                (gate_z[2] + gate_z[3]) / 2,
                gate_z[3],
                gate_z[3],
            ]
        reference_csp = CSP_2D(real_wp_x, real_wp_y, real_wp_z)
        # Create obstacle information
        ob = self.create_obstacles(obs_x, obs_y, gate_x, gate_y, gate_yaw)

        # find out where I am w.r.t. reference spline
        s, d = reference_csp.cartesian_to_frenet(drone_x, drone_y)

        # Generate path
        fplist, best_idx, path = self.planner_core.frenet_optimal_planning(
            reference_csp, s, d, 0.0, 0.0, ob
        )

        # Debug plotting
        if self.DEBUG is True:
            self.DEBUG_draw_state(path, reference_csp, fplist, ob)

        return path, reference_csp, fplist

    def DEBUG_draw_state(
        self,
        fp_best: FrenetPath,
        ref_csp: CSP_2D,
        fplist: List[FrenetPath],
        obstacles: List[Tuple[float, float, float]],
    ) -> None:
        """Draws current 2D state on a plot."""
        if self.DEBUG is False:
            return

        # Clear previous drawing
        self.ax.clear()

        # Reference trajectory
        self.ax.plot(ref_csp.x_sampled, ref_csp.y_sampled, "-b")

        # All predicted trajectories
        for fp in fplist:
            self.ax.plot(fp.x, fp.y)

        # Obstacles
        for i in range(len(obstacles)):
            circle = patches.Circle(
                (obstacles[i][0], obstacles[i][1]),
                obstacles[i][2],
                edgecolor="green",
                facecolor="none",
                lw=2,
            )
            self.ax.add_patch(circle)

        # Best path
        self.ax.plot(fp_best.x, fp_best.y, "-oy")

        # Finally, draw new figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
