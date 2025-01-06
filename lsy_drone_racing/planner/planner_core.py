"""Core functionalities of the planner."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .spline import CSP_2D, FrenetPath, Line_2D, QuinticSpline_2D


class PlannerCore:
    """Class with all the functionalities of the planner."""

    def __init__(
        self,
        MAX_CURVATURE: float = 50.0,
        MAX_ROAD_WIDTH: float = 0.5,
        D_ROAD_W: float = 0.05,
        DT: float = 0.05,
        T_PRED: float = 1.0,
        K_J: float = 0.5,
        K_D: float = 5.0,
    ):
        """Initialize planning parameters.

        Args:
            MAX_CURVATURE:
                Maximum allowed curvature of the trajectory on the Cartesian frame.
                Trajectories with higher curvature (tighter curves) incur a cost penalty.
            MAX_ROAD_WIDTH:
                Maximum value of lateral offset when creating trajectory candidates in the Frenet frame.
            D_ROAD_W:
                Defines the width for sampling the lateral offset in the Frenet frame.
            DT:
                Sampling interval of the planned trajectory.
            T_PRED:
                Duration of the planned trajectory.
            K_J:
                Weight constant for the trajectory's rate of change of acceleration (jerk).
            K_D:
                Weight constant for the terminal deviation from the desired trajectory.
        """
        # Planning parameters
        self.MAX_CURVATURE = MAX_CURVATURE  # Maximum curvature in reciprocal meters (1/m)
        self.MAX_ROAD_WIDTH = MAX_ROAD_WIDTH  # Maximum road width in meters
        self.D_ROAD_W = D_ROAD_W  # Road width sampling length in meters
        self.DT = DT  # Time tick in seconds
        self.T_PRED = T_PRED  # Prediction time in seconds

        # Cost weights
        self.K_J = K_J
        self.K_D = K_D

    def calc_frenet_paths(
        self, s0: float, d0: float, d_d0: float, dd_d0: float
    ) -> List[FrenetPath]:
        """Create path candidates in the Frenet coordinate system.

        Generates a list of possible paths in the Frenet coordinate system by sampling opening and terminal constraints
        within the allowed range.

        Args:
            s0:
                Arc path parameter of the closest point on the target trajectory.
            ##### Longitudinal planning parameters
            vel:
                Current speed of the robot.
            acc:
                Current acceleration of the robot.
            ##### Lateral planning parameters
            d0:
                Current lateral deviation from the target trajectory.
            d_d0:
                Time derivative of the current lateral deviation from the target trajectory.
            dd_d0:
                Second-order time derivative of the current lateral deviation from the target trajectory.

        Returns:
            frenet_paths:
                List of path candidates. At this point, only fields related to
                paths in the Frenet coordinate system are populated.
        """
        frenet_paths = []

        # Generate paths to each offset goal
        for di in np.arange(
            -self.MAX_ROAD_WIDTH, self.MAX_ROAD_WIDTH + self.D_ROAD_W / 10, self.D_ROAD_W
        ):
            # Lateral motion planning
            fp = FrenetPath()

            # lat_qp = QuinticSpline_2D(self.T_PRED, d0, 0.0, 0.0, di)
            lat_qp = Line_2D(self.T_PRED, d0, di)

            fp.t = np.array([t for t in np.arange(0.0, self.T_PRED, self.DT)])
            fp.d = np.array([lat_qp.calc_point(t) for t in fp.t])
            fp.d_ddd = np.array([lat_qp.calc_third_derivative(t) for t in fp.t])

            # Longitudinal motion planning (constant velocity)

            fp.s = fp.t + s0

            Jp = sum(np.power(fp.d_ddd, 2))  # Square of jerk

            fp.cost = self.K_J * Jp + self.K_D * fp.d[-1] ** 2

            frenet_paths.append(fp)

        return frenet_paths

    def calc_global_paths(self, fplist: List[FrenetPath], csp: CSP_2D) -> List[FrenetPath]:
        """Convert paths in the Frenet frame to the Cartesian frame.

        Args:
            fplist:
                Path candidates generated with the `calc_frenet_paths` function.
            csp:
                Reference trajectory used as the basis of the Frenet frame.

        Returns:
            fplist:
                List of path candidates with all fields populated.
        """
        for fp in fplist:
            # Calculate global positions
            for i in range(len(fp.s)):
                ix, iy, iz = csp.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * np.cos(i_yaw + np.pi / 2.0)
                fy = iy + di * np.sin(i_yaw + np.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)
                fp.z.append(iz)

            # Calculate yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(np.hypot(dx, dy))

            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # Calculate curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

        return fplist

    def check_collision(self, fp: FrenetPath, ob: List[Tuple[float, float, float]]) -> bool:
        """Check if the path collides with obstacles.

        Uses sampled points in a FrenetPath object to determine
        if the trajectory collides with any obstacles.

        Args:
            fp:
                FrenetPath object converted to the Cartesian frame using the `calc_global_paths` function.
            ob:
                A list containing tuples representing the obstacles.
                The tuples have the following structure:
                (X coordinate of the center, Y coordinate of the center, diameter of the obstacle).
                Obstacles are modeled as cylinders that are normal to the XY surface, extending infinitely
                on both sides of the XY plane.

        Returns:
            `True` if the sampled points of the path are free from collisions,
            `False` if there is any collision.
        """
        col_num = 0
        for i in range(len(ob)):
            d = np.array(
                [
                    ((ix - ob[i][0]) ** 2 + (iy - ob[i][1]) ** 2) - ob[i][2] ** 2
                    for (ix, iy) in zip(fp.x, fp.y)
                ]
            )

            collision = np.sum(d <= 0)
            col_num += collision
        return col_num

    def check_paths(
        self, fplist: List[FrenetPath], ob: List[Tuple[float, float, float]]
    ) -> List[FrenetPath]:
        """Penalize collisions and excessively tight trajectories.

        Checks for any unrealistic features in candidate trajectories.
        In the original implementation, 'bad' trajectories were eliminated from the selection. However,
        to ensure the existence of trajectory output, we instead heavily penalize violations,
        so that even in cases where the planner cannot find a 'legal' trajectory, the least bad trajectory
        would be selected as the output.

        Args:
            fplist:
                List of FrenetPath objects converted to the Cartesian frame using the `calc_global_paths` function,
                and penalized using the `check_collision` function.
            ob:
                A list containing tuples representing the obstacles. For the structure of the
                tuples, refer to the description of the `check_collision` function above.

        Returns:
            fplist:
                List of FrenetPath objects, with additional costs for
                collisions and maximum curvature violations.
        """
        for i, _ in enumerate(fplist):
            if any([abs(c) > self.MAX_CURVATURE for c in fplist[i].c]):  # Maximum curvature check
                fplist[i].cost += 3000
            fplist[i].cost += 100000 * self.check_collision(fplist[i], ob)

        return fplist

    def frenet_optimal_planning(
        self,
        csp: CSP_2D,
        s0: float,
        d0: float,
        d_d0: float,
        dd_d0: float,
        ob: List[Tuple[float, float, float]],
    ) -> Tuple[List[FrenetPath], int, FrenetPath]:
        """Determine the best trajectory based on the drone's state.

        Args:
            csp:
                Cubic spline object of the target trajectory.
            s0:
                Arc path parameter of the closest point on the target trajectory.
            d0:
                Current lateral deviation from the target trajectory.
            d_d0:
                Time derivative of the current lateral deviation from the target trajectory.
            dd_d0:
                Second-order time derivative of the current lateral deviation from the target trajectory.
            ob:
                A list containing tuples representing the obstacles. For the structure of the
                tuples, refer to the description of the `check_collision` function above.

        Returns:
            Tuple(fplist, best_idx, best_path):
                A tuple containing:
                - A list of all FrenetPath candidates.
                - The index of the best path in the list.
                - The FrenetPath with the least cost, i.e., the best path.
        """
        fplist = self.calc_frenet_paths(s0, d0, d_d0, dd_d0)
        fplist = self.calc_global_paths(fplist, csp)
        fplist = self.check_paths(fplist, ob)

        # Find the minimum cost path
        min_cost = float("inf")
        best_path = None
        best_idx = -1
        for idx, fp in enumerate(fplist):
            if min_cost >= fp.cost:
                best_idx = idx
                min_cost = fp.cost
                best_path = fp
        print(min_cost)
        return fplist, best_idx, best_path
