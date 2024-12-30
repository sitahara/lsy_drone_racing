import math

import numpy as np
from .spline import FrenetPath, Line_2D


class PlannerCore:
    def __init__(
        self,
        MAX_CURVATURE=50.0,
        MAX_ROAD_WIDTH=0.5,
        D_ROAD_W=0.05,
        DT=0.05,
        T_PRED=1.0,
        K_J=0.5,
        K_D=5.0,
    ):
        # Planning parameters
        self.MAX_CURVATURE = MAX_CURVATURE  # maximum curvature [1/m]
        self.MAX_ROAD_WIDTH = MAX_ROAD_WIDTH  # maximum road width [m]
        self.D_ROAD_W = D_ROAD_W  # road width sampling length [m]
        self.DT = DT  # time tick [s]
        self.T_PRED = T_PRED  # prediction time [m]

        # cost weights
        self.K_J = K_J
        self.K_D = K_D

    def calc_frenet_paths(self, s0, d0, d_d0, dd_d0):
        """
        Creates a list of possible paths on frenet coordinate system, by sampling opening and terminal constraints
        within allowed range.
        Parameters
        ----------
        s0 : float
            Arc parth parameter of the closest point on the target trajectory.
        ##### Longitudinal planning parameters
        vel : float
            Current speed of the robot.
        acc : float
            Current acceleration of the robot.
        ##### Lateral planning parameters
        d0 : float
            Current lateral deviation from the target trajectory.
        d_d0 : float
            Time derivative of current lateral deviation from the target trajectory.
        dd_d0 : float
            2nd order time derivative of current lateral deviation from the target trajectory.
        """
        frenet_paths = []

        # generate path to each offset goal
        for di in np.arange(
            -self.MAX_ROAD_WIDTH, self.MAX_ROAD_WIDTH + self.D_ROAD_W / 10, self.D_ROAD_W
        ):
            # Lateral motion planning
            fp = FrenetPath()

            lat_qp = Line_2D(self.T_PRED, d0, di)

            fp.t = np.array([t for t in np.arange(0.0, self.T_PRED, self.DT)])
            fp.d = np.array([lat_qp.calc_point(t) for t in fp.t])
            fp.d_ddd = np.array([lat_qp.calc_third_derivative(t) for t in fp.t])

            # Longitudinal motion planning (constant velocity)

            fp.s = fp.t + s0

            Jp = sum(np.power(fp.d_ddd, 2))  # square of jerk

            fp.cost = self.K_J * Jp + self.K_D * fp.d[-1] ** 2

            frenet_paths.append(fp)

        return frenet_paths

    def calc_global_paths(self, fplist, csp):
        """
        Converts path in frenet frame into actual path in cartesian frame.
        """
        for fp in fplist:
            # calc global positions
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

            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(np.hypot(dx, dy))

            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

        return fplist

    def check_collision(self, fp, ob):
        for i in range(len(ob)):
            d = [
                ((ix - ob[i][0]) ** 2 + (iy - ob[i][1]) ** 2) - ob[i][2] ** 2
                for (ix, iy) in zip(fp.x, fp.y)
            ]

            collision = any([di <= 0 for di in d])

            if collision:
                return False

        return True

    def check_paths(self, fplist, ob):
        ok_ind = []
        for i, _ in enumerate(fplist):
            if any([abs(c) > self.MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
                # continue
                fplist[i].cost += 3000
            if not self.check_collision(fplist[i], ob):
                # continue
                fplist[i].cost += 100000

            ok_ind.append(i)

        return [fplist[i] for i in ok_ind]

    def frenet_optimal_planning(self, csp, s0, d0, d_d0, dd_d0, ob):
        """
        Takes state of things and returns the sampled points of the best trajectory
        Parameters
        ----------
        csp : CSP2D object
            Cubic spline object of the target trajectory.
        s0 : float
            Arc parth parameter of the closest point on the target trajectory.
        ##### Longitudinal planning parameters
        vel : float
            Current speed of the robot.
        acc : float
            Current acceleration of the robot.
        ##### Lateral planning parameters
        d0 : float
            Current lateral deviation from the target trajectory.
        d_d0 : float
            Time derivative of current lateral deviation from the target trajectory.
        dd_d0 : float
            2nd order time derivative of current lateral deviation from the target trajectory.
        ob : list of objects
        """
        fplist = self.calc_frenet_paths(s0, d0, d_d0, dd_d0)
        fplist = self.calc_global_paths(fplist, csp)
        fplist = self.check_paths(fplist, ob)

        # find minimum cost path
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
