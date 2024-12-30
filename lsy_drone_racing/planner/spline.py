import math

import numpy as np
from scipy.interpolate import CubicSpline


class CSP_2D:
    """A class denoting the cubic-spline curve on a 2D surface.
    Since there's no implementation of such spline curve in scipy (only 1D interpolation is available),
    I decided to make it myself.
    """

    def __init__(self, x, y, z):
        """Takes coordinates of knot-points and creates spline objects.

        Parameters
        ----------
        x : list
            x coordinates for data points.
        y : list
            y coordinates for data points.
        z0 : float
            z coordinate at the beginning of the path, usually the height of a gate.
        z1 : list
            z coordinate at the end of the path.
        """
        self.knot_x = x
        self.knot_y = y
        self.knot_z = z
        self.num_knots = len(x)

        # parameter for 1D-spline interpolation of each axis
        ## a little bit expensive, but use L2 norm between points as the curve parameter
        dx = np.diff(self.knot_x)
        dy = np.diff(self.knot_y)
        self.ds = np.hypot(dx, dy)
        self.s = [0]
        self.s.extend(np.cumsum(self.ds))

        # Creation of spline curve objects
        self.spline_x = CubicSpline(self.s, self.knot_x)
        self.spline_y = CubicSpline(self.s, self.knot_y)
        self.spline_z = CubicSpline(self.s, self.knot_z)
        self.s_for_sample = np.linspace(0.0, self.s[-1], 100)
        self.x_sampled = self.spline_x(self.s_for_sample)
        self.y_sampled = self.spline_y(self.s_for_sample)
        self.z_sampled = self.spline_z(self.s_for_sample)

    def calc_position(self, s):
        """
        Given an arc parameter `s`, returns the position at cooresponding location
        """
        return self.spline_x(s), self.spline_y(s), self.spline_z(s)

    def calc_curvature(self, s):
        """calc curvature.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        dx = self.spline_x(s, nu=1)
        ddx = self.spline_x(s, nu=2)
        dy = self.spline_y(s, nu=1)
        ddy = self.spline_y(s, nu=2)
        k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        dx = self.spline_x(s, nu=1)
        dy = self.spline_y(s, nu=1)
        yaw = math.atan2(dy, dx)
        return yaw

    def cartesian_to_frenet(self, x, y):
        """
        Converts cartesian coordinates to those wrt frenet frame on this curve.
        Useful for determining the initial condition of planning.
        Sign of D is calcuted by taking the inner product of error vector and unit vector in D axis.
        Unit vector of D axis is calculated using yaw angle with calc_yaw(s_closest)

        Parameters
        ----------
        x : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        y : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        s : float
            S coordinate of the point on the frenet frame.
        d : float
            D coordinate (deviation) of the point on the frenet frame.
        """
        # To reduce the computational load, use self.x_sampled and self.y_sampled to search for the closest point
        diff_x_squared = (x - self.x_sampled) ** 2
        diff_y_squared = (y - self.y_sampled) ** 2
        # idx_closest = -1
        # val_best = float("inf")
        # for i in range(len(diff_y_squared)):
        #     if val_best > diff_x_squared[i] + diff_y_squared[i]:
        #         val_best = diff_x_squared[i] + diff_y_squared[i]
        #         idx_closest = i
        idx_closest = np.argmin(diff_x_squared + diff_y_squared)
        s_closest = self.s_for_sample[idx_closest]
        idx_closest = idx_closest

        # Convert point into frenet coordinates

        dx = x - self.x_sampled[idx_closest]
        dy = y - self.y_sampled[idx_closest]
        dl = np.sqrt(dx**2 + dy**2)

        yaw_rad = self.calc_yaw(s_closest) + (np.pi / 2)
        D_dx = np.cos(yaw_rad)
        D_dy = np.sin(yaw_rad)
        return s_closest, dl * np.sign(D_dx * dx + D_dy * dy)


class QuarticSpline_2D:
    """
    A class denoting the Quartic(4)-spline curve on a 2D surface.
    Uses Initial
    - Position: d0
    - Velocity (1st order Derivative of value over parameter): d_d0
    - Acceleration (2nd order Derivative of value over parameter): dd_d0
    and Terminal
    - Velocity (1st order Derivative of value over parameter): d_dT
    - Acceleration (2nd order Derivative of value over parameter): dd_dT
    to calculate the coeffecients.
    """

    def __init__(self, T, d0, d_d0, dd_d0, d_dT, dd_dT):
        """
        Calculates coefficients of the Quartic spline in the frenet frame.
        Note that all parameters are denoted in the frenet frame

        Parameters
        ----------
        T: float
            Time horizon of the trajectory. Basically changes the length of the curve generated in this function.
        d0 : float
            Initial position (lateral deviation) of the Quartic curve.
        d_d0 : float
            Initial derivative of the Quartic curve.
        dd_d0 : float
            Initial 2nd-derivative of the Quartic curve.
        d_dT : float
            Terminal derivative of the Quartic curve. Default value
        dd_dT : float
            Terminal derivative of the Quartic curve.
        """
        self.a0 = d0
        self.a1 = d_d0
        self.a2 = dd_d0 / 2.0

        A = np.array([[3 * T**2, 4 * T**3], [6 * T, 12 * T**2]])
        b = np.array([d_dT - self.a1 - 2 * self.a2 * T, dd_dT - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class QuinticSpline_2D:
    """
    A class denoting the Quartic(4)-spline curve on a 2D surface.
    Uses Initial
    - Position: d0
    - Velocity (1st order Derivative of value over parameter): d_d0
    - Acceleration (2nd order Derivative of value over parameter): dd_d0
    and Terminal
    - Position: dT
    - Velocity (1st order Derivative of value over parameter): d_dT
    - Acceleration (2nd order Derivative of value over parameter): dd_dT
    to calculate the coeffecients.
    """

    def __init__(self, T, d0, d_d0, dd_d0, dT, d_dT=0.0, dd_dT=0.0):
        self.a0 = d0
        self.a1 = d_d0
        self.a2 = dd_d0 / 2.0

        A = np.array(
            [[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]]
        )
        b = np.array(
            [
                dT - self.a0 - self.a1 * T - self.a2 * T**2,
                d_dT - self.a1 - 2 * self.a2 * T,
                dd_dT - 2 * self.a2,
            ]
        )
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * t**2
            + self.a3 * t**3
            + self.a4 * t**4
            + self.a5 * t**5
        )

        return xt

    def calc_first_derivative(self, t):
        xt = (
            self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4
        )

        return xt


class Line_2D:
    """
    I'm sick of low-performing Quartic/Quintic splines, I just decided to make a line as
    Trajectory inside frenet frame. No one can stop me hahahahahha
    """

    def __init__(self, T, d0, dT):
        """
        Calculates coefficients of the Quartic spline in the frenet frame.
        Note that all parameters are denoted in the frenet frame

        Parameters
        ----------
        T: float
            Time horizon of the trajectory. Basically changes the length of the curve generated in this function.
        d0 : float
            Initial position (lateral deviation) of the Quartic curve.
        d_dT : float
            Terminal posiion of the curve.
        """
        self.a0 = d0
        self.a1 = (dT - d0) / T

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t

        return xt

    def calc_first_derivative(self, t):
        return self.a1

    def calc_second_derivative(self, t):
        return 0

    def calc_third_derivative(self, t):
        return 0


class FrenetPath:
    """
    An object that stores path on frenet frame.
    """

    def __init__(self):
        self.t = []  # Sampling parameter (corresponds to time)
        self.d = []  # Lateral position
        self.d_d = []  # Change rate of lateral position
        self.d_dd = []  # 2nd order derivative of lateral position
        self.d_ddd = []  # 3rd order derivative of lateral position ("Jerk")
        self.s = []  # Longitudinal position
        self.cost = 0.0  # Cost of this path

        self.x = []  # X Coordinates of the path converted to cartesian coordinates
        self.y = []  # Y Coordinates of the path converted to cartesian coordinates
        self.z = []  # Z Coordinates of the path converted to cartesian coordinates
        self.yaw = []  # Yaw angle of each segment of the sampled path on cartesian coordinates
        self.ds = []  # Distance between each sampled points on the path on cartesian coordinates
        self.c = []  # Curvature at the sampled points of the path, on cartesian coordinates
