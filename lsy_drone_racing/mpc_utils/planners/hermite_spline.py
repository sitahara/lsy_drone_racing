"""Hermite Spline class for generating symbolical smooth paths through waypoints.

Inputs:
    - start_pos: Initial position of the path
    - start_rpy: Initial orientation of the path (roll, pitch, yaw)
    - gates_pos: Positions of the gates to pass through
    - gates_rpy: Orientations of the gates to pass through (roll, pitch, yaw)
    - tangent_scaling: Scaling factor for the tangent vectors (default=3.5). This controls the curvature of the path.
    - debug: Flag to print debug information (default=False)

Methods:
    - updateGates: Update the gates to pass through
    - fitPolynomial: Fit a polynomial to the waypoints
    - fitHermiteSpline: Fit a symbolical Hermite spline to the waypoints. Returns the path and its derivative functions.
    - computeNormals: Compute the tangent vectors at the waypoints from the orientations
    - computeProgress: Compute the progress along the path given the current position and velocity
    - getPathPointsForPlotting: Evaluate the path at equally spaced progress values between theta_0 and theta_end
    - getPolyPath: Evaluate the polynomial path at equally spaced progress values between theta_0 and theta_end
"""

from __future__ import annotations
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
from scipy.integrate import quad


class HermiteSpline:
    """A class to construct a symbolical Hermite Spline and its derivative from the start position through a number of gates for drone path planning.

    Attributes:
    -----------
    waypoints : np.ndarray
        Array of waypoints including start position and gate positions.
    orientations : np.ndarray
        Array of orientations including start orientation and gate orientations.
    tangents : np.ndarray
        Array of tangents computed from orientations.
    debug : bool
        Flag to enable/disable debug mode.
    tangent_scaling : float
        Scaling factor for tangents to control the curvature.
    theta_switch : np.ndarray
        Array of progress values at which segments switch.
    poly_x : np.poly1d
        Polynomial function for x-coordinates.
    poly_y : np.poly1d
        Polynomial function for y-coordinates.
    poly_z : np.poly1d
        Polynomial function for z-coordinates.
    path_function : ca.Function
        CasADi function representing the Hermite spline path.
    dpath_function : ca.Function
        CasADi function representing the derivative of the Hermite spline path.

    Methods:
    --------
    __init__(start_pos, start_rpy, gates_pos, gates_rpy, tangent_scaling=3.5, debug=False):
        Initializes the HermiteSpline with start position, start orientation, gate positions, and gate orientations.

    updateGates(gates_pos, gates_rpy):
        Updates the gate positions and orientations.

    getPathPointsForPlotting(theta_0=0, theta_end=1, num_points=50):
        Returns points and their derivatives for plotting the path.

    getPolyPath(theta_0=0, theta_end=1, num_points=50):
        Returns points along the polynomial path.

    computeProgress(pos, vel, theta_last):
        Computes the progress and rate of progress along the path given the current position and velocity.

    fitPolynomial(t_total=1):
        Fits a polynomial to the waypoints and computes the arc length.

    fitHermiteSpline():
        Fits a Hermite spline to the waypoints and tangents.

    compute_normals(orientations):
        Computes the tangents from the orientations.
    """

    def __init__(
        self,
        start_pos,
        start_rpy,
        gates_pos,
        gates_rpy,
        tangent_scaling=3.5,
        debug=False,
        parametric=False,
        end_at_start=True,
        reverse_start_orientation=True,
    ):
        self.reverse_start_orientation = reverse_start_orientation
        self.end_at_start = end_at_start

        if end_at_start:
            self.waypoints = np.vstack((start_pos, gates_pos, start_pos))
            self.orientations = np.vstack((start_rpy, gates_rpy, start_rpy))
        else:
            self.waypoints = np.vstack((start_pos, gates_pos))
            self.orientations = np.vstack((start_rpy, gates_rpy))
        self.tangents = self.compute_normals(self.orientations)

        self.debug = debug
        self.tangent_scaling = tangent_scaling
        self.parametric = parametric

        # self.fitPolynomial()
        self.calculate_arc_length()

        if self.parametric:
            self.createParametricPath()
        else:
            self.fitHermiteSpline()

    def updateGates(self, gates_pos, gates_rpy):
        if self.end_at_start:
            self.waypoints = np.vstack((self.waypoints[0], gates_pos, self.waypoints[0]))
            self.orientations = np.vstack((self.orientations[0], gates_rpy, self.orientations[0]))
        else:
            self.waypoints = np.vstack((self.waypoints[0], gates_pos))
            self.orientations = np.vstack((self.orientations[0], gates_rpy))
        if self.parametric:
            self.path_params_values = np.concatenate(
                (self.waypoints.flatten(), self.tangents.flatten() * self.tangent_scaling)
            )
        else:
            # self.fitPolynomial()
            self.calculate_arc_length()
            self.fitHermiteSpline()

    def getPathPointsForPlotting(self, theta_0=0, theta_end=1, num_points=50, only_path=False):
        theta_values = np.linspace(theta_0, theta_end, num_points)
        if self.parametric:
            points = np.array(
                [
                    self.path_function(theta=theta, path_params=self.path_params_values)["path"]
                    .full()
                    .flatten()
                    for theta in theta_values
                ]
            )
            if self.debug:
                if not hasattr(self, "debug_path_param_values"):
                    self.debug_path_param_values = self.path_params_values
                else:
                    if not np.allclose(self.path_params_values, self.debug_path_param_values):
                        print("Path was updated")
                        self.debug_path_param_values = self.path_params_values

            if only_path:
                dpoints = None
            else:
                dpoints = np.array(
                    [
                        self.dpath_function(theta=theta, path_params=self.path_params_values)[
                            "dpath"
                        ]
                        .full()
                        .flatten()
                        for theta in theta_values
                    ]
                )
        else:
            points = np.array(
                [self.path_function(theta=theta)["path"].full().flatten() for theta in theta_values]
            )
            if only_path:
                dpoints = None
            else:
                dpoints = np.array(
                    [
                        self.dpath_function(theta=theta)["dpath"].full().flatten()
                        for theta in theta_values
                    ]
                )
        return points, dpoints

    def getPolyPath(self, theta_0=0, theta_end=1, num_points=50):
        theta_values = np.linspace(theta_0, theta_end, num_points)
        points = np.array([[self.poly_x(t), self.poly_y(t), self.poly_z(t)] for t in theta_values])
        return points

    def computeProgress(self, pos, vel, theta_last):
        """Compute the progress and dprogress along the path given the current position and velocity. Searchs around theta_last.

        args:
            pos: Current position
            vel: Current velocity
            theta_last: Last progress value

        returns:
            theta: Current progress along the path
            dtheta: Rate of progress along the path
        """
        if self.parametric:

            def objective(theta):
                path_point = (
                    self.path_function(theta=theta, path_params=self.path_params_values)["path"]
                    .full()
                    .flatten()
                )
                return np.linalg.norm(path_point - pos)
        else:

            def objective(theta):
                path_point = self.path_function(theta=theta)["path"].full().flatten()
                return np.linalg.norm(path_point - pos)

        # Minimize the objective function to find the progress
        theta = minimize(
            objective, theta_last, bounds=[(theta_last - 0.1, theta_last + 0.1)], method="L-BFGS-B"
        )
        theta = theta.x[0]
        # Compute the derivative of the progress
        if self.parametric:
            tangent = (
                self.dpath_function(theta=theta, path_params=self.path_params_values)["path"]
                .full()
                .flatten()
            )
        else:
            tangent = self.dpath_function(theta=theta)["path"].full().flatten()
        dtheta = np.dot(tangent, vel) / np.linalg.norm(tangent)
        return theta, dtheta

    def fitPolynomial(self, t_total=1):
        waypoints = self.waypoints
        degree = waypoints.shape[0] - 1
        t_waypoints = np.linspace(0, t_total, waypoints.shape[0])

        poly_coeffs_x = np.polyfit(t_waypoints, waypoints[:, 0], degree)
        poly_coeffs_y = np.polyfit(t_waypoints, waypoints[:, 1], degree)
        poly_coeffs_z = np.polyfit(t_waypoints, waypoints[:, 2], degree)

        # Create polynomial functions
        poly_x = np.poly1d(poly_coeffs_x)
        poly_y = np.poly1d(poly_coeffs_y)
        poly_z = np.poly1d(poly_coeffs_z)

        # Derivatives of the polynomial functions
        dpoly_x = np.polyder(poly_x)
        dpoly_y = np.polyder(poly_y)
        dpoly_z = np.polyder(poly_z)

        # Arc length integrand
        def arc_length_integrand(t):
            dx_dt = dpoly_x(t)
            dy_dt = dpoly_y(t)
            dz_dt = dpoly_z(t)
            return np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

        # Total arc length
        total_arc_length, _ = quad(arc_length_integrand, 0, 1)

        # Calculate arc length up to each gate (includes start and end points)
        arc_lengths = [quad(arc_length_integrand, 0, t)[0] for t in t_waypoints]

        total_arc_length = arc_lengths[-1]
        self.theta_switch = np.array(arc_lengths) / total_arc_length
        if self.debug:
            print("Polynomial:Arc lengths of each spline segment", arc_lengths)
            print("Polynomial:Progress values at which segments switch", self.theta_switch)

        self.poly_x = poly_x
        self.poly_y = poly_y
        self.poly_z = poly_z
        return None

    def calculate_arc_length(self):
        waypoints = self.waypoints
        tangents = (
            self.tangents * self.tangent_scaling
        )  # Scale the tangents to control the curvature

        def hermite_spline(p0, p1, m0, m1, t):
            """Create a Hermite spline between two points with specified tangents.

            p0, p1: Endpoints of the spline
            m0, m1: Tangents at the endpoints
            t: Parameter (0 <= t <= 1).
            """
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

        # Create Hermite splines for each segment
        splines = []
        segments = len(waypoints) - 1
        for i in range(segments):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            m0 = tangents[i]
            m1 = tangents[i + 1]

            def spline(t, p0=p0, p1=p1, m0=m0, m1=m1):
                return hermite_spline(p0, p1, m0, m1, t)

            splines.append(spline)

        # Calculate the arc length of each spline segment
        arc_lengths = []
        for spline in splines:
            num_points = 500
            t_values = np.linspace(0, 1, num_points)
            arc_length = 0
            for i in range(num_points - 1):
                point1 = spline(t_values[i])
                point2 = spline(t_values[i + 1])
                arc_length += np.linalg.norm(point2 - point1)
            arc_lengths.append(arc_length)

        # Calculate the progress values at which segments switch
        arc_lengths = np.array(arc_lengths)
        arc_lengths[0] *= 1.5  # Increase the arc length of the first segment (Start is slower)
        total_arc_length = np.sum(arc_lengths)
        self.theta_switch = np.cumsum(arc_lengths) / total_arc_length
        self.theta_switch = np.insert(self.theta_switch, 0, 0)

        if self.debug:
            print("Spline:Arc lengths of each spline segment:", arc_lengths)
            print("Spline:Progress values at which segments switch:", self.theta_switch)
        return None

    def fitHermiteSpline(self):
        waypoints = self.waypoints
        tangents = (
            self.tangents * self.tangent_scaling
        )  # Scale the tangents to control the curvature

        # Define the progress parameter
        theta = ca.MX.sym("theta")

        def hermite_spline(p0, p1, m0, m1, t):
            """Create a Hermite spline between two points with specified tangents.

            p0, p1: Endpoints of the spline
            m0, m1: Tangents at the endpoints
            t: Parameter (0 <= t <= 1).
            """
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

        # Create Hermite splines for each segment
        splines = []
        segments = len(waypoints) - 1
        for i in range(segments):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            m0 = tangents[i]
            m1 = tangents[i + 1]
            t = (theta - self.theta_switch[i]) / (self.theta_switch[i + 1] - self.theta_switch[i])
            spline = hermite_spline(p0, p1, m0, m1, t)
            splines.append(spline)

        path = ca.MX.zeros(3)
        for i in range(segments):
            # segment_progress = (theta - self.theta_switch[i]) / (
            #     self.theta_switch[i + 1] - self.theta_switch[i]
            # )
            path += ca.if_else(
                ca.logic_and(theta >= self.theta_switch[i], theta < self.theta_switch[i + 1]),
                splines[i],
                0,
            )
        path += ca.if_else(theta == 1, splines[-1], 0)

        # Create a CasADi function for the path and its derivative
        self.path_function = ca.Function("path", [theta], [path], ["theta"], ["path"])
        dpath = ca.jacobian(self.path_function(theta), theta)
        self.dpath_function = ca.Function("dpath", [theta], [dpath], ["theta"], ["dpath"])

        return None

    def createParametricPath(self):
        """Creates a Hermite spline path that is parametric in theta and the combined waypoints and tangents."""
        # Define the combined parameter vector as a symbolic variable

        path_params = ca.MX.sym("path_params", self.waypoints.size + self.tangents.size)
        self.path_params_values = np.concatenate(
            (self.waypoints.flatten(), self.tangents.flatten() * self.tangent_scaling)
        )
        self.path_params = path_params
        theta = ca.MX.sym("theta")

        # Extract waypoints and tangents from the parameter vector
        waypoints = self.path_params[
            : self.waypoints.size
        ]  # ca.reshape(self.path_params[: self.waypoints.size], self.waypoints.shape)
        tangents = self.path_params[
            self.waypoints.size :
        ]  # ca.reshape(self.path_params[self.waypoints.size :], self.tangents.shape)

        def hermite_spline(p0, p1, m0, m1, t):
            """Create a Hermite spline between two points with specified tangents.

            p0, p1: Endpoints of the spline
            m0, m1: Tangents at the endpoints
            t: Parameter (0 <= t <= 1).
            """
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

        splines = []
        segments = self.waypoints.shape[0] - 1
        k1 = self.waypoints.shape[1]
        k2 = self.tangents.shape[1]
        for i in range(segments):
            p0 = waypoints[i * k1 : (i + 1) * k1]
            p1 = waypoints[(i + 1) * k1 : (i + 2) * k1]
            m0 = tangents[i * k2 : (i + 1) * k2]
            m1 = tangents[(i + 1) * k2 : (i + 2) * k2]
            t = (theta - self.theta_switch[i]) / (self.theta_switch[i + 1] - self.theta_switch[i])
            spline = hermite_spline(p0, p1, m0, m1, t)
            splines.append(spline)

        path = ca.MX.zeros(3)
        for i in range(segments):
            path += ca.if_else(
                ca.logic_and(theta >= self.theta_switch[i], theta < self.theta_switch[i + 1]),
                splines[i],
                0,
            )
        path += ca.if_else(theta == 1, splines[-1], 0)

        # Create a CasADi function for the path and its derivative
        self.path_function = ca.Function(
            "path", [theta, path_params], [path], ["theta", "path_params"], ["path"]
        )
        dpath = ca.jacobian(self.path_function(theta, path_params), theta)
        self.dpath_function = ca.Function(
            "dpath", [theta, path_params], [dpath], ["theta", "path_params"], ["dpath"]
        )
        return None

    def compute_normals(self, orientations):
        tangents = np.zeros((len(orientations), 3))
        for i in range(len(orientations)):
            # Convert RPY to a direction vector (assuming RPY represents the normal direction)
            rot_mat = Rot.from_euler("xyz", orientations[i, :]).as_matrix()
            direction = rot_mat @ np.array([0, 1, 0])
            tangents[i, :] = direction / np.linalg.norm(direction)

        if self.reverse_start_orientation:
            tangents[0, :] = -tangents[0, :]
        return tangents
