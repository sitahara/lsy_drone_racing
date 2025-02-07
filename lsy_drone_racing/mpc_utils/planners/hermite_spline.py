import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad

# # Define the start position and gate positions
# start_pos = np.array([0, 0, 0])
# gate_positions = [np.array([1, 0, 1]), np.array([2, 0, 2]), np.array([1, 0, 3])]
# orientations = [np.array([1, 0, 0]), np.array([1, 0, 0]), np.array([-1, 0, 0])]


# def approximate_arc_length(segment, n=1000):
#     """Approximate the arc length of a segment using a trapezoidal rule."""
#     length = 0
#     t_values = np.linspace(0, 1, n)
#     for i in range(n - 1):
#         p0 = segment(t_values[i])
#         p1 = segment(t_values[i + 1])
#         length += np.linalg.norm(p1 - p0)
#     return length


# # Define the Hermite spline function
# def hermite_spline(p0, p1, m0, m1, t):
#     h00 = 2 * t**3 - 3 * t**2 + 1
#     h10 = t**3 - 2 * t**2 + t
#     h01 = -2 * t**3 + 3 * t**2
#     h11 = t**3 - t**2
#     return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


# # Create the symbolic variable for theta
# theta = ca.SX.sym("theta")

# # Initialize the path
# path_segments = []
# arc_lengths = []
# path_points = []

# # Iterate through the gate positions to create the spline
# for i in range(len(gate_positions) - 1):
#     p0 = gate_positions[i]
#     p1 = gate_positions[i + 1]
#     m0 = orientations[i]
#     m1 = orientations[i + 1]

#     # Create the Hermite spline segment
#     segment_theta = ca.SX.sym(f"segment_theta_{i}")
#     segment = hermite_spline(p0, p1, m0, m1, segment_theta)
#     path_segments.append(segment)

#     # Calculate the arc length of the segment
#     segment_function = ca.Function(f"segment_{i}", [segment_theta], [segment])
#     arc_length = approximate_arc_length(segment_function)
#     print(arc_length)
#     arc_lengths.append(arc_length)

#     # Sample points along the segment
#     num_samples = 100
#     for j in range(num_samples + 1):
#         t = j / num_samples
#         point = ca.Function("point", [segment_theta], [segment])(t).full().flatten()
#         path_points.append(point)

# # Normalize theta to be between 0 and 1 over the entire path
# total_arc_length = sum(arc_lengths)
# normalized_theta = theta * total_arc_length
# normalized_theta = np.linspace(0, 1, len(path_points))

# # Create the interpolant
# print(path_points)
# path_interpolant_x = ca.interpolant(
#     "path", "linear", [normalized_theta.tolist()], path_points[:, 0]
# )

# # Create the symbolic function
# path_function = ca.Function("path", [theta], [path_interpolant_x(theta)])


# # Visualization
# theta_values = np.linspace(0, 1, 1000)
# points = np.array([path_function(t).full().flatten() for t in theta_values])
# # Create the symbolic function for the full path
# path = ca.SX.zeros(3)
# for i, segment in enumerate(path_segments):
#     segment_start = sum(arc_lengths[:i])
#     segment_end = segment_start + arc_lengths[i]
#     segment_theta = (normalized_theta - segment_start) / arc_lengths[i]
#     path += ca.if_else(
#         ca.logic_and(normalized_theta >= segment_start, normalized_theta < segment_end),
#         hermite_spline(
#             gate_positions[i],
#             gate_positions[i + 1],
#             orientations[i],
#             orientations[i + 1],
#             segment_theta,
#         ),
#         0,
#     )

# # Create the symbolic function
# path_function = ca.Function("path", [theta], [path])

# # Visualization
# theta_values = np.linspace(0, 1, 1000)
# points2 = np.array([path_function(t).full().flatten() for t in theta_values])
# print(points.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot(points[:, 0], points[:, 1], points[:, 2], label="Planned Path")
# ax.plot(points2[:, 0], points2[:, 1], points2[:, 2], label="Planned Path2")
# ax.scatter(start_pos[0], start_pos[1], start_pos[2], color="red", label="Start Position")
# # print(gate_positions.shape)
# # ax.scatter(
# #     gate_positions[:][1],
# #     gate_positions[:][1],
# #     gate_positions[:][1],
# #     color="blue",
# #     label="Gate Positions",
# # )
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()
# plt.show()


class HermiteSpline:
    def __init__(self, start_pos, start_rpy, gates_pos, gates_rpy, debug=False):
        self.waypoints = np.vstack((start_pos, gates_pos))  # , start_pos))
        self.orientations = np.vstack((start_rpy, gates_rpy))  # , start_rpy))
        self.tangents = self.compute_normals(self.orientations)
        self.debug = debug

        self.fitPolynomial()

        self.fitHermiteSpline()

    def updateGates(self, gates_pos, gates_rpy):
        self.waypoints = np.vstack((self.waypoints[0], gates_pos))
        self.orientations = np.vstack((self.orientations[0], gates_rpy))
        self.tangents = self.compute_normals(self.orientations)
        self.fitPolynomial()
        self.fitHermiteSpline()

    def getPathPointsForPlotting(self, theta_0=0, theta_end=1, num_points=1000):
        theta_values = np.linspace(theta_0, theta_end, num_points)
        points = np.array([self.path_function(t).full().flatten() for t in theta_values])
        dpoints = np.array([self.dpath_function(t).full().flatten() for t in theta_values])
        return points, dpoints

    def computeProgress(self, pos, vel, theta_last):
        """Compute the progress and dprogress along the path given the current position and velocity.

        args:
            pos: Current position
            vel: Current velocity
            theta_last: Last progress value

        returns:
            progress: Current progress along the path
            dprogress: Rate of progress along the path
        """

        def objective(theta):
            path_point = self.path_function(theta).full().flatten()
            return np.linalg.norm(path_point - pos)

        # Minimize the objective function to find the progress
        progress = minimize(
            objective, theta_last, bounds=[(theta_last - 0.1, theta_last + 0.1)], method="L-BFGS-B"
        )
        progress = progress.x[0]
        # Compute the derivative of the progress
        tangent = self.dpath_function(progress).full().flatten()
        dprogress = np.dot(tangent, vel) / np.linalg.norm(tangent)
        return progress, dprogress

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
            print("Arc lengths of each spline segment", arc_lengths)
            print("Progress values at which segments switch", self.theta_switch)

        self.poly_x = poly_x
        self.poly_y = poly_y
        self.poly_z = poly_z
        return None

    def fitHermiteSpline(self):
        # Step 1: Generate Hermite splines for each path segment
        waypoints = self.waypoints
        tangents = self.tangents

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

        # Create a CasADi function for the path and its derivative
        self.path_function = ca.Function("path", [theta], [path])
        dpath = ca.jacobian(self.path_function(theta), theta)
        self.dpath_function = ca.Function("dpath", [theta], [dpath])

        return None

    def compute_normals(self, orientations):
        tangents = np.zeros((len(orientations), 3))
        for i in range(len(orientations)):
            # Convert RPY to a direction vector (assuming RPY represents the normal direction)
            rot_mat = Rot.from_euler("xyz", orientations[i, :]).as_matrix()
            direction = rot_mat @ np.array([0, 1, 0])
            tangents[i, :] = direction  # / np.linalg.norm(direction)
        # print(tangents)
        return tangents


# import casadi as ca
# import numpy as np

# from scipy import optimize
# from typing import Dict, Any
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import splev, splprep
# from scipy.interpolate import CubicHermiteSpline


# class HermiteSpline:
#     def __init__(self, theta, start_pos, start_rpy, gates_pos, gates_rpy):
#         self.theta = theta
#         self.start_pos = start_pos
#         self.start_rpy = start_rpy
#         self.gates_pos = gates_pos
#         self.gates_rpy = gates_rpy
#         self.waypoints = np.vstack((self.start_pos, self.gates_pos))
#         self.orientations = np.vstack((self.start_rpy, self.gates_rpy))
#         self.tangents = self.compute_normals(self.orientations)
#         self.num_gates = gates_pos.shape[0]
#         # self.path_function, self.dpath_function = self.fitHermiteSpline()

#     def fitPolynomial(self):
#         waypoints = self.waypoints
#         tangents = self.tangents
#         degree = waypoints.shape[0] - 1
#         t = np.linspace(0, 1, waypoints.shape[0])

#         # Fit polynomial for each dimension
#         poly_coeffs_x = np.polyfit(t, waypoints[:, 0], degree)
#         poly_coeffs_y = np.polyfit(t, waypoints[:, 1], degree)
#         poly_coeffs_z = np.polyfit(t, waypoints[:, 2], degree)

#         # Create polynomial functions
#         poly_x = np.poly1d(poly_coeffs_x)
#         poly_y = np.poly1d(poly_coeffs_y)
#         poly_z = np.poly1d(poly_coeffs_z)

#         # Sample points corresponding to equally spaced arc lengths
#         # Corresponds to the number of spline segments
#         num_samples = 1000
#         fitted_coords = np.zeros((num_samples, 3))
#         fitted_tangents = np.zeros((num_samples, 3))
#         for i in range(num_samples):
#             t = i / num_samples
#             fitted_coords[i, :] = [poly_x(t), poly_y(t), poly_z(t)]
#         #    fitted_tangents[i, :] = [dpoly_x(t), dpoly_y(t), dpoly_z(t)]

#         return fitted_coords  # , tangents

#     def update_waypoints(self, gates_pos=None, gates_rpy=None, start_pos=None, start_rpy=None):
#         if gates_pos is not None:
#             self.gates_pos = gates_pos
#             self.gates_rpy = gates_rpy
#         if start_pos and start_rpy is not None:
#             self.start_pos = start_pos
#             self.start_rpy = start_rpy

#         self.waypoints = np.vstack((self.start_pos, self.gates_pos))
#         self.orientations = np.vstack((self.start_rpy, self.gates_rpy))
#         self.tangents = self.compute_normals(self.orientations)
#         self.path_function, self.dpath_function = self.fitHermiteSpline()

#     def fitHermiteSpline(self):
#         waypoints = self.waypoints
#         tangents = self.tangents

#         progress = ca.SX.sym("progress")

#         def hermite_spline(p0, p1, m0, m1, t):
#             h00 = 2 * t**3 - 3 * t**2 + 1
#             h10 = t**3 - 2 * t**2 + t
#             h01 = -2 * t**3 + 3 * t**2
#             h11 = t**3 - t**2
#             return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

#         splines = []
#         lengths = []
#         segments = len(waypoints) - 1
#         for i in range(segments):
#             p0 = waypoints[i]
#             p1 = waypoints[i + 1]
#             m0 = tangents[i]
#             m1 = tangents[i + 1]
#             spline = hermite_spline(p0, p1, m0, m1, progress)
#             splines.append(spline)

#             # Calculate the length of the spline
#             t_sym = ca.SX.sym("t")
#             spline_t = hermite_spline(p0, p1, m0, m1, t_sym)
#             ds = ca.norm_2(ca.jacobian(spline_t, t_sym))
#             arc_length_integrand = ca.Function("arc_length_integrand", [t_sym], [ds])

#             length = quad(lambda t: arc_length_integrand(t).full().flatten()[0], 0, 1)[0]
#             lengths.append(length)

#         total_length = sum(lengths)
#         intervals = np.cumsum([0] + lengths) / total_length

#         path = ca.SX.zeros(3)
#         for i in range(segments):
#             segment_progress = (progress - intervals[i]) / (intervals[i + 1] - intervals[i])
#             path += ca.if_else(
#                 ca.logic_and(progress >= intervals[i], progress < intervals[i + 1]), splines[i], 0
#             )

#         path += ca.if_else(progress == 1, splines[-1], 0)
#         path_function = ca.Function("path", [progress], [path])

#         dpath_value = ca.jacobian(path_function(progress), progress)
#         dpath_function = ca.Function("dpath_function", [progress], [dpath_value])

#         return path_function, dpath_function

#
