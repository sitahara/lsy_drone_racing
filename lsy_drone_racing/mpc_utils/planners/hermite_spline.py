import casadi as ca
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import KroghInterpolator
from scipy.spatial.transform import Rotation as Rot
from scipy.integrate import quad
from scipy import optimize
from typing import Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HermiteSpline:
    def __init__(self, theta, start_pos, start_rpy, gates_pos, gates_rpy):
        self.theta = theta
        self.start_pos = start_pos
        self.start_rpy = start_rpy
        self.gates_pos = gates_pos
        self.gates_rpy = gates_rpy
        self.num_gates = gates_pos.shape[0]
        self.path_function, self.dpath_function = self.fitHermiteSpline()

    def fitPolynomial(self):
        waypoints = np.vstack((self.start_pos, self.gates_pos, self.start_pos))
        tangents = self.compute_normals(waypoints)
        degree = waypoints.shape[0] - 1
        t = np.linspace(0, 1, waypoints.shape[0])

        # # Fit polynomial for each dimension
        # poly_x = CubicHermiteSpline(t, waypoints[:, 0], tangents[:, 0])
        # poly_y = CubicHermiteSpline(t, waypoints[:, 1], tangents[:, 1])
        # poly_z = CubicHermiteSpline(t, waypoints[:, 2], tangents[:, 2])

        # # Derivatives of the polynomial functions
        # dpoly_x = poly_x.derivative()
        # dpoly_y = poly_y.derivative()
        # dpoly_z = poly_z.derivative()
        # Fit polynomial for each dimension
        poly_coeffs_x = np.polyfit(t, waypoints[:, 0], degree)
        poly_coeffs_y = np.polyfit(t, waypoints[:, 1], degree)
        poly_coeffs_z = np.polyfit(t, waypoints[:, 2], degree)

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
        gate_times = np.linspace(0, 1, self.num_gates + 2)
        gate_arc_lengths = [quad(arc_length_integrand, 0, t)[0] for t in gate_times]
        # Sample points corresponding to equally spaced arc lengths
        # Corresponds to the number of spline segments
        num_samples = 200
        arc_lengths = np.linspace(0, total_arc_length, num_samples)
        t_eval = np.zeros(num_samples)

        for i in range(1, num_samples):

            def arc_length_diff(t):
                arc_length, _ = quad(arc_length_integrand, 0, t)
                return arc_length - arc_lengths[i]

            t_eval[i] = optimize.newton(arc_length_diff, t_eval[i - 1])

        # Sample points corresponding to the gate times
        gate_times = [
            optimize.newton(lambda t: quad(arc_length_integrand, 0, t)[0] - al, 0.5)
            for al in gate_arc_lengths
        ]
        # Replace the closest sample point with the gate time
        manual_indices = np.zeros(len(gate_times), dtype=int)
        i = 0
        for gate_time in gate_times:
            # Find the index of the closest sample point to the gate time
            closest_index = np.argmin(np.abs(t_eval - gate_time))

            manual_indices[i] = closest_index
            # t_eval[closest_index] = gate_time
            i += 1
        # print(manual_indices)

        # Evaluate the polynomial at the sampled points
        # t_eval = np.linspace(0, 1, 100)
        x_eval = poly_x(t_eval)
        y_eval = poly_y(t_eval)
        z_eval = poly_z(t_eval)

        # Evaluate the derivatives at the sampled points
        dx_eval = dpoly_x(t_eval)
        dy_eval = dpoly_y(t_eval)
        dz_eval = dpoly_z(t_eval)

        # Combine the evaluated points into a single array
        fitted_coords = np.vstack((x_eval, y_eval, z_eval)).T
        tangents = np.vstack((dx_eval, dy_eval, dz_eval)).T
        # Normalize the tangents
        tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]

        manual_tangents = self.compute_normals(
            np.vstack((self.start_rpy, self.gates_rpy, self.start_rpy))
        )

        # tangents[manual_indices] = manual_tangents

        return fitted_coords, tangents
        print(fitted_coords)

    def fitHermiteSpline(self):
        # Step 1: Generate Hermite splines for each path segment
        waypoints, tangents = self.fitPolynomial()

        # waypoints = np.vstack((self.start_pos, self.gates_pos, self.start_pos))
        # orientations = np.vstack((self.start_rpy, self.gates_rpy, self.start_rpy))
        # tangents = self.compute_normals(orientations)

        # Define the progress parameter
        progress = ca.SX.sym("progress")

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
            spline = hermite_spline(p0, p1, m0, m1, progress)
            splines.append(spline)

        # Define the segments and corresponding progress intervals

        intervals = np.linspace(0, 1, segments + 1)

        # Create a piecewise function for the path
        path = ca.SX.zeros(3)
        for i in range(segments):
            segment_progress = (progress - intervals[i]) / (intervals[i + 1] - intervals[i])
            path += ca.if_else(
                ca.logic_and(progress >= intervals[i], progress < intervals[i + 1]), splines[i], 0
            )

        # Ensure the path reaches the final point at progress = 1
        path += ca.if_else(progress == 1, splines[-1], 0)
        # Create a CasADi function for the path
        path_function = ca.Function("path", [progress], [path])

        progress_values = []
        path_points = []

        for i in range(segments):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            m0 = tangents[i]
            m1 = tangents[i + 1]

            # Define progress interval for the segment
            segment_progress = np.linspace(i / segments, (i + 1) / segments, 100, endpoint=False)
            for t in segment_progress:
                progress_values.append(t)
                path_points.append(hermite_spline(p0, p1, m0, m1, (t - i / segments) * segments))

        # Convert to lists
        progress_values = [float(x) for x in progress_values]

        path_points = [[float(coord) for coord in point] for point in path_points]
        flattened_list = [item for sublist in path_points for item in sublist]
        path_points = flattened_list

        # Create CasADi interpolant for the path
        path_interpolant = ca.interpolant("path", "bspline", [progress_values], path_points)

        # Define symbolic progress variable
        progress = ca.SX.sym("progress")

        # Create symbolic path function
        # path_function = path_interpolant  # (progress)
        # Create symbolic derivative function
        dpath_value = ca.jacobian(path_function(progress), progress)
        dpath_function = ca.Function("dpath_function", [progress], [dpath_value])
        # Step 2: Generate Hermite splines for each path segment
        # path_segments = []
        # theta_norm = self.theta * (self.num_gates + 1)
        # arc_lengths = [0]
        # for i in range(waypoints.shape[0] - 1):
        #     P0 = waypoints[i, :]
        #     P1 = waypoints[i + 1, :]
        #     T0 = tangents[i, :]
        #     T1 = tangents[i + 1, :]
        #     t = (theta_norm - i) / (self.num_gates + 1)
        #     h00, h10, h01, h11 = self.hermite_basis(t)

        #     segment = h00 * P0 + h10 * T0 + h01 * P1 + h11 * T1
        #     path_segments.append(segment)

        #     if i > 0:
        #         arc_lengths.append(
        #             arc_lengths[-1] + ca.norm_2(path_segments[i] - path_segments[i - 1])
        #         )

        # arc_lengths = ca.vertcat(*arc_lengths) / arc_lengths[-1]  # Normalize

        # # Step 3: Combine the path segments into a single function
        # # path_segments = ca.horzcat(*path_segments)  # Combine segments into a single matrix
        # # path_segments = ca.DM(path_segments).full().flatten().tolist()  # Convert to list of floats
        # arc_lengths = ca.DM(arc_lengths).full().flatten().tolist()  # Convert to list of floats
        # print(arc_lengths)
        # path_segments = (
        #     ca.DM(ca.vertcat(*path_segments)).full().flatten().tolist()
        # )  # Convert to list of floats

        # thetas = np.linspace(0, 1, 100).tolist()
        # path_function = ca.interpolant("path", "bspline", thetas, path_segments)

        return path_function, dpath_function

    # def compute_normals(self, orientations):
    #     """Compute the normal vectors from the orientations."""
    #     normals = []
    #     for orientation in orientations:
    #         rot = Rot.from_euler("xyz", orientation)
    #         normal = rot.apply([1, 0, 0])  # Assuming the normal is along the x-axis
    #         normals.append(normal)
    #     return np.array(normals)

    def compute_normals(self, orientations):
        # Compute tangents (normals) from orientations
        tangents = []
        for rpy in orientations:
            # Convert RPY to a direction vector (assuming RPY represents the normal direction)
            direction = np.array(
                [np.cos(rpy[2]) * np.cos(rpy[1]), np.sin(rpy[2]) * np.cos(rpy[1]), np.sin(rpy[1])]
            )
            tangents.append(direction)
        return np.array(tangents)
