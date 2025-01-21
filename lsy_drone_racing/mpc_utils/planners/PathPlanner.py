import casadi as ca
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as Rot
from typing import Dict, Any


class HermiteSplinePathPlanner:
    def __init__(
        self,
        p,
        param_indices,
        current_param_index,
        gates_pos,
        gates_rpy,
        start_pos,
        start_rpy,
        progress,
    ):
        self.num_gates = gates_pos.shape[0]
        self.progress = progress

        self.gates_pos = gates_pos
        self.gates_rpy = gates_rpy
        self.start_pos = start_pos
        self.start_rpy = start_rpy

        # Define symbolic parameters for gate/start positions and orientations
        self.start_pos_sym = ca.MX.sym("start_pos", 3)
        self.start_rpy_sym = ca.MX.sym("start_rpy", 3)
        self.gates_pos_sym = ca.MX.sym("gates_pos", self.num_gates, 3)
        self.gates_rpy_sym = ca.MX.sym("gates_rpy", self.num_gates, 3)
        self.gate_progresses_sym = ca.MX.sym("gate_progresses", self.num_gates)

        # Combine all symbolic parameters into a single path parameter vector
        p_path = ca.vertcat(
            self.start_pos_sym,
            self.start_rpy_sym,
            ca.reshape(self.gates_pos_sym, -1, 1),
            ca.reshape(self.gates_rpy_sym, -1, 1),
            self.gate_progresses_sym,
        )
        if p is not None:
            p = ca.vertcat(p, p_path)
        else:
            p = p_path
        # Add the indices of the path parameters to the dictionary
        param_indices["start_pos"] = np.arange(current_param_index, current_param_index + 3)
        current_param_index += 3
        param_indices["start_rpy"] = np.arange(current_param_index, current_param_index + 3)
        current_param_index += 3
        param_indices["gates_pos"] = np.arange(
            current_param_index, current_param_index + self.num_gates * 3
        )
        current_param_index += self.num_gates * 3
        param_indices["gates_rpy"] = np.arange(
            current_param_index, current_param_index + self.num_gates * 3
        )
        current_param_index += self.num_gates * 3
        param_indices["gate_progresses"] = np.arange(
            current_param_index, current_param_index + self.num_gates
        )
        current_param_index += self.num_gates

        # Store the parameters and indices. Is copied by the dynamics class
        self.p = p
        self.param_indices = param_indices
        self.current_param_index = current_param_index

        # Create Hermite spline with symbolic parameters
        self.path_func, self.dpath_func, self.ddpath_func = self.create_hermite_spline()

        # Initialize gate progresses
        self.gate_progresses = self.calculate_gate_progresses(
            gates_pos, gates_rpy, start_pos, start_rpy
        )

    def create_hermite_spline(self):
        theta = self.progress

        # Hermite spline coefficients
        def hermite_coeff(t):
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            return h00, h10, h01, h11

        # Initialize path and derivatives
        path = ca.MX.zeros(3)
        dpath = ca.MX.zeros(3)
        ddpath = ca.MX.zeros(3)

        # Normalize theta to [0, 1] for the entire path
        theta_norm = theta * (self.num_gates + 1)

        # Combine start, gates, and end positions and orientations
        positions = ca.vertcat(self.start_pos_sym, self.gates_pos_sym, self.start_pos_sym)
        orientations = ca.vertcat(self.start_rpy_sym, self.gates_rpy_sym, self.start_rpy_sym)

        # Compute tangent vectors from rpy (for matching orientation at each gate)
        tangents = self.compute_normals(orientations)

        for i in range(self.num_gates + 1):
            p0 = positions[i, :]
            p1 = positions[i + 1, :]
            m0 = tangents[i, :]
            m1 = tangents[i + 1, :]

            # Normalize theta to [0, 1] for each segment
            t = (theta_norm - i) / (i + 1 - i)
            h00, h10, h01, h11 = hermite_coeff(t)

            # Hermite spline equation
            segment_path = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
            segment_dpath = ca.gradient(segment_path, theta)
            segment_ddpath = ca.gradient(segment_dpath, theta)

            path += ca.if_else((theta_norm >= i) & (theta_norm < i + 1), segment_path, 0)
            dpath += ca.if_else((theta_norm >= i) & (theta_norm < i + 1), segment_dpath, 0)
            ddpath += ca.if_else((theta_norm >= i) & (theta_norm < i + 1), segment_ddpath, 0)

        path_func = ca.Function(
            "path",
            [theta, self.start_pos_sym, self.start_rpy_sym, self.gates_pos_sym, self.gates_rpy_sym],
            [path],
        )
        dpath_func = ca.Function(
            "dpath",
            [theta, self.start_pos_sym, self.start_rpy_sym, self.gates_pos_sym, self.gates_rpy_sym],
            [dpath],
        )
        ddpath_func = ca.Function(
            "ddpath",
            [theta, self.start_pos_sym, self.start_rpy_sym, self.gates_pos_sym, self.gates_rpy_sym],
            [ddpath],
        )

        return path_func, dpath_func, ddpath_func

    def compute_normals(self, orientations):
        """Compute the normal vectors from the orientations."""
        normals = []
        for i in range(orientations.shape[0]):
            orientation = orientations[i, :]
            rot = ca.MX.zeros(3, 3)
            roll, pitch, yaw = orientation[0], orientation[1], orientation[2]
            rot[0, 0] = ca.cos(yaw) * ca.cos(pitch)
            rot[0, 1] = ca.cos(yaw) * ca.sin(pitch) * ca.sin(roll) - ca.sin(yaw) * ca.cos(roll)
            rot[0, 2] = ca.cos(yaw) * ca.sin(pitch) * ca.cos(roll) + ca.sin(yaw) * ca.sin(roll)
            rot[1, 0] = ca.sin(yaw) * ca.cos(pitch)
            rot[1, 1] = ca.sin(yaw) * ca.sin(pitch) * ca.sin(roll) + ca.cos(yaw) * ca.cos(roll)
            rot[1, 2] = ca.sin(yaw) * ca.sin(pitch) * ca.cos(roll) - ca.cos(yaw) * ca.sin(roll)
            rot[2, 0] = -ca.sin(pitch)
            rot[2, 1] = ca.cos(pitch) * ca.sin(roll)
            rot[2, 2] = ca.cos(pitch) * ca.cos(roll)
            normal = rot[:, 0]  # Assuming the normal is along the x-axis
            normals.append(normal)
        return ca.vertcat(*normals)
        # for orientation in orientations:
        #     rot = Rot.from_euler("xyz", orientation)
        #     normal = rot.apply([1, 0, 0])  # Assuming the normal is along the x-axis
        #     normals.append(normal)
        # return np.array(normals)

    def calculate_gate_progresses(self, gates_pos, gates_rpy, start_pos, start_rpy):
        num_gates = gates_pos.shape[0]
        gate_progresses = ca.MX.zeros(num_gates)
        for i in range(num_gates):
            gate_pos = gates_pos[i]
            # Find the progress value that minimizes the distance to the gate position
            progress_values = np.linspace(0, 1, 1000)
            distances = [
                ca.norm_2(self.path_func(p, start_pos, start_rpy, gates_pos, gates_rpy) - gate_pos)
                for p in progress_values
            ]
            gate_progresses[i] = progress_values[np.argmin(distances)]
        return gate_progresses

    def update_gates(self, gates_pos, gates_rpy):
        self.gates_pos = gates_pos
        self.gates_rpy = gates_rpy
        self.gate_progresses = self.calculate_gate_progresses(
            self.gates_pos, self.gates_rpy, self.start_pos, self.start_rpy
        )

    def computeProgress(self, current_pos, current_vel, num_samples=100):
        """Compute the current progress and progress rate along the path."""
        # Discretize the path
        progress_samples = np.linspace(0, 1, num_samples)
        path_positions = np.array(
            [
                self.path_func(p, self.start_pos, self.start_rpy, self.gates_pos, self.gates_rpy)
                for p in progress_samples
            ]
        )

        # Compute the distances from the current position to each sampled point
        distances = np.linalg.norm(path_positions - current_pos, axis=1)

        # Find the index of the closest point
        closest_index = np.argmin(distances)
        closest_progress = progress_samples[closest_index]

        # Optionally, interpolate between the closest points for more accuracy
        if closest_index > 0 and closest_index < num_samples - 1:
            prev_progress = progress_samples[closest_index - 1]
            next_progress = progress_samples[closest_index + 1]
            prev_pos = path_positions[closest_index - 1]
            next_pos = path_positions[closest_index + 1]

            # Linear interpolation
            t = np.dot(current_pos - prev_pos, next_pos - prev_pos) / np.dot(
                next_pos - prev_pos, next_pos - prev_pos
            )
            closest_progress = prev_progress + t * (next_progress - prev_progress)

        # Compute the path tangent at the closest progress
        path_tangent = self.dpath_func(
            closest_progress, self.start_pos, self.start_rpy, self.gates_pos, self.gates_rpy
        )

        # Compute the progress rate as the projection of the current velocity onto the path tangent
        progress_rate = np.dot(current_vel, path_tangent) / np.linalg.norm(path_tangent)

        return closest_progress, progress_rate


class PathPlanner:
    """Simple Path Planner that generates a centerline path through a start point and a series of gate positions and orientations.

    The first derivative of the path at each gate matches the gate's orientation. The same applies to the start point.
    The path is parameterized by a progress variable theta that ranges from 0 to 1.
    When updating gate positions, the path is updated and the current progress is mapped onto the new path.
    The last point of the path coincides with the start position to ensure periodicity.
    The path is parameterized by arc length to ensure smooth progress along the path. The path and derivative of the path are given as casadi functions.

    Attributes:
        gate_positions: Positions of the gates.
        gate_orientations: Orientations of the gates.
        start_position: Start position.
        start_orientation: Start orientation.
        reparameterize: Boolean indicating whether to reparameterize the path by arc length.
        withOrientation: Boolean indicating whether to enforce matching orientation at each gate with the path.
        path: Casadi function representing the path.
        dpath: Casadi function representing the derivative of the path.
        theta: Progress variable ranging from 0 to 1.

    Methods:
        update_gates: Update the gate positions and orientations.
        update_path: Update the path and its derivative by computing a spline through the start and gate positions and orientations.
        compute_normals: Compute the normal vectors from the orientations.
        reparameterize_by_arc_length: Reparameterize the path by arc length using splprep and splev.
        reparameterize_spline_by_arc_length: Reparameterize the path by arc length using CubicHermiteSpline.
    """

    def __init__(
        self,
        gate_positions,
        gate_orientations,
        start_position,
        start_orientation,
        theta: ca.MX,
        withOrientation=True,
        reparameterize=True,
    ):
        """Initialize the path planner with gate positions, gate orientations, start position, and start orientation."""
        self.gate_positions = gate_positions
        self.gate_orientations = gate_orientations
        self.start_position = start_position
        self.start_orientation = start_orientation
        self.theta = theta
        self.reparameterize = reparameterize
        self.withOrientation = withOrientation
        self.path = None
        self.dpath = None
        self.update_path(self.theta)

    def update_parameters(self, **kwargs):
        """Update the parameters of the path planner."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.update_path(self.theta)

    def compute_spline_segment(self, theta, p0, p1, t0, t1, i, n):
        """Compute the Hermite spline for a given segment."""
        t = (theta - i / (n - 1)) * (n - 1)
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t**2 * (3 - 2 * t)
        h11 = t**2 * (t - 1)
        p = h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1
        return p

    def update_path(self, theta: ca.MX):
        """Update the path and its derivative by computing a spline through the start and gate positions and orientations."""
        start_normal = self.compute_normals([self.start_orientation])[0]
        gate_normals = self.compute_normals(self.gate_orientations)
        waypoints = np.vstack([self.start_position, self.gate_positions])
        tangents = np.vstack([start_normal, gate_normals])
        n = len(waypoints)
        path = []
        dpath = []
        for i in range(n - 1):
            # p = self.compute_spline_segment(
            #     theta,
            #     waypoints[i],
            #     waypoints[i + 1],
            #     tangents[i],
            #     tangents[i + 1],
            #     i,
            #     n,
            # )
            # path.append(p)
            t = (theta - i / (n - 1)) * (n - 1)
            h00 = (1 + 2 * t) * (1 - t) ** 2
            h10 = t * (1 - t) ** 2
            h01 = t**2 * (3 - 2 * t)
            h11 = t**2 * (t - 1)
            p = (
                h00 * waypoints[i]
                + h10 * tangents[i]
                + h01 * waypoints[i + 1]
                + h11 * tangents[i + 1]
            )
            dp = (
                (6 * t**2 - 6 * t) * waypoints[i]
                + (3 * t**2 - 4 * t + 1) * tangents[i]
                + (-6 * t**2 + 6 * t) * waypoints[i + 1]
                + (3 * t**2 - 2 * t) * tangents[i + 1]
            )
            path.append(p)
            dpath.append(dp)
        # path_expr = ca.vertcat(*path)
        # self.path = ca.Function("path", [theta], [path_expr])
        # self.dpath = ca.Function("dpath", [theta], [ca.jacobian(path_expr, theta)])
        self.path = ca.Function("path", [theta], [ca.vertcat(*path)])
        self.dpath = ca.Function("dpath", [theta], [ca.vertcat(*dpath)])

        if self.reparameterize:
            self.reparameterize_spline_by_arc_length(theta)

    def getReferenceFrame(self, theta, dpath):
        # Get the tangent vector t from the path planner
        t = dpath(theta)
        t = t / ca.norm_2(t)  # Normalize the tangent vector

        # Compute the normal vector n (assuming the normal is in the xy-plane)
        n = ca.vertcat(-t[1], t[0], 0)

        # Compute the binormal vector b
        b = ca.cross(t, n)

        return t, n, b

    def compute_normals(self, orientations):
        """Compute the normal vectors from the orientations."""
        normals = []
        for orientation in orientations:
            rot = Rot.from_euler("xyz", orientation)
            normal = rot.apply([1, 0, 0])  # Assuming the normal is along the x-axis
            normals.append(normal)
        return np.array(normals)

    def reparameterize_spline_by_arc_length(self, theta, num_points=100):
        """Reparameterize the path by arc length using symbolic expressions."""
        # Sample the path to compute arc lengths
        theta_samples = np.linspace(0, 1, num_points)
        path_samples = np.array(
            [self.path(theta_sample).full().flatten() for theta_sample in theta_samples]
        )

        # Compute the arc lengths
        arc_lengths = np.cumsum(np.sqrt(np.sum(np.diff(path_samples, axis=0) ** 2, axis=1)))
        arc_lengths = np.insert(arc_lengths, 0, 0)  # Insert 0 at the beginning

        # Normalize arc lengths to [0, 1]
        arc_lengths /= arc_lengths[-1]

        # Convert arc_lengths and path_samples to lists of floats
        arc_lengths = arc_lengths.tolist()
        path_samples = path_samples.T.tolist()

        # Create a Casadi interpolant for the reparameterized path
        arc_length_interpolant = ca.interpolant(
            "arc_length_interpolant", "linear", [arc_lengths], [path_samples]
        )

        # Define the reparameterized path function
        reparam_path = arc_length_interpolant(theta)
        self.path = ca.Function("path", [theta], [reparam_path])

        # Define the derivative of the reparameterized path function
        reparam_dpath = ca.jacobian(reparam_path, theta)
        self.dpath = ca.Function("dpath", [theta], [reparam_dpath])

    # def update_path(self, theta: ca.MX):
    #     """Update the path and its derivative by computing a cubic Hermite spline through the start and gate positions and orientations.

    #     Args:
    #         theta: Progress variable defined in the optimization problem and ranging from 0 to 1.
    #     """
    #     # Compute the normals for the start and gate orientations
    #     start_normal = self.compute_normals([self.start_orientation])[0]
    #     gate_normals = self.compute_normals(self.gate_orientations)

    #     # Combine start position and gate positions
    #     waypoints = np.vstack([self.start_position, self.gate_positions])  # , self.start_position])
    #     tangents = np.vstack([start_normal, gate_normals])  # , start_normal])
    #     if self.withOrientation:
    #         # Create the cubic Hermite spline
    #         spline = CubicHermiteSpline(np.linspace(0, 1, len(waypoints)), waypoints, tangents)
    #         # Evaluate the spline at theta
    #         path = spline(theta)
    #         self.path = ca.Function("path", [theta], [ca.vertcat(*path)])
    #         # Evaluate the derivative of the spline at theta
    #         path_derivative = spline.derivative()(theta)
    #         self.dpath = ca.Function("dpath", [theta], [ca.vertcat(*path_derivative)])

    #         if self.reparameterize:
    #             self.reparameterize_spline_by_arc_length(theta)
    #     else:
    #         # Parameterize the path using splines
    #         tck, u = splprep(waypoints.T, s=0, k=3)

    #         # Evaluate the spline at theta
    #         path = splev(theta, tck)
    #         self.path = ca.Function("path", [theta], [ca.vertcat(*path)])

    #         # Evaluate the derivative of the spline at theta
    #         path_derivative = splev(theta, tck, der=1)
    #         self.dpath = ca.Function("dpath", [theta], [ca.vertcat(*path_derivative)])

    #         if self.reparameterize:
    #             self.reparameterize_by_arc_length(theta)

    # def reparameterize_spline_by_arc_length(self, theta, num_points=100):
    #     # Sample the path to compute arc lengths
    #     theta_samples = np.linspace(0, 1, num_points)
    #     path_samples = np.array([self.path(theta).full().flatten() for theta in theta_samples])

    #     # Compute the arc lengths
    #     arc_lengths = np.cumsum(np.sqrt(np.sum(np.diff(path_samples, axis=0) ** 2, axis=1)))
    #     arc_lengths = np.insert(arc_lengths, 0, 0)  # Insert 0 at the beginning

    #     # Normalize arc lengths to [0, 1]
    #     arc_lengths /= arc_lengths[-1]

    #     # Reparameterize the path with respect to arc length
    #     spline = CubicHermiteSpline(
    #         arc_lengths, path_samples, np.gradient(path_samples, arc_lengths, axis=0)
    #     )

    #     # Evaluate the reparameterized spline at theta
    #     path = spline(theta)
    #     self.path = ca.Function("path_function", [theta], [ca.vertcat(*path)])

    #     # Evaluate the derivative of the reparameterized spline at theta
    #     path_derivative = spline.derivative()(theta)
    #     self.dpath = ca.Function("path_gradient_function", [theta], [ca.vertcat(*path_derivative)])

    # def reparameterize_by_arc_length(self, theta, num_points=100):
    #     # Sample the path to compute arc lengths
    #     theta_samples = np.linspace(0, 1, num_points)
    #     path_samples = np.array([self.path(theta).full().flatten() for theta in theta_samples])

    #     # Compute the arc lengths
    #     arc_lengths = np.cumsum(np.sqrt(np.sum(np.diff(path_samples, axis=0) ** 2, axis=1)))
    #     arc_lengths = np.insert(arc_lengths, 0, 0)  # Insert 0 at the beginning

    #     # Normalize arc lengths to [0, 1]
    #     arc_lengths /= arc_lengths[-1]

    #     # Reparameterize the path with respect to arc length
    #     tck, u = splprep(path_samples.T, u=arc_lengths, s=0, k=3)

    #     # Evaluate the reparameterized spline at theta
    #     path = splev(theta, tck)
    #     self.path = ca.Function("path_function", [theta], [ca.vertcat(*path)])

    #     # Evaluate the derivative of the reparameterized spline at theta
    #     path_derivative = splev(theta, tck, der=1)
    #     self.dpath = ca.Function("path_gradient_function", [theta], [ca.vertcat(*path_derivative)])

    # class PathPlanner:

    # def create_centerline(self, gate_positions, gate_orientations):
    #     start_normal = self.compute_normals([self.start_orientation])[0]
    #     gate_normals = self.compute_normals(gate_orientations)
    #     n_gates = gate_positions.shape[0]
    #     path_points = [self.start_position]
    #     tangent_vectors = [start_normal]
    #     for i in range(n_gates - 1):
    #         p0 = gate_positions[i]
    #         p1 = gate_positions[i + 1]
    #         t0 = gate_normals[i]
    #         t1 = gate_normals[i + 1]
    #         for t in np.linspace(0, 1, num=100):
    #             point = self.hermite_spline(p0, p1, t0, t1, t)
    #             path_points.append(point)
    #             tangent_vectors.append(self.compute_tangent(p0, p1))
    #     path_points.append(gate_positions[-1])
    #     tangent_vectors.append(gate_normals[-1])
    #     path_points.append(self.start_position)  # Ensure periodicity
    #     tangent_vectors.append(start_normal)
    #     return path_points, tangent_vectors
    #     def calculate_arc_lengths(self, points):
    #         arc_lengths = [0]
    #         for i in range(1, len(points)):
    #             arc_lengths.append(arc_lengths[-1] + np.linalg.norm(points[i] - points[i - 1]))
    #         return arc_lengths
    #     def interpolate_by_arc_length(self, path_points, arc_lengths, theta):
    #         total_length = arc_lengths[-1]
    #         target_length = theta * total_length
    #         for i in range(1, len(arc_lengths)):
    #             if arc_lengths[i] >= target_length:
    #                 t = (target_length - arc_lengths[i - 1]) / (arc_lengths[i] - arc_lengths[i - 1])
    #                 return (1 - t) * path_points[i - 1] + t * path_points[i]
    #         return path_points[-1]
    #     def compute_tangent(self, p0, p1):
    #         return (p1 - p0) / np.linalg.norm(p1 - p0)
    #     def compute_normals(self, orientations):
    #         normals = []
    #         for rpy in orientations:
    #             rotation = Rot.from_euler("xyz", rpy)
    #             normal = rotation.apply([0, 0, 1])
    #             normals.append(normal)
    #         return normals
    # def hermite_spline(self, p0, p1, t0, t1, t):
    #     h00 = 2 * t**3 - 3 * t**2 + 1
    #     h10 = t**3 - 2 * t**2 + t
    #     h01 = -2 * t**3 + 3 * t**2
    #     h11 = t**3 - t**2
    #     return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1
    #     def reparameterize_by_arc_length(self, path_points):
    #         arc_lengths = self.calculate_arc_lengths(path_points)
    #         total_length = arc_lengths[-1]
    #         num_points = len(path_points)
    #         new_path = []
    #         for i in range(num_points):
    #             theta = i / (num_points - 1)
    #             new_path.append(self.interpolate_by_arc_length(path_points, arc_lengths, theta))
    #         return new_path
    #     def remap_progress(self, old_theta):
    #         old_length = old_theta * self.arc_lengths[-1]
    #         new_theta = old_length / self.arc_lengths[-1]
    #         return new_theta
    #     def get_path(self, theta):
    #         index = int(theta * (len(self.path) - 1))
    #         return self.path[index]
    #     def get_tangent_vector(self, theta):
    #         index = int(theta * (len(self.tangent_vectors) - 1))
    #         return self.tangent_vectors[index]

    # def get_transition_parameter(self):
    #     # Compute the smallest difference between the drone's progress and the progress of any gate
    #     min_diff = ca.inf
    #     for gate_progress in self.gate_progresses:
    #         diff = ca.fabs(self.theta - gate_progress)
    #         min_diff = ca.fmin(min_diff, diff)
    #     return min_diff

    def get_tunnel_dimensions(self):
        # Use the transition parameter in the sigmoid function to smoothly transition between Wn and Wgate
        transition_param = self.get_transition_parameter()
        W = self.Wn + (self.Wgate - self.Wn) / (1 + ca.exp(-10 * (0.5 - transition_param)))
        H = W  # Assuming W(θ) = H(θ)
        return W, H

    def update_constraints(self):
        # Get the desired position on the path
        pd = self.path(self.theta)
