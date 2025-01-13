from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import CubicHermiteSpline
import numpy as np
import casadi as ca


class PathPlanner:
    def __init__(self, gate_positions, gate_orientations, obstacles):
        self.gate_positions = gate_positions
        self.gate_orientations = gate_orientations
        self.obstacles = obstacles
        self.path = None
        self.tangent_vectors = None
        self.update_path()

    def update_gate_positions(self, new_gate_positions, new_gate_orientations):
        self.gate_positions = new_gate_positions
        self.gate_orientations = new_gate_orientations
        self.update_path()

    def update_obstacles(self, new_obstacles):
        self.obstacles = new_obstacles
        # Update path if necessary based on obstacles
        self.update_path()

    def update_path(self):
        self.path, self.tangent_vectors = self.create_centerline(
            self.gate_positions, self.gate_orientations
        )

    def create_centerline(self, gate_positions, gate_orientations):
        # Compute gate normals
        gate_normals = self.compute_gate_normals(gate_orientations)
        n_gates = len(gate_positions)
        path_points = []
        tangent_vectors = []

        for i in range(n_gates - 1):
            p0 = gate_positions[i]
            p1 = gate_positions[i + 1]
            t0 = gate_normals[i]
            t1 = gate_normals[i + 1]
            path_points.append(p0)
            tangent_vectors.append(t0)
            # Add intermediate points using Hermite spline
            for t in np.linspace(0, 1, num=100):
                point = self.hermite_spline(p0, p1, t0, t1, t)
                path_points.append(point)
                tangent_vectors.append(self.compute_tangent(p0, p1))

        path_points.append(gate_positions[-1])
        tangent_vectors.append(gate_normals[-1])

        return path_points, tangent_vectors

    def calculate_arc_lengths(self, points):
        arc_lengths = [0]
        for i in range(1, len(points)):
            arc_lengths.append(arc_lengths[-1] + np.linalg.norm(points[i] - points[i - 1]))
        return arc_lengths

    def compute_tangent(self, p0, p1):
        # Compute the tangent vector between two points
        return (p1 - p0) / np.linalg.norm(p1 - p0)

    def compute_gate_normals(self, gate_orientations):
        normals = []
        for rpy in gate_orientations:
            rotation = Rot.from_euler("xyz", rpy)
            normal = rotation.apply(
                [0, 0, 1]
            )  # Assuming the normal is along the z-axis in the gate's frame
            normals.append(normal)
        return normals

    def get_trajectory(self, theta):
        # Find the appropriate spline segment for the given theta
        for i, spline in enumerate(self.path):
            if spline.x[0] <= theta <= spline.x[1]:
                return spline(theta)
        return self.path[-1](self.path[-1].x[1])

    def hermite_spline(self, p0, p1, t0, t1, t):
        # Hermite spline interpolation
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1

    def get_path(self, theta):
        # Return the position on the path for a given progress variable theta
        index = int(theta * (len(self.path) - 1))
        return self.path[index]

    def get_tangent_vector(self, theta):
        # Return the tangent vector of the path for a given progress variable theta
        index = int(theta * (len(self.tangent_vectors) - 1))
        return self.tangent_vectors[index]


class PathPlanner2:
    def __init__(self, gate_positions, gate_orientations, obstacles, drone_position):
        self.gate_positions = gate_positions
        self.gate_orientations = gate_orientations
        self.obstacles = obstacles
        self.drone_position = drone_position
        self.path = None
        self.tangent_vectors = None
        self.update_path()

    def update_gate_positions(self, new_gate_positions, new_gate_orientations):
        self.gate_positions = new_gate_positions
        self.gate_orientations = new_gate_orientations
        self.update_path()

    def update_obstacles(self, new_obstacles):
        self.obstacles = new_obstacles
        self.update_path()

    def update_drone_position(self, new_drone_position):
        self.drone_position = new_drone_position
        self.update_path()

    def update_path(self):
        self.path, self.tangent_vectors = self.generate_centerline(
            self.gate_positions, self.gate_orientations, self.drone_position
        )
        self.path = self.reparameterize_by_arc_length(self.path)

    def generate_centerline(self, gate_positions, gate_orientations, drone_position):
        gate_normals = self.compute_gate_normals(gate_orientations)
        n_gates = len(gate_positions)
        path_points = [drone_position]
        tangent_vectors = [self.compute_tangent(drone_position, gate_positions[0])]

        for i in range(n_gates):
            p0 = gate_positions[i]
            t0 = gate_normals[i]
            path_points.append(p0)
            tangent_vectors.append(t0)
            if i < n_gates - 1:
                p1 = gate_positions[i + 1]
                t1 = gate_normals[i + 1]
                for t in np.linspace(0, 1, num=100):
                    point = self.hermite_spline(p0, p1, t0, t1, t)
                    path_points.append(point)
                    tangent_vectors.append(self.compute_tangent(p0, p1))

        return path_points, tangent_vectors

    def compute_gate_normals(self, gate_orientations):
        normals = []
        for rpy in gate_orientations:
            rotation = Rot.from_euler("xyz", rpy)
            normal = rotation.apply([0, 0, 1])
            normals.append(normal)
        return normals

    def hermite_spline(self, p0, p1, t0, t1, t):
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1

    def compute_tangent(self, p0, p1):
        return (p1 - p0) / np.linalg.norm(p1 - p0)

    def reparameterize_by_arc_length(self, path_points):
        arc_lengths = self.calculate_arc_lengths(path_points)
        total_length = arc_lengths[-1]
        num_points = len(path_points)
        new_path = []
        for i in range(num_points):
            theta = i / (num_points - 1) * total_length
            new_path.append(self.interpolate_by_arc_length(path_points, arc_lengths, theta))
        return new_path

    def calculate_arc_lengths(self, points):
        arc_lengths = [0]
        for i in range(1, len(points)):
            arc_lengths.append(arc_lengths[-1] + np.linalg.norm(points[i] - points[i - 1]))
        return arc_lengths

    def interpolate_by_arc_length(self, path_points, arc_lengths, theta):
        for i in range(1, len(arc_lengths)):
            if arc_lengths[i] >= theta:
                t = (theta - arc_lengths[i - 1]) / (arc_lengths[i] - arc_lengths[i - 1])
                return (1 - t) * path_points[i - 1] + t * path_points[i]
        return path_points[-1]

    def get_trajectory(self, theta):
        index = int(theta * (len(self.path) - 1))
        return self.path[index]


class Tunnel:
    def __init__(self, path_planner, Wn, Wgate):
        self.path_planner = path_planner
        self.Wn = Wn

        self.Wgate = Wgate

    def get_tunnel_dimensions(self, theta):
        # Use a sigmoid function to smoothly transition between Wn and Wgate
        W = self.Wn + (self.Wgate - self.Wn) / (1 + np.exp(-10 * (theta - 0.5)))
        H = W  # Assuming W(θ) = H(θ)
        return W, H

    def get_tunnel_constraints(self, pk, theta):
        pd = self.path_planner.get_path(theta)
        t = self.path_planner.get_tangent_vector(theta)
        n = np.cross(t, np.array([0, 0, 1]))  # Normal vector
        b = np.cross(t, n)  # Binormal vector

        W, H = self.get_tunnel_dimensions(theta)
        p0 = pd - W * n - H * b

        constraints = [
            ca.dot(pk - p0, n) >= 0,
            2 * H - ca.dot(pk - p0, n) >= 0,
            ca.dot(pk - p0, b) >= 0,
            2 * W - ca.dot(pk - p0, b) >= 0,
        ]

        return constraints

    def get_obstacle_constraints(self, pk):
        constraints = []
        for obstacle in self.path_planner.obstacles:
            obs_pos = obstacle["pos"]
            obs_radius = obstacle["radius"]
            constraints.append(ca.norm_2(pk - obs_pos) >= obs_radius)
        return constraints
