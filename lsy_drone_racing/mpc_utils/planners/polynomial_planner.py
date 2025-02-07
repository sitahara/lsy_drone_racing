import casadi as ca
import numpy as np
from scipy.integrate import quad
from scipy import optimize
from typing import Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splev, splprep
from scipy.interpolate import CubicHermiteSpline


class PolynomialPlanner:
    def __init__(
        self, theta, start_pos, start_rpy, gates_pos, gates_rpy, ts, n_horizon, desired_time=7.0
    ):
        self.theta = theta
        self.desired_time = desired_time
        self.start_pos = start_pos
        self.start_rpy = start_rpy
        self.gates_pos = gates_pos
        self.gates_rpy = gates_rpy
        self.waypoints = np.vstack((self.start_pos, self.gates_pos))
        self.orientations = np.vstack((self.start_rpy, self.gates_rpy))
        self.tangents = self.compute_normals(self.orientations)
        self.ts = ts
        self.n_horizon = n_horizon
        self.fitPolynomial()

    def fitPolynomial(self):
        waypoints = self.waypoints
        tangents = self.tangents
        degree = waypoints.shape[0] - 1
        t = np.linspace(0, self.desired_time, waypoints.shape[0])

        # Fit polynomial for each dimension
        poly_coeffs_x = np.polyfit(t, waypoints[:, 0], degree)
        poly_coeffs_y = np.polyfit(t, waypoints[:, 1], degree)
        poly_coeffs_z = np.polyfit(t, waypoints[:, 2], degree)

        # Create polynomial functions
        poly_x = np.poly1d(poly_coeffs_x)
        poly_y = np.poly1d(poly_coeffs_y)
        poly_z = np.poly1d(poly_coeffs_z)

        self.poly_x = poly_x
        self.poly_y = poly_y
        self.poly_z = poly_z
        return None

        # Sample points corresponding to equally spaced arc lengths
        # Corresponds to the number of spline segments
        num_samples = 1000
        fitted_coords = np.zeros((num_samples, 3))
        fitted_tangents = np.zeros((num_samples, 3))
        for i in range(num_samples):
            t = i / num_samples
            fitted_coords[i, :] = [poly_x(t), poly_y(t), poly_z(t)]
        #    fitted_tangents[i, :] = [dpoly_x(t), dpoly_y(t), dpoly_z(t)]

        return fitted_coords  # , tangents

    def update_waypoints(self, gates_pos=None, gates_rpy=None, start_pos=None, start_rpy=None):
        if gates_pos is not None:
            self.gates_pos = gates_pos
            self.gates_rpy = gates_rpy
        if start_pos and start_rpy is not None:
            self.start_pos = start_pos
            self.start_rpy = start_rpy

        self.waypoints = np.vstack((self.start_pos, self.gates_pos))
        self.orientations = np.vstack((self.start_rpy, self.gates_rpy))
        self.tangents = self.compute_normals(self.orientations)
        self.fitPolynomial()

    def output_xref(self, t0):
        x_ref = np.zeros((self.n_horizon + 1, 3))
        for i in range(self.n_horizon + 1):
            t = t0 + (i + 1) * self.ts
            x_ref[i, :] = np.array([self.poly_x(t), self.poly_y(t), self.poly_z(t)])
        return x_ref.T

    def compute_normals(self, orientations):
        tangents = np.zeros((len(orientations), 3))
        for i in range(len(orientations)):
            # Convert RPY to a direction vector (assuming RPY represents the normal direction)
            roll, pitch, yaw = orientations[i, :]
            direction = np.array(
                [np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)]
            )
            tangents[i, :] = direction / np.linalg.norm(direction)
        print(tangents)
        return tangents
