import unittest
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from lsy_drone_racing.mpc_utils.planners.hermite_spline import HermiteSpline


class TestHermiteSpline(unittest.TestCase):
    def setUp(self):
        self.gate_positions = np.array(
            [[0.45, -1.0, 0.56], [1.0, -1.55, 1.11], [0.0, 0.5, 0.56], [-0.5, -0.5, 1.11]]
        )
        self.gate_orientations = np.array(
            [[0.0, 0.0, 2.35], [0.0, 0.0, -0.78], [0.0, 0.0, 0.0], [0.0, 0.0, 3.14]]
        )
        self.new_gate_positions = self.gate_positions + np.random.normal(0, 0.2, (4, 3))
        self.start_position = np.array([1.0, 1.0, 0.05])
        self.start_orientation = np.array([0, 0, 0])
        self.theta = ca.SX.sym("theta")

    def test_plot_parametric_path(self):
        waypoints = np.vstack(
            (self.start_position, self.gate_positions, self.start_position)
        )  # , start_pos))
        orientations = np.vstack(
            (self.start_orientation, self.gate_orientations, self.start_orientation)
        )  # , start_rpy))
        path = HermiteSpline(
            self.start_position,
            self.start_orientation,
            self.gate_positions,
            self.gate_orientations,
            parametric=True,
        )
        tangents = path.compute_normals(orientations)
        param_values = np.concatenate((waypoints.flatten(), tangents.flatten()))

        path_func = path.path_function
        dpath_func = path.dpath_function

        theta_values = np.linspace(0, 0.9, 1000)
        path_points = np.array(
            [
                path_func(theta=theta, path_params=param_values)["path"].full().flatten()
                for theta in theta_values
            ]
        )
        dpath_points = np.array(
            [dpath_func(theta, param_values).full().flatten() for theta in theta_values]
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            path_points[:, 0], path_points[:, 1], path_points[:, 2], label="Hermite Spline Path"
        )
        ax.scatter(
            self.gate_positions[:, 0],
            self.gate_positions[:, 1],
            self.gate_positions[:, 2],
            color="red",
            label="Gates",
        )
        ax.scatter(
            self.start_position[0],
            self.start_position[1],
            self.start_position[2],
            color="blue",
            label="Start",
        )
        for i in range(len(self.gate_positions)):
            ax.quiver(
                waypoints[i, 0],
                waypoints[i, 1],
                waypoints[i, 2],
                tangents[i, 0],
                tangents[i, 1],
                tangents[i, 2],
                length=0.1,
                color="red",
                label="Gate Tangents" if i == 0 else "",
            )

        ax.legend()
        plt.show()

    def test_compare_path_points(self):
        path1 = HermiteSpline(
            self.start_position,
            self.start_orientation,
            self.gate_positions,
            self.gate_orientations,
            parametric=True,
        )
        path2 = HermiteSpline(
            self.start_position,
            self.start_orientation,
            self.gate_positions,
            self.gate_orientations,
            parametric=False,
        )
        param_values = path1.path_params_values

        path_func1 = path1.path_function
        path_func2 = path2.path_function
        theta_values = np.linspace(0, 0.9, 1000)
        path_points1 = np.array(
            [
                path_func1(theta=theta, path_params=param_values)["path"].full().flatten()
                for theta in theta_values
            ]
        )
        path_points2 = np.array(
            [path_func2(theta=theta)["path"].full().flatten() for theta in theta_values]
        )

        path1.updateGates(self.new_gate_positions, self.gate_orientations)
        path2.updateGates(self.new_gate_positions, self.gate_orientations)
        param_values = path1.path_params_values
        path_func2 = path2.path_function
        path_points1_new = np.array(
            [
                path_func1(theta=theta, path_params=param_values)["path"].full().flatten()
                for theta in theta_values
            ]
        )
        path_points2_new = np.array(
            [path_func2(theta=theta)["path"].full().flatten() for theta in theta_values]
        )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(path_points1[:, 0], path_points1[:, 1], path_points1[:, 2], label="Parametric Path")
        ax.plot(
            path_points2[:, 0], path_points2[:, 1], path_points2[:, 2], label="Nonparametric Path"
        )
        ax.plot(
            path_points1_new[:, 0],
            path_points1_new[:, 1],
            path_points1_new[:, 2],
            label="Parametric Path New",
        )
        ax.plot(
            path_points2_new[:, 0],
            path_points2_new[:, 1],
            path_points2_new[:, 2],
            label="Nonparametric Path New",
        )
        ax.scatter(
            self.gate_positions[:, 0],
            self.gate_positions[:, 1],
            self.gate_positions[:, 2],
            color="red",
            label="Gates",
        )
        ax.scatter(
            self.new_gate_positions[:, 0],
            self.new_gate_positions[:, 1],
            self.new_gate_positions[:, 2],
            color="green",
            label="New Gates",
        )
        ax.scatter(
            self.start_position[0],
            self.start_position[1],
            self.start_position[2],
            color="blue",
            label="Start",
        )
        ax.legend()
        plt.show()
        assert np.allclose(path_points1, path_points2)

    def test_plot_path(self):
        path = HermiteSpline(
            self.start_position,
            self.start_orientation,
            self.gate_positions,
            self.gate_orientations,
            parametric=False,
        )
        waypoints = path.waypoints
        tangents = path.tangents
        # waypoints, tangents = path.fitPolynomial()
        path_func = path.path_function
        dpath_func = path.dpath_function
        # path_func, dpath_func = path.fitHermiteSpline()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Evaluate the path at intervals
        theta_values = np.linspace(0, 0.9, 1000)

        path_points = np.array([path_func(theta).full().flatten() for theta in theta_values])
        dpath_points = np.array([dpath_func(theta).full().flatten() for theta in theta_values])

        ax.plot(
            path_points[:, 0], path_points[:, 1], path_points[:, 2], label="Hermite Spline Path"
        )
        ax.scatter(
            self.gate_positions[:, 0],
            self.gate_positions[:, 1],
            self.gate_positions[:, 2],
            color="red",
            label="Gates",
        )
        ax.scatter(
            self.start_position[0],
            self.start_position[1],
            self.start_position[2],
            color="blue",
            label="Start",
        )

        for i in range(len(self.gate_positions)):
            ax.quiver(
                waypoints[i, 0],
                waypoints[i, 1],
                waypoints[i, 2],
                tangents[i, 0],
                tangents[i, 1],
                tangents[i, 2],
                length=0.1,
                color="red",
                label="Gate Tangents" if i == 0 else "",
            )

        ax.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
