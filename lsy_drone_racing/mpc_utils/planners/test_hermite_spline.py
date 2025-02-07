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
        self.start_position = np.array([1.0, 1.0, 0.05])
        self.start_orientation = np.array([0, 0, 0])
        self.theta = ca.SX.sym("theta")

    # def test_plot_polynomial(self):
    #     path = HermiteSpline(
    #         self.theta,
    #         self.start_position,
    #         self.start_orientation,
    #         self.gate_positions,
    #         self.gate_orientations,
    #     )

    #     path_samples = path.fitPolynomial()

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(path_samples[:, 0], path_samples[:, 1], path_samples[:, 2], label="Planned Path")
    #     ax.scatter(
    #         self.start_position[0],
    #         self.start_position[1],
    #         self.start_position[2],
    #         color="red",
    #         label="Start Position",
    #     )
    #     ax.scatter(
    #         self.gate_positions[:, 0],
    #         self.gate_positions[:, 1],
    #         self.gate_positions[:, 2],
    #         color="blue",
    #         label="Gate Positions",
    #     )
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.legend()
    #     plt.show()

    def test_plot_path(self):
        path = HermiteSpline(
            self.start_position, self.start_orientation, self.gate_positions, self.gate_orientations
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
        theta_values = np.linspace(0, 0.8, 1000)
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
        # Add tangents at the path points to the plot
        # for i in range(len(path_points)):
        #     ax.quiver(
        #         path_points[i, 0],
        #         path_points[i, 1],
        #         path_points[i, 2],
        #         dpath_points[i, 0],
        #         dpath_points[i, 1],
        #         dpath_points[i, 2],
        #         length=0.1,
        #         color="green",
        #         label="Path Tangents" if i == 0 else "",
        #     )
        # gate_normals = path.compute_normals(self.gate_orientations)
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
