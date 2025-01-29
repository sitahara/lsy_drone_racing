import unittest
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from lsy_drone_racing.mpc_utils.Planner.PathPlanner import PathPlanner


class TestPathPlanner(unittest.TestCase):
    def setUp(self):
        self.gate_positions = np.array([[1, 2, 2], [1, 0, 1], [2, 1, 0.5]])
        self.gate_orientations = np.array([[0, 0, 0], [0, 0, np.pi], [0, 0, np.pi / 2]])
        self.start_position = np.array([0, 0, 0])
        self.start_orientation = np.array([0, 0, 0])
        self.theta = ca.MX.sym("theta")
        self.reparameterize = False

    def test_initialization(self):
        planner = PathPlanner(
            self.gate_positions,
            self.gate_orientations,
            self.start_position,
            self.start_orientation,
            self.theta,
            reparameterize=self.reparameterize,
        )
        self.assertIsNotNone(planner.path)
        self.assertIsNotNone(planner.dpath)

    def test_update_path(self):
        planner = PathPlanner(
            self.gate_positions,
            self.gate_orientations,
            self.start_position,
            self.start_orientation,
            self.theta,
            reparameterize=self.reparameterize,
        )
        theta_val = 0.5
        path_val = planner.path(theta_val).full()
        dpath_val = planner.dpath(theta_val).full()
        self.assertEqual(path_val.shape, (3,))
        self.assertEqual(dpath_val.shape, (3,))

    def test_plot_path(self):
        planner = PathPlanner(
            self.gate_positions,
            self.gate_orientations,
            self.start_position,
            self.start_orientation,
            self.theta,
            reparameterize=self.reparameterize,
        )
        theta_samples = np.linspace(0, 1, 100)
        path_samples = np.array([planner.path(theta).full().flatten() for theta in theta_samples])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(path_samples[:, 0], path_samples[:, 1], path_samples[:, 2], label="Planned Path")
        ax.scatter(
            self.start_position[0],
            self.start_position[1],
            self.start_position[2],
            color="red",
            label="Start Position",
        )
        ax.scatter(
            self.gate_positions[:, 0],
            self.gate_positions[:, 1],
            self.gate_positions[:, 2],
            color="blue",
            label="Gate Positions",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    def test_reparameterize_spline_by_arc_length(self):
        planner = PathPlanner(
            self.gate_positions,
            self.gate_orientations,
            self.start_position,
            self.start_orientation,
            self.theta,
            reparameterize=True,
        )
        theta_val = 0.5
        path_val = planner.path(theta_val).full()
        dpath_val = planner.dpath(theta_val).full()
        self.assertEqual(path_val.shape, (3,))
        self.assertEqual(dpath_val.shape, (3,))

    def test_update_parameters(self):
        planner = PathPlanner(
            self.gate_positions,
            self.gate_orientations,
            self.start_position,
            self.start_orientation,
            self.theta,
            reparameterize=self.reparameterize,
        )
        new_gate_positions = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
        planner.update_parameters(gate_positions=new_gate_positions)
        self.assertTrue(np.array_equal(planner.gate_positions, new_gate_positions))


if __name__ == "__main__":
    unittest.main()
