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
        self.Wn = 0.8
        self.Wgate = 0.10

    def test_plot_tunnels(self):
        pathPLanner = HermiteSpline(
            self.start_position,
            self.start_orientation,
            self.gate_positions,
            self.gate_orientations,
            parametric=True,
        )
        tangents = pathPLanner.tangents
        param_values = pathPLanner.path_params_values
        # Calculate the tangent vector
        path_func = pathPLanner.path_function
        path = path_func(self.theta, param_values)
        dpath_func = pathPLanner.dpath_function
        dpath = dpath_func(self.theta, param_values)

        t = dpath / ca.norm_2(dpath)
        n = ca.jacobian(t, self.theta)
        n = n / ca.norm_2(n)
        b = ca.cross(t, n)

        # Create CasADi functions for the path and the vectors
        t_func = ca.Function("t_func", [self.theta], [t])
        n_func = ca.Function("n_func", [self.theta], [n])
        b_func = ca.Function("b_func", [self.theta], [b])

        # Generate points along the path
        theta_values = np.linspace(0, 1, 100)  # Adjust the range and number of points as needed
        path_points = np.array(
            [path_func(theta_val, param_values).full().flatten() for theta_val in theta_values]
        )
        tangent_vectors = np.array(
            [t_func(theta_val).full().flatten() for theta_val in theta_values]
        )
        normal_vectors = np.array(
            [n_func(theta_val).full().flatten() for theta_val in theta_values]
        )
        binormal_vectors = np.array(
            [b_func(theta_val).full().flatten() for theta_val in theta_values]
        )
        d = np.zeros(len(theta_values))
        sigmoid = np.zeros(len(theta_values))
        W = np.zeros(len(theta_values))
        p0_points = np.zeros((len(theta_values), 3))
        p1_points = np.zeros((len(theta_values), 3))
        p2_points = np.zeros((len(theta_values), 3))
        p3_points = np.zeros((len(theta_values), 3))
        for k in range(len(theta_values)):
            d[k] = np.min(np.abs(theta_values[k] - pathPLanner.theta_switch[1:5]))
            sigmoid[k] = 1 / (1 + np.exp(-100 * (d[k] - 0.05)))
            W[k] = self.Wgate + (self.Wn - self.Wgate) * sigmoid[k]
            p0_points[k] = path_points[k] - W[k] * normal_vectors[k] - W[k] * binormal_vectors[k]
            p1_points[k] = path_points[k] + W[k] * normal_vectors[k] - W[k] * binormal_vectors[k]
            p2_points[k] = path_points[k] + W[k] * normal_vectors[k] + W[k] * binormal_vectors[k]
            p3_points[k] = path_points[k] - W[k] * normal_vectors[k] + W[k] * binormal_vectors[k]
        # Calculate the tunnel boundaries
        tunnel_boundaries = [p0_points, p1_points, p2_points, p3_points]

        # Plot the path and the tunnel boundaries
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the path
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], label="Path")

        # Plot the tunnel boundaries
        for i in range(0, len(theta_values), 10):  # Adjust the step size for fewer arrows
            ax.quiver(
                path_points[i, 0],
                path_points[i, 1],
                path_points[i, 2],
                tangent_vectors[i, 0],
                tangent_vectors[i, 1],
                tangent_vectors[i, 2],
                color="r",
                length=0.1,
                normalize=True,
                label="Tangent" if i == 0 else "",
            )
            ax.quiver(
                path_points[i, 0],
                path_points[i, 1],
                path_points[i, 2],
                normal_vectors[i, 0],
                normal_vectors[i, 1],
                normal_vectors[i, 2],
                color="g",
                length=0.1,
                normalize=True,
                label="Normal" if i == 0 else "",
            )
            ax.quiver(
                path_points[i, 0],
                path_points[i, 1],
                path_points[i, 2],
                binormal_vectors[i, 0],
                binormal_vectors[i, 1],
                binormal_vectors[i, 2],
                color="b",
                length=0.1,
                normalize=True,
                label="Binormal" if i == 0 else "",
            )
            ax.plot(
                [path_points[i, 0], p0_points[i, 0]],
                [path_points[i, 1], p0_points[i, 1]],
                [path_points[i, 2], p0_points[i, 2]],
                "k--",
                label="Tunnel Boundary" if i == 0 else "",
            )
            # Plot the rectangle representing the tunnel cross-section
            ax.plot(
                [
                    p0_points[i, 0],
                    p1_points[i, 0],
                    p2_points[i, 0],
                    p3_points[i, 0],
                    p0_points[i, 0],
                ],
                [
                    p0_points[i, 1],
                    p1_points[i, 1],
                    p2_points[i, 1],
                    p3_points[i, 1],
                    p0_points[i, 1],
                ],
                [
                    p0_points[i, 2],
                    p1_points[i, 2],
                    p2_points[i, 2],
                    p3_points[i, 2],
                    p0_points[i, 2],
                ],
                "k-",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    # def test_plot_frenet_frame(self):
    #     path = HermiteSpline(
    #         self.start_position,
    #         self.start_orientation,
    #         self.gate_positions,
    #         self.gate_orientations,
    #         parametric=True,
    #     )
    #     tangents = path.tangents
    #     param_values = path.path_params_values
    #     # Calculate the tangent vector
    #     path_func = path.path_function
    #     dpath_func = path.dpath_function(self.theta, param_values)
    #     ddpath_func = path.ddpath_function
    #     T = dpath_func / ca.norm_2(dpath_func)
    #     # Calculate the derivative of the tangent vector
    #     T_prime = ca.jacobian(T, self.theta)
    #     # Calculate the normal vector
    #     N = T_prime / ca.norm_2(T_prime)
    #     # Calculate the binormal vector
    #     B = ca.cross(T, N)

    #     # Create CasADi functions for the tangent, normal, and binormal vectors
    #     T_func = ca.Function("T_func", [self.theta], [T])
    #     N_func = ca.Function("N_func", [self.theta], [N])
    #     B_func = ca.Function("B_func", [self.theta], [B])

    #     # Generate points along the path
    #     t_values = np.linspace(0, 1, 100)  # Adjust the range and number of points as needed
    #     path_points = np.array(
    #         [path_func(t_val, param_values).full().flatten() for t_val in t_values]
    #     )
    #     tangent_vectors = np.array([T_func(t_val).full().flatten() for t_val in t_values])
    #     normal_vectors = np.array([N_func(t_val).full().flatten() for t_val in t_values])
    #     binormal_vectors = np.array([B_func(t_val).full().flatten() for t_val in t_values])

    #     # Plot the path and the vectors
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")

    #     # Plot the path
    #     ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], label="Path")

    #     # Plot the tangent, normal, and binormal vectors
    #     for i in range(0, len(t_values), 10):  # Adjust the step size for fewer arrows
    #         ax.quiver(
    #             path_points[i, 0],
    #             path_points[i, 1],
    #             path_points[i, 2],
    #             tangent_vectors[i, 0],
    #             tangent_vectors[i, 1],
    #             tangent_vectors[i, 2],
    #             color="r",
    #             length=0.1,
    #             normalize=True,
    #             label="Tangent" if i == 0 else "",
    #         )
    #         ax.quiver(
    #             path_points[i, 0],
    #             path_points[i, 1],
    #             path_points[i, 2],
    #             normal_vectors[i, 0],
    #             normal_vectors[i, 1],
    #             normal_vectors[i, 2],
    #             color="g",
    #             length=0.1,
    #             normalize=True,
    #             label="Normal" if i == 0 else "",
    #         )
    #         ax.quiver(
    #             path_points[i, 0],
    #             path_points[i, 1],
    #             path_points[i, 2],
    #             binormal_vectors[i, 0],
    #             binormal_vectors[i, 1],
    #             binormal_vectors[i, 2],
    #             color="b",
    #             length=0.1,
    #             normalize=True,
    #             label="Binormal" if i == 0 else "",
    #         )

    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.legend()
    #     plt.show()

    # def test_plot_tunnel(self):
    #     waypoints = np.vstack(
    #         (self.start_position, self.gate_positions, self.start_position)
    #     )  # , start_pos))
    #     orientations = np.vstack(
    #         (self.start_orientation, self.gate_orientations, self.start_orientation)
    #     )  # , start_rpy))
    #     path = HermiteSpline(
    #         self.start_position,
    #         self.start_orientation,
    #         self.gate_positions,
    #         self.gate_orientations,
    #         parametric=True,
    #     )
    #     tangents = path.tangents
    #     param_values = path.path_params_values

    #     path_func = path.path_function
    #     dpath_func = path.dpath_function
    #     ddpath_func = path.ddpath_function

    #     theta_values = np.linspace(0, 0.9, 50)

    #     path_points = np.array(
    #         [
    #             path_func(theta=theta, path_params=param_values)["path"].full().flatten()
    #             for theta in theta_values
    #         ]
    #     )
    #     dpath_points = np.array(
    #         [dpath_func(theta, param_values).full().flatten() for theta in theta_values]
    #     )

    #     ddpath_points = np.array(
    #         [ddpath_func(theta, param_values).full().flatten() for theta in theta_values]
    #     )
    #     num_points = len(path_points)
    #     T = np.zeros_like(path_points)
    #     N = np.zeros_like(path_points)
    #     B = np.zeros_like(path_points)
    #     W = np.zeros_like(theta_values)
    #     d = np.zeros_like(theta_values)
    #     sigmoid = np.zeros_like(theta_values)
    #     p0 = np.zeros_like(path_points)
    #     p1 = np.zeros_like(path_points)
    #     p2 = np.zeros_like(path_points)
    #     p3 = np.zeros_like(path_points)

    #     for k in range(num_points):
    #         # Compute the tangent vector
    #         T[k, :] = dpath_points[k, :] / np.linalg.norm(dpath_points[k, :])
    #         # N[k, :] = np.cross(T[k, :], ddpath_points[k, :])
    #         N[k, :] = ddpath_points[k, :] / np.linalg.norm(ddpath_points[k, :])
    #         B[k, :] = np.cross(T[k, :], N[k, :])
    #         # Compute the tunnel corners
    #         d[k] = np.min(np.abs(theta_values[k] - path.theta_switch[1:5]))
    #         sigmoid[k] = 1 / (1 + np.exp(-100 * (d[k] - 0.05)))
    #         W[k] = self.Wgate + (self.Wn - self.Wgate) * sigmoid[k]
    #         p0[k, :] = path_points[k, :] - W[k] * N[k, :] - W[k] * B[k, :]
    #         p1[k, :] = path_points[k, :] + W[k] * N[k, :] - W[k] * B[k, :]
    #         p2[k, :] = path_points[k, :] - W[k] * N[k, :] + W[k] * B[k, :]
    #         p3[k, :] = path_points[k, :] + W[k] * N[k, :] + W[k] * B[k, :]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     # ax.plot(theta_values, d, label="distance")
    #     ax.plot(d, W, label="Width")
    #     ax.plot(d, sigmoid, label="Sigmoid")

    #     ax.set_xlabel("Distance")
    #     ax.set_ylabel("Width and Sigmoid")
    #     ax.set_title("Tunnel Width and Sigmoid Function")
    #     ax.grid(True)

    #     ax.legend()
    #     plt.show()
    #     #     p1[k,:] =
    #     # # Tunnel constraints
    #     # tunnel_constraints = []
    #     # tunnel_constraints.append((pos - p0).T @ n)
    #     # tunnel_constraints.append(2 * H - (pos - p0).T @ n)
    #     # tunnel_constraints.append((pos - p0).T @ b)
    #     # tunnel_constraints.append(2 * W - (pos - p0).T @ b)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(
    #         path_points[:, 0], path_points[:, 1], path_points[:, 2], label="Hermite Spline Path"
    #     )
    #     ind = np.arange(0, num_points, 10)
    #     ax.plot(p0[ind, 0], p0[ind, 1], p0[ind, 2], label="p0")
    #     ax.plot(p1[ind, 0], p1[ind, 1], p1[ind, 2], label="p1")
    #     ax.plot(p2[ind, 0], p2[ind, 1], p2[ind, 2], label="p2")
    #     ax.plot(p3[ind, 0], p3[ind, 1], p3[ind, 2], label="p3")
    #     ax.scatter(
    #         self.gate_positions[:, 0],
    #         self.gate_positions[:, 1],
    #         self.gate_positions[:, 2],
    #         color="red",
    #         label="Gates",
    #     )
    #     ax.scatter(
    #         self.start_position[0],
    #         self.start_position[1],
    #         self.start_position[2],
    #         color="blue",
    #         label="Start",
    #     )
    #     for i in range(len(self.gate_positions)):
    #         ax.quiver(
    #             waypoints[i, 0],
    #             waypoints[i, 1],
    #             waypoints[i, 2],
    #             tangents[i, 0],
    #             tangents[i, 1],
    #             tangents[i, 2],
    #             length=0.1,
    #             color="red",
    #             label="Gate Tangents" if i == 0 else "",
    #         )
    #     for i in range(ind.size):
    #         ax.quiver(
    #             path_points[ind[i], 0],
    #             path_points[ind[i], 1],
    #             path_points[ind[i], 2],
    #             T[ind[i], 0],
    #             T[ind[i], 1],
    #             T[ind[i], 2],
    #             length=0.1,
    #             color="green",
    #             label="T" if i == 0 else "",
    #         )
    #     for i in range(ind.size):
    #         ax.quiver(
    #             path_points[ind[i], 0],
    #             path_points[ind[i], 1],
    #             path_points[ind[i], 2],
    #             N[ind[i], 0],
    #             N[ind[i], 1],
    #             N[ind[i], 2],
    #             length=0.1,
    #             color="blue",
    #             label="N" if i == 0 else "",
    #         )
    #     for i in range(ind.size):
    #         ax.quiver(
    #             path_points[i, 0],
    #             path_points[i, 1],
    #             path_points[i, 2],
    #             B[ind[i], 0],
    #             B[ind[i], 1],  # Updated from B[i, 1] to B[ind[i], 1]
    #             B[ind[i], 2],  # Updated from B[i, 2] to B[ind[i], 2]
    #             length=0.1,
    #             color="orange",
    #             label="B" if i == 0 else "",
    #         )

    #     ax.legend()
    #     plt.show()

    # def test_plot_parametric_path(self):
    #     waypoints = np.vstack(
    #         (self.start_position, self.gate_positions, self.start_position)
    #     )  # , start_pos))
    #     orientations = np.vstack(
    #         (self.start_orientation, self.gate_orientations, self.start_orientation)
    #     )  # , start_rpy))
    #     path = HermiteSpline(
    #         self.start_position,
    #         self.start_orientation,
    #         self.gate_positions,
    #         self.gate_orientations,
    #         parametric=True,
    #     )
    #     tangents = path.compute_normals(orientations)
    #     param_values = np.concatenate((waypoints.flatten(), tangents.flatten()))

    #     path_func = path.path_function
    #     dpath_func = path.dpath_function

    #     theta_values = np.linspace(0, 0.9, 1000)
    #     path_points = np.array(
    #         [
    #             path_func(theta=theta, path_params=param_values)["path"].full().flatten()
    #             for theta in theta_values
    #         ]
    #     )
    #     dpath_points = np.array(
    #         [dpath_func(theta, param_values).full().flatten() for theta in theta_values]
    #     )
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(
    #         path_points[:, 0], path_points[:, 1], path_points[:, 2], label="Hermite Spline Path"
    #     )
    #     ax.scatter(
    #         self.gate_positions[:, 0],
    #         self.gate_positions[:, 1],
    #         self.gate_positions[:, 2],
    #         color="red",
    #         label="Gates",
    #     )
    #     ax.scatter(
    #         self.start_position[0],
    #         self.start_position[1],
    #         self.start_position[2],
    #         color="blue",
    #         label="Start",
    #     )
    #     for i in range(len(self.gate_positions)):
    #         ax.quiver(
    #             waypoints[i, 0],
    #             waypoints[i, 1],
    #             waypoints[i, 2],
    #             tangents[i, 0],
    #             tangents[i, 1],
    #             tangents[i, 2],
    #             length=0.1,
    #             color="red",
    #             label="Gate Tangents" if i == 0 else "",
    #         )

    #     ax.legend()
    #     plt.show()

    # def test_compare_path_points(self):
    #     path1 = HermiteSpline(
    #         self.start_position,
    #         self.start_orientation,
    #         self.gate_positions,
    #         self.gate_orientations,
    #         parametric=True,
    #     )
    #     path2 = HermiteSpline(
    #         self.start_position,
    #         self.start_orientation,
    #         self.gate_positions,
    #         self.gate_orientations,
    #         parametric=False,
    #     )
    #     param_values = path1.path_params_values

    #     path_func1 = path1.path_function
    #     path_func2 = path2.path_function
    #     theta_values = np.linspace(0, 0.9, 1000)
    #     path_points1 = np.array(
    #         [
    #             path_func1(theta=theta, path_params=param_values)["path"].full().flatten()
    #             for theta in theta_values
    #         ]
    #     )
    #     path_points2 = np.array(
    #         [path_func2(theta=theta)["path"].full().flatten() for theta in theta_values]
    #     )

    #     path1.updateGates(self.new_gate_positions, self.gate_orientations)
    #     path2.updateGates(self.new_gate_positions, self.gate_orientations)
    #     param_values = path1.path_params_values
    #     path_func2 = path2.path_function
    #     path_points1_new = np.array(
    #         [
    #             path_func1(theta=theta, path_params=param_values)["path"].full().flatten()
    #             for theta in theta_values
    #         ]
    #     )
    #     path_points2_new = np.array(
    #         [path_func2(theta=theta)["path"].full().flatten() for theta in theta_values]
    #     )

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(path_points1[:, 0], path_points1[:, 1], path_points1[:, 2], label="Parametric Path")
    #     ax.plot(
    #         path_points2[:, 0], path_points2[:, 1], path_points2[:, 2], label="Nonparametric Path"
    #     )
    #     ax.plot(
    #         path_points1_new[:, 0],
    #         path_points1_new[:, 1],
    #         path_points1_new[:, 2],
    #         label="Parametric Path New",
    #     )
    #     ax.plot(
    #         path_points2_new[:, 0],
    #         path_points2_new[:, 1],
    #         path_points2_new[:, 2],
    #         label="Nonparametric Path New",
    #     )
    #     ax.scatter(
    #         self.gate_positions[:, 0],
    #         self.gate_positions[:, 1],
    #         self.gate_positions[:, 2],
    #         color="red",
    #         label="Gates",
    #     )
    #     ax.scatter(
    #         self.new_gate_positions[:, 0],
    #         self.new_gate_positions[:, 1],
    #         self.new_gate_positions[:, 2],
    #         color="green",
    #         label="New Gates",
    #     )
    #     ax.scatter(
    #         self.start_position[0],
    #         self.start_position[1],
    #         self.start_position[2],
    #         color="blue",
    #         label="Start",
    #     )
    #     ax.legend()
    #     plt.show()
    #     assert np.allclose(path_points1, path_points2)

    # def test_plot_path(self):
    #     path = HermiteSpline(
    #         self.start_position,
    #         self.start_orientation,
    #         self.gate_positions,
    #         self.gate_orientations,
    #         parametric=False,
    #     )
    #     waypoints = path.waypoints
    #     tangents = path.tangents
    #     # waypoints, tangents = path.fitPolynomial()
    #     path_func = path.path_function
    #     dpath_func = path.dpath_function
    #     # path_func, dpath_func = path.fitHermiteSpline()

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")

    #     # Evaluate the path at intervals
    #     theta_values = np.linspace(0, 0.9, 1000)

    #     path_points = np.array([path_func(theta).full().flatten() for theta in theta_values])
    #     dpath_points = np.array([dpath_func(theta).full().flatten() for theta in theta_values])

    #     ax.plot(
    #         path_points[:, 0], path_points[:, 1], path_points[:, 2], label="Hermite Spline Path"
    #     )
    #     ax.scatter(
    #         self.gate_positions[:, 0],
    #         self.gate_positions[:, 1],
    #         self.gate_positions[:, 2],
    #         color="red",
    #         label="Gates",
    #     )
    #     ax.scatter(
    #         self.start_position[0],
    #         self.start_position[1],
    #         self.start_position[2],
    #         color="blue",
    #         label="Start",
    #     )

    #     for i in range(len(self.gate_positions)):
    #         ax.quiver(
    #             waypoints[i, 0],
    #             waypoints[i, 1],
    #             waypoints[i, 2],
    #             tangents[i, 0],
    #             tangents[i, 1],
    #             tangents[i, 2],
    #             length=0.1,
    #             color="red",
    #             label="Gate Tangents" if i == 0 else "",
    #         )

    #     ax.legend()
    #     plt.show()


if __name__ == "__main__":
    unittest.main()
