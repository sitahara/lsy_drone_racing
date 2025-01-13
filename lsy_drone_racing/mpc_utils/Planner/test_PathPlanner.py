import pytest
import numpy as np
from lsy_drone_racing.mpc_utils.Planner.PathPlanner import PathPlanner
import matplotlib.pyplot as plt
import time

# FILE: lsy_drone_racing/mpc_utils/Planner/test_PathPlanner.py


class TestPathPlanner:
    def test_create_centerline(self):
        gate_positions = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]])
        gate_orientations = np.array([[0, 0, 0], [0, 0, np.pi / 4], [0, 0, np.pi / 2]])
        obstacles = []

        planner = PathPlanner(gate_positions, gate_orientations, obstacles)
        path, tangent_vectors = planner.create_centerline(gate_positions, gate_orientations)

        assert len(path) > len(gate_positions)
        assert len(tangent_vectors) > len(gate_orientations)
        assert np.allclose(path[0], gate_positions[0])
        assert np.allclose(path[-1], gate_positions[-1])
        assert np.allclose(tangent_vectors[0], planner.compute_gate_normals(gate_orientations)[0])
        assert np.allclose(tangent_vectors[-1], planner.compute_gate_normals(gate_orientations)[-1])

    def test_path_planning_with_obstacles(self):
        gates = [
            {"pos": [0.45, -1.0, 0.56], "rpy": [0.0, 0.0, 2.35]},
            {"pos": [1.0, -1.55, 1.11], "rpy": [0.0, 0.0, -0.78]},
            {"pos": [0.0, 0.5, 0.56], "rpy": [0.0, 0.0, 0.0]},
            {"pos": [-0.5, -0.5, 1.11], "rpy": [0.0, 0.0, 3.14]},
        ]

        obstacles = [
            {"pos": [1.0, -0.5, 1.4]},
            {"pos": [0.5, -1.5, 1.4]},
            {"pos": [0.0, 1.0, 1.4]},
            {"pos": [-0.5, 0.0, 1.4]},
        ]

        gate_positions = np.array([gate["pos"] for gate in gates])
        gate_orientations = np.array([gate["rpy"] for gate in gates])

        planner = PathPlanner(gate_positions, gate_orientations, obstacles)
        path, tangent_vectors = planner.create_centerline(gate_positions, gate_orientations)

        # Verify the path passes through the gates
        for i, gate in enumerate(gates):
            gate_pos = np.array(gate["pos"])
            assert np.allclose(path[i * 101], gate_pos), f"Path does not pass through gate {i + 1}"

        # Verify the path avoids obstacles
        for obstacle in obstacles:
            obs_pos = np.array(obstacle["pos"])
            for point in path:
                assert np.linalg.norm(point - obs_pos) > 0.1, "Path is too close to an obstacle"

        print("Path passes through all gates and avoids obstacles.")
        # Visualization
        self.visualize_path(gates, obstacles, path, title="Initial Path")

        # Update obstacles and gate positions
        new_obstacles = [
            {"pos": [1.0, -0.5, 1.4]},
            {"pos": [0.5, -1.5, 1.4]},
            {"pos": [0.0, 1.0, 1.4]},
            {"pos": [-0.5, 0.0, 1.4]},
            {"pos": [0.5, 0.0, 1.4]},  # New obstacle
        ]

        new_gate_positions = np.array(
            [
                [0.45, -1.0, 0.56],
                [1.0, -1.55, 1.11],
                [0.0, 0.5, 0.56],
                [-0.5, -0.5, 1.11],
                [0.5, 0.5, 0.56],
            ]
        )
        new_gate_orientations = np.array(
            [
                [0.0, 0.0, 2.35],
                [0.0, 0.0, -0.78],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 3.14],
                [0.0, 0.0, 1.57],
            ]
        )

        # Measure the time required for replanning
        start_time = time.time()
        planner.update_gate_positions(new_gate_positions, new_gate_orientations)
        planner.update_obstacles(new_obstacles)
        new_path, new_tangent_vectors = planner.create_centerline(
            new_gate_positions, new_gate_orientations
        )
        end_time = time.time()
        replanning_time = end_time - start_time

        print(f"Replanning time: {replanning_time:.4f} seconds")
        # Visualization
        self.visualize_path(
            gates,
            obstacles,
            path,
            new_gates=new_gate_positions,
            new_obstacles=new_obstacles,
            new_path=new_path,
            title="Updated Path",
        )

    def visualize_path(
        self,
        gates,
        obstacles,
        path,
        new_gates=None,
        new_obstacles=None,
        new_path=None,
        title="Path",
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot gates
        for gate in gates:
            pos = gate["pos"]
            ax.scatter(pos[0], pos[1], pos[2], c="g", marker="o", label="Gate")

        # Plot obstacles
        for obstacle in obstacles:
            pos = obstacle["pos"]
            ax.scatter(pos[0], pos[1], pos[2], c="r", marker="x", label="Obstacle")

        # Plot path
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c="b", label="Path")

        if new_gates is not None:
            # Plot new gates
            for pos in new_gates:
                ax.scatter(pos[0], pos[1], pos[2], c="c", marker="o", label="New Gate")

        if new_obstacles is not None:
            # Plot new obstacles
            for obstacle in new_obstacles:
                pos = obstacle["pos"]
                ax.scatter(pos[0], pos[1], pos[2], c="m", marker="x", label="New Obstacle")

        if new_path is not None:
            # Plot new path
            new_path = np.array(new_path)
            ax.plot(new_path[:, 0], new_path[:, 1], new_path[:, 2], c="y", label="New Path")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        plt.legend()
        plt.show()
