"""Module for Model Predictive Controller implementation."""

import casadi as ca
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as Rmat

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.mpc_utils import AcadosOptimizer, IPOPTOptimizer, BaseDynamics, BaseCost


class MPC(BaseController):
    """Model Predictive Controller implementation using do_mpc."""

    def __init__(
        self,
        initial_obs: NDArray[np.floating],
        initial_info: dict,
        additonal_info: dict = {
            "dynamics_info": {
                "Interface": "Mellinger",  # Mellinger or Thrust
                "ts": 1 / 60,  # Time step for the discrete dynamics, Used also for MPC
                "n_horizon": 60,  # Number of steps in the MPC horizon, used for external cost funtion: parameter definition
                "ControlType": "Thrusts",  # Thrusts, Torques, or MotorRPMs
                "BaseDynamics": "Quaternion",  # Euler or Quaternion
                "useAngVel": True,  # Currently only supports True, False if deul_ang is used instead of w
                "useControlRates": False,  # True if u is used as state and du as control
                "Constraints": {"Obstacles": True, "Gates": False},
                "usePredict": False,  # True if the dynamics are predicted into the future
                "t_predict": 0.05,  # in [s] To compensate control delay, the optimization problem is solved for the current state shifted by t_predict into the future with the current controls
            },
            "constraints_info": {
                "useObstacleConstraints": True,
                "obstacle_diameter": 0.1,
                "useGateConstraints": False,
                "useGoalConstraints": False,
            },
            "optimizer_info": {
                "useMellinger": True,
                "useSoftConstraints": True,
                "soft_penalty": 1e3,
                "useGP": False,
                "useZoro": False,
                "export_dir": "generated_code/mpc",
                "optimizer": "Acados",
            },
            "cost_info": {
                "cost_type": "linear",
                "Qs_pos": np.array([10, 10, 100]),
                "Qs_vel": np.array([0.1, 0.1, 0.1]),
                "Qs_ang": np.array([0.1, 0.1, 0.1]),
                "Qs_dang": np.array([0.1, 0.1, 0.1]),
                "Qs_quat": np.array([0.01, 0.01, 0.01, 0.01]),
                "Rs": np.array([0.01, 0.01, 0.01, 0.01]),
                "Rd": np.array([0.01, 0.01, 0.01, 0.01]),
                "Q_time": np.array([1]),
                "Q_gate": np.array([1, 1, 10, 0.1, 0.1, 0.1]),  # x, y, z, roll, pitch, yaw
            },
        },
    ):
        super().__init__(initial_obs, initial_info)
        self.initial_info = initial_info
        self.initial_obs = initial_obs
        self.additional_info = additonal_info
        self.ts = additonal_info["dynamics_info"]["ts"]
        self.n_horizon = additonal_info["dynamics_info"]["n_horizon"]
        # Init Dynamics including control bounds
        self.dynamics = BaseDynamics(initial_info, initial_obs, additonal_info["dynamics_info"])
        self.costs = BaseCost(self.dynamics, additonal_info["cost_info"])

        # # Init reference trajectory
        # if not additonal_info["dynamics_info"]["dynamics"] == "ThrustTime":
        #     self.x_ref = np.tile(
        #         self.dynamics.x_eq.reshape(self.dynamics.nx, 1), (1, self.n_horizon + 1)
        #     )
        #     self.u_ref = np.tile(
        #         self.dynamics.u_eq.reshape(self.dynamics.nu, 1), (1, self.n_horizon)
        #     )
        #     self.setupCostFunction(additonal_info["cost_info"])
        # else:
        #     self.setupCostTimeFunction(additonal_info["cost_info"])
        # Init Optimizer (acados needs also ipopt for initial guess, ipopt can be used standalone)
        self.ipopt = IPOPTOptimizer(
            dynamics=self.dynamics,
            costs=self.costs,
            optimizer_info=additonal_info["optimizer_info"],
        )
        if additonal_info["optimizer_info"]["optimizer"] == "Acados":
            self.opt = AcadosOptimizer(
                dynamics=self.dynamics,
                costs=self.costs,
                optimizer_info=additonal_info["optimizer_info"],
            )

        self.set_target_trajectory()
        if additonal_info["optimizer_info"]["optimizer"] == "Acados":
            self.obs = initial_obs
            self.current_state = np.concatenate(
                [initial_obs["pos"], initial_obs["vel"], initial_obs["rpy"], initial_obs["ang_vel"]]
            )
            self.updateTargetTrajectory()
            self.ipopt.step(self.current_state, self.costs.x_ref, self.costs.u_ref)
            self.opt.x_guess = self.ipopt.x_guess
            self.opt.u_guess = self.ipopt.u_guess
            self.n_step = 0

    def compute_control(
        self, obs: NDArray[np.floating], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The action either for the thrust or mellinger interface.
        """
        self.obs = obs
        self.current_state = np.concatenate([obs["pos"], obs["vel"], obs["rpy"], obs["ang_vel"]])
        # Updates x_ref, the current target trajectory and upcounts the trajectory tick
        self.updateTargetTrajectory()
        if self.opt is None:
            action = self.ipopt.step(self.current_state, self.costs.x_ref, self.costs.u_ref)
        else:
            action = self.opt.step(self.current_state, obs, self.costs.x_ref, self.costs.u_ref)

        print(f"Current position: {self.current_state[:3]}")
        print(f"Desired position: {self.costs.x_ref[:3, 1]}")
        print(f"Next position: {action[:3]}")

        return action.flatten()

    def set_target_trajectory(self, t_total: float = 8) -> None:
        """Set the target trajectory for the MPC controller."""
        self.n_step = 0  # current step for the target trajectory
        self.t_total = t_total
        waypoints = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.8, 0.5, 0.2],
                [0.55, -0.8, 0.4],
                [0.2, -1.8, 0.65],
                [1.1, -1.35, 1.0],
                [0.2, 0.0, 0.65],
                [0.0, 0.75, 0.525],
                [0.0, 0.75, 1.1],
                [-0.5, -0.5, 1.1],
                [-0.5, -1.0, 1.1],
            ]
        )
        # self.t_total = t_total
        t = np.linspace(0, t_total, len(waypoints))
        self.target_trajectory = CubicSpline(t, waypoints)
        # Generate points along the spline for visualization

        t_vis = np.linspace(0, t_total - 1, 100)
        spline_points = self.target_trajectory(t_vis)
        try:
            # Plot the spline as a line in PyBullet
            for i in range(len(spline_points) - 1):
                p.addUserDebugLine(
                    spline_points[i],
                    spline_points[i + 1],
                    lineColorRGB=[1, 0, 0],  # Red color
                    lineWidth=2,
                    lifeTime=0,  # 0 means the line persists indefinitely
                    physicsClientId=0,
                )
        except p.error:
            ...  # Ignore errors if PyBullet is not available
        return None

    def updateTargetTrajectory(self):
        """Update the target trajectory for the MPC controller."""
        current_time = self.n_step * self.ts
        t_horizon = np.linspace(
            current_time, current_time + self.n_horizon * self.ts, self.n_horizon + 1
        )

        # Evaluate the spline at the time points
        pos_des = self.target_trajectory(t_horizon).T
        # Handle the case where the end time exceeds the total time
        if t_horizon[-1] > self.t_total:
            last_value = self.target_trajectory(self.t_total).reshape(3, 1)
            n_repeat = np.sum(t_horizon > self.t_total)
            pos_des[:, -n_repeat:] = np.tile(last_value, (1, n_repeat))
        # print(reference_trajectory_horizon)
        self.costs.x_ref[:3, :] = pos_des
        self.n_step += 1
        return None

    # def setupCostFunction(self, cost_info: dict) -> None:
    #     """Setup the cost function for the MPC controller."""
    #     # Define the cost function parameters
    #     costs = {}
    #     costs["cost_type"] = cost_info["cost_type"]
    #     if self.dynamics.nx == 13:
    #         cost_info["Qs"] = np.concatenate(
    #             [
    #                 cost_info["Qs_pos"],
    #                 cost_info["Qs_vel"],
    #                 cost_info["Qs_quat"],
    #                 cost_info["Qs_dang"],
    #             ]
    #         )
    #         cost_info["Qt"] = np.concatenate(
    #             [
    #                 cost_info["Qt_pos"],
    #                 cost_info["Qt_vel"],
    #                 cost_info["Qt_quat"],
    #                 cost_info["Qt_dang"],
    #             ]
    #         )
    #     else:
    #         cost_info["Qs"] = np.concatenate(
    #             [
    #                 cost_info["Qs_pos"],
    #                 cost_info["Qs_vel"],
    #                 cost_info["Qs_ang"],
    #                 cost_info["Qs_dang"],
    #             ]
    #         )
    #         cost_info["Qt"] = np.concatenate(
    #             [
    #                 cost_info["Qt_pos"],
    #                 cost_info["Qt_vel"],
    #                 cost_info["Qt_ang"],
    #                 cost_info["Qt_dang"],
    #             ]
    #         )
    #     if self.dynamics.useControlRates:
    #         R = np.diag(cost_info["Rd"] / self.dynamics.u_scal)
    #         Qs = np.diag(np.concatenate([cost_info["Qs"], cost_info["Rs"]]) / self.dynamics.x_scal)
    #         Qt = np.diag(np.concatenate([cost_info["Qt"], cost_info["Rs"]]) / self.dynamics.x_scal)
    #     else:
    #         R = np.diag(cost_info["Rs"] / self.dynamics.u_scal)
    #         Qs = np.diag(cost_info["Qs"] / self.dynamics.x_scal)
    #         Qt = np.diag(cost_info["Qt"] / self.dynamics.x_scal)

    #     costs["R"] = R
    #     costs["Qs"] = Qs
    #     costs["Qt"] = Qt
    #     self.costs = costs
    #     self.costs["cost_function"] = self.baseCost  # cost function for the MPC

    # def baseCost(self, x, x_ref, u, u_ref):
    #     """Base Cost function for the MPC controller."""
    #     return ca.mtimes([(x - x_ref).T, self.costs["Qs"], (x - x_ref)]) + ca.mtimes(
    #         [(u - u_ref).T, self.costs["R"], (u - u_ref)]
    #     )

    # def setupCostTimeFunction(self, cost_info: dict) -> None:
    #     """Setup the cost function for the MPC controller."""
    #     # Define the cost function parameters
    #     costs = {}
    #     costs["cost_type"] = "External"
    #     costs["Qs_pos"] = np.diag(cost_info["Qs_pos"])
    #     costs["Qs_vel"] = np.diag(cost_info["Qs_vel"])
    #     costs["Qs_ang"] = np.diag(cost_info["Qs_ang"])
    #     costs["Qs_dang"] = np.diag(cost_info["Qs_dang"])
    #     costs["R"] = np.diag(cost_info["Rs"])
    #     costs["Q_time"] = cost_info["Q_time"]
    #     costs["Q_gate"] = cost_info["Q_gate"]
    #     self.costs = costs
    #     self.costs["cost_function"] = self.timeCost

    # def timeCost(self, x, u, p):
    #     vel = x[3:6]
    #     cost_vel = ca.mtimes([vel.T, self.costs["Qs_vel"], vel])

    #     ang = x[6:9]
    #     cost_ang = ca.mtimes([ang.T, self.costs["Qs_ang"], ang])

    #     dang = x[9:12]
    #     cost_dang = ca.mtimes([dang.T, self.costs["Qs_dang"], dang])

    #     t = x[self.dynamics.state_indices["time"]]
    #     cost_time = t * self.costs["Q_time"] * t

    #     cost_u = ca.mtimes([u.T, self.costs["R"], u])

    #     pos_rpy = ca.vertcat(x[:3], x[6:9])
    #     err = ca.if_else(
    #         x[self.dynamics.state_indices["gate_passed"]],
    #         ca.norm_2(pos_rpy - p[self.dynamics.param_indices["subsequent_gate"]]),
    #         ca.norm_2(pos_rpy - p[self.dynamics.param_indices["next_gate"]]),
    #     )
    #     cost_gate = ca.mtimes([err.T, self.costs["Q_gate"], err])

    #     return cost_gate + cost_vel + cost_ang + cost_dang + cost_u + cost_time
