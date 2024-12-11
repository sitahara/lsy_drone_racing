"""Module for Model Predictive Controller implementation using do_mpc."""

import casadi as ca
import do_mpc
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import BaseController
from lsy_drone_racing.control.mpc_utils import (
    W1,
    R_body_to_inertial_symb,
    W1_symb,
    W2_dot_symb,
    W2_symb,
    rpm_to_torques_mat,
)


class MPCController(BaseController):
    """Model Predictive Controller implementation using do_mpc."""

    def __init__(self, initial_obs: NDArray[np.floating], initial_info: dict):  # noqa: D107
        super().__init__(initial_obs, initial_info)
        self._tick = 0
        self.t_step = 1 / initial_info["env_freq"]  # Time step
        self.n_horizon = 20  # Prediction horizon
        # Define the system parameters
        self.mass = initial_info["drone_mass"]  # kg
        self.g = 9.81  # m/s^2
        self.ct = 3.1582e-10  # N/RPM^2, lift coefficient
        self.cd = 7.9379e-12  # N/RPM^2, drag coefficient
        arm_length = 0.046  # m, arm length
        self.c_tau_xy = arm_length * self.ct / np.sqrt(2)  # torque coefficient

        self.rpmToTorqueMat = rpm_to_torques_mat(self.c_tau_xy, self.cd)  # Torque matrix

        self.J = np.diag(
            [1.395e-5, 1.436e-5, 2.173e-5]
        )  # kg*m^2 , diag(I_xx,I_yy,I_zz), moment of inertia
        self.J_inv = np.linalg.inv(self.J)  # inverse of moment of inertia

        # Initialize the MPC model
        self.model = do_mpc.model.Model("continuous")

        # Define state variables
        pos = self.model.set_variable(
            var_type="_x", var_name="pos", shape=(3, 1)
        )  # x, y, z, linear position
        vel = self.model.set_variable(
            var_type="_x", var_name="vel", shape=(3, 1)
        )  # dx, dy, dz, linear velocity
        eul_ang = self.model.set_variable(
            var_type="_x", var_name="eul_ang", shape=(3, 1)
        )  # euler angles roll, pitch, yaw
        deul_ang = self.model.set_variable(
            var_type="_x", var_name="deul_ang", shape=(3, 1)
        )  # droll, dpitch, dyaw in eul_ang

        # Define Control variables
        # Note: The control variables are the collective thrust and body torques [thrust, tau_x, tau_y, tau_z].
        thrust = self.model.set_variable(
            var_type="_u", var_name="thrust", shape=(1, 1)
        )  # torque of each rotor
        torques = self.model.set_variable(
            var_type="_u", var_name="torques", shape=(3, 1)
        )  # torque of each rotor

        # Note: torques = rpm_to_torques_mat @ rpm
        # Note: thrust = ct * np.sum(rpm)

        # Define Expressions for the dynamics
        w = self.model.set_expression(
            "dang_body", W1_symb(eul_ang) @ deul_ang
        )  # Body Angular velocity

        # Define Dynamics in world frame as euler angles
        self.model.set_rhs("pos", vel)  # dpos = vel
        self.model.set_rhs(
            "vel",
            ca.vertcat(0, 0, -self.g)
            + R_body_to_inertial_symb(eul_ang) @ ca.vertcat(0, 0, thrust / self.mass),
        )

        self.model.set_rhs("eul_ang", deul_ang)  # deul_ang = deul_ang
        self.model.set_rhs(
            "deul_ang",
            W2_dot_symb(eul_ang, deul_ang) @ w
            + W2_symb(eul_ang) @ (self.J_inv @ (ca.cross(self.J @ w, w) + torques)),
        )

        # Define variables and expressions needed for the objective function
        pos_des = self.model.set_variable(var_type="_tvp", var_name="pos_des", shape=(3, 1))
        pos_err = pos - pos_des

        # Stage cost (Lagrange term): position error and euler angle velocity is penalized
        self.model.set_expression(
            "state_cost_l",
            pos_err.T @ ca.diag([1, 1, 1]) @ pos_err
            + deul_ang.T @ ca.diag([0.01, 0.01, 0.01]) @ deul_ang,
        )
        # Terminal cost (Mayer term): position error and euler angle velocity is penalized
        self.model.set_expression(
            "state_cost_m",
            pos_err.T @ ca.diag([1, 1, 1]) @ pos_err
            + deul_ang.T @ ca.diag([0.01, 0.01, 0.01]) @ deul_ang,
        )
        # Setup the model (hereafter, the expressions and variables are fixed)
        self.model.setup()

        # Initialize the MPC controller
        self.mpc = do_mpc.controller.MPC(self.model)
        # Set MPC parameters
        setup_mpc = {
            "n_horizon": self.n_horizon,
            "n_robust": 0,
            "t_step": self.t_step,
            "state_discretization": "collocation",
            "collocation_type": "radau",
            "collocation_deg": 2,  # optimize the degree of the collocation
            "collocation_ni": 2,  # optimize the number of the collocation
            "store_full_solution": True,
            # "open_loop": False,
            "nlpsol_opts": {
                # "ipopt.linear_solver": "ma27",
                "ipopt.max_iter": 20,
                "ipopt.tol": 1e-7,
                "ipopt.print_level": 0,  # Suppress IPOPT output
                "print_time": 0,  # Suppress IPOPT timing output
            },
        }
        self.mpc.set_param(**setup_mpc)

        # Define the constraints and scaling
        # Rotor rates, these are the base of the constraints
        rpm_min = 4070.3**2  # lower bound for rotor rates
        rpm_max = (4070.3 + 0.2685 * 65535) ** 2  # upper bound for rotor rates

        # thrust
        thrust_lb = 0.4 * self.mass * self.g  # lower bound for thrust to avoid tumbling
        thrust_ub = self.ct * 4 * rpm_max  # upper bound for thrust
        self.mpc.scaling["_u", "thrust"] = thrust_ub - thrust_lb
        self.mpc.bounds["lower", "_u", "thrust"] = thrust_lb
        self.mpc.bounds["upper", "_u", "thrust"] = thrust_ub

        # torques (note: removed factor two from the torque constraints)
        torque_xy_lb = -self.c_tau_xy * (
            rpm_max - rpm_min
        )  # lower bound for torque in x and y direction
        torque_xy_ub = self.c_tau_xy * (rpm_max - rpm_min)
        torque_z_lb = -self.cd * (rpm_max - rpm_min)  # lower bound for torque in z direction
        torque_z_ub = self.cd * (rpm_max - rpm_min)
        # Note that the bounds needs to be optimized so that the interdependency of the torques and thrust is considered
        self.mpc.scaling["_u", "torques"] = np.array(
            [torque_xy_ub - torque_xy_lb, torque_xy_ub - torque_xy_lb, torque_z_ub - torque_z_lb]
        ).T
        self.mpc.bounds["lower", "_u", "torques"] = np.array(
            [torque_xy_lb, torque_xy_lb, torque_z_lb]
        ).T
        self.mpc.bounds["upper", "_u", "torques"] = np.array(
            [torque_xy_ub, torque_xy_ub, torque_z_ub]
        ).T
        # States
        x_y_max = 2
        z_max = 2
        self.mpc.scaling["_x", "pos"] = np.array([2 * x_y_max, 2 * x_y_max, z_max]).T
        self.mpc.bounds["lower", "_x", "pos"] = np.array([-x_y_max, -x_y_max, 0.1]).T
        self.mpc.bounds["upper", "_x", "pos"] = np.array([x_y_max, x_y_max, z_max]).T

        vel_scale = 10
        self.mpc.scaling["_x", "vel"] = vel_scale

        eul_ang_scale = np.pi
        self.mpc.scaling["_x", "eul_ang"] = eul_ang_scale

        deul_ang_scale = 10
        self.mpc.scaling["_x", "deul_ang"] = deul_ang_scale

        # Define the objective function
        self.mpc.set_objective(
            mterm=self.model.aux["state_cost_m"], lterm=self.model.aux["state_cost_l"]
        )
        # Note: set_rterm peniallizes the control input changes
        # Note: not influenced by scaling, hence scale manually
        self.mpc.set_rterm(thrust=0.01, torques=0.01)  # Control regularization

        # Set the initial guess for the control variables
        initial_control = np.array([0, 0.0, 0.0, 0.0])  # Example: [thrust, tau_x, tau_y, tau_z]
        self.mpc.u0 = initial_control

        # Set the target trajectory
        self.set_target_trajectory()
        self.mpc.set_tvp_fun(self.updateTargetTrajectory)

        # Setup the MPC
        self.mpc.setup()
        # Set the initial guess for the state variables
        self.mpc.x0["pos"] = initial_obs["pos"]
        self.mpc.x0["vel"] = initial_obs["vel"]
        self.mpc.x0["eul_ang"] = initial_obs["rpy"]
        self.mpc.x0["deul_ang"] = initial_obs["ang_vel"]

        print("initial_obs: ", self.mpc.x0["pos"])

        self.mpc.set_initial_guess()

    def compute_control(
        self, obs: NDArray[np.floating], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [thrust, tau_des] as a numpy array.
        """
        # Get current observations
        pos = obs["pos"]  # position in world frame
        vel = obs["vel"]  # velocity in world frame
        eul_ang = obs["rpy"]  # euler angles roll, pitch, yaw
        deul_ang = obs["ang_vel"]  # angular velocity in body frame
        current_state = np.concatenate([pos, vel, eul_ang, deul_ang])
        # Get the control input from the MPC

        u = self.mpc.make_step(current_state)

        print("current time:", self.mpc.t0)

        next_poss = self.mpc.data.prediction(("_x", "pos"))[:, :, 0]
        for k in range(self.n_horizon):
            p.addUserDebugLine(
                next_poss[:, k],
                next_poss[:, k + 1],
                lineColorRGB=[0, 0, 1],  # Blue color
                lineWidth=2,
                lifeTime=self.t_step * 4,  # 0 means the line persists indefinitely
                physicsClientId=0,
            )

        # Extract the next predicted states from the MPC
        next_pos = self.mpc.data.prediction(("_x", "pos"))[:, 1, 0]
        print("next_pos: ", next_pos)
        next_vel = self.mpc.data.prediction(("_x", "vel"))[:, 1, 0]
        acc = (
            self.mpc.data.prediction(("_x", "vel"))[:, 1, 0]
            - self.mpc.data.prediction(("_x", "vel"))[:, 0, 0]
        ) / self.t_step
        next_eul_ang = self.mpc.data.prediction(("_x", "eul_ang"))[:, 1, 0]
        next_deul_ang = self.mpc.data.prediction(("_x", "deul_ang"))[:, 1, 0]
        # u = np.array([self.ct * np.sum(rpm), self.rpmToTorqueMat @ rpm])

        # action: Full-state command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] to follow.
        # where ax, ay, az are the acceleration in world frame, rrate, prate, yrate are the roll, pitch, yaw rate in body frame

        rpy_rate = W1(next_eul_ang) @ next_deul_ang  # convert euler angle rate to body frame
        action = np.concatenate([next_pos, next_vel, acc, [next_eul_ang[2]], rpy_rate.flatten()])
        return action.flatten()

    def updateTargetTrajectory(self, t_now):
        """Time-varying parameter function to update pos_des."""
        # print("t_now: ", t_now)
        tvp_template = self.mpc.get_tvp_template()
        for k in range(self.n_horizon + 1):
            t_now_k = min((t_now + (k + 1) * self.t_step), self.t_total)
            # print("t_now_k: ", t_now_k)
            tvp_template["_tvp", k, "pos_des"] = self.target_trajectory(t_now_k)
            # Plot the spline as a line in PyBullet
            # if k < self.n_horizon:
            #     start_point = self.target_trajectory(t_now_k)
            #     end_point = self.target_trajectory(t_now_k + 1)
            #     p.addUserDebugLine(
            #         start_point.flatten(),
            #         end_point.flatten(),
            #         lineColorRGB=[0, 0, 1],  # Blue color
            #         lineWidth=2,
            #         lifeTime=self.t_step * 2,  # 0 means the line persists indefinitely
            #         physicsClientId=0,
            #     )

        return tvp_template

    def set_target_trajectory(self, t_total=9):
        """Set the target trajectory for the MPC controller."""
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
        self.t_total = t_total
        t = np.linspace(0, self.t_total, len(waypoints))
        self.target_trajectory = CubicSpline(t, waypoints)

        # Generate points along the spline for visualization
        t_vis = np.linspace(0, self.t_total - 1, 100)
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
