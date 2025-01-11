from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import TYPE_CHECKING
from lsy_drone_racing.mpc_utils.dynamics_classes import BaseDynamics
import numpy as np
import casadi as ca


class BaseCost:
    def __init__(
        self,
        dynamics: BaseDynamics,
        cost_info: dict = {
            "cost_type": "linear",
            "Qs_pos": np.array([1, 1, 10]),
            "Qs_vel": np.array([0.1, 0.1, 0.1]),
            "Qs_ang": np.array([0.1, 0.1, 0.1]),
            "Qs_dang": np.array([0.1, 0.1, 0.1]),
            "Qs_quat": np.array([0.01, 0.01, 0.01, 0.01]),
            "Rs": np.array([0.01, 0.01, 0.01, 0.01]),
            "Rd": np.array([0.01, 0.01, 0.01, 0.01]),
            "Q_time": np.array([1]),
            "Q_gate": np.array([1, 1, 10, 0.1, 0.1, 0.1]),  # x, y, z, roll, pitch, yaw
        },
    ):
        self.dynamics = dynamics
        self.cost_info = cost_info
        self.cost_type = cost_info.get("cost_type", "linear")

        if self.cost_type == "linear":
            self.setupLinearCosts()
        elif self.cost_type == "external":
            self.setupExternalCosts()

    def setupLinearCosts(self):
        """Setup the linear (Quadratic) costs of form: (sum_i ||x_i-x_ref_i||_{Qs}^2 + ||u_i-u_ref_i||_{R}^2) + ||x_N-x_ref_N||_{Qt}^2."""
        Qs_pos = self.cost_info.get("Qs_pos", np.array([1, 1, 10]))
        Qs_vel = self.cost_info.get("Qs_vel", np.array([0.1, 0.1, 0.1]))
        Qs_ang = self.cost_info.get("Qs_ang", np.array([0.1, 0.1, 0.1]))
        Qs_dang = self.cost_info.get("Qs_dang", np.array([0.1, 0.1, 0.1]))
        Qs_quat = self.cost_info.get("Qs_quat", np.array([0.01, 0.01, 0.01, 0.01]))
        Qt_pos = self.cost_info.get("Qt_pos", Qs_pos)
        Qt_vel = self.cost_info.get("Qt_vel", Qs_vel)
        Qt_ang = self.cost_info.get("Qt_ang", Qs_ang)
        Qt_dang = self.cost_info.get("Qt_dang", Qs_dang)
        Qt_quat = self.cost_info.get("Qt_quat", Qs_quat)

        Rs = self.cost_info.get("Rs", np.array([0.01, 0.01, 0.01, 0.01]))
        Rd = self.cost_info.get("Rd", np.array([0.01, 0.01, 0.01, 0.01]))
        if self.dynamics.baseDynamics == "Euler":
            Qs = np.concatenate([Qs_pos, Qs_vel, Qs_ang, Qs_dang])
            Qt = np.concatenate([Qt_pos, Qt_vel, Qt_ang, Qt_dang])
        elif self.dynamics.baseDynamics == "Quaternion":
            Qs = np.concatenate([Qs_pos, Qs_vel, Qs_quat, Qs_dang])
            Qt = np.concatenate([Qt_pos, Qt_vel, Qt_quat, Qt_dang])
        else:
            raise ValueError("Base dynamics not recognized.")

        if self.dynamics.useControlRates:
            Qs = np.concatenate([Qs, Rs])
            Qt = np.concatenate([Qt, Rs])
            R = Rd
        else:
            R = Rs

        self.Qs = np.diag(Qs)
        self.Qt = np.diag(Qt)
        self.R = np.diag(R)
        self.x_ref = np.tile(
            self.dynamics.x_eq.reshape(self.dynamics.nx, 1), self.dynamics.n_horizon + 1
        )
        print(self.x_ref.shape)
        self.u_ref = np.tile(
            self.dynamics.u_eq.reshape(self.dynamics.nu, 1), self.dynamics.n_horizon
        )
        print(self.u_ref.shape)
        self.stageCostFunc = self.LQ_stageCost
        self.terminalCostFunc = self.LQ_terminalCost

    def LQ_stageCost(self, x, u, x_ref, u_ref):
        """Compute the LQR cost."""
        return ca.mtimes([(x - x_ref).T, self.Qs, x - x_ref]) + ca.mtimes(
            [(u - u_ref).T, self.R, u - u_ref]
        )

    def LQ_terminalCost(self, x, x_ref):
        """Compute the LQR cost."""
        return ca.mtimes([(x - x_ref).T, self.Qt, x - x_ref])
