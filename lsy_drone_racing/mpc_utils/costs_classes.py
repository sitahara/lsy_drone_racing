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
            "cost_type": "linear",  # "MPCC",
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
        elif self.cost_type == "MPCC":
            self.setupMPCCCosts()
        else:
            raise ValueError("Cost type not recognized.")

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

    def setupMPCCCosts(self):
        """Setup the cost function for the MPCCpp controller.
        We are using the EXTERNAL interface of acados to define the cost function.
        The cost function has 6 components:
        1. Lag error: The error between the current position and the desired position
        2. Contour error: The error between the current position and the desired contour
        3. Body angular velocity: The angular velocity of the body
        5. Thrust rate: The rate of change of the thrust
        4. Progress rate: The L2 norm rate of progress along the path
        6. Progress rate: The negative L1 rate of progress along the path.
        """
        cost_info = self.cost_info
        self.cost_type = "external"

        # Lag error weights
        Ql = cost_info.get("Ql", 1)
        self.Ql = ca.diag([Ql, Ql, Ql])
        # Contour error weights
        Qc = cost_info.get("Qc", 1)
        self.Qc = ca.diag([Qc, Qc, Qc])
        # Body Angular velocity weights
        Qw = cost_info.get("Qw", 1)
        self.Qw = ca.diag([Qw, Qw, Qw])
        # Progress rate weights
        self.Qmu = cost_info.get("Qmu", 1)
        # Thrust rate weights
        Rdf = cost_info.get("Rdf", 1)
        self.Rdf = ca.diag([Rdf, Rdf, Rdf, Rdf])
        # Progress rate weights
        self.Rdprogress = cost_info.get("Rdprogress", 1)
        self.stageCostFunc = self.MPCC_stage_cost
        self.terminalCostFunc = self.MPCC_terminalCost

    def get_tangent_vector(self, pd, theta):
        # Compute the tangent vector of the path at theta
        dpd_dtheta = ca.gradient(pd, theta)
        tangent = dpd_dtheta / ca.norm_2(dpd_dtheta)
        return tangent

    def MPCC_stage_cost(self, x, u, p):
        pos = x[self.dynamics.state_indices["pos"]]
        vel = x[self.dynamics.state_indices["vel"]]
        quat = x[self.dynamics.state_indices["quat"]]
        w = x[self.dynamics.state_indices["w"]]
        f = x[self.dynamics.state_indices["f"]]
        progress = x[self.dynamics.state_indices["progress"]]
        dprogress = x[self.dynamics.state_indices["dprogress"]]
        df = u[self.dynamics.control_indices["df"]]
        ddprogress = u[self.dynamics.control_indices["ddprogress"]]

        # Desired position and tangent vector on the path
        pd = p["pd"]
        tangent_line = self.get_tangent_vector(pd, progress)
        pos_err = pos - pd

        # Lag error
        lag_err = ca.dot(pos_err, tangent_line) * tangent_line
        lag_cost = ca.mtimes([lag_err.T, self.Ql, lag_err])

        # Contour error
        contour_err = pos_err - lag_err
        contour_cost = ca.mtimes([contour_err.T, self.Qc, contour_err])

        # Body angular velocity cost
        w_cost = ca.mtimes([w.T, self.Qw, w])

        # Progress rate cost
        dprogress_cost = -dprogress * self.qmu

        # Thrust rate cost
        thrust_rate_cost = ca.mtimes([df.T, self.Rdf, df])

        # Total stage cost
        stage_cost = lag_cost + contour_cost + w_cost + dprogress_cost + thrust_rate_cost

        return stage_cost

    # def setupMPCCCosts(self):
    #     """Setup the MPC costs."""
    #     cost_info = self.cost_info
    #     Qc = cost_info.get("Qc", np.array([1, 1, 10]))
    #     self.stageCostFunc = self.MPC_stageCost
    #     self.terminalCostFunc = self.MPC_terminalCost

    # def stageCost_MPCC(self, x, u, x_ref, u_ref):
    #     """Compute the MPC cost."""
    #     pos = x[self.dynamics.state_indices["pos"]]
    #     theta = x[self.dynamics.state_indices["theta"]]

    #     # Countouring cost
    #     contour_err = self.compute_contour_error(pos, theta)
    #     contour_cost = ca.mtimes([contour_err.T, self.Qc, contour_err])
    #     # Lag cost
    #     lag_err = self.compute_lag_error(pos, theta)
    #     lag_cost = ca.mtimes([lag_err.T, self.Ql, lag_err])
    #     # Body angular velocity cost
    #     w = x[self.dynamics.state_indices["w"]]
    #     ang_vel_cost = ca.mtimes([w.T, self.Qs_dang, w])
    #     # Control rate cost
    #     du = x[self.dynamics.state_indices["du"]]
    #     du_cost = ca.mtimes([du.T, self.R, du])
    #     # Progress derivative costs
    #     dprogress_cost_L2 = ca.mtimes([dtheta.T, self.Qp, dtheta])
    #     dprogress_cost_L1 = -progress_weight * dtheta

    #     return (
    #         lag_cost + contour_cost + ang_vel_cost + du_cost + dprogress_cost_L2 + dprogress_cost_L1
    #     )
