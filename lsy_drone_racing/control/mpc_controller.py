from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

from lsy_drone_racing.control import BaseController

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

import do_mpc
#from scipy.spatial.transform import Rotation as R
#from casadi import vertcat, horzcat, sin, cos


class MPCController(BaseController):
    """Model Predictive Controller implementation using do_mpc."""

    def __init__(self, initial_obs: NDArray[np.floating], initial_info: dict):
        super().__init__(initial_obs, initial_info)


        # Initialize the do_mpc model
        self.model = do_mpc.model.Model('continuous')
        
        # Define state variables
        pos = self.model.set_variable(var_type='_x', var_name='pos', shape=(3, 1)) # x, y, z
        vel = self.model.set_variable(var_type='_x', var_name='vel', shape=(3, 1)) # dx, dy, dz
        phi = self.model.set_variable(var_type='_x', var_name='phi', shape=(3, 1)) # roll, pitch, yaw
        dphi = self.model.set_variable(var_type='_x', var_name='phi', shape=(3, 1)) # droll, dpitch, dyaw

        # Define expressions
        w = self.model.set_expression('w', self.W1(phi) @ dphi)
        Rb = self.model.set_expression('Rb', self.R_body_to_inertial(phi))
        W2 = self.model.set_expression('W2', self.W2(phi))
        dW2 = self.model.set_expression('dW2', self.W2_dot(phi, dphi))

        # Define control variables
        torques = self.model.set_variable(var_type='_u', var_name='torques', shape=(3, 1)) # tau_x, tau_y, tau_z
        thrust = self.model.set_variable(var_type='_u', var_name='thrust', shape=(1, 1)) # F

        # Define TVP variables
        target_pos = self.model.set_variable(var_type='_tvp', var_name='target_pos', shape=(3, 1))
        target_phi = self.model.set_variable(var_type='_tvp', var_name='target_phi', shape=(3, 1))
        
        # Define parameters
        mass = 0.27  # kg
        g = 9.81  # m/s^2
        vg = np.array([0, 0, -g]).T
         # arm_length = 0.042 # m
        J = np.diag([1.395e-5, 1.436e-5, 2.173e-5]) # kg*m^2 , diag(J_xx,J_yy,J_zz) 
        J_inv = np.linalg.inv(J)
        

        # Define Dynamics
        self.model.set_rhs('Rb', Rb @ self.S(w))
        self.model.set_rhs('pos', vel)
        self.model.set_rhs('vel', vg + Rb @ np.array([0, 0, thrust]).T / mass)
        self.model.set_rhs('phi', dphi)
        self.model.set_rhs('dphi', dW2 @ w + W2 @ (J_inv @ (np.cross((J @ w), w) + torques)))
        
        self.model.setup()
        
        # Initialize the MPC controller
        self.mpc = do_mpc.controller.MPC(self.model)
        
        # Set MPC parameters
        setup_mpc = {
            'n_horizon': 20,
            'n_robust': 5,
            'c_horizon': 3,
            't_step': 0.04,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 2,
            'store_full_solution': False,
            'open_loop': False,
        }
        self.mpc.set_param(**setup_mpc)

        # TODO: Adjust the scaling factors, soft- and hard- constraints
        # Scale the variables

        self.mpc.scaling['_x', 'pos'] = 1
        self.mpc.scaling['_x', 'vel'] = 1
        self.mpc.scaling['_x', 'phi'] = 10
        self.mpc.scaling['_x', 'dphi'] = 100

        self.mpc.scaling['_u', 'torques'] = 100
        self.mpc.scaling['_u', 'thrust'] = 100

        # Set the constraints
        # self.mpc.bounds['lower', '_u', 'torques'] = -1
        # self.mpc.bounds['upper', '_u', 'torques'] = 1
        # self.mpc.bounds['lower', '_u', 'thrust'] = 0
        # self.mpc.bounds['upper', '_u', 'thrust'] = 1

        # self.mpc.bounds['lower', '_x', 'pos'] = -10
        # self.mpc.bounds['upper', '_x', 'pos'] = 10

        # Soft constraints
        # self.mpc.set_nl_cons()

        
        # Define the objective functions
        # State costs, TODO: tune the weights, add mterm
        self.mpc.set_objective(mterm=None, lterm=self.objective_function(pos, target_pos, phi, target_phi, thrust, torques))
        # Actuation costs
        self.mpc.set_rterm(torques=1e-2, thrust=1e-2)


        



        # Time-varying reference trajectory
        self.target_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.get_target_trajectory)

        
        self.mpc.setup()
        

    def R_body_to_inertial(self, rpy):
        """Compute the rotation matrix from roll, pitch, yaw angles."""
        phi, theta, psi = rpy[0], rpy[1], rpy[2]
        R = np.array([
            [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
            [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.sin(phi) * np.cos(theta)],
            [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi), np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), np.cos(phi) * np.cos(theta)]
          ])
        #R_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        return R.T # Transpose the matrix to convert from body to inertial frame
    
    def W1(self, rpy):
        """Compute the W1 matrix from euler angles."""
        phi, theta, psi = rpy[0], rpy[1], rpy[2]
        W1_matrix = np.array([
            [1, 0, -np.sin(theta)],
            [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
            [0, -np.sin(phi), np.cos(phi) * np.cos(theta)]
        ])
        return W1_matrix
    def W2(self, rpy):
        """Compute the W2 matrix from euler angles."""
        phi, theta, psi = rpy[0], rpy[1], rpy[2]
        W2_matrix = np.array([ 1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta),
                               0, np.cos(phi), -np.sin(phi),
                               0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)])
        return W2_matrix
    def W2_dot(self, rpy, rpy_dot):
        """Compute the time derivative of the W2 matrix."""
        phi, theta, psi = rpy[0], rpy[1], rpy[2]
        phi_dot, theta_dot, psi_dot = rpy_dot[0], rpy_dot[1], rpy_dot[2]
        
        W2_dot_matrix = np.array([
            [0, np.cos(phi) * np.tan(theta) * phi_dot + np.sin(phi) * (1 / np.cos(theta))**2 * theta_dot, -np.sin(phi) * np.tan(theta) * phi_dot + np.cos(phi) * (1 / np.cos(theta))**2 * theta_dot],
            [0, -np.sin(phi) * phi_dot, -np.cos(phi) * phi_dot],
            [0, np.cos(phi) / np.cos(theta) * phi_dot + np.sin(phi) * np.sin(theta) / (np.cos(theta))**2 * theta_dot, -np.sin(phi) / np.cos(theta) * phi_dot + np.cos(phi) * np.sin(theta) / (np.cos(theta))**2 * theta_dot]
        ])
        return W2_dot_matrix
    
    def S(vector):
        """Calculate the skew-symmetric matrix from a vector."""
        assert vector.shape == (3,), "Input vector must be a 3-element vector."      
        x, y, z = vector
        skew_matrix = np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])
        
        return skew_matrix
    def Rdes(self, roll_des, f_des):
        """Compute the desired attitude matrix."""
        "args: phi_des: desired roll angle, f_des:  translational acceleration"
        "returns: desired attitude matrix"
        z_des = f_des / np.linalg.norm(f_des)
        n_des = np.array([np.cos(roll_des), np.sin(roll_des), 0]).T
        y_des = np.cross(z_des, n_des) / np.linalg.norm(np.cross(z_des, n_des))
        x_des = np.cross(y_des, z_des)
        return np.array([x_des, y_des, z_des])
    
    def Fdes(self,fdes,z):
        """Compute the desired collective thrust."""
        "args: fdes: desired translational acceleration , z: current body z axis"
        return fdes.T @ z
    
    def objective_function(self, pos, target_pos, phi, target_phi, thrust, torques):
        """Define the objective function for the MPC."""
        """args: pos: current position, target_pos: desired position, 
                 phi: current orientation, target_phi: desired orientation, 
                 thrust: current total thrust, torques: current torques"""
        #
        thrust_des = 0.27*9.81
        # Weights
        Q_pos = np.diag([10, 10, 10])  # Position weight
        Q_ori = np.diag([1, 1, 1])     # Orientation weight

        # Position Control
        cost_pos = (pos - target_pos).T @Q_pos @ (pos - target_pos)

        # Attitude Control Cost
        R_des = self.Rdes(target_phi[0], thrust_des)
        R_body = self.R_body_to_inertial(phi)
        orientation_err = 0.5 * (R_des @ R_body.T - R_body @ R_des.T)
        orientation_err = np.array([orientation_err[2, 1], orientation_err[0, 2], orientation_err[1, 0]])
        cost_orientation = orientation_err.T @ Q_ori @ orientation_err

        return cost_pos + cost_orientation 

    def compute_control(
        self, obs: NDArray[np.floating], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone."""
        
        # Update the state of the simulator
        #self.simulator.x0 = obs
        #self.update_target_trajectory(self.get_target_trajectory)

        # Make a prediction
        u0 = self.mpc.make_step(obs)
        
        return u0
    
    # def update_target_trajectory(self, get_target_trajectory: Callable[[float], tuple[NDArray[np.floating], NDArray[np.floating]]]):
    #     """Update the target trajectory in the MPC."""
    #     def tvp_fun(t_now):
    #         tvp_template = self.mpc.get_tvp_template()
    #         for k in range(self.mpc.n_horizon):
    #             t = t_now + k * self.mpc.t_step
    #             target_pos, target_rpy = get_target_trajectory(t)
    #             tvp_template['_tvp', k, 'target_pos'] = target_pos
    #             tvp_template['_tvp', k, 'target_rpy'] = target_rpy
    #         return tvp_template
        
    #     self.mpc.set_tvp_fun(tvp_fun)

    def get_target_trajectory(self,t: float):
        # Example implementation: return the target position and orientation for the given time
        self.target_template['_tvp', 'target_pos'] = np.array([[1.0], [2.0], [3.0]])
        self.target_template['_tvp', 'target_phi'] = np.array([[0.1], [0.2], [0.3]])
        return self.target_template
    