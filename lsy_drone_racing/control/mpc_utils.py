"""This module contains utility functions for model predictive control (MPC) in drone racing."""

import casadi as ca
import numpy as np


def R_body_to_inertial_symb(rpy: ca.SX) -> ca.SX:
    """Compute the rotation matrix from body frame to inertial frame.

    Args:
        rpy (ca.SX): Roll, pitch, and yaw angles.

    Returns:
        ca.SX: Rotation matrix from body to inertial frame.
    """
    phi, theta, psi = rpy[0], rpy[1], rpy[2]
    Rm = ca.vertcat(
        ca.horzcat(ca.cos(theta) * ca.cos(psi), ca.cos(theta) * ca.sin(psi), -ca.sin(theta)),
        ca.horzcat(
            ca.sin(phi) * ca.sin(theta) * ca.cos(psi) - ca.cos(phi) * ca.sin(psi),
            ca.sin(phi) * ca.sin(theta) * ca.sin(psi) + ca.cos(phi) * ca.cos(psi),
            ca.sin(phi) * ca.cos(theta),
        ),
        ca.horzcat(
            ca.cos(phi) * ca.sin(theta) * ca.cos(psi) + ca.sin(phi) * ca.sin(psi),
            ca.cos(phi) * ca.sin(theta) * ca.sin(psi) - ca.sin(phi) * ca.cos(psi),
            ca.cos(phi) * ca.cos(theta),
        ),
    )
    return Rm.T  # Transpose the matrix to convert from body to inertial frame


def rpm_to_thrust(ct, rpm1, rpm2, rpm3, rpm4):
    """Compute the thrust from the rotor rates.

    Args:
        ct (float): Thrust coefficient.
        rpm1 (float): Rotor rate 1.
        rpm2 (float): Rotor rate 2.
        rpm3 (float): Rotor rate 3.
        rpm4 (float): Rotor rate 4.

    Returns:
        float: Total thrust.
    """
    return ct * (rpm1 + rpm2 + rpm3 + rpm4)


def rpm_to_torques_mat(c_tau_xy, cd):
    """Creates a matrix to compute the torques in the body frame from the rotor rates."""

    return np.array(
        [
            [-c_tau_xy, -c_tau_xy, c_tau_xy, c_tau_xy],
            [-c_tau_xy, c_tau_xy, c_tau_xy, -c_tau_xy],
            [-cd, cd, -cd, cd],
        ]
    )


def W1_symb(rpy: ca.SX) -> ca.SX:
    """Compute the W1 matrix from euler angles using casadi symbolic."""
    phi, theta = rpy[0], rpy[1]
    W1_matrix = ca.vertcat(
        ca.horzcat(1, 0, -ca.sin(theta)),
        ca.horzcat(0, ca.cos(phi), ca.sin(phi) * ca.cos(theta)),
        ca.horzcat(0, -ca.sin(phi), ca.cos(phi) * ca.cos(theta)),
    )
    return W1_matrix


def W1(eul_ang):
    """Compute the W1 matrix from euler angles."""
    phi, theta = eul_ang[0], eul_ang[1]
    W1_matrix = np.array(
        [
            [1, 0, -np.sin(theta)],
            [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
            [0, -np.sin(phi), np.cos(phi) * np.cos(theta)],
        ]
    )
    return W1_matrix


def W2_dot_symb(phi: ca.SX, dphi: ca.SX) -> ca.SX:
    """Compute the time derivative of the W2 matrix."""
    return W2_symb(phi) @ ca.skew(dphi)


def W2_symb(rpy: ca.SX) -> ca.SX:
    """Compute the W2 matrix from euler angles."""
    phi, theta = rpy[0], rpy[1]
    W2_matrix = ca.vertcat(
        ca.horzcat(1, ca.sin(phi) * ca.tan(theta), ca.cos(phi) * ca.tan(theta)),
        ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
        ca.horzcat(0, ca.sin(phi) / ca.cos(theta), ca.cos(phi) / ca.cos(theta)),
    )
    return W2_matrix
