"""This module contains utility functions for model predictive control (MPC) in drone racing."""

from typing import Union

import casadi as ca
import numpy as np


def shuffleQuat(q: ca.MX) -> ca.MX:
    """Shuffle the quaternion from [qw, qx, qy, qz ] to [qx, qy, qz,qw]."""
    return ca.vertcat(q[1], q[2], q[3], q[0])


def quaternion_conjugate(q: ca.MX) -> ca.MX:
    """Compute the conjugate of a quaternion [qx,qy,qz,qw]."""
    return ca.vertcat(-q[0], -q[1], -q[2], q[3])


def quaternion_rotation(q: ca.MX, v: ca.MX) -> ca.MX:
    """Rotate a vector by a quaternion [qx,qy,qz,qw]."""
    t = 2 * ca.cross(q[:-1], v)
    return v + q[-1] * t + ca.cross(q[:-1], t)


def quaternion_product(q: ca.MX, r: ca.MX) -> ca.MX:
    """Compute the product of two quaternions."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    rx, ry, rz, rw = r[0], r[1], r[2], r[3]
    return ca.vertcat(
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
        qw * rw - qx * rx - qy * ry - qz * rz,
    )


def quaternion_to_rotation_matrix(q: ca.MX) -> ca.MX:
    """Convert a quaternion [qx, qy, qz, qw] to a rotation matrix."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    R = ca.MX(3, 3)
    R[0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R


def quaternion_to_euler(quat: ca.MX) -> ca.MX:
    """Convert a quaternion [qx, qy, qz, qw] to Euler angles [roll, pitch, yaw]."""
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = ca.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = ca.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = ca.atan2(siny_cosp, cosy_cosp)

    return ca.vertcat(roll, pitch, yaw)


def R_body_to_inertial(rpy: Union[np.ndarray, ca.SX]) -> Union[np.ndarray, ca.SX]:
    """Compute the rotation matrix from body frame to inertial frame.

    Args:
        rpy (Union[np.ndarray, ca.SX]): Roll, pitch, yaw angles.

    Returns:
        Union[np.ndarray, ca.SX]: Rotation matrix from body frame to inertial frame.
    """
    phi, theta, psi = rpy[0], rpy[1], rpy[2]
    if isinstance(phi, float):
        Rm = np.array(
            [
                [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
                [
                    np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                    np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                    np.sin(phi) * np.cos(theta),
                ],
                [
                    np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi),
                    np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
                    np.cos(phi) * np.cos(theta),
                ],
            ]
        )
    else:
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


def Rbi(phi: ca.MX, theta: ca.MX, psi: ca.MX) -> ca.MX:
    """Create a rotation matrix from euler angles.

    This represents the extrinsic X-Y-Z (or quivalently the intrinsic Z-Y-X (3-2-1)) euler angle
    rotation.

    Args:
        phi: roll (or rotation about X).
        theta: pitch (or rotation about Y).
        psi: yaw (or rotation about Z).

    Returns:
        R: casadi Rotation matrix
    """
    rx = ca.blockcat([[1, 0, 0], [0, ca.cos(phi), -ca.sin(phi)], [0, ca.sin(phi), ca.cos(phi)]])
    ry = ca.blockcat(
        [[ca.cos(theta), 0, ca.sin(theta)], [0, 1, 0], [-ca.sin(theta), 0, ca.cos(theta)]]
    )
    rz = ca.blockcat([[ca.cos(psi), -ca.sin(psi), 0], [ca.sin(psi), ca.cos(psi), 0], [0, 0, 1]])
    return rz @ ry @ rx


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


def rpm_to_torques_mat(c_tau_xy: float, cd: float) -> np.ndarray:
    """Creates a matrix to compute the torques in the body frame from the rotor rates."""
    return np.array(
        [
            [-c_tau_xy, -c_tau_xy, c_tau_xy, c_tau_xy],
            [-c_tau_xy, c_tau_xy, c_tau_xy, -c_tau_xy],
            [-cd, cd, -cd, cd],
        ]
    )


def W1(eul_ang: np.ndarray) -> np.ndarray:
    """Compute the numpy W1 matrix from euler angles. w = W1 * drpy."""
    phi, theta = eul_ang[0], eul_ang[1]
    return np.array(
        [
            [1, 0, -np.sin(theta)],
            [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
            [0, -np.sin(phi), np.cos(phi) * np.cos(theta)],
        ]
    )


def W2(eul_ang: np.ndarray) -> np.ndarray:
    """Compute the W2 matrix from euler angles. drpy = W2 * w."""
    phi, theta = eul_ang[0], eul_ang[1]
    return np.array(
        [
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ]
    )


def W1s(eul_ang: ca.MX) -> ca.MX:
    """Compute the symbolic W1 matrix from euler angles."""
    return ca.vertcat(
        ca.horzcat(1, 0, -ca.sin(eul_ang[1])),
        ca.horzcat(0, ca.cos(eul_ang[0]), ca.sin(eul_ang[0]) * ca.cos(eul_ang[1])),
        ca.horzcat(0, -ca.sin(eul_ang[0]), ca.cos(eul_ang[0]) * ca.cos(eul_ang[1])),
    )


def dW1s(eul_ang: ca.MX, deul_ang: ca.MX) -> ca.MX:
    """Compute the time derivative of the symbolical W1 matrix."""
    return W1s(eul_ang) @ ca.skew(deul_ang)


def W2s(eul_ang: ca.MX) -> ca.MX:
    """Compute the W2 matrix from euler angles."""
    return ca.vertcat(
        ca.horzcat(
            1, ca.sin(eul_ang[0]) * ca.tan(eul_ang[1]), ca.cos(eul_ang[0]) * ca.tan(eul_ang[1])
        ),
        ca.horzcat(0, ca.cos(eul_ang[0]), -ca.sin(eul_ang[0])),
        ca.horzcat(
            0, ca.sin(eul_ang[0]) / ca.cos(eul_ang[1]), ca.cos(eul_ang[0]) / ca.cos(eul_ang[1])
        ),
    )


def dW2s(eul_ang: ca.MX, deul_ang: ca.MX) -> ca.MX:
    """Compute the time derivative of the symbolical W2 matrix."""
    return W2s(eul_ang) @ ca.skew(deul_ang)


def rungeKuttaExpr(x, u, dt, fc) -> ca.MX:
    """Perform one step of the 4th order Runge-Kutta integration method.

    Args:
        x (ca.MX): The state vector.
        u (ca.MX): The control input vector.
        dt (float): The time step.
        fc (ca.Function): The continuous function to discretize.

    Returns:
        ca.MX: The discrete dynamics function.
    """
    k1 = fc(x, u)
    k2 = fc(x + dt / 2 * k1, u)
    k3 = fc(x + dt / 2 * k2, u)
    k4 = fc(x + dt * k3, u)
    xn = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xn


def rungeKuttaFcn(nx, nu, dt, fc) -> ca.Function:
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    xn = rungeKuttaExpr(x, u, dt, fc)
    return ca.Function("disc_dyn", [x, u], [xn], ["x", "u"], ["xn"])
