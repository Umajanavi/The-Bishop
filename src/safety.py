"""
Joint-limit logic for the Franka Panda, ported from FaceFollow_Franka_v3_2D_CAM.py (the face
follower that runs on the same robot). Pure maths: no robot, no camera.

How it keeps the joints away from their limits:
  - A damped Jacobian turns the wanted hand motion into joint velocities, so the arm never
    makes sudden large joint moves near a singular pose.
  - A 7-joint arm has one spare degree of freedom (the "null space"). Joints in the outer 40 %
    of their range are pulled back toward the middle along it. This reshapes the elbow without
    moving the hand. Joints inside the inner 60 % get no pull.
  - A hand motion that pushes a joint toward its limit is slowed down below 0.30 rad from the
    limit and stopped at 0.12 rad (the same goes for a nearly singular pose).
  - If that holds the wanted motion back, the move reports "blocked" so the caller can stop.
"""

import math
from typing import Tuple

import numpy as np

# Joint limits of the Panda (rad).
JOINT_LIMITS_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_LIMITS_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

NULLSPACE_ZONE = 0.6                # joints within this fraction of their range get no pull
NULLSPACE_GAIN_RAD_S = 0.3          # pull speed at the very edge of the range
JOINT_SLOW_MARGIN_RAD = 0.30        # moving toward a limit slows below this distance ...
JOINT_HARD_MARGIN_RAD = 0.12        # ... and stops at this distance
SIGMA_SLOW = 0.05                   # same for the smallest singular value of the Jacobian
SIGMA_HARD = 0.02
JOINT_MAX_SPEED_RAD_S = 0.5
JACOBIAN_DAMPING = 0.05
BLOCKED_SCALE = 0.05                # the wanted motion slowed below this factor counts as blocked

# Franka Panda Denavit-Hartenberg parameters.
_DH_A = (0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0)
_DH_D = (0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0, 0.107)
_DH_ALPHA = (0.0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2, 0.0)


def _dh(a, d, alpha, theta) -> np.ndarray:
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def panda_fk(q) -> np.ndarray:
    """4x4 flange pose in the base frame (metres)."""
    pose = np.eye(4)
    for a, d, alpha, theta in zip(_DH_A, _DH_D, _DH_ALPHA, list(q) + [0.0]):
        pose = pose @ _dh(a, d, alpha, theta)
    return pose


def panda_jacobian(q, eps: float = 1e-6) -> np.ndarray:
    """6x7 Jacobian of the flange (linear rows 0-2, angular rows 3-5, base frame)."""
    q = np.array(q, dtype=float)
    pose = panda_fk(q)
    jacobian = np.zeros((6, 7))

    for joint in range(7):
        moved = q.copy()
        moved[joint] += eps
        moved_pose = panda_fk(moved)
        jacobian[:3, joint] = (moved_pose[:3, 3] - pose[:3, 3]) / eps
        delta = moved_pose[:3, :3] @ pose[:3, :3].T
        jacobian[3:, joint] = np.array([
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ]) / (2 * eps)

    return jacobian


def joint_margins(q) -> np.ndarray:
    """Distance (rad) of each joint to its nearest limit."""
    q = np.asarray(q, dtype=float)
    return np.minimum(q - JOINT_LIMITS_MIN, JOINT_LIMITS_MAX - q)


def nullspace_pull(q) -> np.ndarray:
    """Joint velocity that pulls joints in the outer part of their range back toward the middle."""
    q = np.array(q, dtype=float)
    middle = 0.5 * (JOINT_LIMITS_MIN + JOINT_LIMITS_MAX)
    half = 0.5 * (JOINT_LIMITS_MAX - JOINT_LIMITS_MIN)
    fraction = (q - middle) / half
    excess = np.clip((np.abs(fraction) - NULLSPACE_ZONE) / (1.0 - NULLSPACE_ZONE), 0.0, 1.5)
    return -np.sign(fraction) * excess * NULLSPACE_GAIN_RAD_S


def _limit_factor(q, qdot) -> Tuple[float, int]:
    """Factor 0..1 that slows a joint motion approaching a limit, and the limiting joint."""
    factor, which = 1.0, -1
    for i in range(7):
        if qdot[i] > 1e-4:
            margin = JOINT_LIMITS_MAX[i] - q[i]
        elif qdot[i] < -1e-4:
            margin = q[i] - JOINT_LIMITS_MIN[i]
        else:
            continue
        f = min(1.0, max(0.0, (margin - JOINT_HARD_MARGIN_RAD) / (JOINT_SLOW_MARGIN_RAD - JOINT_HARD_MARGIN_RAD)))
        if f < factor:
            factor, which = f, i
    return factor, which


def joint_velocity_command(q, twist6, max_speed: float = JOINT_MAX_SPEED_RAD_S) -> Tuple[np.ndarray, str, bool]:
    """
    Joint velocities that give the hand the twist [v; omega] (base frame), plus the null-space
    motion that pulls the joints away from their limits.
    Returns (joint velocities, note, blocked). `blocked` is True when the limits hold back the
    requested motion.
    """
    q = np.array(q, dtype=float)
    twist6 = np.asarray(twist6, dtype=float)
    jacobian = panda_jacobian(q)
    _u, singular_values, vt = np.linalg.svd(jacobian)
    q_task = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + JACOBIAN_DAMPING ** 2 * np.eye(6), twist6)
    q_null = vt[-1] * float(vt[-1] @ nullspace_pull(q))

    notes = []
    scale = 1.0
    limit_factor, which = _limit_factor(q, q_task + q_null)
    if limit_factor < 1.0:
        scale = min(scale, limit_factor)
        notes.append(f"joint {which + 1} near its limit")
    singular_factor = min(1.0, max(0.0, (float(singular_values[5]) - SIGMA_HARD) / (SIGMA_SLOW - SIGMA_HARD)))
    if singular_factor < 1.0:
        scale = min(scale, singular_factor)
        notes.append("near a singular pose")
    blocked = scale < BLOCKED_SCALE
    qdot = scale * q_task + q_null

    # Never push a joint inside the hard margin further toward its limit.
    for i in range(7):
        if (qdot[i] > 0 and JOINT_LIMITS_MAX[i] - q[i] < JOINT_HARD_MARGIN_RAD) or \
           (qdot[i] < 0 and q[i] - JOINT_LIMITS_MIN[i] < JOINT_HARD_MARGIN_RAD):
            if abs(qdot[i]) > 1e-3:
                blocked = True
            qdot[i] = 0.0

    # Uniform scaling keeps the direction of the hand motion.
    biggest = float(np.max(np.abs(qdot)))
    if biggest > max_speed:
        qdot *= max_speed / biggest

    return qdot, ("SLOWED: " + ", ".join(notes)) if notes else "", blocked
