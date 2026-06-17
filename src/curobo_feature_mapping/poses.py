# SPDX-License-Identifier: Apache-2.0
"""Camera pose helpers (prediction, trajectory files)."""

from pathlib import Path

import numpy as np

from curobo.types import DeviceCfg, Pose


def predict_pose(prev: Pose, prev2: Pose) -> Pose:
    """Constant-velocity prediction: apply the last relative motion again.

    With camera-to-world poses ``T``, the relative motion between the two
    previous frames is ``dT = T[prev2]^-1 @ T[prev]``; the predicted next pose is
    ``T[prev] @ dT``.
    """
    delta = prev2.inverse().multiply(prev)
    return prev.multiply(delta)


QUAT_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, av = a[0], a[1:]
    bw, bv = b[0], b[1:]
    return np.concatenate(
        ([aw * bw - av @ bv], aw * bv + bw * av + np.cross(av, bv))
    )


def rotvec_to_quat(rv: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector to (w, x, y, z) quaternion."""
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return QUAT_IDENTITY.copy()
    axis = rv / theta
    return np.concatenate(([np.cos(theta / 2.0)], np.sin(theta / 2.0) * axis))


def integrate_gyro(samples: list) -> np.ndarray:
    """Integrate gyro samples into a body-frame rotation quaternion (w,x,y,z).

    ``samples`` is a list of ``(t_seconds, angular_velocity_xyz)`` with rates in
    the camera body frame. Body rates compose on the right:
    ``R_{k+1} = R_k @ exp(omega * dt)``.
    """
    q = QUAT_IDENTITY.copy()
    for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        q = quat_mul(q, rotvec_to_quat(0.5 * (w0 + w1).astype(np.float64) * dt))
    return q / np.linalg.norm(q)


def load_trajectory(path: str, device_cfg: DeviceCfg) -> list:
    """Load per-frame poses (``x y z qw qx qy qz`` per line) from a text file."""
    rows = np.loadtxt(str(Path(path).expanduser()), dtype=np.float32, ndmin=2)
    if rows.shape[1] != 7:
        raise ValueError(
            f"trajectory file must have 7 columns (x y z qw qx qy qz), got {rows.shape[1]}"
        )
    return [Pose.from_list(r.tolist(), device_cfg=device_cfg) for r in rows]
