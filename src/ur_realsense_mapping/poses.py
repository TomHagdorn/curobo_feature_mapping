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


def load_trajectory(path: str, device_cfg: DeviceCfg) -> list:
    """Load per-frame poses (``x y z qw qx qy qz`` per line) from a text file."""
    rows = np.loadtxt(str(Path(path).expanduser()), dtype=np.float32, ndmin=2)
    if rows.shape[1] != 7:
        raise ValueError(
            f"trajectory file must have 7 columns (x y z qw qx qy qz), got {rows.shape[1]}"
        )
    return [Pose.from_list(r.tolist(), device_cfg=device_cfg) for r in rows]
