#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a cuRobo robot config (YAML) from a UR URDF.

cuRobo ships only ``ur10e`` configs, but the simulation defaults to ``ur5e``
(``curobo_feature_mapping_simulation/launch/sim.launch.py``). This script fits
collision spheres to the robot meshes with cuRobo's ``RobotBuilder`` and writes
a config that ``MotionPlannerCfg.create(robot=<path>)`` can load.

Run it in the ROS-node venv (py3.12, has curobo). It needs a plain URDF and the
directory that ``package://`` mesh paths resolve against.

First produce a ur5e URDF from the installed ur_description (ROS sourced):

    source /opt/ros/jazzy/setup.bash
    xacro $(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro \
        ur_type:=ur5e name:=ur > /tmp/ur5e.urdf

Then fit + save the cuRobo config:

    python scripts/gen_robot_cfg.py \
        --urdf /tmp/ur5e.urdf \
        --mesh-root $(ros2 pkg prefix ur_description)/share \
        --out config/ur5e.yml

``--mesh-root`` is the PARENT of the ``ur_description`` directory, because the
URDF references ``package://ur_description/meshes/...`` and cuRobo strips the
``package://`` prefix then joins the remainder onto mesh-root.
"""

import argparse
from pathlib import Path

from curobo.robot_builder import RobotBuilder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", required=True, help="Path to the (expanded) URDF.")
    ap.add_argument(
        "--mesh-root", default="",
        help="Dir that package:// mesh paths resolve against (parent of ur_description).",
    )
    ap.add_argument("--out", required=True, help="Output .yml path.")
    ap.add_argument("--tool-frame", default="tool0", help="End-effector / tool frame.")
    ap.add_argument("--sphere-density", type=float, default=1.0)
    args = ap.parse_args()

    builder = RobotBuilder(
        urdf_path=args.urdf,
        asset_path=args.mesh_root,
        tool_frames=[args.tool_frame],
    )
    print(f"Links: {builder.collision_link_names}")
    builder.fit_collision_spheres(sphere_density=args.sphere_density)
    builder.compute_collision_matrix()
    print(f"Fitted {builder.num_spheres} collision spheres.")

    config = builder.build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    builder.save(config, str(out), include_cspace=True)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
