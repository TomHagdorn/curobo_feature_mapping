# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around cuRobo's :class:`MotionPlanner` for the ROS node.

Keeps all cuRobo planning specifics (config, world updates, result -> ROS
trajectory conversion) out of ``map_publisher.py``. The node owns one
:class:`ArmPlanner`, refreshes its collision world from the live cuRobo map
mesh, and asks it to plan to a tool pose or joint goal.

cuRobo public API used (new ``_src`` layout, post warp-1.13):

- ``curobo.motion_planner.MotionPlanner`` / ``MotionPlannerCfg.create``
- ``planner.update_world(Scene(mesh=[Mesh(...)]))`` from ``curobo.scene``
- ``planner.plan_pose(GoalToolPose, JointState)`` / ``planner.plan_cspace``
- result ``TrajOptSolverResult.get_interpolated_plan()`` -> ``JointState``
- ``planner.joint_names`` (robot joint order),
  ``planner.trajopt_solver.config.interpolation_dt`` (waypoint spacing)
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Mesh, Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose


def _squeeze_to_2d(t: "torch.Tensor") -> "torch.Tensor":
    """Drop leading singleton dims until a (T, dof) tensor remains."""
    while t.ndim > 2:
        t = t[0]
    return t


class PlanResult:
    """Plain container for a planned trajectory (cuRobo types kept off the node)."""

    def __init__(
        self,
        joint_names: List[str],
        positions: "torch.Tensor",   # (T, dof)
        velocities: Optional["torch.Tensor"],  # (T, dof) or None
        dt: float,
        plan_time: float,
    ):
        self.joint_names = joint_names
        self.positions = positions
        self.velocities = velocities
        self.dt = dt
        self.plan_time = plan_time

    @property
    def num_points(self) -> int:
        return int(self.positions.shape[0])


class ArmPlanner:
    """Owns a cuRobo MotionPlanner and converts results to plain arrays."""

    def __init__(
        self,
        robot_config: str,
        device: str = "cuda:0",
        num_trajopt_seeds: int = 4,
        num_ik_seeds: int = 32,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        collision_activation_distance: float = 0.02,
        collision_cache: Optional[dict] = None,
        use_cuda_graph: bool = True,
    ):
        self._device_cfg = DeviceCfg(device=torch.device(device))
        # Mesh cache must hold the (decimated) map mesh; obb cache covers any
        # cuboids. Sized generously so update_world never reallocates under a
        # CUDA graph.
        if collision_cache is None:
            collision_cache = {"obb": 32, "mesh": 16}
        cfg = MotionPlannerCfg.create(
            robot=robot_config,
            num_trajopt_seeds=num_trajopt_seeds,
            num_ik_seeds=num_ik_seeds,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            optimizer_collision_activation_distance=collision_activation_distance,
            collision_cache=collision_cache,
            use_cuda_graph=use_cuda_graph,
            device_cfg=self._device_cfg,
        )
        self._planner = MotionPlanner(cfg)
        self._joint_names: List[str] = list(self._planner.joint_names)
        self._dt = float(self._planner.trajopt_solver.config.interpolation_dt)

    # ------------------------------------------------------------------ #
    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def tool_frames(self) -> List[str]:
        return list(self._planner.tool_frames)

    def warmup(self):
        """Compile CUDA graphs / prime caches so the first real plan is fast."""
        self._planner.warmup()

    # ------------------------------------------------------------------ #
    def set_world_from_mesh(self, vertices, faces, name: str = "curobo_map"):
        """Replace the planner's collision world with a single mesh.

        ``vertices``/``faces`` are numpy arrays in the planner's base frame
        (the node passes the map mesh already expressed in ``world_frame``,
        which must equal the robot base_link).
        """
        mesh = Mesh(
            name=name,
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # identity; verts in base frame
            vertices=vertices.tolist() if hasattr(vertices, "tolist") else vertices,
            faces=faces.tolist() if hasattr(faces, "tolist") else faces,
        )
        self._planner.update_world(Scene(mesh=[mesh]))

    def clear_world(self):
        """Empty collision world (planning against self-collision only)."""
        self._planner.update_world(Scene())

    # ------------------------------------------------------------------ #
    def _current_state(self, current_q: Optional[Sequence[float]]) -> JointState:
        # Fall back to the robot's default (retract) config until a real joint
        # state is available, so a plan can be requested right after startup.
        if current_q is None:
            return self._planner.default_joint_state.clone().unsqueeze(0)
        q = torch.as_tensor(
            current_q, dtype=torch.float32, device=self._device_cfg.device
        ).view(1, -1)
        return JointState.from_position(q, joint_names=self._joint_names)

    def reorder_to_planner(
        self, names: Sequence[str], positions: Sequence[float]
    ) -> Optional[List[float]]:
        """Reorder a (names, positions) pair into the planner's joint order.

        Returns None if any planner joint is missing from ``names`` (e.g. the
        first /joint_states message before all joints are reported).
        """
        lookup = dict(zip(names, positions))
        try:
            return [float(lookup[j]) for j in self._joint_names]
        except KeyError:
            return None

    # ------------------------------------------------------------------ #
    def plan_to_pose(
        self,
        position_xyz: Sequence[float],
        quat_wxyz: Sequence[float],
        current_q: Sequence[float],
        tool_frame: Optional[str] = None,
        max_attempts: int = 5,
    ) -> Optional[PlanResult]:
        """Plan to a Cartesian tool pose (in the robot base frame)."""
        frame = tool_frame or self.tool_frames[0]
        dev = self._device_cfg.device
        goal_pose = Pose(
            position=torch.as_tensor(position_xyz, dtype=torch.float32, device=dev).view(1, 3),
            quaternion=torch.as_tensor(quat_wxyz, dtype=torch.float32, device=dev).view(1, 4),
        )
        goal = GoalToolPose.from_poses({frame: goal_pose})
        t0 = time.time()
        result = self._planner.plan_pose(
            goal, self._current_state(current_q), max_attempts=max_attempts
        )
        return self._to_plan_result(result, time.time() - t0)

    def plan_to_joint(
        self,
        goal_q: Sequence[float],
        current_q: Sequence[float],
        max_attempts: int = 5,
    ) -> Optional[PlanResult]:
        """Plan to a joint configuration (joints in the planner's order)."""
        q = torch.as_tensor(
            goal_q, dtype=torch.float32, device=self._device_cfg.device
        ).view(1, -1)
        goal_state = JointState.from_position(q, joint_names=self._joint_names)
        t0 = time.time()
        result = self._planner.plan_cspace(
            goal_state, self._current_state(current_q), max_attempts=max_attempts
        )
        return self._to_plan_result(result, time.time() - t0)

    def plan_grasp_to_pose(
        self,
        grasp_xyz: Sequence[float],
        quat_wxyz: Sequence[float],
        current_q: Sequence[float],
        tool_frame: Optional[str] = None,
        approach: float = 0.15,
        lift: float = 0.15,
    ):
        """Plan a grasp MOTION (approach -> grasp -> lift) to a tool pose.

        Wraps cuRobo's ``plan_grasp``: it plans the three motion segments given
        a grasp pose; it does NOT detect grasps or close a gripper. The grasp
        pose is the contact pose; cuRobo synthesizes the approach (``approach``
        m back along the tool +Z) and the lift (``lift`` m) automatically.

        Returns ``(PlanResult | None, status_str)``.
        """
        frame = tool_frame or self.tool_frames[0]
        dev = self._device_cfg.device
        grasp_pose = Pose(
            position=torch.as_tensor(grasp_xyz, dtype=torch.float32, device=dev).view(1, 3),
            quaternion=torch.as_tensor(quat_wxyz, dtype=torch.float32, device=dev).view(1, 4),
        )
        goal = GoalToolPose.from_poses({frame: grasp_pose})  # num_goalset=1
        t0 = time.time()
        res = self._planner.plan_grasp(
            goal,
            self._current_state(current_q),
            grasp_approach_offset=-abs(approach),
            grasp_lift_offset=-abs(lift),
            # tool0 has no collision spheres in the ur5e config, but disabling
            # the contact link is the documented way to let it reach an obstacle
            # (the queried object is part of the collision map).
            disable_collision_links=[frame],
        )
        status = res.status if res is not None else "plan_grasp returned None"
        if res is None or not bool(res.success.any()):
            return None, status

        segments = [
            (res.approach_interpolated_trajectory, res.approach_interpolated_last_tstep),
            (res.grasp_interpolated_trajectory, res.grasp_interpolated_last_tstep),
            (res.lift_interpolated_trajectory, res.lift_interpolated_last_tstep),
        ]
        pos_list, vel_list, has_vel = [], [], True
        for traj, last in segments:
            if traj is None:
                continue
            p = _squeeze_to_2d(traj.position)
            n = int(last.view(-1)[0]) if last is not None else p.shape[0]
            pos_list.append(p[:n].detach().cpu())
            if traj.velocity is not None:
                vel_list.append(_squeeze_to_2d(traj.velocity)[:n].detach().cpu())
            else:
                has_vel = False
        positions = torch.cat(pos_list, dim=0)
        velocities = torch.cat(vel_list, dim=0) if (has_vel and vel_list) else None
        return PlanResult(self._joint_names, positions, velocities, self._dt,
                          time.time() - t0), status

    # ------------------------------------------------------------------ #
    def _to_plan_result(self, result, plan_time: float) -> Optional[PlanResult]:
        if result is None or torch.count_nonzero(result.success) == 0:
            return None
        traj: JointState = result.get_interpolated_plan()
        # Interpolated tensors carry leading (batch, seed) dims, e.g.
        # (1, 1, T, dof); drop them down to (T, dof).
        positions = _squeeze_to_2d(traj.position).detach().cpu()
        velocities = (
            _squeeze_to_2d(traj.velocity).detach().cpu()
            if traj.velocity is not None
            else None
        )
        names = traj.joint_names if traj.joint_names is not None else self._joint_names
        return PlanResult(
            joint_names=list(names),
            positions=positions,
            velocities=velocities,
            dt=self._dt,
            plan_time=plan_time,
        )
