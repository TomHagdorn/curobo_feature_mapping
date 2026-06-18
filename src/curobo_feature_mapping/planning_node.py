# SPDX-License-Identifier: Apache-2.0
"""ROS planning glue for the curobo_feature_mapping node.

`PlanningServer` owns everything that turns the in-process cuRobo planner
(`motion_planner.ArmPlanner`) into ROS interfaces: the `~/move_to_pose` action
(reach or grasp, plan +/- execute), the `~/plan_to_pose` service, the
`/joint_states` cache, and the `FollowJointTrajectory` client used to run plans
on the controller.

It is composed by `CuroboMapPublisher` (same node, same process) so it shares
the live in-GPU map directly — the map and the prompt resolver are injected as
callables, so this stays decoupled from how the map is built:

    server = create_planning_server(
        node,
        world_frame=..., device=..., tf_buffer=...,
        mesh_provider=node._extract_mesh_np,  # () -> (vertices, faces)
        query_fn=node._run_query,             # (prompt) -> dict with "centroid"
    )

`create_planning_server` returns None (after logging) when planning is disabled,
the cuRobo motion modules are missing, or no robot config is available, so the
node keeps mapping + answering feature queries regardless.
"""

import json
import time
from pathlib import Path
from typing import Callable, Optional

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from sensor_msgs.msg import JointState as JointStateMsg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Pose TF transform helper; only needed when a target_pose names a frame other
# than world_frame. Importing tf2_geometry_msgs also registers the conversions.
try:
    from tf2_geometry_msgs import do_transform_pose_stamped
except ImportError:  # pragma: no cover
    do_transform_pose_stamped = None

# Controller action (ros-<distro>-control-msgs). Without it, planning still
# works; only on-robot execution is disabled.
try:
    from control_msgs.action import FollowJointTrajectory
except ImportError:  # pragma: no cover
    FollowJointTrajectory = None

# Planning interfaces live in the optional curobo_feature_mapping_interfaces
# colcon package (build + source it). Without them the action/service are off.
try:
    from curobo_feature_mapping_interfaces.action import MoveToPose
    from curobo_feature_mapping_interfaces.srv import PlanToPose
except ImportError:  # pragma: no cover
    MoveToPose = None
    PlanToPose = None

# ArmPlanner pulls in cuRobo's motion-planning modules; keep optional so a
# mapping-only checkout (or one without a robot config) still loads.
try:
    from curobo_feature_mapping.motion_planner import ArmPlanner
except ImportError:  # pragma: no cover
    ArmPlanner = None

# Mesh decimation (shared with the MoveIt export path); only used to cap the
# collision mesh handed to the planner.
try:
    from curobo_feature_mapping.moveit_export import decimate
    _DECIMATE = True
except ImportError:  # pragma: no cover
    _DECIMATE = False

# Tool orientation (wxyz) that points a UR tool0 +Z axis straight down — the
# default approach pose for a standoff/grasp above a queried object (180 deg
# about X).
_TOOL_DOWN_WXYZ = [0.0, 1.0, 0.0, 0.0]


def _default_robot_config() -> str:
    # config/ur5e.yml sits at the package root; this module is under
    # <pkg>/src/curobo_feature_mapping/. parents[2] == <pkg>.
    return str(Path(__file__).resolve().parents[2] / "config" / "ur5e.yml")


def create_planning_server(
    node,
    *,
    world_frame: str,
    device: str,
    tf_buffer,
    mesh_provider: Callable,
    query_fn: Callable,
) -> "Optional[PlanningServer]":
    """Build a PlanningServer, or return None (logging why) if unavailable."""
    log = node.get_logger()
    if not node.get_parameter("enable_planning").value:
        return None
    if ArmPlanner is None:
        log.warning(
            "enable_planning is true but ArmPlanner could not be imported "
            "(cuRobo motion modules missing); planning disabled"
        )
        return None
    robot_config = node.get_parameter("robot_config").value or _default_robot_config()
    if not Path(robot_config).is_file():
        log.error(
            f"robot_config not found: {robot_config} (generate it with "
            "scripts/gen_robot_cfg.py); planning disabled"
        )
        return None
    return PlanningServer(
        node, robot_config,
        world_frame=world_frame, device=device, tf_buffer=tf_buffer,
        mesh_provider=mesh_provider, query_fn=query_fn,
    )


class PlanningServer:
    """Owns the cuRobo planner and its ROS action/service interfaces."""

    def __init__(
        self,
        node,
        robot_config: str,
        *,
        world_frame: str,
        device: str,
        tf_buffer,
        mesh_provider: Callable,
        query_fn: Callable,
    ):
        self._node = node
        self._world_frame = world_frame
        self._tf_buffer = tf_buffer
        self._mesh_provider = mesh_provider  # () -> (vertices, faces)
        self._query_fn = query_fn            # (prompt) -> dict
        self._latest_js = None               # (names, positions) cache
        self._exec_client = None
        log = node.get_logger()

        log.info(f"Loading cuRobo MotionPlanner ({robot_config})...")
        self._planner = ArmPlanner(
            robot_config,
            device=device,
            collision_activation_distance=float(
                node.get_parameter("plan_collision_activation_distance").value
            ),
        )
        self._planner.warmup()
        log.info(f"MotionPlanner ready; joints={self._planner.joint_names}")

        cb = ReentrantCallbackGroup()
        node.create_subscription(
            JointStateMsg,
            node.get_parameter("joint_states_topic").value,
            self._on_joint_states,
            10,
        )
        if FollowJointTrajectory is not None:
            self._exec_client = ActionClient(
                node, FollowJointTrajectory,
                node.get_parameter("controller_action").value, callback_group=cb,
            )
        else:
            log.warning(
                "control_msgs not found (install ros-<distro>-control-msgs); "
                "plan execution disabled (planning still works)"
            )

        if PlanToPose is not None:
            node.create_service(
                PlanToPose, "~/plan_to_pose", self._srv_plan_to_pose, callback_group=cb
            )
        if MoveToPose is not None:
            self._move_action = ActionServer(
                node, MoveToPose, "~/move_to_pose",
                execute_callback=self._execute_move,
                goal_callback=lambda _g: GoalResponse.ACCEPT,
                cancel_callback=lambda _g: CancelResponse.ACCEPT,
                callback_group=cb,
            )
        else:
            log.warning(
                "MoveToPose interface not found (build "
                "curobo_feature_mapping_interfaces); ~/move_to_pose unavailable"
            )

    def _param(self, name):
        return self._node.get_parameter(name).value

    # ------------------------------------------------------------------ #
    def _on_joint_states(self, msg: JointStateMsg):
        self._latest_js = (list(msg.name), list(msg.position))

    def _refresh_world(self) -> str:
        """Push the current map mesh into the planner's collision world.

        world_frame must equal the robot base frame for poses to line up; the
        sim sets both to base_link. Never fails: with no map yet the world is
        cleared (free-space / self-collision-only planning). Returns a short
        status string describing the world used.
        """
        vertices, faces = self._mesh_provider()
        if vertices is None:
            self._planner.clear_world()
            return "empty (no map surface yet)"
        max_tris = int(self._param("mesh_max_triangles"))
        if _DECIMATE and len(faces) > max_tris:
            vertices, faces = decimate(vertices, faces, max_tris)
        self._planner.set_world_from_mesh(vertices, faces)
        return f"mesh ({len(faces)} triangles)"

    def _resolve_goal(self, prompt, target_pose: PoseStamped):
        """Return ((position_xyz, quat_wxyz, source, is_prompt), error_or_None).

        Returns the RAW target (no standoff/clearance applied — the caller adds
        those per mode): prompt non-empty -> the feature-map centroid (downward
        tool); otherwise target_pose (TF'd into world_frame if needed).
        """
        if prompt:
            result = self._query_fn(prompt)
            if "centroid" not in result:
                return None, result.get("error", f"no match for prompt '{prompt}'")
            c = result["centroid"]
            # x,y from the centroid, z from the matched top surface (so a
            # top-down offset clears the object instead of landing mid-volume).
            top_z = float(result.get("top_z", c[2]))
            pos = [float(c[0]), float(c[1]), top_z]
            return (pos, list(_TOOL_DOWN_WXYZ), f"prompt:{prompt}", True), None

        ps = target_pose
        frame = ps.header.frame_id or self._world_frame
        if frame != self._world_frame:
            if do_transform_pose_stamped is None:
                return None, (
                    f"target_pose is in '{frame}' but tf2_geometry_msgs is "
                    f"unavailable; send the pose in '{self._world_frame}'"
                )
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._world_frame, frame, rclpy.time.Time(), timeout=Duration(seconds=0.5)
                )
                ps = do_transform_pose_stamped(ps, tf)
            except Exception as exc:  # noqa: BLE001
                return None, f"could not transform target_pose {frame}->{self._world_frame}: {exc}"
        p, q = ps.pose.position, ps.pose.orientation
        return ([p.x, p.y, p.z], [q.w, q.x, q.y, q.z], "pose", False), None

    def _plan(self, prompt, target_pose, standoff, tool_frame, grasp=False, grasp_lift=0.0):
        """Resolve the goal, refresh the world, and plan. Returns (plan, info).

        Vertical offset above the raw target depends on mode:
        - grasp: + grasp_clearance only (cuRobo's plan_grasp adds the approach);
          the object is in the collision map, so the contact pose sits just above
          the matched surface (no gripper to reach inside it).
        - reach to a prompt: + standoff (aim above the queried object).
        - reach to an explicit pose: no offset (use the pose as given).
        """
        info = {}
        goal, err = self._resolve_goal(prompt, target_pose)
        if err:
            return None, {"error": err}
        pos, quat, source, is_prompt = goal
        if grasp:
            pos = [pos[0], pos[1], pos[2] + float(self._param("grasp_clearance"))]
        elif is_prompt:
            sd = standoff if standoff > 0 else float(self._param("plan_standoff"))
            pos = [pos[0], pos[1], pos[2] + sd]
        info["goal"] = {"position": pos, "quaternion": quat, "source": source}
        info["mode"] = "grasp" if grasp else "reach"
        info["world"] = self._refresh_world()
        if grasp:
            lift = grasp_lift if grasp_lift > 0 else float(self._param("plan_standoff"))
            plan, status = self._planner.plan_grasp_to_pose(
                pos, quat, self._current_q(), tool_frame=tool_frame or None, lift=lift
            )
            info["grasp_status"] = status
        else:
            plan = self._planner.plan_to_pose(
                pos, quat, self._current_q(), tool_frame=tool_frame or None
            )
        if plan is None:
            return None, {"error": "no feasible trajectory found", **info}
        info.update(n_points=plan.num_points, plan_time=round(plan.plan_time, 3),
                    duration=round(plan.num_points * plan.dt, 3))
        return plan, info

    def _current_q(self):
        """Current arm configuration in the planner's joint order.

        Falls back to the planner's default (retract) config until joint_states
        arrives, so a plan can still be requested right after startup.
        """
        if self._latest_js is not None:
            names, positions = self._latest_js
            q = self._planner.reorder_to_planner(names, positions)
            if q is not None:
                return q
        return None  # ArmPlanner falls back to the robot's default config

    def _plan_to_joint_trajectory(self, plan, vel_scale: float) -> JointTrajectory:
        jt = JointTrajectory()
        jt.joint_names = plan.joint_names
        vel_scale = vel_scale if 0.0 < vel_scale <= 1.0 else 1.0
        dt = plan.dt / vel_scale  # slower motion -> wider time spacing
        for i in range(plan.num_points):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in plan.positions[i].tolist()]
            if plan.velocities is not None:
                pt.velocities = [float(v) * vel_scale for v in plan.velocities[i].tolist()]
            t = (i + 1) * dt
            pt.time_from_start = DurationMsg(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            jt.points.append(pt)
        return jt

    # ------------------------------------------------------------------ #
    def _srv_plan_to_pose(self, request, response):
        plan, info = self._plan(
            request.prompt, request.target_pose, request.standoff, request.tool_frame
        )
        response.success = plan is not None
        if plan is not None:
            response.planned_trajectory = self._plan_to_joint_trajectory(plan, 1.0)
        response.message = json.dumps(info)
        return response

    def _execute_move(self, goal_handle):
        req = goal_handle.request
        result = MoveToPose.Result()

        def feedback(phase, progress=0.0, tracking_error=0.0):
            fb = MoveToPose.Feedback()
            fb.phase, fb.progress, fb.tracking_error = phase, float(progress), float(tracking_error)
            goal_handle.publish_feedback(fb)

        feedback("planning")
        vel_scale = req.max_velocity_scale if req.max_velocity_scale > 0 else float(
            self._param("plan_max_velocity_scale")
        )
        plan, info = self._plan(
            req.prompt, req.target_pose, req.standoff, req.tool_frame,
            grasp=req.grasp, grasp_lift=req.grasp_lift,
        )
        if plan is None:
            goal_handle.abort()
            result.success, result.message = False, json.dumps(info)
            return result

        traj = self._plan_to_joint_trajectory(plan, vel_scale)
        result.planned_trajectory = traj

        if not req.execute:
            feedback("done", 1.0)
            goal_handle.succeed()
            result.success, result.message = True, json.dumps({**info, "executed": False})
            return result

        ok, exec_info = self._execute_trajectory(traj, goal_handle, feedback)
        info.update(exec_info)
        if ok:
            goal_handle.succeed()
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.success = ok
        result.message = json.dumps({**info, "executed": True})
        return result

    def _execute_trajectory(self, traj: JointTrajectory, goal_handle, feedback):
        """Forward to the controller's FollowJointTrajectory action with feedback."""
        if self._exec_client is None:
            return False, {"exec_error": "no controller action client (control_msgs missing)"}
        if not self._exec_client.wait_for_server(timeout_sec=5.0):
            return False, {"exec_error": "controller action server unavailable"}

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj
        send_future = self._exec_client.send_goal_async(goal_msg)
        ctrl_handle = self._spin_until_done(send_future)
        if ctrl_handle is None or not ctrl_handle.accepted:
            return False, {"exec_error": "controller rejected the trajectory"}

        result_future = ctrl_handle.get_result_async()
        total = traj.points[-1].time_from_start
        total_s = total.sec + total.nanosec * 1e-9
        start = time.time()
        max_track = 0.0
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                self._spin_until_done(ctrl_handle.cancel_goal_async())
                return False, {"exec_error": "cancelled", "max_tracking_error": max_track}
            elapsed = time.time() - start
            err = self._tracking_error(traj, elapsed)
            max_track = max(max_track, err)
            feedback("executing", min(elapsed / total_s, 1.0) if total_s > 0 else 0.0, err)
            time.sleep(0.1)

        feedback("done", 1.0, max_track)
        status = result_future.result().result
        ok = status.error_code == 0
        return ok, {"max_tracking_error": round(max_track, 4),
                    "controller_error_code": int(status.error_code)}

    def _tracking_error(self, traj: JointTrajectory, elapsed: float) -> float:
        """Max |planned(elapsed) - measured| over the arm joints, in rad."""
        if self._latest_js is None:
            return 0.0
        names, positions = self._latest_js
        lookup = dict(zip(names, positions))
        idx = 0
        for i, pt in enumerate(traj.points):
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if t <= elapsed:
                idx = i
            else:
                break
        planned = traj.points[idx].positions
        errs = [abs(lookup[j] - planned[k]) for k, j in enumerate(traj.joint_names) if j in lookup]
        return max(errs) if errs else 0.0

    def _spin_until_done(self, future, timeout: float = 30.0):
        """Block on a future while the executor (other threads) keeps spinning."""
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            time.sleep(0.02)
        return future.result() if future.done() else None
