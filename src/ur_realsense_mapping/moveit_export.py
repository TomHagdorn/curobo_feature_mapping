# SPDX-License-Identifier: Apache-2.0
"""Convert cuRobo meshes to MoveIt 2 planning-scene messages."""

import numpy as np
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import Mesh, MeshTriangle

try:
    import fast_simplification
except ImportError:  # pragma: no cover
    fast_simplification = None


def decimate(vertices: np.ndarray, faces: np.ndarray, max_triangles: int):
    """Reduce triangle count for MoveIt's collision checker (FCL).

    Returns the inputs unchanged when already small enough or when
    ``fast_simplification`` is not installed.
    """
    if len(faces) <= max_triangles or fast_simplification is None:
        return vertices, faces
    target_reduction = 1.0 - max_triangles / len(faces)
    v, f = fast_simplification.simplify(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
        target_reduction=target_reduction,
    )
    return v, f


def mesh_to_msg(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    msg = Mesh()
    msg.vertices = [
        Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in vertices
    ]
    msg.triangles = [
        MeshTriangle(vertex_indices=[int(a), int(b), int(c)]) for a, b, c in faces
    ]
    return msg


def make_collision_object(
    vertices: np.ndarray,
    faces: np.ndarray,
    frame_id: str,
    object_id: str = "curobo_map",
) -> CollisionObject:
    co = CollisionObject()
    co.header.frame_id = frame_id
    co.id = object_id
    pose = Pose()
    pose.orientation.w = 1.0  # Pose() defaults to all-zero, an invalid quaternion
    co.meshes = [mesh_to_msg(vertices, faces)]
    co.mesh_poses = [pose]
    co.operation = CollisionObject.ADD
    return co


def make_planning_scene_diff(collision_object: CollisionObject) -> PlanningScene:
    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects.append(collision_object)
    return scene
