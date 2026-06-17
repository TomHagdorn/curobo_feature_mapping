"""Volumetric mapping with cuRobo from a RealSense camera on a UR robot."""

from curobo_feature_mapping.poses import load_trajectory, predict_pose
from curobo_feature_mapping.realsense_bag import RealsenseBag

__all__ = ["RealsenseBag", "load_trajectory", "predict_pose"]
