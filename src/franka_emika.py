import numpy as np
import franky as fk
from scipy.spatial.transform import Rotation
from src.robot import Robot
from src.transforms import *

class FrankaEmika(Robot):
    def __init__(self, wait_for_retry, ip="172.16.0.2"):
        super().__init__(wait_for_retry)
        self._arm = fk.Robot(ip)
        self._gripper = fk.Gripper(ip)

        self._arm.recover_from_errors()
        self._arm.relative_dynamics_factor = 0.1

    def _get_pose(self):
        self._arm.recover_from_errors()
        pose = self._arm.current_cartesian_state.pose.end_effector_pose.matrix
        return fix_rotation_error(pose * 1000)

    def _move_to(self, pose):
        self._arm.stop()
        self._arm.recover_from_errors()
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        motion = fk.CartesianMotion(fk.Affine(pose[:3, 3] * 0.001, quat), reference_type=fk.ReferenceType.Absolute)
        self._arm.move(motion)

    def get_joints(self):
        # Only called while the viewing pose is being set, never between the moves of a game:
        # on this arm, reading the state right before a long Cartesian move made that move abort.
        try:
            return np.array(self._arm.current_joint_state.position, dtype=float)
        except Exception as error:
            print(f"Couldn't read the joint angles, joint-limit protection is off: {error}")
            return None

    def move_to_joints(self, joints):
        self._arm.stop()
        self._arm.recover_from_errors()
        self._arm.move(fk.JointMotion(np.asarray(joints, dtype=float).tolist()))

    def _close_gripper(self):
        self._arm.recover_from_errors()
        self._gripper.grasp(0.015, 0.02, 20.0)

    def _open_gripper(self):
        self._arm.recover_from_errors()
        self._gripper.move(0.02, 0.02)
