from abc import ABC, abstractmethod


class Robot(ABC):
    def __init__(self, wait_for_retry):
        self._wait_for_retry = wait_for_retry

    def get_pose(self):
        return self._retry(self._get_pose)

    def move_to(self, pose):
        self._retry(self._move_to, pose)

    def close_gripper(self):
        self._retry(self._close_gripper)

    def open_gripper(self):
        self._retry(self._open_gripper)

    def get_joints(self):
        # Optional: the arm's joint angles, or None if this robot can't report them.
        # Without them the joint-limit protection in the motion planner stays off.
        return None

    def move_to_joints(self, joints):
        raise NotImplementedError

    def quit(self):
        pass

    def _retry(self, action, *args):
        while True:
            try:
                return action(*args)
            except Exception as e:
                self._wait_for_retry(e)

    @abstractmethod
    def _get_pose(self):
        pass

    @abstractmethod
    def _move_to(self, pose):
        pass

    @abstractmethod
    def _close_gripper(self):
        pass

    @abstractmethod
    def _open_gripper(self):
        pass
