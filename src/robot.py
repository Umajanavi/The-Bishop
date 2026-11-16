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
