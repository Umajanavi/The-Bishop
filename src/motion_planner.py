import os
import chess
import numpy as np
from config import *
from src.board import *
from src.transforms import *
from src.types import *


class MotionPlanner:
    def __init__(self, robot):
        self.robot = robot
        self.viewing_pose = None
        self.camera_to_base = None
        self.configuration = None

    def set_viewing_pose(self):
        self.viewing_pose = self.robot.get_pose()
        self.camera_to_base = self.viewing_pose @ CAMERA_TO_FLANGE

    def has_saved_viewing_pose(self):
        return os.path.exists("viewing_pose.npy")

    def save_viewing_pose(self):
        np.save("viewing_pose.npy", self.viewing_pose)

    def move_to_saved_viewing_pose(self):
        self.robot.move_to(np.load("viewing_pose.npy"))

    def look_at_board(self):
        self.robot.move_to(self.viewing_pose)
        self.configuration = RobotConfiguration(pose=self.viewing_pose, is_gripper_closed=False)

    def plan_transfer(self, state, target_board):
        surplus, wanted = diff_board(state.board, target_board)

        for square in wanted:
            if state.board.piece_at(square) is None:
                piece = self._get_source_piece(state, square, surplus, wanted)
                if piece is not None:
                    return Transfer(piece=piece, square=square)

        for square in wanted:
            if state.board.piece_at(square) is not None:
                return self._park(state, square)

        for square in surplus:
            if square not in wanted:
                return self._park(state, square)

        return None

    def make_transfer(self, state, transfer):
        reference_yaw = self._reference_yaw(state)
        pick_pose = self._get_pick_pose(self._snap_pose_yaw(transfer.piece.pose, reference_yaw))
        place_pose = self._get_place_pose(state, transfer.square, reference_yaw)

        yield self.robot.open_gripper()
        yield from self._move_to(state, self._approach(pick_pose))
        yield from self._move_to(state, pick_pose)
        yield from self._close_gripper()

        yield from self._move_to(state, self._approach(pick_pose))
        yield from self._move_to(state, self._approach(place_pose))
        yield from self._move_to(state, place_pose)
        yield from self._open_gripper()

        yield from self._move_to(state, self._approach(place_pose))

    def _get_source_piece(self, state, square, surplus, wanted):
        movable = [
            piece for piece in state.pieces
            if piece.square is None or piece.square in surplus
        ]

        candidates = [piece for piece in movable if piece.type == wanted[square]]

        if not candidates:
            return None

        point = square_to_surface(square)[:2]
        return min(candidates, key=lambda piece: (
            piece.square is None,
            np.linalg.norm(piece.pose[:2, 3] - point),
        ))

    def _park(self, state, square):
        return Transfer(piece=self._get_piece_on_square(state.pieces, square), square=None)

    def _get_piece_on_square(self, pieces, square):
        for piece in pieces:
            if piece.square == square:
                return piece

        raise RuntimeError("No piece detected on " + chess.square_name(square) + ".")

    def _get_place_pose(self, state, square, reference_yaw):
        position = self._get_target_position(state, square)
        target_pose = self._snap_pose_yaw(roll_pitch_yaw_to_pose(position, yaw=0), reference_yaw)
        return self._get_pick_pose(target_pose)

    def _get_pick_pose(self, pose):
        return move_along_axis(pose, 2, ROBOT_GRIP_HEIGHT) @ roll_pitch_yaw_to_pose([0.0, 0.0, 0.0], roll=np.pi)

    def _reference_yaw(self, state):
        reference_pose = np.linalg.inv(state.surface_to_camera) @ np.linalg.inv(self.camera_to_base) @ self.viewing_pose
        return np.arctan2(reference_pose[1, 0], reference_pose[0, 0])

    def _snap_pose_yaw(self, pose, reference_yaw):
        yaw = np.arctan2(pose[1, 0], pose[0, 0])
        snapped_yaw = reference_yaw + (yaw - reference_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return roll_pitch_yaw_to_pose(pose[:3, 3], yaw=snapped_yaw)

    def _get_target_position(self, state, square):
        if square is None:
            if state.parking_spot is None:
                raise RuntimeError("Nowhere left to put a piece.")
            return state.parking_spot[:, 3]

        return square_to_surface(square)

    def _approach(self, pose):
        return move_along_axis(pose, 2, -ROBOT_APPROACH_HEIGHT)

    def _move_to(self, state, pose):
        base_pose = self.camera_to_base @ state.surface_to_camera @ pose
        self.configuration = self.configuration._replace(pose=base_pose)
        yield self.configuration
        self.robot.move_to(base_pose)

    def _close_gripper(self):
        self.configuration = self.configuration._replace(is_gripper_closed=True)
        yield self.configuration
        self.robot.close_gripper()

    def _open_gripper(self):
        self.configuration = self.configuration._replace(is_gripper_closed=False)
        yield self.configuration
        self.robot.open_gripper()
