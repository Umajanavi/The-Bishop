import os
import chess
import numpy as np
from config import *
from src.board import *
from src.transforms import *
from src.types import *

# Joint-limit protection (only active when the robot can report its joint angles).
# The last joint (the wrist) turns the gripper. Like the face follower, it may use the inner
# 60 % of its range freely; a grasp that would push it into the outer 40 % is turned by a
# quarter or half turn instead, which holds the square icon just as well.
WRIST_LIMIT_RAD = 2.8973
WRIST_FREE_ZONE = 0.6


class MotionPlanner:
    def __init__(self, robot):
        self.robot = robot
        self.viewing_pose = None
        self.viewing_joints = None
        self.camera_to_base = None
        self.configuration = None
        self.return_to_viewing_joints = False
        self.moved_since_look = False

    def set_viewing_pose(self):
        self.viewing_pose = self.robot.get_pose()
        self.viewing_joints = self.robot.get_joints()
        self.return_to_viewing_joints = self.viewing_joints is not None
        self.moved_since_look = False
        self.camera_to_base = self.viewing_pose @ CAMERA_TO_FLANGE

    def has_saved_viewing_pose(self):
        return os.path.exists("viewing_pose.npy")

    def save_viewing_pose(self):
        np.save("viewing_pose.npy", self.viewing_pose)

    def move_to_saved_viewing_pose(self):
        self.robot.move_to(np.load("viewing_pose.npy"))

    def look_at_board(self):
        self.robot.move_to(self.viewing_pose)
        self._restore_viewing_joints()
        self.configuration = RobotConfiguration(pose=self.viewing_pose, is_gripper_closed=False)

    def _restore_viewing_joints(self):
        """
        The same hand pose can be reached with different arm shapes, and after moving pieces the
        arm comes back in a slightly different shape each time. Over a game that drift carries
        the elbow joints to their limits. So after pieces were moved, put the arm back into the
        exact joint angles it had when the viewing pose was set (the face follower's "return to
        the start pose"). If that ever fails it switches itself off and the game goes on as before.
        """
        if not self.return_to_viewing_joints or not self.moved_since_look:
            return

        self.moved_since_look = False
        try:
            self.robot.move_to_joints(self.viewing_joints)
        except Exception as error:
            print(f"Joint-space return to the viewing pose switched off: {error}")
            self.return_to_viewing_joints = False

        self.robot.move_to(self.viewing_pose)

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
        reference_yaw += self._wrist_relief(state, transfer, reference_yaw)
        pick_pose = self._get_pick_pose(self._snap_pose_yaw(transfer.piece.pose, reference_yaw))
        place_pose = self._get_place_pose(state, transfer.square, reference_yaw)
        self.moved_since_look = True

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

    def _wrist_relief(self, state, transfer, reference_yaw):
        """
        Extra turn (0, a quarter turn either way, or a half turn) for the gripper in this
        transfer, so the wrist joint stays in the free middle of its range. The same turn is
        used for picking and placing, so the piece is put down exactly as it would have been.
        """
        if self.viewing_joints is None:
            return 0.0

        def furthest_wrist_angle(turn):
            yaw = reference_yaw + turn
            poses = [
                self._get_pick_pose(self._snap_pose_yaw(transfer.piece.pose, yaw)),
                self._get_place_pose(state, transfer.square, yaw),
            ]

            wrist = start = self.viewing_joints[6]
            furthest = 0.0
            for pose in poses:
                # The arm turns the short way round from wherever the wrist currently is.
                target = self._predict_wrist(self.camera_to_base @ state.surface_to_camera @ pose)
                wrist += (target - wrist + np.pi) % (2 * np.pi) - np.pi
                furthest = max(furthest, abs(wrist))

            # It must also be able to turn the short way back to the viewing pose afterwards.
            if abs(start - wrist) > np.pi - 0.3:
                return np.inf

            return furthest

        if furthest_wrist_angle(0.0) <= WRIST_FREE_ZONE * WRIST_LIMIT_RAD:
            return 0.0

        return min((0.0, np.pi / 2, -np.pi / 2, np.pi), key=furthest_wrist_angle)

    def _predict_wrist(self, base_pose):
        """
        Roughly the angle the wrist joint needs for a gripper pose in the base frame. Swinging the
        arm around its base turns the gripper by the same angle, and the wrist joint turns the
        gripper about its own axis, so the wrist has to make up whatever is left.
        """
        swing = np.arctan2(base_pose[1, 3], base_pose[0, 3]) - np.arctan2(self.viewing_pose[1, 3], self.viewing_pose[0, 3])
        neutral = (
            roll_pitch_yaw_to_pose([0.0, 0.0, 0.0], yaw=swing)
            @ self.viewing_pose
            @ roll_pitch_yaw_to_pose([0.0, 0.0, 0.0], yaw=-self.viewing_joints[6])
        )
        relative = neutral[:3, :3].T @ base_pose[:3, :3]
        return np.arctan2(relative[1, 0], relative[0, 0])

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
