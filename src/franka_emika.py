import time
import numpy as np
import franky as fk
from scipy.spatial.transform import Rotation
from src.robot import Robot
from src.safety import joint_margins, joint_velocity_command, panda_fk
from src.transforms import *

# "servo":     move the hand the way the face follower does, with joint velocities and the joint-limit
#              logic in src/safety.py (null-space pull, slowing near limits, hard margin).
# "cartesian": the original franky Cartesian move, with no joint-limit logic.
# "dry_run":   reads the robot and prints what the servo move would do, but never moves the arm.
MOVE_MODE = "servo"
DEBUG_SERVO = True                  # print one line at the start and end of every servo move

SERVO_INTERVAL_S = 0.05             # how often the loop runs and sends a velocity command
SERVO_REFRESH_S = 0.4               # an unchanged command is re-sent at least this often
POSITION_GAIN = 3.5                 # 1/s: wanted hand speed per metre still to go along the path
LATERAL_GAIN = 6.0                  # 1/s: pulls the hand back onto the straight path to the target
ROTATION_GAIN = 2.0                 # 1/s: wanted turning speed per radian of orientation error
MAX_LINEAR_SPEED_M_S = 0.10
SOFT_LANDING_BASE_M_S = 0.010       # speed limit near the target is this plus ...
SOFT_LANDING_GAIN = 1.0             # ... this (1/s) times the distance still to go
MAX_ANGULAR_SPEED_RAD_S = 0.35
MAX_JOINT_SPEED_RAD_S = 0.20        # franky clips joints at 0.26 rad/s at the 0.1 dynamics factor used here
RAMP_UP_S = 1.0                     # speed builds up over this time at the start of a move
POSITION_TOLERANCE_M = 0.001        # the real arm creeps about 0.5 mm between moves even at rest, so 1 mm is as exact as it gets
ROTATION_TOLERANCE_RAD = 0.015
CLOSE_ENOUGH_M = 0.003              # if the arm cannot get closer than this and has stopped improving, the move counts as done
CLOSE_ENOUGH_RAD = 0.05
MIN_COMMAND_CHANGE_RAD_S = 0.0005   # the corrections near the target are tiny but must still be sent
MOVE_TIMEOUT_S = 40.0
STALL_WINDOW_S = 2.0                # every this many seconds the arm must have got at least ...
STALL_MIN_PROGRESS_M = 0.004        # ... this much closer (2 mm/s). A joint limit slows the arm to a crawl, never to a halt.
STALL_IGNORE_BELOW_M = 0.005        # the last few millimetres of a move are slow by design
BLOCKED_TIMEOUT_S = 1.0             # a joint limit may hold the move back this long before it is stopped
MODEL_TOLERANCE_M = 0.004           # the kinematic model must agree with the robot's own pose this well
MODEL_TOLERANCE_RAD = 0.03
CONTACT_ZONE_M = 0.008              # touchdown is only recognised this close to the target (e.g. the table surface)
CONTACT_WINDOW_S = 0.4              # the hand must have been commanded to move over this long ...
CONTACT_MIN_COMMANDED_M = 0.0012    # ... by at least this far in total ...
CONTACT_MOVED_FRACTION = 0.25       # ... yet moved less than this fraction of it
NO_ERROR_DETAILS ="no error details available"


class FrankaEmika(Robot):
    def __init__(self, wait_for_retry, ip="172.16.0.2"):
        super().__init__(wait_for_retry)
        self._arm = fk.Robot(ip)
        self._gripper = fk.Gripper(ip)
        self._flange_to_hand = None

        self._arm.recover_from_errors()
        self._arm.relative_dynamics_factor = 0.1

    def _get_pose(self):
        self._arm.recover_from_errors()
        pose = self._arm.current_cartesian_state.pose.end_effector_pose.matrix
        return fix_rotation_error(pose * 1000)

    def _move_to(self, pose):
        if MOVE_MODE in ("servo", "dry_run"):
            self._servo_to(pose)
        else:
            self._cartesian_to(pose)

    def _cartesian_to(self, pose):
        self._arm.stop()
        self._arm.recover_from_errors()
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        motion = fk.CartesianMotion(fk.Affine(pose[:3, 3] * 0.001, quat), reference_type=fk.ReferenceType.Absolute)
        self._arm.move(motion)

    def _servo_to(self, pose):
        target = np.array(pose, dtype=float)
        target[:3, 3] *= 0.001

        self._arm.stop()
        self._arm.recover_from_errors()
        self._learn_hand_offset()

        began = last_sent = time.time()
        last_velocity = np.zeros(7)
        blocked_since = None
        start = path = None
        checkpoint_progress, checkpoint_time, note = None, began, ""
        commands_sent = 0
        announced = False
        history = []                     # (time, hand position, commanded linear speed) of the last cycles

        try:
            while True:
                now = time.time()
                joints = self._read_joints()
                hand = self._read_hand_pose()
                self._check_model(joints, hand)

                if start is None:
                    start = hand[:3, 3].copy()
                    path = target[:3, 3] - start
                    length = float(np.linalg.norm(path))
                    path = path / length if length > 1e-6 else None

                position_error = target[:3, 3] - hand[:3, 3]
                rotation_error = Rotation.from_matrix(target[:3, :3] @ hand[:3, :3].T).as_rotvec()
                distance, angle = np.linalg.norm(position_error), np.linalg.norm(rotation_error)

                # Touchdown: close to the target, commanded to move, but the hand is no longer moving. The
                # hand is resting on the board or a piece, so stop pushing instead of building up force.
                if (distance < CONTACT_ZONE_M and distance > POSITION_TOLERANCE_M and angle < CLOSE_ENOUGH_RAD
                        and now - began > RAMP_UP_S + 0.5
                        and self._touched_down(history, hand[:3, 3], now)):
                    if DEBUG_SERVO:
                        print(f"[servo] touchdown: the hand stopped {distance * 1000:.1f} mm short of the target "
                              f"(resting on something), so the move ends here")
                    break

                if commands_sent:
                    reason = self._robot_error_reason()
                    if reason != NO_ERROR_DETAILS and distance < CONTACT_ZONE_M:
                        if DEBUG_SERVO:
                            print(f"[servo] the robot's contact detection tripped {distance * 1000:.1f} mm from the target "
                                  f"({reason}); treated as touchdown")
                        break
                    if reason != NO_ERROR_DETAILS:
                        raise RuntimeError(
                            f"The robot stopped itself ({reason}) after {now - began:.1f} s, with {distance * 1000:.1f} mm "
                            f"and {np.degrees(angle):.1f} degrees still to go. Hand at z = {hand[2, 3] * 1000:.1f} mm, "
                            f"target z = {target[2, 3] * 1000:.1f} mm (robot base frame). "
                            "A cartesian_reflex is the robot's contact detection: the hand pressed on something.")

                if DEBUG_SERVO and not announced:
                    announced = True
                    print(f"[servo] start: {distance * 1000:.1f} mm and {np.degrees(angle):.1f} deg to go | "
                          f"joint margins {np.round(joint_margins(joints), 2).tolist()}")

                if distance < POSITION_TOLERANCE_M and angle < ROTATION_TOLERANCE_RAD:
                    break
                if now - began > MOVE_TIMEOUT_S:
                    if distance < CLOSE_ENOUGH_M and angle < CLOSE_ENOUGH_RAD:
                        break
                    raise RuntimeError(
                        f"The arm did not reach its target in {MOVE_TIMEOUT_S:.0f} s "
                        f"(still {distance * 1000:.1f} mm and {np.degrees(angle):.1f} degrees away)")

                # Stalled: a joint limit can slow the arm to a crawl without ever blocking it outright.
                progress = distance + 0.05 * angle
                if checkpoint_progress is None:
                    checkpoint_progress, checkpoint_time = progress, now
                elif now - checkpoint_time >= STALL_WINDOW_S:
                    if distance < CLOSE_ENOUGH_M and angle < CLOSE_ENOUGH_RAD and checkpoint_progress - progress < POSITION_TOLERANCE_M:
                        break
                    if checkpoint_progress - progress < STALL_MIN_PROGRESS_M and progress > STALL_IGNORE_BELOW_M:
                        raise RuntimeError(
                            f"The arm is barely moving ({note.replace('SLOWED: ', '') or 'no joint limit reported'}), "
                            f"still {distance * 1000:.1f} mm and {np.degrees(angle):.1f} degrees from its target. "
                            f"Joint margins: {np.round(joint_margins(joints), 2).tolist()}")
                    checkpoint_progress, checkpoint_time = progress, now

                # Follow the straight line from where the move started to the target: speed along it, and a
                # stronger pull back onto it, so the hand carrying a piece doesn't wander off the path.
                if path is None:
                    wanted = position_error * POSITION_GAIN
                else:
                    progress_along = float(np.dot(hand[:3, 3] - start, path))
                    along = np.clip((length - progress_along) * POSITION_GAIN, -MAX_LINEAR_SPEED_M_S, MAX_LINEAR_SPEED_M_S)
                    sideways = (start + progress_along * path) - hand[:3, 3]
                    wanted = path * along + sideways * LATERAL_GAIN

                ramp = 1.0 if MOVE_MODE == "dry_run" else min(1.0, (now - began) / RAMP_UP_S)
                # Soft landing: the closer to the target, the slower, so a light touch of a piece or the board
                # at the end of a descent doesn't trip the robot's contact detection.
                landing_speed = SOFT_LANDING_BASE_M_S + SOFT_LANDING_GAIN * distance
                linear = _limited(wanted, min(MAX_LINEAR_SPEED_M_S, landing_speed)) * ramp
                angular = _limited(rotation_error * ROTATION_GAIN, MAX_ANGULAR_SPEED_RAD_S) * ramp
                history.append((now, hand[:3, 3].copy(), float(np.linalg.norm(linear))))
                del history[:-12]

                velocity, note, blocked = joint_velocity_command(
                    joints, np.concatenate([linear, angular]), max_speed=MAX_JOINT_SPEED_RAD_S)

                if blocked:
                    blocked_since = blocked_since or now
                    if now - blocked_since > BLOCKED_TIMEOUT_S:
                        raise RuntimeError(
                            (note.replace("SLOWED: ", "") or "a joint limit") + " - this move is not possible from here. "
                            f"Joint margins: {np.round(joint_margins(joints), 2).tolist()}")
                else:
                    blocked_since = None

                if MOVE_MODE == "dry_run":
                    print(f"[servo dry run] would send joint velocities {np.round(velocity, 3).tolist()} rad/s "
                          f"(largest {np.max(np.abs(velocity)):.3f}) {note}; the arm is not moved")
                    return

                if np.max(np.abs(velocity - last_velocity)) >= MIN_COMMAND_CHANGE_RAD_S or now - last_sent >= SERVO_REFRESH_S:
                    try:
                        self._arm.move(fk.JointVelocityMotion(velocity.astype(float)), asynchronous=True)
                    except Exception as error:
                        if distance < CONTACT_ZONE_M:
                            if DEBUG_SERVO:
                                print(f"[servo] the robot refused a command {distance * 1000:.1f} mm from the target "
                                      f"({self._robot_error_reason()}); treated as touchdown")
                            break
                        raise RuntimeError(
                            f"The robot stopped itself ({self._robot_error_reason()}) with {distance * 1000:.1f} mm "
                            f"and {np.degrees(angle):.1f} degrees still to go: {error}") from error
                    last_velocity, last_sent = velocity, now
                    commands_sent += 1

                time.sleep(SERVO_INTERVAL_S)
        finally:
            # Only stop a motion that was started. Stopping a robot that is already at rest would
            # start a new velocity-control session for nothing.
            if commands_sent:
                self._stop_servo()
            if DEBUG_SERVO:
                print(f"[servo] done after {time.time() - began:.1f} s, {commands_sent} velocity commands")

    @staticmethod
    def _touched_down(history, position, now):
        """True if the hand was commanded to move over the last moments but has hardly moved at all."""
        recent = [entry for entry in history if now - entry[0] <= CONTACT_WINDOW_S]
        if len(recent) < 2 or now - recent[0][0] < CONTACT_WINDOW_S * 0.75:
            return False
        elapsed = now - recent[0][0]
        commanded = float(np.mean([entry[2] for entry in recent])) * elapsed
        moved = float(np.linalg.norm(position - recent[0][1]))
        return commanded > CONTACT_MIN_COMMANDED_M and moved < CONTACT_MOVED_FRACTION * commanded

    def _stop_servo(self):
        try:
            self._arm.move(fk.JointVelocityStopMotion())
        except Exception as error:
            if "User stopped" in str(error) or "Reflex" in str(error):
                return                  # already stopped (by the user or by the robot's own reflex)
            try:
                self._arm.move(fk.JointVelocityMotion(np.zeros(7)), asynchronous=True)
                self._arm.join_motion()
            except Exception as second_error:
                print(f"Warning: could not stop the arm cleanly: {error} / {second_error}")
                print(f"[servo] why the robot is in reflex: {self._robot_error_reason()}")

    def _robot_error_reason(self):
        """Ask franky/libfranka which errors the robot reports (best effort, for the console only)."""
        found = []
        for owner_name in ("_arm", "state"):
            try:
                owner = self._arm if owner_name == "_arm" else self._arm.state
            except Exception:
                continue
            for attribute in ("current_errors", "last_motion_errors"):
                try:
                    value = getattr(owner, attribute)
                    value = value() if callable(value) else value
                    if str(value) not in ("", "[]", "None"):
                        found.append(f"{attribute}: {value}")
                except Exception:
                    pass
        return "; ".join(found) or NO_ERROR_DETAILS

    def _read_joints(self):
        return np.array(self._arm.current_joint_state.position, dtype=float)

    def _read_hand_pose(self):
        return np.array(self._arm.current_cartesian_state.pose.end_effector_pose.matrix, dtype=float)

    def _learn_hand_offset(self):
        """
        The robot reports the pose of its hand (end effector). The kinematic model works on the
        flange, so the fixed transform between the two is read from the robot once, which also
        covers whatever gripper and end-effector setup is configured in Desk.
        """
        if self._flange_to_hand is None:
            self._flange_to_hand = np.linalg.inv(panda_fk(self._read_joints())) @ self._read_hand_pose()

    def _check_model(self, joints, hand):
        """The joint-limit logic is only as good as the kinematic model: stop if it disagrees with the robot."""
        model = panda_fk(joints) @ self._flange_to_hand
        offset = np.linalg.norm(model[:3, 3] - hand[:3, 3])
        turn = np.linalg.norm(Rotation.from_matrix(model[:3, :3] @ hand[:3, :3].T).as_rotvec())
        if offset > MODEL_TOLERANCE_M or turn > MODEL_TOLERANCE_RAD:
            raise RuntimeError(
                f"The kinematic model is {offset * 1000:.1f} mm and {np.degrees(turn):.1f} degrees off the robot's own pose. "
                "Set MOVE_MODE = \"cartesian\" in src/franka_emika.py to use the original move.")

    def _close_gripper(self):
        self._arm.recover_from_errors()
        self._gripper.grasp(0.015, 0.02, 20.0)

    def _open_gripper(self):
        self._arm.recover_from_errors()
        self._gripper.move(0.02, 0.02)


def _limited(vector, maximum):
    length = np.linalg.norm(vector)
    return vector if length <= maximum else vector * (maximum / length)
