
import chess, cv2, math
import numpy as np
from config import *
from pathlib import Path
from src.types import *
from src.board import *
from src.transforms import *
from ultralytics import YOLO

RUNS_DIR = Path(__file__).parent.parent / "yolo" / "runs"


class Vision:
    def __init__(self):
        self.capture = cv2.VideoCapture(CAMERA_INDEX)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        detector_parameters = cv2.aruco.DetectorParameters()
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.tag_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5),
            detector_parameters
        )

        self.piece_detector = YOLO(RUNS_DIR / "obb" / "icon" / "weights" / "best.pt")
        self.piece_classifier = YOLO(RUNS_DIR / "classify" / "train" / "weights" / "best.pt")

    def set_autofocus_locked(self, locked: bool):
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0 if locked else 1)

    def detect_state(self, game_board: chess.Board):
        self.capture.grab()
        successful, frame = self.capture.read()
        if not successful:
            raise RuntimeError("Couldn't capture image")

        clean_frame = frame.copy()
        tags = self._detect_tags(frame)
        surface_to_camera = self._get_surface_to_camera(tags)

        if surface_to_camera is None:
            return DetectedState(frame=frame, board=None, pieces=None, parking_spot=None, surface_to_camera=None)

        pieces = self._detect_pieces(frame, clean_frame, surface_to_camera)
        parking_spot = self._get_parking_spot(frame, pieces, surface_to_camera)

        detected_board = set_board_pieces(game_board, pieces)

        return DetectedState(frame=frame, board=detected_board, pieces=pieces, parking_spot=parking_spot, surface_to_camera=surface_to_camera)

    def _detect_tags(self, frame):
        detected_corners, detected_ids, _ = self.tag_detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, detected_corners, detected_ids)
 
        tags = {}
        if detected_ids is not None:
            for corners, id in zip(detected_corners, detected_ids.flatten()):
                tags.setdefault(id, []).append(corners.reshape(4, 2))

        return tags

    def _get_surface_to_camera(self, tags):
        expected_ids = list(SURFACE_TAG_POINTS.keys())
        if set(expected_ids) - set(tags.keys()):
            return None

        success, rotation_vector, translation_vector = cv2.solvePnP(
            np.concatenate([SURFACE_TAG_POINTS[id] for id in expected_ids]),
            np.concatenate([tags[id][0] for id in expected_ids]),
            CAMERA_MATRIX,
            DIST_COEFFICIENTS
        )

        if not success:
            raise RuntimeError("Failed to solve PnP.")

        return rotation_vector_to_pose(rotation_vector, translation_vector)

    def _detect_pieces(self, frame, clean_frame, surface_to_camera):
        boxes = self.piece_detector.predict(clean_frame, verbose=False)[0].obb.xywhr.cpu().numpy()
        poses = [self._get_piece_pose(box, surface_to_camera) for box in boxes]
        boxes, poses = self._remove_false_positives(boxes, poses)
        if not boxes:
            return []

        crops = [
            self._crop_icon(clean_frame, centre_x, centre_y, max(width, height) * ICON_CROP_MARGIN_RATIO, math.degrees(rotation))
            for centre_x, centre_y, width, height, rotation in boxes
        ]
        classifications = self.piece_classifier.predict(crops, verbose=False)

        pieces = []
        for classification, box, pose in zip(classifications, boxes, poses):
            piece_name = classification.names[classification.probs.top1]
            piece_type = self._parse_piece_name(piece_name)
            pieces.append(DetectedPiece(type=piece_type, square=surface_to_square(pose), pose=pose))
            self._draw_piece(frame, box, surface_to_camera, pose, piece_type)

        return pieces

    def _remove_false_positives(self, boxes, poses):
        surface_limit = SURFACE_TAG_SPAN / 2 + SURFACE_TAG_SIZE / 2
        tag_centres = [points[:, :2].mean(axis=0) for points in SURFACE_TAG_POINTS.values()]
        tag_clearance = PIECE_DIAMETER / 2 + SURFACE_TAG_SIZE / 2

        kept_boxes, kept_poses = [], []
        for box, pose in zip(boxes, poses):
            centre = pose[:2, 3]
            outside_surface = np.any(np.abs(centre) > surface_limit)
            on_tag = any(np.linalg.norm(centre - tag_centre) < tag_clearance for tag_centre in tag_centres)
            if not outside_surface and not on_tag:
                kept_boxes.append(box)
                kept_poses.append(pose)

        return kept_boxes, kept_poses

    def _crop_icon(self, image, centre_x, centre_y, side_length, angle_degrees):
        transform = cv2.getRotationMatrix2D((centre_x, centre_y), angle_degrees, ICON_CROP_SIZE_PX / side_length)
        transform[:, 2] += ICON_CROP_SIZE_PX / 2 - np.array([centre_x, centre_y])
        return cv2.warpAffine(image, transform, (ICON_CROP_SIZE_PX, ICON_CROP_SIZE_PX), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _parse_piece_name(self, name):
        colour, piece_name = name.split("_")
        return chess.Piece(chess.PIECE_NAMES.index(piece_name), colour == "white")

    def _get_piece_pose(self, box, surface_to_camera):
        centre_x, centre_y, _, _, rotation = box
        centre = pixel_to_surface(np.array([centre_x, centre_y]), surface_to_camera, PIECE_HEIGHT)

        return roll_pitch_yaw_to_pose(centre, yaw=-rotation)

    def _get_parking_spot(self, frame, pieces, surface_to_camera):
        half_board = 4 * SQUARE_SIZE
        half_tag = SURFACE_TAG_SIZE / 2
        clearance = PIECE_DIAMETER / 2 + PARKING_TAG_MARGIN
        half_span_x, half_span_y = SURFACE_TAG_SPAN / 2

        inner_x = half_board + PIECE_DIAMETER / 2 + PARKING_BOARD_MARGIN
        outer_x = half_span_x + half_tag - clearance
        limit_y = half_span_y - half_tag - clearance

        parked = [piece.pose[:2, 3] for piece in pieces if piece.square is None]
        min_distance = PIECE_DIAMETER + PARKING_PIECE_CLEARANCE

        for x in np.arange(inner_x, outer_x + 1, PARKING_RESOLUTION):
            for side in (-1, 1):
                for y in np.arange(-limit_y, limit_y + 1, PARKING_RESOLUTION):
                    point = np.array([side * x, y])
                    if all(np.linalg.norm(point - other) >= min_distance for other in parked):
                        self._draw_parking_spot(frame, surface_to_camera, point)
                        return make_pose(np.eye(3), [point[0], point[1], 0.0])

        return None

    def _draw_piece(self, frame, box, surface_to_camera, pose, piece_type):
        centre_x, centre_y, width, height, rotation = box
        corners = cv2.boxPoints(((centre_x, centre_y), (width, height), math.degrees(rotation)))
        cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 0, 255), 1)

        center_pixel = camera_to_pixel(surface_to_camera @ pose)
        label = f"{'w' if piece_type.color else 'b'}.{chess.piece_name(piece_type.piece_type)}"
        cv2.circle(frame, center_pixel, 5, (0, 0, 255), -1)
        cv2.putText(frame, label, (center_pixel[0] + 8, center_pixel[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    def _draw_parking_spot(self, frame, surface_to_camera, point):
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        outline = [
            camera_to_pixel(surface_to_camera @ make_pose(np.eye(3), [
                point[0] + PIECE_DIAMETER / 2 * np.cos(angle),
                point[1] + PIECE_DIAMETER / 2 * np.sin(angle),
                0.0,
            ]))
            for angle in angles
        ]

        cv2.polylines(frame, [np.array(outline)], True, (0, 255, 0), 1)

    def quit(self):
        self.capture.release()

