import numpy as np

CAMERA_INDEX = 0
CAMERA_MATRIX = np.array([
    [ 952.157379300,  0.000000000,  660.198984117],
    [ 0.000000000,  947.518900298,  352.744430098],
    [ 0.000000000,  0.000000000,  1.000000000],
])

DIST_COEFFICIENTS = np.array([
    0.043934230,
    -0.238380021,
    -0.001029113,
    -0.000264255,
    0.253064833,
])

CAMERA_TO_FLANGE = np.array([
    [ 0.025259962, -0.999531351, -0.017291960,  53.770767447],
    [ 0.999680636,  0.025269138, -0.000312309, -8.376543839],
    [ 0.000749116, -0.017278548,  0.999850434, -35.451649079],
    [ 0.000000000,  0.000000000,  0.000000000,  1.000000000],
])


ROBOT_APPROACH_HEIGHT = 35
ROBOT_GRIP_HEIGHT = 25

SQUARE_SIZE = 34.875
SURFACE_TAG_SIZE = 15.0 
SURFACE_TAG_SPAN = np.array([591, 265])

PIECE_HEIGHT = 6
PIECE_DIAMETER = 30

ICON_CROP_SIZE_PX = 64
ICON_CROP_MARGIN_RATIO = 1.25

PARKING_BOARD_MARGIN = 20.0
PARKING_TAG_MARGIN = 2.0
PARKING_PIECE_CLEARANCE = 2.0
PARKING_RESOLUTION = 2.0


def _surface_tag_points(corner):
    center_x, center_y = SURFACE_TAG_SPAN / 2 * corner
    half_tag = SURFACE_TAG_SIZE / 2
    return np.array([
        [center_x - half_tag, center_y + half_tag, 0.0],
        [center_x + half_tag, center_y + half_tag, 0.0],
        [center_x + half_tag, center_y - half_tag, 0.0],
        [center_x - half_tag, center_y - half_tag, 0.0],
    ])

SURFACE_TAG_POINTS = {
    0: _surface_tag_points([-1,  1]),
    1: _surface_tag_points([ 1,  1]),
    2: _surface_tag_points([ 1, -1]),
    3: _surface_tag_points([-1, -1]),
}

TOTAL_PIECES = 34 # 16 white, 16 black + 2 extra queens