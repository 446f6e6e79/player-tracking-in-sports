"""
FIBA court geometry: world-frame landmark points and polyline primitives.

World frame:
    origin: center of the court at floor level (midline ∩ center-circle center).
    +X:     toward the "right" hoop (long axis of the court).
    +Y:     toward the stands side (short axis, away from the bench).
    +Z:     up (perpendicular to the floor).

FIBA dimensions (mm):
    court     28000 × 15000
    lane      5800 × 4900
    rim       1575 inside the endline, 3050 above the floor
    3pt arc   6750 radius from rim center, straight portion 900 from sideline
    center    circle radius 1800
    backboard 1800 wide × 1050 tall, front face 1200 inside the endline,
              bottom edge 2900 above the floor.
"""
from dataclasses import dataclass
import numpy as np

# Named landmark points used by extrinsic calibration (PnP) and the click UI.
WORLD_LANDMARKS_MM: dict[str, np.ndarray] = {
    # Court corners
    "corner_left_bench":   np.array([-14000.0, -7500.0,    0.0], dtype=np.float32),
    "corner_right_bench":  np.array([ 14000.0, -7500.0,    0.0], dtype=np.float32),
    "corner_left_stands":  np.array([-14000.0,  7500.0,    0.0], dtype=np.float32),
    "corner_right_stands": np.array([ 14000.0,  7500.0,    0.0], dtype=np.float32),

    # Center line and center circle
    "center_circle_bench":  np.array([     0.0, -1800.0,   0.0], dtype=np.float32),
    "center_circle_stands": np.array([     0.0,  1800.0,   0.0], dtype=np.float32),
    "center_line_bench":    np.array([     0.0, -7500.0,   0.0], dtype=np.float32),
    "center_line_stands":   np.array([     0.0,  7500.0,   0.0], dtype=np.float32),

    # Free-throw lane anchors (lane half-width = 4900/2 = 2450; ft line = endline ± 5800)
    "lane_left_endline_bench":   np.array([-14000.0, -2450.0, 0.0], dtype=np.float32),
    "lane_left_ft_bench":        np.array([ -8200.0, -2450.0, 0.0], dtype=np.float32),
    "lane_left_endline_stands":  np.array([-14000.0,  2450.0, 0.0], dtype=np.float32),
    "lane_left_ft_stands":       np.array([ -8200.0,  2450.0, 0.0], dtype=np.float32),
    "lane_right_endline_bench":  np.array([ 14000.0, -2450.0, 0.0], dtype=np.float32),
    "lane_right_ft_bench":       np.array([  8200.0, -2450.0, 0.0], dtype=np.float32),
    "lane_right_endline_stands": np.array([ 14000.0,  2450.0, 0.0], dtype=np.float32),
    "lane_right_ft_stands":      np.array([  8200.0,  2450.0, 0.0], dtype=np.float32),

    # Hoops and floor under the rim (rim center 1575 mm inside endline, 3050 mm up)
    "hoop_left_rim":          np.array([-12425.0, 0.0, 3050.0], dtype=np.float32),
    "floor_under_hoop_left":  np.array([-12425.0, 0.0,    0.0], dtype=np.float32),
    "hoop_right_rim":         np.array([ 12425.0, 0.0, 3050.0], dtype=np.float32),
    "floor_under_hoop_right": np.array([ 12425.0, 0.0,    0.0], dtype=np.float32),

    # Three-point line: straight portion 900 mm from the sideline (y = ±6600);
    # apex on the midline, 6750 mm from the rim center toward half-court.
    "three_pt_left_endline_bench":   np.array([-14000.0, -6600.0, 0.0], dtype=np.float32),
    "three_pt_left_apex":            np.array([ -5675.0,     0.0, 0.0], dtype=np.float32),
    "three_pt_left_endline_stands":  np.array([-14000.0,  6600.0, 0.0], dtype=np.float32),
    "three_pt_right_endline_bench":  np.array([ 14000.0, -6600.0, 0.0], dtype=np.float32),
    "three_pt_right_apex":           np.array([  5675.0,     0.0, 0.0], dtype=np.float32),
    "three_pt_right_endline_stands": np.array([ 14000.0,  6600.0, 0.0], dtype=np.float32),

    # Backboard outer corners. Front face 1200 mm inside the endline (x = ±12800),
    # 1800 mm wide centered on the midline (y = ±900), bottom edge 2900 mm and
    # top edge 3950 mm above the floor.
    "backboard_left_top_bench":      np.array([-12800.0, -900.0, 3950.0], dtype=np.float32),
    "backboard_left_top_stands":     np.array([-12800.0,  900.0, 3950.0], dtype=np.float32),
    "backboard_left_bottom_bench":   np.array([-12800.0, -900.0, 2900.0], dtype=np.float32),
    "backboard_left_bottom_stands":  np.array([-12800.0,  900.0, 2900.0], dtype=np.float32),
    "backboard_right_top_bench":     np.array([ 12800.0, -900.0, 3950.0], dtype=np.float32),
    "backboard_right_top_stands":    np.array([ 12800.0,  900.0, 3950.0], dtype=np.float32),
    "backboard_right_bottom_bench":  np.array([ 12800.0, -900.0, 2900.0], dtype=np.float32),
    "backboard_right_bottom_stands": np.array([ 12800.0,  900.0, 2900.0], dtype=np.float32),
}

# Source of truth for the named landmark set. Click order = insertion order.
LANDMARKS: tuple[str, ...] = tuple(WORLD_LANDMARKS_MM)

# Public court-geometry constants — single source of truth for callers outside
# this module (visualization, calibration UI, scene renderers).
COURT_HALF_LENGTH_MM = 14000.0  # Half the length of the FIBA court (mm)
COURT_HALF_WIDTH_MM = 7500.0    # Half the width of the FIBA court (mm)
MM_PER_METER = 1000.0           # Scalar used by renderers that work in meters.

# Court line primitives (used for rendering)
_HALF_LEN = COURT_HALF_LENGTH_MM
_HALF_WID = COURT_HALF_WIDTH_MM
_LANE_HALF_W = 2450.0           # Half the width of the free-throw lane (mm)
_LANE_X_INNER = 8200.0          # X coordinate of the inner edge of the ft line (mm)
_THREEPT_R = 6750.0             # Radius of the three-point arc from the rim center (mm)
_THREEPT_STRAIGHT_Y = 6600.0    # Y coordinate of the straight portion of the three-point line (mm)
_CENTER_CIRCLE_R = 1800.0       # Radius of the center circle (mm)
_BACKBOARD_X = 12800.0          # X coordinate of the backboard front face (mm)
_BACKBOARD_HALF_W = 900.0       # Half the width of the backboard (mm)
_BACKBOARD_Z_BOTTOM = 2900.0    # Z coordinate of the backboard bottom edge (mm)
_BACKBOARD_Z_TOP = 3950.0       # Z coordinate of the backboard top edge (mm)
# Rim center X positions
LEFT_RIM_X = -12425.0           # X coordinate of the left rim center (mm)
RIGHT_RIM_X = 12425.0           # X coordinate of the right rim center (mm)


@dataclass(frozen=True)
class CourtLine:
    """A named polyline primitive on the court, with metadata about whether it's closed (loop) or open."""
    name: str
    closed: bool
    pts: np.ndarray  # (N, 3) float32, world mm


def _circle(cx: float, cy: float, cz: float, r: float, N: int = 64) -> np.ndarray:
    """Generate a circle primitive as an (N, 3) array of XYZ points (mm)."""
    # Divide the full circle into N equal segments, excluding the endpoint to avoid duplicate points for closed loops.
    thetas = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    
    # Compute the XYZ coordinates of the circle points based on the center (cx, cy, cz) and radius r.
    return np.stack([
        cx + r * np.cos(thetas),
        cy + r * np.sin(thetas),
        np.full_like(thetas, cz),
    ], axis=1).astype(np.float32)


def _three_point_polyline(rim_x: float, n: int = 64) -> np.ndarray:
    """3pt line: consists of a straight portion parallel to the sideline, then an arc of radius _THREEPT_R around the rim center"""
    y_end = _THREEPT_STRAIGHT_Y
    dx = float(np.sqrt(_THREEPT_R * _THREEPT_R - y_end * y_end))
    endline_x = -_HALF_LEN if rim_x < 0 else _HALF_LEN
    theta_max = float(np.arctan2(y_end, dx))
    if rim_x < 0:
        thetas = np.linspace(-theta_max, theta_max, n)
    else:
        thetas = np.linspace(np.pi + theta_max, np.pi - theta_max, n)
    arc = np.stack([
        rim_x + _THREEPT_R * np.cos(thetas),
        _THREEPT_R * np.sin(thetas),
        np.zeros_like(thetas),
    ], axis=1)
    return np.vstack([
        [[endline_x, -y_end, 0.0]],
        arc,
        [[endline_x,  y_end, 0.0]],
    ]).astype(np.float32)


def _court_outline() -> np.ndarray:
    """Court perimeter as a closed loop (mm)."""
    return np.array([
        [-_HALF_LEN, -_HALF_WID, 0],
        [ _HALF_LEN, -_HALF_WID, 0],
        [ _HALF_LEN,  _HALF_WID, 0],
        [-_HALF_LEN,  _HALF_WID, 0],
    ], dtype=np.float32)


def _midline() -> np.ndarray:
    """Midline as an open polyline (mm)."""
    return np.array([[0, -_HALF_WID, 0], [0, _HALF_WID, 0]], dtype=np.float32)


def _center_circle(n: int = 64) -> np.ndarray:
    """Center circle as a closed loop (mm)."""
    return _circle(0.0, 0.0, 0.0, _CENTER_CIRCLE_R, n)


def _lane_left() -> np.ndarray:
    """Left lane as a closed loop (mm)."""
    return np.array([
        [-_HALF_LEN,     -_LANE_HALF_W, 0],
        [-_LANE_X_INNER, -_LANE_HALF_W, 0],
        [-_LANE_X_INNER,  _LANE_HALF_W, 0],
        [-_HALF_LEN,      _LANE_HALF_W, 0],
    ], dtype=np.float32)


def _lane_right() -> np.ndarray:
    """Right lane as a closed loop (mm)."""
    return np.array([
        [ _HALF_LEN,     -_LANE_HALF_W, 0],
        [ _LANE_X_INNER, -_LANE_HALF_W, 0],
        [ _LANE_X_INNER,  _LANE_HALF_W, 0],
        [ _HALF_LEN,      _LANE_HALF_W, 0],
    ], dtype=np.float32)


def _backboard_left() -> np.ndarray:
    """Left backboard as a closed loop (mm)."""
    x = -_BACKBOARD_X
    return np.array([
        [x, -_BACKBOARD_HALF_W, _BACKBOARD_Z_BOTTOM],
        [x,  _BACKBOARD_HALF_W, _BACKBOARD_Z_BOTTOM],
        [x,  _BACKBOARD_HALF_W, _BACKBOARD_Z_TOP],
        [x, -_BACKBOARD_HALF_W, _BACKBOARD_Z_TOP],
    ], dtype=np.float32)


def _backboard_right() -> np.ndarray:
    """Right backboard as a closed loop (mm)."""
    x = _BACKBOARD_X
    return np.array([
        [x, -_BACKBOARD_HALF_W, _BACKBOARD_Z_BOTTOM],
        [x,  _BACKBOARD_HALF_W, _BACKBOARD_Z_BOTTOM],
        [x,  _BACKBOARD_HALF_W, _BACKBOARD_Z_TOP],
        [x, -_BACKBOARD_HALF_W, _BACKBOARD_Z_TOP],
    ], dtype=np.float32)


def court_lines() -> list[CourtLine]:
    """
    Return a list of CourtLine representing the standard FIBA court geometry, with points in world mm coordinates.
    This is the entry point for other modules to get access to the canonical court geometry primitives.
    """
    return [
        CourtLine("court_outline",  True,  _court_outline()),
        CourtLine("midline",        False, _midline()),
        CourtLine("center_circle",  True,  _center_circle()),
        CourtLine("lane_left",      True,  _lane_left()),
        CourtLine("lane_right",     True,  _lane_right()),
        CourtLine("three_pt_left",  False, _three_point_polyline(LEFT_RIM_X)),
        CourtLine("three_pt_right", False, _three_point_polyline(RIGHT_RIM_X)),
    ]
