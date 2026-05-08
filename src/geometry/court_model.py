"""
3D world coordinates of the court landmarks (FIBA spec, units: millimeters).

World frame:
    origin: center of the court at floor level (midline ∩ center-circle center).
    +X:     toward the "right" hoop (long axis of the court).
    +Y:     toward the stands side (short axis, away from the bench).
    +Z:     up (perpendicular to the floor).

FIBA dimensions used:
    court     28000 × 15000 mm
    lane      5800 × 4900 mm
    rim       1575 mm inside the endline, 3050 mm above the floor
    3pt arc   6750 mm radius from the rim center, straight portion 900 mm from sideline
    center    circle radius 1800 mm
"""
import numpy as np

from src.geometry.calibration_landmarks import LANDMARKS


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
}

assert set(WORLD_LANDMARKS_MM) == set(LANDMARKS), (
    "WORLD_LANDMARKS_MM keys must match LANDMARKS exactly. "
    f"Missing: {set(LANDMARKS) - set(WORLD_LANDMARKS_MM)}, "
    f"extra: {set(WORLD_LANDMARKS_MM) - set(LANDMARKS)}"
)
