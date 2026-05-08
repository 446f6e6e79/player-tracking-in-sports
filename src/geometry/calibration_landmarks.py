"""
UI metadata for the landmark click picker.

The court 3D model and the canonical `LANDMARKS` ordering live in
`src.geometry.court`; this module only adds the normalized inset-diagram
positions and the click-mapping type alias used by the picker UI.
"""
from src.geometry.court import LANDMARKS


# Normalized [0, 1] coordinates on the court-diagram PNG, used to render the
# next-target highlight in the click picker. Values are arbitrary visual hints —
# the **label** is what tells the user which physical point to click.
LANDMARK_DIAGRAM_NORM: dict[str, tuple[float, float]] = {
    # Court corners (4)
    "corner_left_bench":   (0.020, 0.900),
    "corner_right_bench":  (0.980, 0.900),
    "corner_left_stands":  (0.020, 0.150),
    "corner_right_stands": (0.980, 0.150),

    # Center line (4)
    "center_circle_bench":   (0.500, 0.615),
    "center_circle_stands":  (0.500, 0.435),
    "center_line_bench":     (0.500, 0.900),
    "center_line_stands":    (0.500, 0.150),

    # Free-throw lane anchors (8)
    "lane_left_endline_bench":   (0.020, 0.615),
    "lane_left_ft_bench":        (0.205, 0.615),
    "lane_left_endline_stands":  (0.020, 0.455),
    "lane_left_ft_stands":       (0.205, 0.455),
    "lane_right_endline_bench":  (0.980, 0.615),
    "lane_right_ft_bench":       (0.790, 0.615),
    "lane_right_endline_stands": (0.980, 0.455),
    "lane_right_ft_stands":      (0.790, 0.455),

    # Hoops + floor-below anchors (4)
    "hoop_left_rim":          (0.065, 0.520),
    "floor_under_hoop_left":  (0.065, 0.520),
    "hoop_right_rim":         (0.930, 0.520),
    "floor_under_hoop_right": (0.930, 0.520),

    # Three-point arcs (6)
    "three_pt_left_endline_bench":   (0.020, 0.840),
    "three_pt_left_apex":            (0.300, 0.510),
    "three_pt_left_endline_stands":  (0.020, 0.220),
    "three_pt_right_endline_bench":  (0.980, 0.840),
    "three_pt_right_apex":           (0.700, 0.510),
    "three_pt_right_endline_stands": (0.980, 0.220),

    # Backboard outer corners (8). Top vs. bottom is invisible in this top-down
    # inset, so the (norm_x, norm_y) offsets are just disambiguation hints —
    # the label tells the user which corner to click.
    "backboard_left_top_bench":      (0.055, 0.555),
    "backboard_left_top_stands":     (0.055, 0.485),
    "backboard_left_bottom_bench":   (0.075, 0.555),
    "backboard_left_bottom_stands":  (0.075, 0.485),
    "backboard_right_top_bench":     (0.945, 0.555),
    "backboard_right_top_stands":    (0.945, 0.485),
    "backboard_right_bottom_bench":  (0.925, 0.555),
    "backboard_right_bottom_stands": (0.925, 0.485),
}

# Drift guard: every world landmark must have a diagram coord, and vice-versa.
assert set(LANDMARK_DIAGRAM_NORM) == set(LANDMARKS), (
    "LANDMARK_DIAGRAM_NORM keys must match LANDMARKS exactly. "
    f"Missing: {set(LANDMARKS) - set(LANDMARK_DIAGRAM_NORM)}, "
    f"extra: {set(LANDMARK_DIAGRAM_NORM) - set(LANDMARKS)}"
)


# A landmark -> clicked pixel mapping; None means the user skipped that landmark.
LandmarkClicks = dict[str, tuple[float, float] | None]
