from dataclasses import dataclass


@dataclass(frozen=True)
class ScalePair:
    """Two landmarks whose physical distance is fixed by the FIBA court spec."""
    a: str
    b: str
    distance_mm: float


# FIBA dimensions: court 28 m × 15 m, lane 5.8 m × 4.9 m, hoop 3.05 m above floor.
# Each pair contributes one estimate of the metric scale; the final scale is
# their mean, so misclicks on a single landmark only nudge the average.
SCALE_PAIRS: list[ScalePair] = [
    # Court sidelines (28 m)
    ScalePair("corner_left_bench",  "corner_right_bench",  28000.0),
    ScalePair("corner_left_stands", "corner_right_stands", 28000.0),
    # Court endlines / center line (15 m)
    ScalePair("corner_left_bench",  "corner_left_stands",  15000.0),
    ScalePair("corner_right_bench", "corner_right_stands", 15000.0),
    ScalePair("center_line_bench",  "center_line_stands",  15000.0),
    # Free-throw lane length (5.8 m)
    ScalePair("lane_left_endline_bench",   "lane_left_ft_bench",   5800.0),
    ScalePair("lane_left_endline_stands",  "lane_left_ft_stands",  5800.0),
    ScalePair("lane_right_endline_bench",  "lane_right_ft_bench",  5800.0),
    ScalePair("lane_right_endline_stands", "lane_right_ft_stands", 5800.0),
    # Free-throw lane width (4.9 m)
    ScalePair("lane_left_endline_bench",  "lane_left_endline_stands",  4900.0),
    ScalePair("lane_left_ft_bench",       "lane_left_ft_stands",       4900.0),
    ScalePair("lane_right_endline_bench", "lane_right_endline_stands", 4900.0),
    ScalePair("lane_right_ft_bench",      "lane_right_ft_stands",      4900.0),
    # Hoop drop (3.05 m, vertical)
    ScalePair("hoop_left_rim",  "floor_under_hoop_left",  3050.0),
    ScalePair("hoop_right_rim", "floor_under_hoop_right", 3050.0),
]

# Ordered dict mapping landmark -> (norm_x, norm_y) on the court-diagram PNG.
# Iteration order is the click order; values are normalized [0, 1] coordinates
# for the inset diagram (scaled to pixel coords at render time).
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
}

LANDMARKS: tuple[str, ...] = tuple(LANDMARK_DIAGRAM_NORM)

# A landmark -> clicked pixel mapping; None means the user skipped that landmark.
LandmarkClicks = dict[str, tuple[float, float] | None]
