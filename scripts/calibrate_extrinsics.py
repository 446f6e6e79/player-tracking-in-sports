"""
Two-camera calibration of camera extrinsics from cross-camera correspondences.
The user clicks the same set of physical landmarks in both cameras (no world
knowledge required, just "click the same thing in each view").

Pipeline:
    1. The anchor camera (default cam_13) defines the world frame: R=I, T=0.
    2. User clicks the LANDMARKS list in both anchor and the other camera.
    3. Essential matrix + recoverPose -> relative pose of the second camera
       (||t||=1). Triangulate the clicked landmarks in the anchor's frame.
    4. Set absolute scale by demanding |rim_center - floor_under_rim| = 3050 mm.
    5. Per-camera reprojection diagnostics, then a single y/N prompt before
       atomically writing both JSONs via save_extrinsics.

Usage:
    python scripts/calibrate_extrinsics.py                       # cam_13 + cam_4
    python scripts/calibrate_extrinsics.py --camera cam_2        # cam_13 + cam_2
    python scripts/calibrate_extrinsics.py --anchor cam_4 --camera cam_2
"""
import argparse
import functools
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src.geometry.load_camera_data import (
    get_distortion,
    get_intrinsics,
    load_camera_data,
    save_extrinsics,
)
from src.geometry.triangulation import recover_relative_pose
from src.utils.video import get_frames, open_video

# Metric scale target: rim center to floor distance in mm.
HOOP_HEIGHT_MM = 3050.0

# Pairs of landmarks to use for absolute scale estimation
SCALE_PAIRS: list[tuple[str, str]] = [
    ("hoop_left_rim", "floor_under_hoop_left"),
    ("hoop_right_rim", "floor_under_hoop_right"),
]

STATUS_BAR_HEIGHT    = 50   # px — header strip stacked above the camera frame; the click callback subtracts it to translate window→camera coords, so this must stay shared with the renderer.
CLICK_DEBOUNCE_S     = 0.25 # s — drop EVENT_LBUTTONDOWN events arriving faster than this (macOS Cocoa fires duplicates)

# Ordered dict mapping landmark -> (norm_x, norm_y) on the court-diagram PNG. 
# Iteration order is the click order;
# values are normalized [0, 1] coordinates for the inset diagram (scaled later to pixel coords).
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
    
    # Three-point ARCS (6)
    "three_pt_left_endline_bench":   (0.020, 0.840),
    "three_pt_left_apex":            (0.300, 0.510),
    "three_pt_left_endline_stands":  (0.020, 0.220),
    "three_pt_right_endline_bench":  (0.980, 0.840),
    "three_pt_right_apex":           (0.700, 0.510),
    "three_pt_right_endline_stands": (0.980, 0.220), 
}
LANDMARKS: tuple[str, ...] = tuple(LANDMARK_DIAGRAM_NORM)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-camera extrinsics calibration via cross-camera correspondences.")
    p.add_argument("--anchor", default="cam_13",
                   help="Camera defining the world frame (R=I, t=0). Default: cam_13.")
    p.add_argument("--camera", default="cam_4",
                   help="The other camera to calibrate against the anchor. Default: cam_4.")
    p.add_argument("--input-dir", default="data/videos",
                   help="Directory containing the source videos (outN.mp4).")
    p.add_argument("--display-max-dim", type=int, default=1280,
                   help="Longest side of the display window in pixels.")
    return p.parse_args()


def _load_first_frame(camera_id: str, input_dir: Path) -> np.ndarray:
    """Load the first frame of the given camera's video."""
    filename = camera_id.replace("cam_", "out") + ".mp4"
    path = input_dir / filename
    
    # Open the video and read the first frame
    cap = open_video(str(path))
    try:
        frames, _ = get_frames(cap, max_frames=1)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frame from {path}")
    return frames[0]

# The LRU cache allows us to load and resize the diagram inset once
@functools.lru_cache(maxsize=1)
def _load_diagram_inset(
    diagram_inset_width: int = 320,
    path: Path = REPO_ROOT / "media" / "court-diagram.drawio.png",
) -> np.ndarray:
    """
    Load the transparent court-diagram PNG and resize to `diagram_inset_width`
    while preserving aspect ratio.
    """
    if not path.exists():
        raise FileNotFoundError(f"Court diagram image not found: {path}")
    # Read the diagram preserving alpha (transparency) channel.
    diagram = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if diagram is None:
        raise RuntimeError(f"OpenCV failed to decode {path}")
    if diagram.ndim != 3 or diagram.shape[2] != 4:
        raise RuntimeError(f"Expected an RGBA PNG, got shape {diagram.shape}")
    
    # Resize the diagram to fit the desired width while preserving aspect ratio.
    h0, w0 = diagram.shape[:2]
    scale = diagram_inset_width / w0
    new_w = diagram_inset_width
    new_h = max(1, int(round(h0 * scale)))
    return cv2.resize(diagram, (new_w, new_h))


def _insert_field_diagram(
    base_canvas: np.ndarray, 
    field_diagram: np.ndarray, 
    top_left_position: tuple[int, int],
    margin: int = 20,
) -> None:
    
    x0, y0 = top_left_position
    h, w = field_diagram.shape[:2]
    roi = base_canvas[y0:y0 + h, x0:x0 + w].astype(np.float32)

    alpha = field_diagram[..., 3:4].astype(np.float32) / 255.0
    rgb = field_diagram[..., :3].astype(np.float32)

    base_canvas[y0:y0 + h, x0:x0 + w] = (alpha * rgb + (1.0 - alpha) * roi).astype(np.uint8)


def _diagram_pixel(
    label: str, 
    inset_w: int, 
    inset_h: int
) -> tuple[int, int] | None:
    """Given a landmark label, return the scaled pixel coordinates for that landmark in the diagram inset, or None if the label is unknown."""
    # Get the normalized (fx, fy) coordinates from the LANDMARK_DIAGRAM_NORM dict
    norm = LANDMARK_DIAGRAM_NORM.get(label)
    if norm is None:
        return None
    # Scale the normalized coordinates to the actual pixel dimensions of the inset
    norm_x, norm_y = norm
    return (int(round(norm_x * inset_w)), int(round(norm_y * inset_h)))


def _draw_diagram_dot(
    inset: np.ndarray,
    label: str,
    color: tuple[int, int, int, int],
    radius: int=4,
    ordinal: int | None = None,
) -> None:
    """Place a colored dot on the diagram inset for `label`. If `ordinal` is given,
    annotate it as `#{ordinal}` in the same color."""
    ih, iw = inset.shape[:2]
    target = _diagram_pixel(label, iw, ih)
    if target is None:
        return
    cv2.circle(inset, target, radius, color, thickness=-1)
    if ordinal is not None:
        cv2.putText(inset, f"#{ordinal}",
                    (target[0] + radius + 2, target[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def _render_overlay(
    camera_frame_resized: np.ndarray,
    camera_id: str,
    clicks: dict[str, tuple[float, float] | None],
    next_index: int,
    scale: float,
    diagram_inset_margin: int = 20,
) -> np.ndarray:
    """
    Compose the picker view: camera frame as the canvas with clicked points,
    plus the diagram inset acting as a progress map (green=clicked with
    ordinal, red=skipped, blue=current target).
    """
    # Start with the resized camera frame and a fresh diagram inset.
    # _load_diagram_inset is cached; copy so the per-frame dots don't accumulate
    # into the cached source image.
    cam = camera_frame_resized.copy()
    diagram_inset = _load_diagram_inset().copy()
    ih, iw = diagram_inset.shape[:2]
    inset_top_left = (
        max(0, cam.shape[1] - iw - diagram_inset_margin),
        STATUS_BAR_HEIGHT + diagram_inset_margin,
    )

    # Single pass: per landmark, draw to the camera frame and/or the diagram
    # depending on whether it's clicked, skipped, current, or not yet reached.
    for i, label in enumerate(LANDMARKS):
        click = clicks.get(label)
        # If already clicked, draw a green dot on the camera frame + diagram, with ordinal.
        if click is not None:
            ix, iy = click
            dx, dy = int(round(ix * scale)), int(round(iy * scale))
            cv2.circle(cam, (dx, dy), 4, (0, 255, 0), thickness=-1)
            cv2.putText(cam, f"#{i + 1}", (dx + 6, dy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            _draw_diagram_dot(diagram_inset, label, (0, 255, 0, 255), ordinal=i + 1)
        
        # If it is the next target to click, draw a blue dot on the diagram.
        elif i == next_index:
            _draw_diagram_dot(diagram_inset, label, (255, 0, 0, 255))
        # If it is skipped (user pressed 'n'), draw a red dot on the diagram.
        elif i < next_index:
            _draw_diagram_dot(diagram_inset, label, (0, 0, 255, 255))

    # Status bar contents — drawn on a separate strip above the camera frame.
    if next_index < len(LANDMARKS):
        current_label = LANDMARKS[next_index]
        msg = f"{camera_id}  [{next_index + 1}/{len(LANDMARKS)}] click {current_label}"
    else:
        current_label = None
        msg = f"{camera_id}  all landmarks recorded — Enter to continue"
    keys = "keys: u=undo  n=not visible  q=quit"

    _insert_field_diagram(cam, diagram_inset, inset_top_left)

    # Header strip stacked on top of the camera frame so it never obscures clicks.
    header = np.zeros((STATUS_BAR_HEIGHT, cam.shape[1], 3), dtype=cam.dtype)
    cv2.putText(header, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, keys, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([header, cam])


def collect_clicks(
    camera_id: str,
    frame: np.ndarray,
    display_max_dim: int,
) -> tuple[dict[str, tuple[float, float] | None], str]:
    """
    Walk the user through the LANDMARKS list, collecting one click per visible
    landmark (or `n` to skip a landmark, `s` to skip this camera entirely, `q`
    to abort). Returns (clicks, status) where status is 'done', 'skipped', or 'quit'.
    """
    h, w = frame.shape[:2]
    scale = display_max_dim / max(h, w) if max(h, w) > display_max_dim else 1.0
    display_size = (int(round(w * scale)), int(round(h * scale)))
    base_display = cv2.resize(frame, display_size) if scale != 1.0 else frame.copy()

    clicks: dict[str, tuple[float, float] | None] = {label: None for label in LANDMARKS}
    order: list[str] = []  # labels in click order, for undo
    state = {"index": 0}
    last_click_time = [0.0]   # one-element list so the closure can mutate it

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["index"] >= len(LANDMARKS):
            return
        if y < STATUS_BAR_HEIGHT:
            return  # click landed in the external header strip, not the camera frame
        now = time.monotonic()
        if now - last_click_time[0] < CLICK_DEBOUNCE_S:
            return  # macOS HighGUI fires duplicate LBUTTONDOWN events; drop the bounce
        last_click_time[0] = now
        label = LANDMARKS[state["index"]]
        clicks[label] = (x / scale, (y - STATUS_BAR_HEIGHT) / scale)
        order.append(label)
        state["index"] += 1

    window = f"calibrate_{camera_id}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            overlay = _render_overlay(base_display, camera_id, clicks, state["index"], scale)
            cv2.imshow(window, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return clicks, "quit"
            if key == ord("n") and state["index"] < len(LANDMARKS):
                label = LANDMARKS[state["index"]]
                clicks[label] = None
                order.append(label)
                state["index"] += 1
            if key == ord("u") and order:
                last = order.pop()
                clicks[last] = None
                state["index"] = max(0, state["index"] - 1)
            if key in (13, 10) and state["index"] >= len(LANDMARKS):
                return clicks, "done"
    finally:
        cv2.destroyWindow(window)


def _common_labels(clicks_a: dict[str, tuple[float, float] | None],
                   clicks_b: dict[str, tuple[float, float] | None]) -> list[str]:
    return [lbl for lbl in LANDMARKS if clicks_a.get(lbl) is not None and clicks_b.get(lbl) is not None]


def _stack(clicks: dict[str, tuple[float, float] | None], labels: list[str]) -> np.ndarray:
    return np.array([clicks[l] for l in labels], dtype=np.float32)

def main() -> None:
    # Parse and validate command-line args
    args = parse_args()
    if args.anchor == args.camera:
        print(f"--anchor and --camera must be different (both are {args.anchor}).")
        return
    
    anchor, cam_b = args.anchor, args.camera
    cameras = [anchor, cam_b]
    input_dir = Path(args.input_dir)

    # Load camera data and first frames for both.
    cams: dict[str, dict] = {}
    for c_id in cameras:
        data = load_camera_data(c_id)
        cams[c_id] = {
            "mtx": get_intrinsics(data),
            "dist": get_distortion(data),
            "frame": _load_first_frame(c_id, input_dir),
        }

    # Collect clicks for both cameras
    all_clicks: dict[str, dict[str, tuple[float, float] | None]] = {}
    for c_id in cameras:
        print(f"\n=== Click landmarks in {c_id} ===")
        clicks, status = collect_clicks(c_id, cams[c_id]["frame"], args.display_max_dim)
        if status == "quit":
            print("Quit requested. No JSONs written.")
            return
        if status == "skipped":
            print(f"Skipped {c_id}. Cannot proceed without both cameras.")
            return
        all_clicks[c_id] = clicks

    # Phase 2: pair (anchor <-> cam_b) -> relative pose + triangulated structure.
    pair_labels = _common_labels(all_clicks[anchor], all_clicks[cam_b])
    if len(pair_labels) < 8:
        print(f"Need >=8 common landmarks between {anchor} and {cam_b}, got {len(pair_labels)}.")
        return

    pts_anchor = _stack(all_clicks[anchor], pair_labels)
    pts_b = _stack(all_clicks[cam_b], pair_labels)

    R_b, t_b, inlier_mask, p_anchor_norm, p_b_norm = recover_relative_pose(
        pts_anchor, pts_b,
        cams[anchor]["mtx"], cams[anchor]["dist"],
        cams[cam_b]["mtx"], cams[cam_b]["dist"],
    )
    n_inliers = int(inlier_mask.sum())
    print(f"\n{anchor} <-> {cam_b}: {n_inliers}/{len(pair_labels)} inliers from recoverPose")
    if n_inliers < 6:
        print(f"Too few inliers ({n_inliers}). Re-click with cleaner landmarks.")
        return

    P_anchor = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float32)
    P_b = np.hstack([R_b, t_b]).astype(np.float32)
    Xh = cv2.triangulatePoints(P_anchor, P_b, p_anchor_norm.T, p_b_norm.T)
    X = (Xh[:3] / Xh[3]).T  # (N, 3) in anchor frame, scale arbitrary

    structure: dict[str, np.ndarray] = {
        label: X[i] for i, label in enumerate(pair_labels) if inlier_mask[i]
    }

    # Phase 3: anchor metric scale from (rim, floor_under_rim) pairs.
    distances = []
    for top, bottom in SCALE_PAIRS:
        if top in structure and bottom in structure:
            distances.append(float(np.linalg.norm(structure[top] - structure[bottom])))
    if not distances:
        print("No (rim, floor_under_rim) pair was triangulated — cannot fix metric scale.")
        return
    mean_d = float(np.mean(distances))
    scale = HOOP_HEIGHT_MM / mean_d
    print(f"Scale anchor: mean rim-to-floor distance {mean_d:.4f} (arbitrary units) "
          f"-> scale = {scale:.4f}  (target {HOOP_HEIGHT_MM} mm)")

    # Apply scale to translation and structure.
    tvec_b_scaled = (t_b * scale).astype(np.float32)
    structure_scaled = {l: (p * scale).astype(np.float32) for l, p in structure.items()}

    # Final extrinsics for each camera (in anchor's world frame).
    rvec_anchor = np.zeros((3, 1), dtype=np.float32)
    tvec_anchor = np.zeros((3, 1), dtype=np.float32)
    rvec_b, _ = cv2.Rodrigues(R_b)
    rvec_b = rvec_b.astype(np.float32)

    extrinsics: dict[str, tuple[np.ndarray, np.ndarray]] = {
        anchor: (rvec_anchor, tvec_anchor),
        cam_b: (rvec_b, tvec_b_scaled),
    }

    # Phase 4: per-camera reprojection diagnostics.
    print("\n=== Reprojection residuals (px) ===")
    reproj_per_camera: dict[str, dict[str, tuple[float, float]]] = {}
    for cid in cameras:
        rvec, tvec = extrinsics[cid]
        labels = [l for l in LANDMARKS if l in structure_scaled and all_clicks[cid].get(l) is not None]
        if not labels:
            print(f"{cid}: no labels with both 3D structure and clicks — skipping diagnostic")
            continue
        world = np.array([structure_scaled[l] for l in labels], dtype=np.float32)
        clicked = _stack(all_clicks[cid], labels)
        proj, _ = cv2.projectPoints(world, rvec, tvec, cams[cid]["mtx"], cams[cid]["dist"])
        proj = proj.reshape(-1, 2)
        residuals = np.linalg.norm(proj - clicked, axis=1)
        print(f"{cid}: mean={residuals.mean():.2f}  max={residuals.max():.2f}  n={len(labels)}")
        for i, label in enumerate(labels):
            print(f"    {label:<24s} residual={residuals[i]:.2f}")
        reproj_per_camera[cid] = {label: (float(proj[i, 0]), float(proj[i, 1])) for i, label in enumerate(labels)}

    # Phase 5: atomic save.
    answer = input("\nWrite extrinsics for both cameras? [y/N]: ").strip().lower()
    if answer != "y":
        print("Discarded. No JSONs modified.")
        return
    for cid, (rvec, tvec) in extrinsics.items():
        save_extrinsics(cid, rvec, tvec)
        print(f"Wrote rvec/tvec to data/camera_data/{cid}.json")


if __name__ == "__main__":
    main()
