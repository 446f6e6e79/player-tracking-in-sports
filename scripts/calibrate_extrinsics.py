"""
Multi-view calibration of camera extrinsics from cross-camera correspondences.
The user clicks the same set of physical landmarks in each camera (no world
coordinates required). 

The pipeline:
  1. Anchor camera (first item of --cameras, default cam_13): R = I, t = 0.
  2. Pair (anchor <-> camera B): essential matrix + recoverPose -> relative pose
     of B (||t||=1). Triangulate the clicked landmarks in the anchor's frame.
  3. Camera C: solvePnP against the triangulated 3D structure -> pose of C in
     the anchor's frame, sharing scale with B automatically.
  4. Set absolute scale by demanding |rim_center - floor_under_rim| = 3050 mm.
  5. Show per-camera reprojection diagnostics, then a single y/N prompt before
     atomically writing all three JSONs via save_extrinsics.
    Corners:       corner_left_bench / corner_left_stands / corner_right_bench / corner_right_stands
    Center:        center_court / center_line_bench / center_line_stands
    Lane (area):   lane_{left,right}_{endline,ft}_{bench,stands}   (8 corners)
    Hoops:         hoop_left_rim / hoop_right_rim
    Floor anchors: floor_under_hoop_left / floor_under_hoop_right
    3-point arc:   three_pt_{left,right}_apex
    FT circle:     ft_circle_{left,right}_apex

The two (rim, floor_under_rim) pairs double as the metric scale anchors.

Usage:
    python scripts/calibrate_extrinsics.py
    python scripts/calibrate_extrinsics.py --cameras cam_13 cam_2 cam_4
"""
import argparse
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
from src.geometry.triangulation import compute_extrinsics, recover_relative_pose
from src.utils.video import get_frames, open_video


HOOP_HEIGHT_MM = 3050.0

SCALE_PAIRS: list[tuple[str, str]] = [
    ("hoop_left_rim", "floor_under_hoop_left"),
    ("hoop_right_rim", "floor_under_hoop_right"),
]

COURT_DIAGRAM_PATH   = REPO_ROOT / "media" / "court-diagram.drawio.png"
DIAGRAM_INSET_WIDTH  = 320  # px — inset width on the camera frame; height follows aspect
DIAGRAM_INSET_MARGIN = 12   # px — gap from camera-frame edges
STATUS_BAR_HEIGHT    = 50   # px — translucent strip at the top of the camera frame
DIAGRAM_DOT_RADIUS   = 4    # px — current-landmark highlight on the inset
CLICK_DEBOUNCE_S     = 0.25 # s — drop EVENT_LBUTTONDOWN events arriving faster than this (macOS Cocoa fires duplicates)

# Single source of truth: ordered dict mapping landmark label -> (x_frac, y_frac)
# on the court-diagram PNG. Diagram convention:
#   y small => stands (top of image), y large => bench (bottom).
#   x small => left basket, x large => right basket.
# Iteration order is the click order; LANDMARKS below is a derived view.
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
    p = argparse.ArgumentParser(description="Multi-view extrinsics calibration via cross-camera correspondences.")
    p.add_argument("--cameras", nargs="+", default=["cam_13", "cam_2", "cam_4"],
                   help="Cameras to calibrate. The first is the world-frame anchor (R=I, t=0).")
    p.add_argument("--input-dir", default="data/videos",
                   help="Directory containing the source videos (outN.mp4).")
    p.add_argument("--display-max-dim", type=int, default=1280,
                   help="Longest side of the display window in pixels.")
    return p.parse_args()


def video_path_for(camera_id: str, input_dir: Path) -> Path:
    num = camera_id.split("_", 1)[1]
    return input_dir / f"out{num}.mp4"


def _load_first_frame(camera_id: str, input_dir: Path) -> np.ndarray:
    path = video_path_for(camera_id, input_dir)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = open_video(str(path))
    try:
        frames, _ = get_frames(cap, max_frames=1)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frame from {path}")
    return frames[0]


def _load_diagram_inset() -> np.ndarray:
    """
    Load the transparent court-diagram PNG and resize to DIAGRAM_INSET_WIDTH while
    preserving aspect ratio. Returns the BGRA image (alpha kept for blending).
    """
    if not COURT_DIAGRAM_PATH.exists():
        raise FileNotFoundError(f"Court diagram image not found: {COURT_DIAGRAM_PATH}")
    diagram = cv2.imread(str(COURT_DIAGRAM_PATH), cv2.IMREAD_UNCHANGED)
    if diagram is None:
        raise RuntimeError(f"OpenCV failed to decode {COURT_DIAGRAM_PATH}")
    if diagram.ndim != 3 or diagram.shape[2] != 4:
        raise RuntimeError(f"Expected an RGBA PNG, got shape {diagram.shape}")
    h0, w0 = diagram.shape[:2]
    scale = DIAGRAM_INSET_WIDTH / w0
    new_w = DIAGRAM_INSET_WIDTH
    new_h = max(1, int(round(h0 * scale)))
    return cv2.resize(diagram, (new_w, new_h))


def _paste_inset(canvas: np.ndarray, inset_bgra: np.ndarray, top_left: tuple[int, int]) -> None:
    """Alpha-blend `inset_bgra` (BGRA) onto BGR `canvas` in place at top_left."""
    x0, y0 = top_left
    h, w = inset_bgra.shape[:2]
    roi = canvas[y0:y0 + h, x0:x0 + w].astype(np.float32)
    alpha = inset_bgra[..., 3:4].astype(np.float32) / 255.0
    rgb = inset_bgra[..., :3].astype(np.float32)
    canvas[y0:y0 + h, x0:x0 + w] = (alpha * rgb + (1.0 - alpha) * roi).astype(np.uint8)


def _diagram_pixel(label: str, inset_w: int, inset_h: int) -> tuple[int, int] | None:
    """Map a landmark label to pixel coords inside the inset image."""
    norm = LANDMARK_DIAGRAM_NORM.get(label)
    if norm is None:
        return None
    fx, fy = norm
    return (int(round(fx * inset_w)), int(round(fy * inset_h)))


def _render_overlay(
    camera_frame_resized: np.ndarray,
    camera_id: str,
    clicks: dict[str, tuple[float, float] | None],
    next_index: int,
    scale: float,
    inset_bgra: np.ndarray,
    inset_top_left: tuple[int, int],
    reprojected: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Compose the picker view: camera frame as the canvas, with already-clicked
    points drawn as green dots, a translucent status bar across the top, and the
    transparent court-diagram alpha-blended in the top-right corner with the
    current landmark highlighted.
    """
    cam = camera_frame_resized.copy()

    # Picked-point overlays.
    for i, label in enumerate(LANDMARKS):
        click = clicks.get(label)
        if click is None:
            continue
        ix, iy = click
        dx, dy = int(round(ix * scale)), int(round(iy * scale))
        cv2.circle(cam, (dx, dy), 4, (0, 255, 0), thickness=-1)
        cv2.putText(cam, f"#{i + 1} {label}", (dx + 6, dy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # Reprojection markers.
    if reprojected is not None:
        for _label, (rx, ry) in reprojected.items():
            dx, dy = int(round(rx * scale)), int(round(ry * scale))
            cv2.drawMarker(cam, (dx, dy), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                           markerSize=12, thickness=1)

    # Status bar contents — drawn on a separate strip above the camera frame.
    if next_index < len(LANDMARKS):
        current_label = LANDMARKS[next_index]
        msg = f"{camera_id}  [{next_index + 1}/{len(LANDMARKS)}] click {current_label}"
    else:
        current_label = None
        msg = f"{camera_id}  all landmarks recorded — Enter to continue"
    keys = "keys: u=undo  n=not visible  s=skip camera  q=quit"

    # Diagram inset with current-landmark highlight.
    inset = inset_bgra.copy()
    if current_label is not None:
        ih, iw = inset.shape[:2]
        target = _diagram_pixel(current_label, iw, ih)
        if target is not None:
            cv2.circle(inset, target, DIAGRAM_DOT_RADIUS, (0, 0, 255, 255), thickness=-1)
    _paste_inset(cam, inset, inset_top_left)

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
    cam_w, cam_h = base_display.shape[1], base_display.shape[0]

    inset = _load_diagram_inset()
    inset_h, inset_w = inset.shape[:2]
    inset_x0 = max(0, cam_w - inset_w - DIAGRAM_INSET_MARGIN)
    inset_y0 = STATUS_BAR_HEIGHT + DIAGRAM_INSET_MARGIN
    inset_top_left = (inset_x0, inset_y0)

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
            overlay = _render_overlay(base_display, camera_id, clicks, state["index"], scale,
                                      inset, inset_top_left)
            cv2.imshow(window, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return clicks, "quit"
            if key == ord("s"):
                return clicks, "skipped"
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


def show_reprojection(
    camera_id: str,
    frame: np.ndarray,
    clicks: dict[str, tuple[float, float] | None],
    reprojected: dict[str, tuple[float, float]],
    display_max_dim: int,
) -> None:
    h, w = frame.shape[:2]
    scale = display_max_dim / max(h, w) if max(h, w) > display_max_dim else 1.0
    display_size = (int(round(w * scale)), int(round(h * scale)))
    base_display = cv2.resize(frame, display_size) if scale != 1.0 else frame.copy()
    cam_w = base_display.shape[1]
    inset = _load_diagram_inset()
    inset_h, inset_w = inset.shape[:2]
    inset_top_left = (max(0, cam_w - inset_w - DIAGRAM_INSET_MARGIN),
                      STATUS_BAR_HEIGHT + DIAGRAM_INSET_MARGIN)
    overlay = _render_overlay(base_display, f"{camera_id} (reprojection)",
                              clicks, len(LANDMARKS), scale,
                              inset, inset_top_left,
                              reprojected=reprojected)
    cv2.putText(overlay, "Enter to continue",
                (10, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    window = f"reproj_{camera_id}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.imshow(window, overlay)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10, ord("q")):
                break
    finally:
        cv2.destroyWindow(window)


def _common_labels(clicks_a: dict[str, tuple[float, float] | None],
                   clicks_b: dict[str, tuple[float, float] | None]) -> list[str]:
    return [lbl for lbl in LANDMARKS if clicks_a.get(lbl) is not None and clicks_b.get(lbl) is not None]


def _stack(clicks: dict[str, tuple[float, float] | None], labels: list[str]) -> np.ndarray:
    return np.array([clicks[l] for l in labels], dtype=np.float32)


def main() -> None:
    args = parse_args()
    if len(args.cameras) != 3:
        print(f"--cameras must list exactly three camera ids (got {len(args.cameras)}). "
              "The chain pipeline assumes anchor + 2 others.")
        return

    anchor, cam_b, cam_c = args.cameras
    input_dir = Path(args.input_dir)

    # Load camera data and first frames for all three.
    cams: dict[str, dict] = {}
    for cid in args.cameras:
        data = load_camera_data(cid)
        cams[cid] = {
            "mtx": get_intrinsics(data),
            "dist": get_distortion(data),
            "frame": _load_first_frame(cid, input_dir),
        }

    # Phase 1: collect clicks per camera.
    all_clicks: dict[str, dict[str, tuple[float, float] | None]] = {}
    for cid in args.cameras:
        print(f"\n=== Click landmarks in {cid} ===")
        clicks, status = collect_clicks(cid, cams[cid]["frame"], args.display_max_dim)
        if status == "quit":
            print("Quit requested. No JSONs written.")
            return
        if status == "skipped":
            print(f"Skipped {cid}. Cannot proceed without all three cameras.")
            return
        all_clicks[cid] = clicks

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

    # Phase 3: PnP cam_c against the triangulated structure.
    pnp_labels = [l for l in LANDMARKS if l in structure and all_clicks[cam_c].get(l) is not None]
    if len(pnp_labels) < 4:
        print(f"Need >=4 triangulated landmarks visible in {cam_c}, got {len(pnp_labels)}.")
        return

    world_pts = np.array([structure[l] for l in pnp_labels], dtype=np.float32)
    image_pts = _stack(all_clicks[cam_c], pnp_labels)
    rvec_c, tvec_c = compute_extrinsics(world_pts, image_pts, cams[cam_c]["mtx"], cams[cam_c]["dist"])
    print(f"{anchor} <- {cam_c}: PnP solved using {len(pnp_labels)} landmarks")

    # Phase 4: anchor metric scale from (rim, floor_under_rim) pairs.
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

    # Apply scale to translations and structure.
    tvec_b_scaled = (t_b * scale).astype(np.float32)
    tvec_c_scaled = (tvec_c.reshape(3, 1) * scale).astype(np.float32)
    structure_scaled = {l: (p * scale).astype(np.float32) for l, p in structure.items()}

    # Final extrinsics for each camera (in anchor's world frame).
    rvec_anchor = np.zeros((3, 1), dtype=np.float32)
    tvec_anchor = np.zeros((3, 1), dtype=np.float32)
    rvec_b, _ = cv2.Rodrigues(R_b)
    rvec_b = rvec_b.astype(np.float32)
    rvec_c = rvec_c.astype(np.float32).reshape(3, 1)

    extrinsics: dict[str, tuple[np.ndarray, np.ndarray]] = {
        anchor: (rvec_anchor, tvec_anchor),
        cam_b: (rvec_b, tvec_b_scaled),
        cam_c: (rvec_c, tvec_c_scaled),
    }

    # Phase 5: per-camera reprojection diagnostics.
    print("\n=== Reprojection residuals (px) ===")
    reproj_per_camera: dict[str, dict[str, tuple[float, float]]] = {}
    for cid in args.cameras:
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

    # Show overlays so the user can visually verify.
    for cid in args.cameras:
        if cid in reproj_per_camera:
            show_reprojection(cid, cams[cid]["frame"], all_clicks[cid],
                              reproj_per_camera[cid], args.display_max_dim)

    # Phase 6: atomic save.
    answer = input("\nWrite extrinsics for all three cameras? [y/N]: ").strip().lower()
    if answer != "y":
        print("Discarded. No JSONs modified.")
        return
    for cid, (rvec, tvec) in extrinsics.items():
        save_extrinsics(cid, rvec, tvec)
        print(f"Wrote rvec/tvec to data/camera_data/{cid}.json")


if __name__ == "__main__":
    main()
