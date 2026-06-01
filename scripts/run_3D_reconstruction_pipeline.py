"""
Run the 3D reconstruction pipeline end-to-end on three cameras.
The pipeline steps are:
    1. Load each camera's tracking JSON.
    2. Rectify bbox-center points using the camera intrinsics.
    3. Triangulate per-class points across all cameras into a TriangulationOutput.
    4. Persist triangulation.json (always).
    5. Optional renders: top-down minimap MP4, FIFA-style radar overlay on
       camera A's source video, and a 3D matplotlib animation MP4.

Usage examples:
    python scripts/run_3D_reconstruction_pipeline.py --cameras cam_4 cam_13 cam_2
    python scripts/run_3D_reconstruction_pipeline.py --cameras cam_4 cam_13 cam_2 \\
        --render-minimap --overlay-video --render-3d-graph
"""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.calibration.camera_data import CameraData
from src.config import get_config
from src.cli import (
    add_force_arg,
    add_max_frames_arg,
    add_output_dir_arg,
    max_frames_or_none,
)
from src.geometry.rectification import rectify_tracking_output
from src.geometry.smoothing import smooth_triangulation
from src.geometry.triangulation import triangulate_rectified_outputs
from src.paths import CameraPaths, ReconstructionPaths, preflight_output_paths
from src.types.tracking import TrackingOutput
from src.utils.logging import configure_logging, get_logger
from src.utils.video_io import load_video_frames
from src.visualization.scene_3d import produce_3d_scene_video
from src.visualization.video_render import (
    produce_minimap_video,
    produce_radar_overlay_video,
)


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the 3D reconstruction pipeline on three cameras."
    )
    p.add_argument("--cameras", nargs=3, required=True, metavar=("CAM_A", "CAM_B", "CAM_C"),
                   help="Exactly three camera ids, e.g. --cameras cam_4 cam_13 cam_2.")
    add_output_dir_arg(p)
    p.add_argument("--render-minimap", action="store_true",
                   help="Also render a stand-alone top-down minimap MP4.")
    p.add_argument("--overlay-video", action="store_true",
                   help="Also render a radar-overlay MP4 on top of camera A's source video.")
    p.add_argument("--render-3d-graph", action="store_true",
                   help="Also render a 3D matplotlib animation MP4 of the triangulated scene.")
    add_max_frames_arg(p, help_text="-1 to read every frame, otherwise stop after N frames (only consulted for --overlay-video).")
    add_force_arg(p)
    return p.parse_args()


def run_3d_reconstruction_pipeline(
    cameras: tuple[str, str, str],
    *,
    output_dir: str | None = None,
    render_minimap: bool = False,
    overlay_video: bool = False,
    render_3d_graph: bool = False,
    max_frames: int | None = None,
    force: bool = False,
) -> None:
    """Programmatic entry point for the 3D reconstruction pipeline."""
    cam_a, cam_b, cam_c = cameras
    logger.info("Running 3D reconstruction pipeline for cameras %s, %s, %s...",
                cam_a, cam_b, cam_c)

    # Set up paths for each camera and the overall reconstruction outputs
    cam_paths = {
        cam_id: CameraPaths.for_camera(cam_id, results_dir=output_dir)
        for cam_id in (cam_a, cam_b, cam_c)
    }

    # Ensure all required camera tracking JSONs exist before doing any expensive compute
    recon = ReconstructionPaths.for_cameras(
        (cam_a, cam_b, cam_c),
        results_dir=output_dir,
    )
    recon.mkdir()

    output_paths_to_check: list[Path] = [recon.triangulation_json]
    if render_minimap:
        output_paths_to_check.append(recon.minimap_video)
    if overlay_video:
        output_paths_to_check.append(recon.overlay_video)
    if render_3d_graph:
        output_paths_to_check.append(recon.scene_3d_video)

    # Check for existing outputs before doing any expensive compute, to avoid overwriting results
    preflight_output_paths(output_paths_to_check, force)

    # 1. Load tracking JSONs and camera calibrations for all cameras
    cams: dict[str, CameraData] = {}
    trackings: dict[str, TrackingOutput] = {}
    for cam_id in (cam_a, cam_b, cam_c):
        json_path = cam_paths[cam_id].tracking_json
        if not json_path.exists():
            raise FileNotFoundError(f"Tracking JSON not found: {json_path}")
        cams[cam_id] = CameraData.load(cam_id)
        trackings[cam_id] = TrackingOutput.read(str(json_path))
        logger.info("Loaded %s: %d tracking frames",
                    cam_id, len(trackings[cam_id].frames))

    # 2. Rectify bbox-center points per camera
    logger.info("Rectifying tracking points...")
    rectified = {
        cam_id: rectify_tracking_output(trackings[cam_id], cams[cam_id])
        for cam_id in cams
    }

    # 3. Triangulate across both cameras
    logger.info("Triangulating across cameras...")
    triangulation = triangulate_rectified_outputs(cams, rectified)
    logger.info("Triangulated %d frames across %s",
                len(triangulation.frames), triangulation.camera_ids)
    logger.info("Smoothing 3D tracks with forward Kalman filter...")
    gs = get_config().geometry.smoothing
    triangulation = smooth_triangulation(
        triangulation,
        std_pos=gs.std_pos,
        std_vel=gs.std_vel,
        std_meas=gs.std_meas,
    )

    # 4. Persist the triangulation output (JSON)
    logger.info("Writing triangulation output to %s...", recon.triangulation_json)
    triangulation.write(str(recon.triangulation_json), overwrite=force)

    # 5. Optional renders
    if render_minimap:
        logger.info("Saving top-down minimap video to %s...", recon.minimap_video)
        produce_minimap_video(triangulation, str(recon.minimap_video))

    if overlay_video:
        overlay_source = cam_paths[cam_a].video
        if not overlay_source.exists():
            raise FileNotFoundError(f"Video not found: {overlay_source}")
        logger.info("Loading %s frames from %s for radar overlay...",
                    cam_a, overlay_source)
        frames_color, fps = load_video_frames(overlay_source, max_frames=max_frames)
        logger.info("Loaded %d frames at %.2f fps", len(frames_color), fps)

        logger.info("Saving radar overlay video to %s...", recon.overlay_video)
        produce_radar_overlay_video(
            frames_color,
            triangulation,
            str(recon.overlay_video),
        )

    if render_3d_graph:
        logger.info("Saving 3D scene animation to %s...", recon.scene_3d_video)
        produce_3d_scene_video(triangulation, str(recon.scene_3d_video))


def main() -> None:
    configure_logging()
    args = parse_args()
    run_3d_reconstruction_pipeline(
        cameras=tuple(args.cameras),
        output_dir=args.output_dir,
        render_minimap=args.render_minimap,
        overlay_video=args.overlay_video,
        render_3d_graph=args.render_3d_graph,
        max_frames=max_frames_or_none(args.max_frames),
        force=args.force,
    )


if __name__ == "__main__":
    main()
