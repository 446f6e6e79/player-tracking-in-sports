from collections import defaultdict
from dataclasses import replace

from src.types.tracking import FrameTrackedDetections, TrackedDetection, TrackingOutput

BALL_CLASS = "Ball"


def _rank_labels_per_track(
    tracking_output: TrackingOutput,
) -> tuple[dict[int, list[tuple[str, float]]], dict[int, int]]:
    """
    For each track, rank labels by total confidence across the track's detections.
    
    Parameters:
        - tracking_output: The input tracking output containing frames and detections.
    Returns:
        - ranked: A dictionary mapping track_id to a list of (class_name, total_confidence) tuples, 
        sorted by confidence in descending order.
        - length_by_track: A dictionary mapping track_id to the number of detections in that track, 
        used for tie-breaking in label assignment.
    """
    # Store total confidence per class within each track, store track length for tie-breaking.
    conf_by_track: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # Store track length for tie-breaking.
    length_by_track: dict[int, int] = defaultdict(int)

    for frame_detection in tracking_output.frames:
        for detection in frame_detection.detections:
            track_id = detection.track_id
            class_name = detection.class_name
            confidence = detection.confidence

            # Accumulate confidence for this detection's class within its track.
            conf_by_track[track_id][class_name] += confidence
            length_by_track[track_id] += 1

    ranked: dict[int, list[tuple[str, float]]] = {
        tid: sorted(label_conf.items(), key=lambda x: x[1], reverse=True)
        for tid, label_conf in conf_by_track.items()
    }
    return ranked, dict(length_by_track)


def _assign_stable_labels(
    ranked: dict[int, list[tuple[str, float]]],
    length_by_track: dict[int, int],
) -> dict[int, str]:
    """
    Assign a single stable label to each track. 
    Each track "claims" its highest-confidence label that isn't already claimed by a higher-priority track.

    Parameters:
        - ranked: A dictionary mapping track_id to a list of (class_name, total_confidence) tuples, sorted by confidence in descending order.
        - length_by_track: A dictionary mapping track_id to the number of detections in that track, used for tie-breaking in label assignment.
    Returns:
        - A dictionary mapping track_id to its assigned stable class_name.
    """
    # Sort tracks by the confidence of their top label
    priority = sorted(
        ranked.keys(),
        key=lambda tid: (ranked[tid][0][1], length_by_track[tid]),
        reverse=True,
    )

    claimed: set[str] = set()
    resolved: dict[int, str] = {}

    # Greedily assign each track its highest-confidence label that is still available
    for track_id in priority:
        for class_name, _conf in ranked[track_id]:
            # If the class is the ball, or if it hasn't been claimed by a higher-priority track, claim it for this track.
            if class_name == BALL_CLASS or class_name not in claimed:
                resolved[track_id] = class_name
                # If we claimed a non-ball class, mark it as claimed so lower-priority tracks can't use it.
                if class_name != BALL_CLASS:
                    claimed.add(class_name)
                break
        # If we went through all the labels and couldn't claim any (e.g. all were taken by higher-priority tracks)
        # Assign the top label anyway, even though it's a conflict
        else:
            resolved[track_id] = ranked[track_id][0][0]
            print(
                f"[label_resolution] track {track_id} had no free label; "
                f"keeping conflicting top choice {resolved[track_id]!r}"
            )

    return resolved


def _dedupe_within_frame(detections: list[TrackedDetection]) -> list[TrackedDetection]:
    """Remove duplicate detections of the same class within a single frame, keeping only the one with the highest confidence."""
    # Map from class_name to the best detection of that class in this frame
    best_by_label: dict[str, TrackedDetection] = {}
    # List of detections to keep (all ball detections + best non-ball detections)
    survivors: list[TrackedDetection] = []

    for detection in detections:
        # Keep all ball detections
        if detection.class_name == BALL_CLASS:
            survivors.append(detection)
            continue
        
        # Get current detection's class and confidence
        class_name = detection.class_name
        confidence = detection.confidence

        current_best = best_by_label.get(class_name)
        if current_best is None or detection.confidence > current_best.confidence:
            best_by_label[detection.class_name] = detection
    # Add the best non-ball detections to the survivors list
    survivors.extend(best_by_label.values())
    return survivors


def resolve_track_labels(tracking_output: TrackingOutput) -> TrackingOutput:
    """
        Resolve a single stable label for each track in the tracking output.
        This allows us to correct inconsistent labeling across frames, where the same track might 
        be labeled as "Player 1" in one frame and "Player 2" in another.
        
        The resolution process involves:
        1. Ranking the labels for each track by their total confidence across all detections in that track.
        2. Greedily assigning each track its highest-confidence label that isn't already claimed by a higher-priority track
           (where priority is determined by the confidence of the track's top label).
        3. Rewriting the tracking output with the resolved stable labels, and deduplicating any conflicting detections within each frame.
           (this way we ensure that each label appears at most once per frame)
    """
    ranked, length_by_track = _rank_labels_per_track(tracking_output)
    resolved = _assign_stable_labels(ranked, length_by_track)

    new_frames: list[FrameTrackedDetections] = []
    for fd in tracking_output.frames:
        rewritten: list[TrackedDetection] = []
        for detection in fd.detections:
            if detection.track_id not in resolved:
                rewritten.append(detection)
                continue
            stable = resolved[detection.track_id]
            if stable == detection.class_name:
                rewritten.append(detection)
            else:
                rewritten.append(replace(detection, class_name=stable))

        rewritten = _dedupe_within_frame(rewritten)
        new_frames.append(FrameTrackedDetections(frame_index=fd.frame_index, detections=rewritten))

    return TrackingOutput(
        source=tracking_output.source,
        camera_id=tracking_output.camera_id,
        fps=tracking_output.fps,
        frames=new_frames,
    )
