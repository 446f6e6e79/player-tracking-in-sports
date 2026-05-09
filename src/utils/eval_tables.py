import pandas as pd
from IPython.display import display

from src.types.evaluation import DetectionMetrics, TrackingMetrics


def show_detection_table(results: dict[tuple[str, str], DetectionMetrics]) -> None:
    """Display a detection metrics table grouped by method, with one sub-row per camera.

    Parameters:
        - results: mapping of (method, camera) to DetectionMetrics.
    """
    rows = []
    for (method, camera), result in results.items():
        rows.append({
            "Method": method, "Camera": camera,
            "TP": result.tp, "FP": result.fp, "FN": result.fn,
            "Precision": round(result.precision, 3), "Recall": round(result.recall, 3),
            "F1": round(result.f1, 3), "Mean IoU": round(result.mean_iou, 3),
        })
    display(pd.DataFrame(rows).set_index(["Method", "Camera"]))


def show_identity_table(results: dict[tuple[str, str], TrackingMetrics]) -> None:
    """Display an identity (IDF1) metrics table grouped by tracker, with one sub-row per camera.

    Parameters:
        - results: mapping of (tracker, camera) to TrackingMetrics.
    """
    rows = []
    for (tracker, camera), result in results.items():
        id_ = result.identity
        rows.append({
            "Tracker": tracker, "Camera": camera,
            "TP": id_.tp, "FP": id_.fp, "FN": id_.fn,
            "IDP": round(id_.precision, 3), "IDR": round(id_.recall, 3), "IDF1": round(id_.f1, 3),
        })
    display(pd.DataFrame(rows).set_index(["Tracker", "Camera"]))


def show_hota_table(results: dict[tuple[str, str], TrackingMetrics]) -> None:
    """Display a HOTA metrics table grouped by tracker, with one sub-row per camera.

    Per-alpha breakdowns are omitted; use HOTAMetrics.hota_per_alpha etc. for those.

    Parameters:
        - results: mapping of (tracker, camera) to TrackingMetrics.
    """
    rows = []
    for (tracker, camera), result in results.items():
        h = result.hota
        rows.append({
            "Tracker": tracker, "Camera": camera,
            "HOTA": round(h.hota, 3), "DetA": round(h.deta, 3),
            "AssA": round(h.assa, 3), "LocA": round(h.loca, 3),
        })
    display(pd.DataFrame(rows).set_index(["Tracker", "Camera"]))
