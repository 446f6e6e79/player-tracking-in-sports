"""Pydantic schema for the centralised application config (config.yaml)."""

from pydantic import BaseModel


class ModelsConfig(BaseModel):
    hf_repo_id: str
    yolo_filename: str
    osnet_filename: str
    osnet_model_name: str


class ClassesConfig(BaseModel):
    ball_class_id: int
    ball_class_name: str


class PipelineConfig(BaseModel):
    chunk_size: int
    yolo_batch_size: int


class YoloDetectionConfig(BaseModel):
    conf_threshold: float
    inference_size: int
    iou_threshold: float
    batch_size: int


class TwoPassConfig(BaseModel):
    player_inference_size: int
    ball_inference_size: int
    ball_conf_threshold: float
    merge_nms_iou: float


class NmsConfig(BaseModel):
    iou_threshold: float


class Mog2Config(BaseModel):
    learning_rate: float
    detect_shadows: bool


class ImageProcessingConfig(BaseModel):
    clahe_clip_limit: float
    clahe_tile_grid: tuple[int, int]
    blob_min_area: int
    blob_max_area: int


class DetectionConfig(BaseModel):
    yolo: YoloDetectionConfig
    two_pass: TwoPassConfig
    nms: NmsConfig
    mog2: Mog2Config
    image_processing: ImageProcessingConfig


class DeepSortConfig(BaseModel):
    max_iou_distance: float
    max_appearance_distance: float
    max_age: int
    n_init: int
    feature_budget: int


class SortConfig(BaseModel):
    max_iou_distance: float
    max_age: int
    n_init: int


class AppearanceConfig(BaseModel):
    input_size: tuple[int, int]
    embed_dim: int


class TrajectorySmoothingConfig(BaseModel):
    max_gap: int
    ema_alpha: float


class TrackingConfig(BaseModel):
    deep_sort: DeepSortConfig
    sort: SortConfig
    appearance: AppearanceConfig
    trajectory_smoothing: TrajectorySmoothingConfig


class TriangulationConfig(BaseModel):
    ground_inlier_mm: float
    ball_reproj_px: float
    min_views: int


class GeometrySmoothingConfig(BaseModel):
    max_gap: int
    gate_chi2: float
    std_pos: float
    std_vel: float
    std_meas: float


class GeometryConfig(BaseModel):
    triangulation: TriangulationConfig
    smoothing: GeometrySmoothingConfig


class CalibrationConfig(BaseModel):
    solve_pnp_min_points: int
    rmse_min_points: int
    display_max_dim: int


class AugmentationConfig(BaseModel):
    mosaic: float
    mixup: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float


class TrainingConfig(BaseModel):
    model: str
    epochs: int
    patience: int
    imgsz: int
    batch: int
    name: str
    out: str
    augmentation: AugmentationConfig


class AppConfig(BaseModel):
    models: ModelsConfig
    classes: ClassesConfig
    pipeline: PipelineConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    geometry: GeometryConfig
    calibration: CalibrationConfig
    training: TrainingConfig
