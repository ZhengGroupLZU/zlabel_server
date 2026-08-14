from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ZLabel Server"
    version: str = "1.0.0"

    database_url: str = "./zlabel_server.db"
    # segmentation model used for inference
    model_name: Literal["SAM", "SlimSAM", "EdgeSAM", "SAM2", "SAM3"] = "EdgeSAM"
    # directory containing the ONNX model files (assets/onnx)
    model_dir: str = "assets/onnx"
    # ONNXRuntime execution provider: CPU / CUDA (falls back to CPU if unavailable)
    ort_backend: Literal["CPU", "CUDA"] = "CPU"
    ort_threads: int = 8
    # SAM3 PCS detection thresholds
    sam3_conf: float = 0.25
    sam3_iou: float = 0.7

    contour_min_points: int = 10
    contour_max_points: int = 100
    contour_max_iterations: int = 10

    oplist_host: str = "http://127.0.0.1:5244"
    oplist_username: str = ""
    oplist_password: str = ""
    oplist_proj_dir: str = "/zlabel_server/projects"
    oplist_proj_name: str = ""

    image_cache_size: int = 100
    min_contour_area_ratio: float = 3.0e-5

    model_config = SettingsConfigDict(
        env_prefix="ZLSERVER_",
        env_file=".env.onnx",
        env_file_encoding="utf-8",
    )

    @property
    def oplist_zlabel_save_dir(self):
        return f"{self.oplist_proj_dir}/{self.oplist_proj_name}/zlabel"


SETTINGS = Settings()
