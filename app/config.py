from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ZLabel Server"
    version: str = "1.0.0"

    database_url: str = "./zlabel_server.db"
    # large model for batch rectangle inference
    model_name: Literal["SAM", "MobileSAM", "SAM2", "SAM3"] = "SAM3"
    model_path: str = ""
    # small minor model for point inference
    # only used for SAM3, which has not point input
    minor_model_name: Literal["SAM", "MobileSAM", "SAM2", "SAM3"] = "MobileSAM"
    minor_model_path: str = ""

    oplist_host: str = "http://127.0.0.1:5244"
    oplist_username: str = ""
    oplist_password: str = ""
    oplist_proj_dir: str = "/zlabel_server/projects"
    oplist_proj_name: str = ""

    image_cache_size: int = 100
    min_contour_area_ratio: float = 3.0e-5

    model_config = SettingsConfigDict(
        env_prefix="ZLSERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @property
    def oplist_zlabel_save_dir(self):
        return f"{self.oplist_proj_dir}/{self.oplist_proj_name}/zlabel"


SETTINGS = Settings()
