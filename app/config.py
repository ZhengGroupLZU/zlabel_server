from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ZLabel Server"
    version: str = "1.0.0"

    database_url: str = "./zlabel_server.db"
    model_name: Literal["SAM", "EdgeSAM", "SAM2"] = "EdgeSAM"
    encoder_path: str = "assets/edge_sam_3x_encoder.onnx"
    decoder_path: str = "assets/edge_sam_3x_decoder.onnx"

    oplist_host: str = "http://127.0.0.1:5244"
    oplist_username: str = ""
    oplist_password: str = ""
    oplist_proj_dir: str = "/zlabel_server/projects"
    oplist_proj_name: str = ""

    image_cache_size: int = 100

    model_config = SettingsConfigDict(env_prefix="ZLSERVER_")

    @property
    def oplist_zlabel_save_dir(self):
        return f"{self.oplist_proj_dir}/{self.oplist_proj_name}/zlabel"


SETTINGS = Settings()
