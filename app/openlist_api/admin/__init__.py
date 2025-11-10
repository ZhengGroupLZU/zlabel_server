"""管理员API模块。

包含所有管理员功能，包括元信息、用户、存储、驱动、设置和任务管理。
"""

from .driver import DriverAPI
from .meta import MetaAPI
from .setting import SettingAPI
from .storage import StorageAPI
from .task import TaskAPI
from .user import UserAPI

__all__ = [
    "MetaAPI",
    "UserAPI",
    "StorageAPI",
    "DriverAPI",
    "SettingAPI",
    "TaskAPI",
]
