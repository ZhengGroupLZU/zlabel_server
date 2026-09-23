"""Logger used across the inference assets.

Kept separate from ``app.core.logging`` so the inference worker (a separate
process, and the only consumer of these modules) does not depend on the API
layer. The class is the one v1 used (``app/logger.py``).
"""

import logging

from rich.logging import RichHandler

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


class ZLogger(logging.Logger):
    def __init__(self, name: str, level: int = logging.DEBUG) -> None:
        super().__init__(name, level)
        self.format = "%(asctime)s-%(message)s"
        self.addHandler(RichHandler(rich_tracebacks=True))
