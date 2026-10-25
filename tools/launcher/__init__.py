"""NInfer Launcher：本地 WebUI 启动器（见 docs/launcher.md）。"""

from .config import FIELDS, Store, build_argv, builtin_presets, validate  # noqa: F401

__all__ = ["FIELDS", "Store", "build_argv", "builtin_presets", "validate"]