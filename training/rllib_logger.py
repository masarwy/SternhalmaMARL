"""Custom RLlib logger implementations."""

from __future__ import annotations

from ray.tune.logger import Logger


class MinimalRayLogger(Logger):
    """Lightweight Logger to avoid Ray's deprecated legacy loggers."""

    def _init(self) -> None:
        return

    def on_result(self, result: dict) -> None:
        del result
        return

    def flush(self) -> None:
        return

    def close(self) -> None:
        return
