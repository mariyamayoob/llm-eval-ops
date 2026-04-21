"""Offline eval package.

Avoid re-export imports at package import time to prevent circular-import traps.
Import concrete types from `evals.contracts` / `evals.runner` directly.
"""

from __future__ import annotations

__all__: list[str] = []
