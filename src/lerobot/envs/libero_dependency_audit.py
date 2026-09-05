"""Lightweight optional dependency audit for LIBERO simulation smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class LiberoDependencyStatus:
    name: str
    available: bool
    reason: str

    @property
    def status(self) -> str:
        return "available" if self.available else "missing"

    def as_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "available": self.available, "status": self.status, "reason": self.reason}


OPTIONAL_LIBERO_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("gymnasium", "Gymnasium vector/env API used by the LIBERO wrapper."),
    ("mujoco", "MuJoCo physics runtime required by the real simulator."),
    ("robosuite", "Robot simulation layer used underneath LIBERO tasks."),
    ("libero", "LIBERO benchmark/task package and OffScreenRenderEnv."),
    ("pyarrow", "Parquet metadata backend for reading LeRobot dataset task tables."),
    ("fastparquet", "Alternate parquet metadata backend when pyarrow is not installed."),
)


def audit_libero_optional_dependencies() -> dict[str, LiberoDependencyStatus]:
    """Return structured availability for optional LIBERO smoke dependencies.

    This function intentionally uses importlib metadata checks only so importing it
    never imports MuJoCo, LIBERO, Gymnasium, or parquet backends.
    """

    audit = {
        name: LiberoDependencyStatus(
            name=name,
            available=find_spec(name) is not None,
            reason=reason,
        )
        for name, reason in OPTIONAL_LIBERO_DEPENDENCIES
    }
    audit["parquet_backend"] = LiberoDependencyStatus(
        name="parquet_backend",
        available=audit["pyarrow"].available or audit["fastparquet"].available,
        reason="At least one parquet reader backend is required to inspect meta/tasks.parquet.",
    )
    return audit


def missing_real_libero_dependencies(
    audit: dict[str, LiberoDependencyStatus] | None = None,
) -> list[str]:
    audit = audit or audit_libero_optional_dependencies()
    required = ("gymnasium", "mujoco", "robosuite", "libero")
    return [name for name in required if not audit[name].available]


def parquet_backend_available(audit: dict[str, LiberoDependencyStatus] | None = None) -> bool:
    audit = audit or audit_libero_optional_dependencies()
    return audit["parquet_backend"].available


__all__ = [
    "LiberoDependencyStatus",
    "OPTIONAL_LIBERO_DEPENDENCIES",
    "audit_libero_optional_dependencies",
    "missing_real_libero_dependencies",
    "parquet_backend_available",
]
