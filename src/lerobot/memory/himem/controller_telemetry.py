"""Read-only controller interface capture; no controller invocation or effective-action claim."""
from __future__ import annotations

import numpy as np


def capture_controller_fields(controller, data, *, phase: str) -> dict:
    if phase not in ("PRE_STEP", "POST_STEP"):
        raise ValueError("E_TELEMETRY_PHASE")
    result = {"phase": phase, "controller_effective_action7": None,
              "controller_effective_status": "NOT_OBSERVED",
              "control_frequency_hz": float(controller.control_freq), "fields": {}}
    for name in ("goal_pos", "goal_ori", "torques", "input_min", "input_max", "output_min", "output_max"):
        value = getattr(controller, name, None)
        if value is not None:
            array = np.asarray(value).copy()
            if not np.isfinite(array).all():
                raise ValueError("E_TELEMETRY_FINITE")
            result["fields"][name] = {"shape": list(array.shape), "numeric": array.tolist()}
    for name in ("ctrl", "qpos", "qvel"):
        array = np.asarray(getattr(data, name)).copy()
        if not np.isfinite(array).all():
            raise ValueError("E_TELEMETRY_FINITE")
        result["fields"][name] = {"shape": list(array.shape), "numeric": array.tolist()}
    return result
