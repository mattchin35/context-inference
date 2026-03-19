"""Compatibility shim for the top-level switching module.

Stage 1 of the switching reorganization keeps the historical import path
`src.behavior_modeling.switching` alive while the implementation now lives in
`src.behavior_modeling.agent_switching.switching`.
"""

from src.behavior_modeling.agent_switching.switching import (
    SWITCH_TRIAL_DF_COLUMNS,
    _validate_switching_mode,
    apply_inactive_updates,
    build_active_strategy_labels,
    run_switched_agent_session,
)

__all__ = [
    "SWITCH_TRIAL_DF_COLUMNS",
    "_validate_switching_mode",
    "apply_inactive_updates",
    "build_active_strategy_labels",
    "run_switched_agent_session",
]
