"""Validated cache-payload bridge for computed LFP summary component arrays.

This module has no numerical computations. It verifies already-computed named
arrays, preserves their objects and axis order, and makes the cache schema
explicit for each component.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.neural_analysis.lfp_summary.pipeline import ComponentPayload


@dataclass(frozen=True)
class ArrayContract:
    """One immutable NPZ array contract.

    Parameters
    ----------
    axes : tuple[str, ...]
        Ordered named axes. Axis lengths must agree globally for equal names.
    units : str
        Physical units or an explicit categorical/count semantic label.
    """

    axes: tuple[str, ...]
    units: str


def _contracts(
    entries: Mapping[str, tuple[tuple[str, ...], str]],
) -> Mapping[str, ArrayContract]:
    """Freeze explicit array contracts without allocating numerical arrays.

    Parameters
    ----------
    entries : Mapping[str, tuple[tuple[str, ...], str]]
        Array names mapped to axis tuples and units.

    Returns
    -------
    Mapping[str, ArrayContract]
        Read-only named array contracts.
    """
    return MappingProxyType(
        {name: ArrayContract(axes, units) for name, (axes, units) in entries.items()}
    )


POWER_ARRAY_SCHEMA = _contracts(
    {
        "trial_indices": (("trial",), "trial-table-row"),
        "site_ids": (("site",), "stable-site-id"),
        "site_voltage_units": (("site",), "source-voltage-unit"),
        "condition_names": (("condition",), "condition-name"),
        "condition_membership": (("trial", "condition"), "boolean"),
        "filter_membership": (("trial",), "boolean"),
        "condition_trial_count": (("condition",), "trial"),
        "condition_effective_trial_count": (("condition", "site"), "trial"),
        "condition_unstable": (("condition", "site"), "boolean"),
        "frequency_hz": (("frequency",), "Hz"),
        "epoch_names": (("epoch",), "epoch-name"),
        "objective_valid": (("site", "trial"), "boolean"),
        "site_valid": (("site", "trial"), "boolean"),
        "user_excluded": (("trial",), "boolean"),
        "exclusion_reason_code": (("site", "trial"), "reason-code"),
        "psd_linear": (("site", "trial", "epoch", "frequency"), "source-voltage-unit^2/Hz"),
        "psd_valid": (("site", "trial", "epoch"), "boolean"),
        "session_reference_psd_linear": (("site", "frequency"), "source-voltage-unit^2/Hz"),
        "presession_reference_psd_linear": (("site", "frequency"), "source-voltage-unit^2/Hz"),
        "presession_reference_available": (("site",), "boolean"),
        "normalized_psd_session_db": (("site", "trial", "epoch", "frequency"), "dB"),
        "normalized_psd_presession_db": (("site", "trial", "epoch", "frequency"), "dB"),
        "band_names": (("band",), "band-name"),
        "band_power_linear": (("site", "trial", "epoch", "band"), "source-voltage-unit^2/Hz"),
        "band_power_session_db": (("site", "trial", "epoch", "band"), "dB"),
        "band_power_presession_db": (("site", "trial", "epoch", "band"), "dB"),
        "trial_rms": (("site", "trial"), "source-voltage-unit"),
        "trial_peak_to_peak": (("site", "trial"), "source-voltage-unit"),
        "relative_time_s": (("time",), "s"),
        "source_trace": (("site", "trial", "time"), "source-voltage-unit"),
    }
)

SYNCHRONY_ARRAY_SCHEMA = _contracts(
    {
        "trial_indices": (("trial",), "trial-table-row"),
        "site_ids": (("site",), "stable-site-id"),
        "site_voltage_units": (("site",), "source-voltage-unit"),
        "condition_names": (("condition",), "condition-name"),
        "condition_membership": (("trial", "condition"), "boolean"),
        "filter_membership": (("trial",), "boolean"),
        "frequency_hz": (("frequency",), "Hz"),
        "epoch_names": (("epoch",), "epoch-name"),
        "band_names": (("band",), "band-name"),
        "relative_time_s": (("time",), "s"),
        "site_valid": (("site", "trial"), "boolean"),
        "pair_valid": (("pair", "trial"), "boolean"),
        "site_exclusion_count": (("site",), "trial"),
        "pair_exclusion_count": (("pair",), "trial"),
        "pair_site_a_ids": (("pair",), "stable-site-id"),
        "pair_site_b_ids": (("pair",), "stable-site-id"),
        "itpc": (("condition", "site", "frequency", "time"), "dimensionless"),
        "itpc_effective_trial_count": (("condition", "site", "frequency", "time"), "trial"),
        "ispc": (("condition", "pair", "frequency", "time"), "dimensionless"),
        "ispc_phase_offset_rad": (("condition", "pair", "frequency", "time"), "rad"),
        "ispc_effective_trial_count": (("condition", "pair", "frequency", "time"), "trial"),
        "itpc_band_mean": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_ci_low": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_bootstrap_q25": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_bootstrap_median": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_bootstrap_q75": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_ci_high": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_band_trial_count": (("condition", "site", "epoch", "band"), "trial"),
        "itpc_unstable": (("condition", "site", "epoch", "band"), "boolean"),
        "ispc_band_mean": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_ci_low": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_bootstrap_q25": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_bootstrap_median": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_bootstrap_q75": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_ci_high": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_band_trial_count": (("condition", "pair", "epoch", "band"), "trial"),
        "ispc_unstable": (("condition", "pair", "epoch", "band"), "boolean"),
        "plv_by_frequency": (("trial", "pair", "epoch", "frequency"), "dimensionless"),
        "plv_phase_offset_rad": (("trial", "pair", "epoch", "frequency"), "rad"),
        "plv_valid_sample_count": (("trial", "pair", "epoch", "frequency"), "sample"),
        "plv_valid_sample_fraction": (("trial", "pair", "epoch", "frequency"), "fraction"),
        "plv_computable": (("trial", "pair", "epoch", "frequency"), "boolean"),
        "plv_band_mean": (("trial", "pair", "epoch", "band"), "dimensionless"),
        "source_trace": (("site", "trial", "time"), "source-voltage-unit"),
        "band_filtered_trace": (("site", "trial", "band", "time"), "source-voltage-unit"),
        "hilbert_phase_rad": (("site", "trial", "band", "time"), "rad"),
    }
)

SPIKE_PHASE_ARRAY_SCHEMA = _contracts(
    {
        "unit_ids": (("unit",), "stable-unit-id"),
        "population_ids": (("population",), "population-id"),
        "trial_indices": (("trial",), "trial-table-row"),
        "site_ids": (("site",), "stable-site-id"),
        "site_voltage_units": (("site",), "source-voltage-unit"),
        "condition_names": (("condition",), "condition-name"),
        "condition_membership": (("trial", "condition"), "boolean"),
        "filter_membership": (("trial",), "boolean"),
        "epoch_names": (("epoch",), "epoch-name"),
        "band_names": (("band",), "band-name"),
        "frequency_hz": (("frequency",), "Hz"),
        "relative_time_s": (("time",), "s"),
        "phase_bin_edges_rad": (("phase_bin_edge",), "rad"),
        "ppc": (("unit", "condition", "site", "epoch", "frequency"), "dimensionless"),
        "resultant_length": (("unit", "condition", "site", "epoch", "frequency"), "dimensionless"),
        "preferred_phase_rad": (("unit", "condition", "site", "epoch", "frequency"), "rad"),
        "spike_count": (("unit", "condition", "site", "epoch", "frequency"), "spike"),
        "computable": (("unit", "condition", "site", "epoch", "frequency"), "boolean"),
        "reliable": (("unit", "condition", "site", "epoch", "frequency"), "boolean"),
        "null_eligible": (("unit", "condition", "site", "epoch", "frequency"), "boolean"),
        "eligible_trial_count": (("unit", "condition", "site", "epoch", "frequency"), "trial"),
        "null_exceedance_count": (("unit", "condition", "site", "epoch", "frequency"), "shuffle"),
        "permutation_count": (("unit", "condition", "site", "epoch", "frequency"), "shuffle"),
        "p_value": (("unit", "condition", "site", "epoch", "frequency"), "probability"),
        "q_value": (("unit", "condition", "site", "epoch", "frequency"), "probability"),
        "significant": (("unit", "condition", "site", "epoch", "frequency"), "boolean"),
        "null_mean": (("unit", "condition", "site", "epoch", "frequency"), "dimensionless"),
        "null_std": (("unit", "condition", "site", "epoch", "frequency"), "dimensionless"),
        "null_p025": (("unit", "condition", "site", "epoch", "frequency"), "dimensionless"),
        "null_p50": (("unit", "condition", "site", "epoch", "frequency"), "dimensionless"),
        "null_p975": (("unit", "condition", "site", "epoch", "frequency"), "dimensionless"),
        "ppc_band_mean": (("unit", "condition", "site", "epoch", "band"), "dimensionless"),
        "representative_phase_hist_count": (
            (("unit", "condition", "site", "epoch", "band", "phase_bin"), "spike")
        ),
        "relative_spike_times_s": (("spike",), "s"),
        "relative_spike_time_offsets": (("unit", "trial_offset"), "spike-offset"),
        "source_trace": (("site", "trial", "time"), "source-voltage-unit"),
        "band_filtered_trace": (("site", "trial", "band", "time"), "source-voltage-unit"),
        "hilbert_phase_rad": (("site", "trial", "band", "time"), "rad"),
        "selected_low_unit_ids": (("condition", "site", "epoch", "band"), "stable-unit-id"),
        "selected_high_unit_ids": (("condition", "site", "epoch", "band"), "stable-unit-id"),
        "illustrative_low_trial_indices": (
            (("condition", "site", "epoch", "band"), "trial-table-row")
        ),
        "illustrative_high_trial_indices": (
            (("condition", "site", "epoch", "band"), "trial-table-row")
        ),
        "illustrative_trial_indices": (("condition", "site", "epoch", "band"), "trial-table-row"),
    }
)

COMPONENT_ARRAY_SCHEMAS = MappingProxyType(
    {
        "power": POWER_ARRAY_SCHEMA,
        "synchrony": SYNCHRONY_ARRAY_SCHEMA,
        "spike_phase": SPIKE_PHASE_ARRAY_SCHEMA,
    }
)

_UNICODE_ARRAYS = frozenset(
    {
        "unit_ids",
        "population_ids",
        "site_ids",
        "site_voltage_units",
        "condition_names",
        "epoch_names",
        "band_names",
        "pair_site_a_ids",
        "pair_site_b_ids",
        "selected_low_unit_ids",
        "selected_high_unit_ids",
        "exclusion_reason_code",
    }
)
_INTEGER_UNITS = frozenset(
    {"trial", "sample", "spike", "shuffle", "spike-offset", "trial-table-row"}
)


def build_component_payload(component: str, arrays: Mapping[str, np.ndarray]) -> ComponentPayload:
    """Build one cache-ready component payload without changing any array object.

    Parameters
    ----------
    component : str
        ``"power"``, ``"synchrony"``, or ``"spike_phase"``.
    arrays : Mapping[str, numpy.ndarray]
        Already-computed arrays with exactly the component's named axes and
        physical units. This function never casts, reshapes, or copies arrays.

    Returns
    -------
    ComponentPayload
        Pipeline-compatible arrays and JSON-ready manifest entry.
    """
    schema = _component_schema(component)
    payload = ComponentPayload(
        arrays=dict(arrays),
        manifest_entry={
            "file_name": f"{component}.npz",
            "state": "complete",
            "array_schema": _json_schema(schema),
            "warnings": [],
        },
    )
    validate_component_payload(component, payload)
    return payload


def validate_component_payload(component: str, payload: ComponentPayload) -> None:
    """Validate a payload's exact names, dtypes, named axes, and packed spikes.

    Parameters
    ----------
    component : str
        ``"power"``, ``"synchrony"``, or ``"spike_phase"``.
    payload : ComponentPayload
        Pipeline payload with unmodified NumPy array objects and manifest entry.

    Returns
    -------
    None
        Validation changes neither arrays nor payload metadata.

    Raises
    ------
    ValueError
        If an array is absent, extra, unsafe, semantically mistyped, or has an
        incompatible named axis or packed-spike offset.
    """
    schema = _component_schema(component)
    if not isinstance(payload, ComponentPayload):
        raise ValueError("payload must be a ComponentPayload")
    _validate_manifest_entry(component, payload.manifest_entry, schema)
    _validate_array_names(payload.arrays, schema)
    dimensions = _validate_array_contracts(payload.arrays, schema)
    if component == "spike_phase":
        _validate_packed_spike_offsets(payload.arrays, dimensions)


def _component_schema(component: str) -> Mapping[str, ArrayContract]:
    """Return one immutable component contract.

    Parameters
    ----------
    component : str
        Requested component identity.

    Returns
    -------
    Mapping[str, ArrayContract]
        Read-only array schema for the component.
    """
    try:
        return COMPONENT_ARRAY_SCHEMAS[component]
    except KeyError as error:
        raise ValueError(f"unknown LFP summary component: {component}") from error


def _json_schema(schema: Mapping[str, ArrayContract]) -> dict[str, dict[str, object]]:
    """Convert immutable schema constants to JSON-ready metadata.

    Parameters
    ----------
    schema : Mapping[str, ArrayContract]
        Immutable named array contracts.

    Returns
    -------
    dict[str, dict[str, object]]
        Ordinary dictionaries with JSON list axes and unit strings.
    """
    return {
        name: {"axes": list(contract.axes), "units": contract.units}
        for name, contract in schema.items()
    }


def _validate_manifest_entry(
    component: str,
    entry: Mapping[str, object],
    schema: Mapping[str, ArrayContract],
) -> None:
    """Bind a payload to its exact component filename and immutable schema.

    Parameters
    ----------
    component : str
        Component identity used to derive the only allowed cache basename.
    entry : Mapping[str, object]
        Candidate component manifest entry.
    schema : Mapping[str, ArrayContract]
        Immutable schema expected for the component.

    Returns
    -------
    None
        Entry metadata is inspected only and never changed.
    """
    if not isinstance(entry, Mapping):
        raise ValueError("component manifest entry must be a mapping")
    expected_filename = f"{component}.npz"
    if entry.get("file_name") != expected_filename:
        raise ValueError("component file name must be its safe component NPZ basename")
    if entry.get("array_schema") != _json_schema(schema):
        raise ValueError("component manifest array schema does not match its contract")


def _validate_array_names(
    arrays: Mapping[str, np.ndarray],
    schema: Mapping[str, ArrayContract],
) -> None:
    """Reject missing and undeclared cache arrays.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Candidate component arrays.
    schema : Mapping[str, ArrayContract]
        Required named array contracts.

    Returns
    -------
    None
        Array objects remain untouched.
    """
    actual = set(arrays)
    required = set(schema)
    missing = sorted(required - actual)
    extras = sorted(actual - required)
    if missing:
        raise ValueError(f"missing required array: {missing[0]}")
    if extras:
        raise ValueError(f"undeclared extra array: {extras[0]}")


def _validate_array_contracts(
    arrays: Mapping[str, np.ndarray],
    schema: Mapping[str, ArrayContract],
) -> dict[str, int]:
    """Validate array dtypes, ranks, and equal-length named axes.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Candidate arrays with no permitted extras or omissions.
    schema : Mapping[str, ArrayContract]
        Required axes and units for each named array.

    Returns
    -------
    dict[str, int]
        Global named-axis lengths for packed-spike validation.
    """
    dimensions: dict[str, int] = {}
    for name, contract in schema.items():
        array = arrays[name]
        if not isinstance(array, np.ndarray):
            raise ValueError(f"array {name} must be a NumPy array")
        _validate_dtype(name, array, contract.units)
        if array.ndim != len(contract.axes):
            raise ValueError(f"array-axis mismatch for {name}")
        for axis, size in zip(contract.axes, array.shape):
            prior_size = dimensions.get(axis)
            if prior_size is not None and prior_size != size:
                raise ValueError(f"array-axis mismatch for {name}: axis {axis}")
            dimensions[axis] = size
    return dimensions


def _validate_dtype(name: str, array: np.ndarray, units: str) -> None:
    """Enforce safe dtype families for categorical, mask, count, and value arrays.

    Parameters
    ----------
    name : str
        Named component array.
    array : numpy.ndarray
        Array whose dtype is inspected without casting.
    units : str
        Contract units determining semantic dtype requirements.

    Returns
    -------
    None
        No array data are changed.
    """
    if array.dtype.kind == "O":
        raise ValueError(f"array {name} has object or pickle dtype")
    if name in _UNICODE_ARRAYS:
        if array.dtype.kind != "U":
            raise ValueError(f"array {name} must use fixed Unicode dtype")
        return
    if units == "boolean":
        if array.dtype.kind != "b":
            raise ValueError(f"array {name} must use boolean dtype")
        return
    if units in _INTEGER_UNITS:
        if array.dtype.kind not in {"i", "u"}:
            raise ValueError(f"array {name} must use integer dtype")
        return
    if array.dtype.kind != "f":
        raise ValueError(f"array {name} must use floating dtype")


def _validate_packed_spike_offsets(
    arrays: Mapping[str, np.ndarray],
    dimensions: Mapping[str, int],
) -> None:
    """Validate unit-by-trial offsets into one globally packed spike-time axis.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Validated spike-phase arrays including global packed times and offsets.
    dimensions : Mapping[str, int]
        Global named-axis lengths established by the component contract.

    Returns
    -------
    None
        Packed times and offsets retain their stored values and shape.
    """
    offsets = arrays["relative_spike_time_offsets"]
    packed_times = arrays["relative_spike_times_s"]
    expected_offset_size = dimensions["trial"] + 1
    if offsets.shape != (dimensions["unit"], expected_offset_size):
        raise ValueError("relative_spike_time_offsets must have axes (unit, trial_offset)")
    if packed_times.shape != (dimensions["spike"],):
        raise ValueError("relative_spike_times_s must have axis (spike,)")
    if offsets[0, 0] != 0:
        raise ValueError("relative_spike_time_offsets must start at global zero")
    if np.any(np.diff(offsets, axis=1) < 0):
        raise ValueError("relative_spike_time_offsets must be nondecreasing")
    if not np.array_equal(offsets[:-1, -1], offsets[1:, 0]):
        raise ValueError("relative_spike_time_offsets must connect unit boundaries")
    if offsets[-1, -1] != packed_times.size:
        raise ValueError("relative_spike_time_offsets must end at packed spike count")
