"""Contract tests for payloads bridging numerical summaries to cache I/O."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from src.neural_analysis.lfp_summary_payloads import (
    SYNCHRONY_ARRAY_SCHEMA,
    build_component_payload,
    validate_component_payload,
)
from src.neural_analysis.lfp_summary_pipeline import ComponentPayload


_AXIS_LENGTHS = {
    "unit": 2,
    "population": 1,
    "site": 1,
    "pair": 1,
    "trial": 1,
    "trial_offset": 2,
    "condition": 1,
    "epoch": 1,
    "band": 1,
    "frequency": 1,
    "time": 1,
    "phase_bin": 1,
    "phase_bin_edge": 2,
    "spike": 2,
}


def _schema(
    axes_and_units: Mapping[str, tuple[tuple[str, ...], str]],
) -> dict[str, dict[str, object]]:
    """Return JSON-ready named-axis contracts.

    Parameters
    ----------
    axes_and_units : Mapping[str, tuple[tuple[str, ...], str]]
        Array names mapped to their ordered axes and physical units.

    Returns
    -------
    dict[str, dict[str, object]]
        JSON-compatible manifest ``array_schema`` entries.
    """
    return {
        name: {"axes": list(axes), "units": units}
        for name, (axes, units) in axes_and_units.items()
    }


_POWER = _schema(
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
        "psd_linear": (
            (("site", "trial", "epoch", "frequency"), "source-voltage-unit^2/Hz")
        ),
        "psd_valid": (("site", "trial", "epoch"), "boolean"),
        "session_reference_psd_linear": (
            (("site", "frequency"), "source-voltage-unit^2/Hz")
        ),
        "presession_reference_psd_linear": (
            (("site", "frequency"), "source-voltage-unit^2/Hz")
        ),
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

_SYNCHRONY = _schema(
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
        "itpc_ci_high": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_bootstrap_q25": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_bootstrap_median": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_bootstrap_q75": (("condition", "site", "epoch", "band"), "dimensionless"),
        "itpc_band_trial_count": (("condition", "site", "epoch", "band"), "trial"),
        "itpc_unstable": (("condition", "site", "epoch", "band"), "boolean"),
        "ispc_band_mean": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_ci_low": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_ci_high": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_bootstrap_q25": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_bootstrap_median": (("condition", "pair", "epoch", "band"), "dimensionless"),
        "ispc_bootstrap_q75": (("condition", "pair", "epoch", "band"), "dimensionless"),
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

_SPIKE_PHASE = _schema(
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
        "resultant_length": (
            (("unit", "condition", "site", "epoch", "frequency"), "dimensionless")
        ),
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
            ("condition", "site", "epoch", "band"),
            "trial-table-row",
        ),
        "illustrative_high_trial_indices": (
            ("condition", "site", "epoch", "band"),
            "trial-table-row",
        ),
        "illustrative_trial_indices": (("condition", "site", "epoch", "band"), "trial-table-row"),
    }
)

_SCHEMAS = {"power": _POWER, "synchrony": _SYNCHRONY, "spike_phase": _SPIKE_PHASE}
_BOOLEAN_UNITS = {"boolean"}
_INTEGER_UNITS = {"trial", "sample", "spike", "shuffle", "spike-offset", "trial-table-row"}
_UNICODE_NAMES = {
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


def _arrays_for(component: str) -> dict[str, np.ndarray]:
    """Return tiny safe arrays satisfying every named array in one component contract.

    Parameters
    ----------
    component : str
        One of ``"power"``, ``"synchrony"``, or ``"spike_phase"``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Numeric, boolean, and fixed-Unicode arrays with documented axes. The
        spike offsets start at zero and end at the packed spike count.
    """
    arrays: dict[str, np.ndarray] = {}
    for name, contract in _SCHEMAS[component].items():
        axes = contract["axes"]
        shape = tuple(_AXIS_LENGTHS[axis] for axis in axes)
        units = contract["units"]
        if name in _UNICODE_NAMES:
            arrays[name] = np.full(shape, name, dtype="<U32")
        elif units in _BOOLEAN_UNITS:
            arrays[name] = np.ones(shape, dtype=bool)
        elif units in _INTEGER_UNITS:
            arrays[name] = np.ones(shape, dtype=np.int64)
        else:
            arrays[name] = np.ones(shape, dtype=np.float64)
    if component == "spike_phase":
        arrays["relative_spike_time_offsets"] = np.array(
            [[0, 1], [1, 2]],
            dtype=np.int64,
        )
    return arrays


@pytest.mark.parametrize("component", ("power", "synchrony", "spike_phase"))
def test_build_component_payload_declares_required_arrays_with_exact_axes_and_units(
    component: str,
) -> None:
    """Each cache payload bridges every documented named array without reshaping.

    Parameters
    ----------
    component : str
        Component contract selected by pytest parametrization.
    """
    arrays = _arrays_for(component)

    payload = build_component_payload(component, arrays)

    assert isinstance(payload, ComponentPayload)
    assert payload.arrays.keys() == arrays.keys()
    for name, array in arrays.items():
        assert payload.arrays[name] is array
    assert payload.manifest_entry["array_schema"] == _SCHEMAS[component]
    validate_component_payload(component, payload)


@pytest.mark.parametrize("component", ("power", "synchrony", "spike_phase"))
def test_payload_rejects_missing_object_shape_mismatch_and_undeclared_arrays(
    component: str,
) -> None:
    """Payload validation enforces complete safe arrays and shared named-axis sizes.

    Parameters
    ----------
    component : str
        Component contract selected by pytest parametrization.
    """
    arrays = _arrays_for(component)
    missing_name = next(iter(arrays))
    missing = dict(arrays)
    missing.pop(missing_name)
    with pytest.raises(ValueError, match=missing_name):
        build_component_payload(component, missing)

    object_arrays = _arrays_for(component)
    object_arrays[missing_name] = np.array([{"unsafe": True}], dtype=object)
    with pytest.raises(ValueError, match="object|pickle|safe"):
        build_component_payload(component, object_arrays)

    mismatched = _arrays_for(component)
    shared_axis_name = next(
        name
        for name, contract in _SCHEMAS[component].items()
        if "site" in contract["axes"] and name != "site_ids"
    )
    site_axis = _SCHEMAS[component][shared_axis_name]["axes"].index("site")
    mismatch_shape = list(mismatched[shared_axis_name].shape)
    mismatch_shape[site_axis] = 2
    mismatched[shared_axis_name] = np.ones(tuple(mismatch_shape), dtype=np.float64)
    with pytest.raises(ValueError, match=shared_axis_name):
        build_component_payload(component, mismatched)

    unexpected = _arrays_for(component)
    unexpected["undeclared_extra"] = np.ones(1, dtype=np.float64)
    with pytest.raises(ValueError, match="undeclared_extra|undeclared|extra"):
        build_component_payload(component, unexpected)


@pytest.mark.parametrize(
    ("component", "forbidden_name"),
    (
        ("synchrony", "wavelet_coefficients"),
        ("spike_phase", "null_ppc_draws"),
    ),
)
def test_payload_rejects_full_temporary_wavelet_or_null_draw_arrays(
    component: str,
    forbidden_name: str,
) -> None:
    """Full wavelet tensors and shuffle draws are temporary, not cache payloads.

    Parameters
    ----------
    component : str
        Component that must reject the named temporary computation product.
    forbidden_name : str
        Full-resolution wavelet or null-draw array name that must not be cached.
    """
    arrays = _arrays_for(component)
    arrays[forbidden_name] = np.ones((1, 1, 1, 1), dtype=np.complex64)

    with pytest.raises(ValueError, match=forbidden_name):
        build_component_payload(component, arrays)


@pytest.mark.parametrize("component", ("power", "synchrony", "spike_phase"))
def test_payload_identity_masks_and_counts_use_contract_dtypes(component: str) -> None:
    """Identity coordinates, masks, and counts retain safe explicit semantic dtypes.

    Parameters
    ----------
    component : str
        Component contract selected by pytest parametrization.
    """
    payload = build_component_payload(component, _arrays_for(component))

    for name, contract in _SCHEMAS[component].items():
        array = payload.arrays[name]
        units = contract["units"]
        if name in _UNICODE_NAMES:
            assert array.dtype.kind == "U"
        elif units in _BOOLEAN_UNITS:
            assert array.dtype == np.dtype(bool)
        elif units in _INTEGER_UNITS:
            assert array.dtype.kind in {"i", "u"}


def test_synchrony_payload_declares_saved_quantiles_counts_and_no_bootstrap_draw_axis() -> None:
    """Synchrony persists compact scalar summaries, never the transient bootstrap draws."""
    expected_axes = {
        "itpc_bootstrap_q25": ("condition", "site", "epoch", "band"),
        "itpc_bootstrap_median": ("condition", "site", "epoch", "band"),
        "itpc_bootstrap_q75": ("condition", "site", "epoch", "band"),
        "itpc_band_trial_count": ("condition", "site", "epoch", "band"),
        "ispc_bootstrap_q25": ("condition", "pair", "epoch", "band"),
        "ispc_bootstrap_median": ("condition", "pair", "epoch", "band"),
        "ispc_bootstrap_q75": ("condition", "pair", "epoch", "band"),
        "ispc_band_trial_count": ("condition", "pair", "epoch", "band"),
    }

    assert set(expected_axes).issubset(SYNCHRONY_ARRAY_SCHEMA)
    for name, axes in expected_axes.items():
        contract = SYNCHRONY_ARRAY_SCHEMA[name]
        assert contract.axes == axes
        assert contract.units == (
            "trial" if name.endswith("trial_count") else "dimensionless"
        )
    assert all("bootstrap_values" not in name for name in SYNCHRONY_ARRAY_SCHEMA)


def test_spike_payload_offsets_index_each_unit_trial_and_bound_packed_spikes() -> None:
    """Packed spike offsets are integer, monotonic, and bounded by each unit's data.

    The offsets use ``trial_offset = trial + 1`` so every unit includes both
    its initial zero and its final packed-spike count.
    """
    payload = build_component_payload("spike_phase", _arrays_for("spike_phase"))
    offsets = payload.arrays["relative_spike_time_offsets"]
    packed_times = payload.arrays["relative_spike_times_s"]

    assert offsets.dtype.kind in {"i", "u"}
    assert offsets.shape == (_AXIS_LENGTHS["unit"], _AXIS_LENGTHS["trial"] + 1)
    assert offsets[0, 0] == 0
    assert np.all(np.diff(offsets, axis=1) >= 0)
    assert np.array_equal(offsets[:-1, -1], offsets[1:, 0])
    assert offsets[-1, -1] == packed_times.size


def test_spike_payload_requires_distinct_low_and_high_illustrative_trials() -> None:
    """Matched exemplars must not silently reuse one unit's trial for both panels."""
    arrays = _arrays_for("spike_phase")
    arrays["illustrative_low_trial_indices"][:] = 3
    arrays["illustrative_high_trial_indices"][:] = 9
    arrays["illustrative_trial_indices"][:] = 9

    payload = build_component_payload("spike_phase", arrays)

    assert payload.arrays["illustrative_low_trial_indices"].dtype == np.dtype(np.int64)
    assert payload.arrays["illustrative_high_trial_indices"].dtype == np.dtype(np.int64)
    assert payload.arrays["illustrative_low_trial_indices"].shape == (1, 1, 1, 1)
    assert payload.arrays["illustrative_high_trial_indices"].shape == (1, 1, 1, 1)
    assert payload.arrays["illustrative_trial_indices"][0, 0, 0, 0] == 9

    legacy_only = dict(arrays)
    legacy_only.pop("illustrative_low_trial_indices")
    legacy_only.pop("illustrative_high_trial_indices")
    with pytest.raises(ValueError, match="illustrative_(low|high)_trial_indices"):
        build_component_payload("spike_phase", legacy_only)


def test_payload_validation_rejects_forged_schema_or_component_filename() -> None:
    """Validation must bind safe arrays to the exact manifest schema and basename."""

    payload = build_component_payload("power", _arrays_for("power"))
    forged_schema = ComponentPayload(
        payload.arrays,
        {**payload.manifest_entry, "array_schema": {}},
    )
    with pytest.raises(ValueError, match="schema"):
        validate_component_payload("power", forged_schema)

    unsafe_filename = ComponentPayload(
        payload.arrays,
        {**payload.manifest_entry, "file_name": "../power.npz"},
    )
    with pytest.raises(ValueError, match="file|name|basename"):
        validate_component_payload("power", unsafe_filename)
