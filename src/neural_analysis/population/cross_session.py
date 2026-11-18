from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_DECODER_RUN_INDEX = 0
SUPPORTED_ANALYSIS_REGIONS = {"HPC", "V1", "PFC"}
BASE_CONDITIONS = [
    "correct_rewarded",
    "incorrect",
    "omission",
    "switch",
    "stay",
]


def normalize_region_name_for_filename(region_name: str) -> str:
    """
    Normalize a brain-region label for analysis filenames.

    Parameters
    ----------
    region_name : str
        Brain-region label. Supported values are ``"HPC"``, ``"V1"``, and
        ``"PFC"``, case-insensitive.

    Returns
    -------
    str
        Canonical region label for filenames.
    """

    normalized_region = str(region_name).strip().upper()
    if normalized_region not in SUPPORTED_ANALYSIS_REGIONS:
        raise ValueError(
            f"Unsupported region {region_name!r}. Expected one of {sorted(SUPPORTED_ANALYSIS_REGIONS)}."
        )
    return normalized_region


def make_region_analysis_filename(region_name: str | None, base_filename: str) -> str:
    """
    Prefix an analysis filename with a region tag when one is supplied.

    Parameters
    ----------
    region_name : str | None
        Optional brain-region label. ``None`` preserves legacy filenames.
    base_filename : str
        Base analysis filename including extension.

    Returns
    -------
    str
        Legacy filename or region-prefixed filename.
    """

    if region_name is None:
        return base_filename
    return f"{normalize_region_name_for_filename(region_name)}_{base_filename}"


def make_analysis_filename_tag(region_name: str | None, date_filename_tag: str | None) -> str | None:
    """
    Combine region and date selectors into one plot filename tag.

    Parameters
    ----------
    region_name : str | None
        Optional brain-region label.
    date_filename_tag : str | None
        Optional date-selection filename tag.

    Returns
    -------
    str | None
        Combined filename tag, or ``None`` when no tag components are supplied.
    """

    tag_parts: list[str] = []
    if region_name is not None:
        tag_parts.append(normalize_region_name_for_filename(region_name))
    if date_filename_tag is not None:
        tag_parts.append(str(date_filename_tag))
    if not tag_parts:
        return None
    return "_".join(tag_parts)


def make_analysis_title_suffix(region_name: str | None, date_label: str | None) -> str | None:
    """
    Combine region and date selectors into one plot-title suffix.

    Parameters
    ----------
    region_name : str | None
        Optional brain-region label.
    date_label : str | None
        Optional human-readable date-selection label.

    Returns
    -------
    str | None
        Combined title suffix, or ``None`` when no components are supplied.
    """

    suffix_parts: list[str] = []
    if region_name is not None:
        suffix_parts.append(normalize_region_name_for_filename(region_name))
    if date_label is not None:
        suffix_parts.append(str(date_label))
    if not suffix_parts:
        return None
    return ", ".join(suffix_parts)


def find_session_analysis_csvs(mouse_root: Path | str, region_name: str | None = None) -> pd.DataFrame:
    """
    Find per-session analysis CSVs under a mouse data directory.

    Parameters
    ----------
    mouse_root : Path | str
        Root directory containing one or more session folders. Each session must have a
        ``processed`` directory with ``state_decodability_analysis.csv`` and
        ``correct_rewarded_decoding_performance.csv``.
    region_name : str | None, optional
        Optional region label. If supplied, discovers region-prefixed files such
        as ``HPC_state_decodability_analysis.csv``.

    Returns
    -------
    pd.DataFrame
        One row per discovered session with columns:
        ``session_dir``, ``processed_dir``, ``decodability_csv_path``,
        and ``decoder_performance_csv_path``.
    """

    mouse_root = Path(mouse_root)
    decodability_filename = make_region_analysis_filename(region_name, "state_decodability_analysis.csv")
    decoder_performance_filename = make_region_analysis_filename(
        region_name,
        "correct_rewarded_decoding_performance.csv",
    )
    decodability_paths = sorted(mouse_root.rglob(decodability_filename))
    session_rows: list[dict[str, Path]] = []
    for decodability_path in decodability_paths:
        processed_dir = decodability_path.parent
        decoder_performance_path = processed_dir / decoder_performance_filename
        if not decoder_performance_path.exists():
            continue
        session_rows.append(
            {
                "session_dir": processed_dir.parent,
                "processed_dir": processed_dir,
                "decodability_csv_path": decodability_path,
                "decoder_performance_csv_path": decoder_performance_path,
            }
        )

    return pd.DataFrame(session_rows)


def load_state_decodability_sessions(session_csv_paths: list[Path] | tuple[Path, ...]) -> pd.DataFrame:
    """
    Load per-session state decodability CSVs into one cross-session table.

    Parameters
    ----------
    session_csv_paths : list[Path] | tuple[Path, ...]
        Paths to per-session ``state_decodability_analysis.csv`` files.

    Returns
    -------
    pd.DataFrame
        One row per ``session x trial_condition`` with columns:
        ``session``, ``mouse``, ``date``, ``region``, ``trial_condition``,
        ``cv_score_before``, ``cv_score_after``, ``p_value_before``, and ``p_value_after``.
    """

    rows: list[pd.DataFrame] = []
    for csv_path in session_csv_paths:
        session_df = pd.read_csv(csv_path)
        required_columns = {
            "session_id",
            "mouse",
            "date",
            "region",
            "condition",
            "cv_score_pre",
            "cv_score_post",
            "p_value_pre",
            "p_value_post",
        }
        missing_columns = required_columns - set(session_df.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

        rows.append(
            session_df.rename(
                columns={
                    "session_id": "session",
                    "condition": "trial_condition",
                    "cv_score_pre": "cv_score_before",
                    "cv_score_post": "cv_score_after",
                    "p_value_pre": "p_value_before",
                    "p_value_post": "p_value_after",
                }
            )[
                [
                    "session",
                    "mouse",
                    "date",
                    "region",
                    "trial_condition",
                    "cv_score_before",
                    "cv_score_after",
                    "p_value_before",
                    "p_value_after",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "session",
                "mouse",
                "date",
                "region",
                "trial_condition",
                "cv_score_before",
                "cv_score_after",
                "p_value_before",
                "p_value_after",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def load_state_decoder_performance_sessions(
    session_csv_paths: list[Path] | tuple[Path, ...],
    decoder_run_index: int | None = DEFAULT_DECODER_RUN_INDEX,
) -> pd.DataFrame:
    """
    Load one decoder run from each per-session repeated-decoder CSV.

    Parameters
    ----------
    session_csv_paths : list[Path] | tuple[Path, ...]
        Paths to per-session ``correct_rewarded_decoding_performance.csv`` files.
    decoder_run_index : int | None, optional
        Decoder-run index to extract from each session table. If ``None``, all decoder runs
        are loaded.

    Returns
    -------
    pd.DataFrame
        One row per ``session x trial_condition`` when ``decoder_run_index`` is an integer,
        or one row per ``session x trial_condition x decoder_run`` when ``decoder_run_index``
        is ``None``. Columns are:
        ``session``, ``mouse``, ``date``, ``region``, ``trial_condition``,
        ``decoder_run_index``, ``test_accuracy_before``, and ``test_accuracy_after``.
    """

    rows: list[dict[str, object]] = []
    for csv_path in session_csv_paths:
        session_df = pd.read_csv(csv_path)
        required_columns = {"session_id", "mouse", "date", "region", "decoder_run"}
        missing_columns = required_columns - set(session_df.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

        if decoder_run_index is None:
            selected_rows = session_df.copy()
        else:
            selected_rows = session_df.loc[session_df["decoder_run"] == int(decoder_run_index)]
            if selected_rows.shape[0] != 1:
                raise ValueError(
                    f"{csv_path} must contain exactly one row for decoder_run == {decoder_run_index}."
                )

        for _, selected_row in selected_rows.iterrows():
            for condition_name in BASE_CONDITIONS:
                rows.append(
                    {
                        "session": selected_row["session_id"],
                        "mouse": selected_row["mouse"],
                        "date": selected_row["date"],
                        "region": selected_row["region"],
                        "trial_condition": condition_name,
                        "decoder_run_index": int(selected_row["decoder_run"]),
                        "test_accuracy_before": selected_row[f"{condition_name}_test_accuracy_pre"],
                        "test_accuracy_after": selected_row[f"{condition_name}_test_accuracy_post"],
                    }
                )

    return pd.DataFrame(rows)


def save_cross_session_tables(
    output_dir: Path | str,
    decodability_df: pd.DataFrame,
    decoder_df: pd.DataFrame,
    region_name: str | None = None,
) -> tuple[Path, Path]:
    """
    Save the cross-session decodability and decoder-performance CSVs.

    Parameters
    ----------
    output_dir : Path | str
        Directory where the cross-session CSVs should be written.
    decodability_df : pd.DataFrame
        Cross-session state decodability table with one row per session and trial condition.
    decoder_df : pd.DataFrame
        Cross-session decoder-performance table with one row per session and trial condition.
    region_name : str | None, optional
        Optional region label used to namespace saved cross-session CSVs. ``None``
        preserves legacy filenames.

    Returns
    -------
    tuple[Path, Path]
        Saved decodability CSV path and saved decoder-performance CSV path.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decodability_path = output_dir / make_region_analysis_filename(
        region_name,
        "state_decodability_analysis_cross_session.csv",
    )
    decoder_path = output_dir / make_region_analysis_filename(
        region_name,
        "correct_rewarded_state_decoding_performance_cross_session.csv",
    )
    decodability_df.to_csv(decodability_path, index=False)
    decoder_df.to_csv(decoder_path, index=False)
    return decodability_path, decoder_path


def _parse_date_series(date_values: pd.Series, table_name: str) -> pd.Series:
    """
    Parse one table's date column using the project date convention.

    Parameters
    ----------
    date_values : pd.Series
        One-dimensional date values with shape ``(n_rows,)``. Dates are expected
        in ``YYYY-MM-DD`` format.
    table_name : str
        Human-readable table name used in error messages.

    Returns
    -------
    pd.Series
        Datetime values with shape ``(n_rows,)`` aligned to ``date_values``.
    """

    try:
        return pd.to_datetime(date_values, format="%Y-%m-%d", errors="raise")
    except ValueError as error:
        raise ValueError(f"{table_name} contains dates that are not in YYYY-MM-DD format.") from error


def _parse_optional_date(date_value: str | None, field_name: str) -> pd.Timestamp | None:
    """
    Parse an optional date-selection boundary.

    Parameters
    ----------
    date_value : str | None
        Date string in ``YYYY-MM-DD`` format, or ``None``.
    field_name : str
        Human-readable field name used in error messages.

    Returns
    -------
    pd.Timestamp | None
        Parsed date boundary, or ``None`` when no boundary was supplied.
    """

    if date_value is None:
        return None
    try:
        return pd.to_datetime(date_value, format="%Y-%m-%d", errors="raise")
    except ValueError as error:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format.") from error


def _normalize_include_dates(include_dates: list[str] | tuple[str, ...] | None) -> list[str] | None:
    """
    Normalize an optional explicit date list while preserving user order.

    Parameters
    ----------
    include_dates : list[str] | tuple[str, ...] | None
        Dates to include, each in ``YYYY-MM-DD`` format.

    Returns
    -------
    list[str] | None
        Unique date strings in supplied order, or ``None`` if no explicit list
        was supplied.
    """

    if include_dates is None:
        return None
    normalized_dates = list(dict.fromkeys(str(date_value) for date_value in include_dates))
    for date_value in normalized_dates:
        _parse_optional_date(date_value, "include_dates")
    return normalized_dates


def filter_table_by_date_selection(
    table_df: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    include_dates: list[str] | tuple[str, ...] | None = None,
    table_name: str = "table",
) -> pd.DataFrame:
    """
    Filter one cross-session table by date.

    Parameters
    ----------
    table_df : pd.DataFrame
        Cross-session table with shape ``(n_rows, n_columns)`` and a ``date``
        column using ``YYYY-MM-DD`` strings.
    start_date : str | None, optional
        Inclusive lower date boundary in ``YYYY-MM-DD`` format.
    end_date : str | None, optional
        Inclusive upper date boundary in ``YYYY-MM-DD`` format.
    include_dates : list[str] | tuple[str, ...] | None, optional
        Explicit dates to include. When combined with boundaries, rows must pass
        both selectors.
    table_name : str, optional
        Human-readable table name used in error messages.

    Returns
    -------
    pd.DataFrame
        Filtered copy of ``table_df`` with the original columns and row order.
    """

    if "date" not in table_df.columns:
        raise ValueError(f"{table_name} is missing required 'date' column.")

    parsed_dates = _parse_date_series(table_df["date"], table_name=table_name)
    start_timestamp = _parse_optional_date(start_date, "start_date")
    end_timestamp = _parse_optional_date(end_date, "end_date")
    if start_timestamp is not None and end_timestamp is not None and start_timestamp > end_timestamp:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    selection_mask = pd.Series(True, index=table_df.index)
    if start_timestamp is not None:
        selection_mask = selection_mask & (parsed_dates >= start_timestamp)
    if end_timestamp is not None:
        selection_mask = selection_mask & (parsed_dates <= end_timestamp)

    normalized_include_dates = _normalize_include_dates(include_dates)
    if normalized_include_dates is not None:
        selection_mask = selection_mask & table_df["date"].astype(str).isin(normalized_include_dates)

    filtered = table_df.loc[selection_mask].copy()
    if filtered.empty:
        raise ValueError(f"Date selection left no rows in {table_name}.")
    return filtered


def select_cross_session_date_range(
    decodability_df: pd.DataFrame,
    decoder_df: pd.DataFrame,
    decoder_all_runs_df: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    include_dates: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply one date selection to all cross-session plotting tables.

    Parameters
    ----------
    decodability_df : pd.DataFrame
        Cross-session decodability table with one row per session and condition.
    decoder_df : pd.DataFrame
        Cross-session decoder table for the selected decoder run.
    decoder_all_runs_df : pd.DataFrame
        Cross-session decoder table containing all decoder runs.
    start_date : str | None, optional
        Inclusive lower date boundary in ``YYYY-MM-DD`` format.
    end_date : str | None, optional
        Inclusive upper date boundary in ``YYYY-MM-DD`` format.
    include_dates : list[str] | tuple[str, ...] | None, optional
        Explicit dates to include. Requested dates must be present in the loaded
        decodability table.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(filtered_decodability_df, filtered_decoder_df,
        filtered_decoder_all_runs_df, selected_dates_df)``. The summary table has
        one row per selected date and assumes one session per date.
    """

    normalized_include_dates = _normalize_include_dates(include_dates)
    if normalized_include_dates is not None:
        available_dates = set(decodability_df["date"].astype(str)) if "date" in decodability_df.columns else set()
        missing_dates = [date_value for date_value in normalized_include_dates if date_value not in available_dates]
        if missing_dates:
            raise ValueError(f"Requested dates are not available: {missing_dates}")

    filtered_decodability = filter_table_by_date_selection(
        decodability_df,
        start_date=start_date,
        end_date=end_date,
        include_dates=normalized_include_dates,
        table_name="decodability_df",
    )
    filtered_decoder = filter_table_by_date_selection(
        decoder_df,
        start_date=start_date,
        end_date=end_date,
        include_dates=normalized_include_dates,
        table_name="decoder_df",
    )
    filtered_decoder_all_runs = filter_table_by_date_selection(
        decoder_all_runs_df,
        start_date=start_date,
        end_date=end_date,
        include_dates=normalized_include_dates,
        table_name="decoder_all_runs_df",
    )

    summary_rows: list[dict[str, object]] = []
    selected_dates = sorted(filtered_decodability["date"].astype(str).unique().tolist())
    for date_value in selected_dates:
        date_decodability = filtered_decodability.loc[filtered_decodability["date"].astype(str) == date_value]
        date_decoder = filtered_decoder.loc[filtered_decoder["date"].astype(str) == date_value]
        date_decoder_all_runs = filtered_decoder_all_runs.loc[
            filtered_decoder_all_runs["date"].astype(str) == date_value
        ]
        sessions = sorted(date_decodability["session"].astype(str).unique().tolist())
        if len(sessions) != 1:
            raise ValueError(f"Date {date_value} maps to multiple sessions: {sessions}")
        summary_rows.append(
            {
                "date": date_value,
                "session": sessions[0],
                "n_decodability_rows": int(date_decodability.shape[0]),
                "n_decoder_rows": int(date_decoder.shape[0]),
                "n_decoder_all_run_rows": int(date_decoder_all_runs.shape[0]),
            }
        )

    selected_dates_df = pd.DataFrame(summary_rows)
    return filtered_decodability, filtered_decoder, filtered_decoder_all_runs, selected_dates_df




def make_output_filename(base_name: str, filename_tag: str | None) -> str:
    """
    Build a PNG filename with an optional date-selection tag.

    Parameters
    ----------
    base_name : str
        Filename stem without extension.
    filename_tag : str | None
        Optional suffix appended before ``.png``.

    Returns
    -------
    str
        PNG filename.
    """

    if filename_tag is None:
        return f"{base_name}.png"
    return f"{base_name}_{filename_tag}.png"




















