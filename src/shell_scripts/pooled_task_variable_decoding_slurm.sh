#!/usr/bin/env bash

#SBATCH --partition=normal
#SBATCH --job-name=pooled_task_decoding
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gs/gsfs0/users/mchin1/logs/pooled_task_decoding_%A_%a.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.chin@einsteinmed.edu

set -euo pipefail


fail() {
    printf 'pooled_task_variable_decoding_slurm.sh: %s\n' "$*" >&2
    exit 2
}


if [[ -n "${SLURM_JOB_ID-}" ]]; then
    if [[ "${SLURM_CPUS_PER_TASK-}" != "1" ]]; then
        fail "scheduled execution requires exactly one Slurm CPU per task"
    fi
    if [[ -z "${SLURM_SUBMIT_DIR-}" ]] || [[ ! -d "$SLURM_SUBMIT_DIR" ]]; then
        fail "SLURM_SUBMIT_DIR must identify the submitted repository root"
    fi
    working_directory="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
    mode="scheduled"
else
    working_directory="$(pwd -P)"
    mode="login"
fi

repository_root="$(git -C "$working_directory" rev-parse --show-toplevel 2>/dev/null)" || \
    fail "working directory is not inside a Git repository"
repository_root="$(cd -- "$repository_root" && pwd -P)"
if [[ "$working_directory" != "$repository_root" ]]; then
    fail "submission and login commands must run from the repository root"
fi

required_paths=(
    pyproject.toml
    uv.lock
    src/neural_analysis/task_decoding/pooled/__init__.py
    src/neural_analysis/task_decoding/pooled/pipeline.py
    src/neural_analysis/task_decoding/pooled/run_batch.py
    src/neural_analysis/task_decoding/pooled/run_session.py
    src/neural_analysis/task_decoding/pooled/slurm.py
    src/shell_scripts/pooled_task_variable_decoding_slurm.sh
)
for required_path in "${required_paths[@]}"; do
    git -C "$repository_root" ls-files --error-unmatch "$required_path" >/dev/null 2>&1 || \
        fail "required execution dependency is not tracked: $required_path"
done

tracked_status="$(git -C "$repository_root" status --porcelain --untracked-files=no)" || \
    fail "unable to inspect tracked repository status"
if [[ -n "$tracked_status" ]]; then
    fail "repository has tracked changes; commit or restore them before submission"
fi
untracked_python="$(
    git -C "$repository_root" ls-files --others --exclude-standard -- \
        ':(glob)src/neural_analysis/task_decoding/pooled/*.py'
)" || fail "unable to inspect untracked pooled task-decoding source"
if [[ -n "$untracked_python" ]]; then
    fail "pooled task-decoding package contains untracked Python source"
fi

command -v uv >/dev/null 2>&1 || fail "uv is unavailable on PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd -- "$repository_root"

if [[ "$mode" == "scheduled" ]]; then
    if [[ -n "${POOLED_DECODING_EXPECTED_COMMIT-}" ]]; then
        current_commit="$(git rev-parse HEAD)" || fail "unable to resolve scheduled commit"
        if [[ "$current_commit" != "$POOLED_DECODING_EXPECTED_COMMIT" ]]; then
            fail "scheduled checkout no longer matches the submitted commit"
        fi
    fi
    case "${1-}" in
        _execute-prepared)
            exec uv run --frozen --no-sync --offline \
                python -m src.neural_analysis.task_decoding.pooled.run_session "$@"
            ;;
        _execute-array)
            exec uv run --frozen --no-sync --offline \
                python -m src.neural_analysis.task_decoding.pooled.run_batch "$@"
            ;;
        *)
            fail "scheduled execution accepts only _execute-prepared or _execute-array"
            ;;
    esac
fi

case "${1-}" in
    submit-new|submit-resume|status)
        exec uv run --frozen --no-sync --offline \
            python -m src.neural_analysis.task_decoding.pooled.slurm "$@"
        ;;
    batch-dry-run|submit-array|batch-status)
        exec uv run --frozen --no-sync --offline \
            python -m src.neural_analysis.task_decoding.pooled.run_batch "$@"
        ;;
    *)
        fail "unsupported pooled task-decoding login command"
        ;;
esac
