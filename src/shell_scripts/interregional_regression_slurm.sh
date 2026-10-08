#!/usr/bin/env bash

#SBATCH --partition=unlimited
#SBATCH --job-name=interregional_poisson
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gs/gsfs0/users/mchin1/logs/interregional_regression_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.chin@einsteinmed.edu

set -euo pipefail


fail() {
    # Report configuration errors before the scientific runner starts.
    printf 'interregional_regression_slurm.sh: %s\n' "$*" >&2
    exit 2
}


if [[ ! "${SLURM_CPUS_PER_TASK-}" =~ ^[0-9]+$ ]]; then
    fail "SLURM_CPUS_PER_TASK must be a positive integer"
fi
if [[ "$SLURM_CPUS_PER_TASK" -ne 8 ]]; then
    fail "SLURM_CPUS_PER_TASK must equal the reviewed eight-CPU allocation"
fi

# Slurm runs a spooled script copy, so the checkout must come from submission
# metadata rather than the script's runtime location.
if [[ -z "${SLURM_SUBMIT_DIR-}" ]]; then
    fail "SLURM_SUBMIT_DIR must identify the submitted repository root"
fi
if [[ ! -d "$SLURM_SUBMIT_DIR" ]]; then
    fail "SLURM_SUBMIT_DIR is not an existing repository directory"
fi
submit_directory="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
repository_root="$(
    git -C "$submit_directory" rev-parse --show-toplevel 2>/dev/null
)" || fail "SLURM_SUBMIT_DIR is not inside a Git repository"
repository_root="$(cd -- "$repository_root" && pwd -P)"
if [[ "$submit_directory" != "$repository_root" ]]; then
    fail "SLURM_SUBMIT_DIR must be the repository root"
fi

required_paths=(
    pyproject.toml
    uv.lock
    src/__init__.py
    src/neural_analysis/__init__.py
    src/neural_analysis/interregional/__init__.py
    src/neural_analysis/interregional/run_session.py
    src/shell_scripts/interregional_regression_slurm.sh
)
for required_path in "${required_paths[@]}"; do
    git -C "$repository_root" ls-files --error-unmatch "$required_path" \
        >/dev/null 2>&1 || fail "required execution dependency is not tracked: $required_path"
done

tracked_status="$(
    git -C "$repository_root" status --porcelain --untracked-files=no
)" || fail "unable to inspect repository tracked status"
if [[ -n "$tracked_status" ]]; then
    fail "repository has tracked changes; commit or restore them before submission"
fi

command -v uv >/dev/null 2>&1 || fail "uv is unavailable on PATH"

# This runner has one Python process and measured threaded linear algebra, so
# the numerical backends may use the complete Slurm CPU allocation.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1

cd -- "$repository_root"
repository_commit="$(git rev-parse HEAD)"

printf 'Cluster host: %s\n' "$(hostname)"
printf 'UTC start: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Repository root: %s\n' "$repository_root"
printf 'Repository commit: %s\n' "$repository_commit"
printf 'Repository status: tracked-clean\n'
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID-unavailable}"
printf 'SLURM_JOB_NAME=%s\n' "${SLURM_JOB_NAME-unavailable}"
printf 'SLURM_CPUS_PER_TASK=%s\n' "$SLURM_CPUS_PER_TASK"
printf 'Thread limits: OMP=%s MKL=%s OPENBLAS=%s\n' \
    "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS"
uv --version
printf 'Launcher arguments:'
printf ' %q' "$@"
printf '\n'

# Replacing the shell preserves the runner's exit status and lets Slurm TERM
# reach it directly. A retry is an ordinary fresh `new` invocation.
exec uv run --frozen --no-sync --offline \
    python -m src.neural_analysis.interregional.run_session "$@"
