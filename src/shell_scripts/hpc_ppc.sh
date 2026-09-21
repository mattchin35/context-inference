#!/usr/bin/env bash

#SBATCH --partition=unlimited
#SBATCH --job-name=ppc_cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gs/gsfs0/users/mchin1/logs/ppc_cluster_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthew.chin@einsteinmed.edu

set -euo pipefail


fail() {
    # Report one configuration error before any Python process starts.
    printf 'hpc_ppc.sh: %s\n' "$*" >&2
    exit 2
}


if [[ ! "${SLURM_CPUS_PER_TASK-}" =~ ^[0-9]+$ ]]; then
    fail "SLURM_CPUS_PER_TASK must be a positive integer"
fi
if [[ "$SLURM_CPUS_PER_TASK" -ne 8 ]]; then
    fail "this reviewed wrapper requires exactly eight Slurm CPUs per task"
fi

# Slurm executes a spooled copy of this file, so the script location cannot
# identify the submitted checkout. Require submission from the repository root.
if [[ -z "${SLURM_SUBMIT_DIR-}" ]]; then
    fail "SLURM_SUBMIT_DIR must identify the submitted repository"
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
if [[ ! -f "$repository_root/pyproject.toml" ]] || \
    [[ ! -f "$repository_root/src/neural_analysis/lfp_spike_phase_launcher.py" ]]; then
    fail "repository does not contain the Spike-phase launcher"
fi
tracked_status="$(
    git -C "$repository_root" status --porcelain --untracked-files=no
)" || fail "unable to inspect repository tracked status"
if [[ -n "$tracked_status" ]]; then
    fail "repository has tracked changes; commit or restore them before submission"
fi

# Bash validates only the execution-resource field it owns. Scientific CLI
# validation remains in the Python launcher.
explicit_worker_count=""
arguments=("$@")
for ((argument_index = 0; argument_index < ${#arguments[@]}; argument_index++)); do
    argument="${arguments[argument_index]}"
    if [[ "$argument" == "--workers" ]]; then
        if [[ -n "$explicit_worker_count" ]]; then
            fail "--workers may be specified only once"
        fi
        if ((argument_index + 1 >= ${#arguments[@]})); then
            fail "--workers requires a value"
        fi
        explicit_worker_count="${arguments[argument_index + 1]}"
        ((argument_index += 1))
    elif [[ "$argument" == --workers=* ]]; then
        if [[ -n "$explicit_worker_count" ]]; then
            fail "--workers may be specified only once"
        fi
        explicit_worker_count="${argument#--workers=}"
    fi
done
if [[ -n "$explicit_worker_count" ]]; then
    if [[ ! "$explicit_worker_count" =~ ^[0-9]+$ ]]; then
        fail "--workers value must be a positive integer"
    fi
    if [[ "$explicit_worker_count" -ne "$SLURM_CPUS_PER_TASK" ]]; then
        fail "--workers must equal SLURM_CPUS_PER_TASK"
    fi
fi

command -v uv >/dev/null 2>&1 || fail "uv is unavailable on PATH"

# Prevent NumPy/SciPy backends from multiplying the explicit process count.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

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

# Replacing the shell preserves launcher exit codes and makes Slurm TERM reach
# the Python process directly so it can persist resumable state.
exec uv run --frozen --no-sync --offline \
    python -m src.neural_analysis.lfp_spike_phase_launcher "$@"
