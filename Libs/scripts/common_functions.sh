#!/bin/bash

# =============================================================================
# @file        common_functions.sh
# @brief       Common utility functions for neuron simulation scripts
# @author      NeuronSim Research Team
# @version     1.0.0
# @date        2024-01-01
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

# Standard logging function with timestamps and colors
log() {
    local level="$1"
    local message="$2"
    local color=""

    case "$level" in
        "INFO")  color="$BLUE" ;;
        "SUCCESS") color="$GREEN" ;;
        "WARNING") color="$YELLOW" ;;
        "ERROR") color="$RED" ;;
        *) color="$NC" ;;
    esac

    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message${NC}"
}

# Convenience function for simple log messages (defaults to INFO level)
log_message() {
    log "INFO" "$1"
}

# Enhanced logging with file output
log_with_file() {
    local level="$1"
    local message="$2"
    local log_file="$3"

    local timestamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
    local log_entry="$timestamp [$level] $message"

    echo "$log_entry"

    if [[ -n "$log_file" && -w "$(dirname "$log_file")" ]]; then
        echo "$log_entry" >> "$log_file"
    fi
}

# =============================================================================
# Error Handling Functions
# =============================================================================

# Standard error exit function
error_exit() {
    local message="$1"
    local exit_code="${2:-1}"

    log "ERROR" "$message"
    exit "$exit_code"
}

# Warning function (non-fatal)
warning_msg() {
    local message="$1"
    log "WARNING" "$message"
}

# Success message
success_msg() {
    local message="$1"
    log "SUCCESS" "$message"
}

# Resolve Conda from the caller's environment or an explicit portable override.
conda_activate_script() {
    if [[ -n "${NEURONSIM_CONDA_ACTIVATE:-}" ]]; then
        printf '%s\n' "${NEURONSIM_CONDA_ACTIVATE}"
        return 0
    fi
    if [[ -n "${CONDA_EXE:-}" ]]; then
        printf '%s\n' "$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)/etc/profile.d/conda.sh"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        printf '%s\n' "$(conda info --base)/etc/profile.d/conda.sh"
        return 0
    fi
    return 1
}

# Strict conda activation (hw env)
require_conda_hw() {
    local conda_activate
    conda_activate="$(conda_activate_script || true)"
    if [[ -z "$conda_activate" || ! -f "$conda_activate" ]]; then
        error_exit "A portable Conda activation script is required; set NEURONSIM_CONDA_ACTIVATE"
    fi
    # shellcheck disable=SC1090
    set +u
    if ! source "$conda_activate" 2>/dev/null || ! conda activate hw 2>/dev/null; then
        set -u
        error_exit "Failed to activate conda environment: hw"
    fi
    set -u
}

require_conda_env() {
    local env_name="$1"
    local conda_activate
    conda_activate="$(conda_activate_script || true)"
    if [[ -z "$conda_activate" || ! -f "$conda_activate" ]]; then
        error_exit "A portable Conda activation script is required; set NEURONSIM_CONDA_ACTIVATE"
    fi
    set +u
    if ! source "$conda_activate" 2>/dev/null || ! conda activate "$env_name" 2>/dev/null; then
        set -u
        error_exit "Failed to activate conda environment: $env_name"
    fi
    set -u
}

# Strict YAML scalar getter
yaml_get_strict() {
    local yaml_path="$1"
    local key_path="$2"
    python3 - "$yaml_path" "$key_path" <<'PY'
import sys, yaml
path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
node = cfg
for part in key.split("."):
    if not isinstance(node, dict) or part not in node:
        raise SystemExit(f"Missing key '{key}' in {path}")
    node = node[part]
if isinstance(node, (dict, list)):
    raise SystemExit(f"Expected scalar at '{key}' in {path}")
print(node)
PY
}

# Normalize SLURM time strings loaded through YAML.
# PyYAML interprets values like 24:00:00 as integer seconds, while Slurm expects
# HH:MM:SS or D-HH:MM:SS. This helper converts numeric seconds back to HH:MM:SS.
normalize_slurm_time() {
    local raw_value="$1"
    python3 - "$raw_value" <<'PY'
import sys

raw = sys.argv[1].strip()
if not raw:
    raise SystemExit("Empty SLURM time value")

if ":" in raw or "-" in raw:
    print(raw)
    raise SystemExit(0)

try:
    total_seconds = int(float(raw))
except ValueError:
    print(raw)
    raise SystemExit(0)

if total_seconds < 0:
    raise SystemExit(f"Invalid negative SLURM time: {raw}")

hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
PY
}

project_root_from_scripts() {
    if [[ -n "${PROJECT_ROOT:-}" ]]; then
        echo "${PROJECT_ROOT}"
        return 0
    fi

    local helper_dir
    helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "$(cd "${helper_dir}/../.." && pwd)"
}

read_project_prefix() {
    local project_root
    project_root="$(project_root_from_scripts)"

    local prefix_file="${project_root}/PROJECT_PREFIX"
    if [[ ! -f "${prefix_file}" ]]; then
        error_exit "PROJECT_PREFIX not found: ${prefix_file}"
    fi

    local prefix
    prefix="$(tr -d '\r\n' < "${prefix_file}")"
    if [[ ! "${prefix}" =~ ^[A-Z][A-Za-z]{2,7}$ ]]; then
        error_exit "Invalid PROJECT_PREFIX '${prefix}'. Expected 3-8 letters with leading uppercase."
    fi

    echo "${prefix}"
}

slurm_job_name() {
    local suffix="$1"
    local prefix="${PROJECT_PREFIX:-}"
    if [[ -z "${prefix}" ]]; then
        prefix="$(read_project_prefix)"
    fi
    echo "${prefix}_${suffix}"
}

# Optional YAML scalar getter with explicit default
yaml_get_or_default() {
    local yaml_path="$1"
    local key_path="$2"
    local default_value="$3"
    python3 - "$yaml_path" "$key_path" "$default_value" <<'PY'
import sys, yaml
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
node = cfg
for part in key.split("."):
    if not isinstance(node, dict) or part not in node:
        print(default)
        raise SystemExit(0)
    node = node[part]
if isinstance(node, (dict, list)):
    print(default)
else:
    print(node)
PY
}

# Strict YAML list getter
yaml_get_list_strict() {
    local yaml_path="$1"
    local key_path="$2"
    python3 - "$yaml_path" "$key_path" <<'PY'
import sys, yaml
path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
node = cfg
for part in key.split("."):
    if not isinstance(node, dict) or part not in node:
        raise SystemExit(f"Missing key '{key}' in {path}")
    node = node[part]
if not isinstance(node, list):
    raise SystemExit(f"Expected list at '{key}' in {path}")
print(" ".join(str(v) for v in node))
PY
}

# Extract list-of-dict field (e.g., variants[*].id) as space-separated list
yaml_get_list_of_dict_field() {
    local yaml_path="$1"
    local key_path="$2"     # path to list node
    local field="$3"        # field name inside each dict
    python3 - "$yaml_path" "$key_path" "$field" <<'PY'
import sys, yaml
path, key, field = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
node = cfg
for part in key.split("."):
    if not isinstance(node, dict) or part not in node:
        raise SystemExit(f"Missing key '{key}' in {path}")
    node = node[part]
if not isinstance(node, list):
    raise SystemExit(f"Expected list at '{key}' in {path}")
values = []
for idx, item in enumerate(node):
    if not isinstance(item, dict) or field not in item:
        raise SystemExit(f"Item {idx} in '{key}' missing field '{field}'")
    values.append(str(item[field]))
print(" ".join(values))
PY
}

normalize_override_list() {
    local raw_value="${1:-}"
    python3 - "$raw_value" <<'PY'
import sys

raw = sys.argv[1].strip()
if not raw:
    print("")
    raise SystemExit(0)

tokens = []
seen = set()
for part in raw.replace(",", " ").split():
    token = part.strip()
    if not token or token in seen:
        continue
    seen.add(token)
    tokens.append(token)
print(" ".join(tokens))
PY
}

resolve_override_list() {
    local raw_override="${1:-}"
    shift
    python3 - "$raw_override" "$@" <<'PY'
import sys

override_raw = sys.argv[1].strip()
default_items = [item for item in sys.argv[2:] if item]

if not override_raw:
    print(" ".join(default_items))
    raise SystemExit(0)

requested = []
seen = set()
for part in override_raw.replace(",", " ").split():
    token = part.strip()
    if not token or token in seen:
        continue
    seen.add(token)
    requested.append(token)

missing = [token for token in requested if token not in default_items]
if missing:
    raise SystemExit(
        "Unknown override values: "
        + ", ".join(missing)
        + ". Available: "
        + ", ".join(default_items)
    )

print(" ".join(requested))
PY
}

normalize_slurm_gpu_id() {
    local raw_value="${1:-}"
    python3 - "$raw_value" <<'PY'
import sys

raw = sys.argv[1].strip()
if not raw:
    print("None")
    raise SystemExit(0)

lowered = raw.lower()
if lowered in {"none", "null", "auto"}:
    print("None")
else:
    print(raw)
PY
}

# Render template string with placeholders
yaml_render_template() {
    local template="$1"
    local task="$2"
    local baseline="$3"
    local seed="$4"
    python3 - "$template" "$task" "$baseline" "$seed" <<'PY'
import sys
tpl, task, baseline, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
out = tpl.replace("${task}", task).replace("${baseline}", baseline).replace("${seed}", seed)
print(out)
PY
}

# Generate Slurm log out/err paths; ensures directory exists
# Usage: make_slurm_log_paths <category> <task> <name...>
make_slurm_log_paths() {
    local category="$1"; local task="$2"; shift 2
    local subpath="${*// /_}"
    local base="${PROJECT_ROOT:-$(pwd)}/Output/logs/${category}/${task}/${subpath}"
    mkdir -p "$base"
    echo "${base}/%j.out ${base}/%j.err"
}

# Generate unified benchmark log path
# Usage: make_benchmark_log_path <category> <task> <exp_name> <seed>
# Example: make_benchmark_log_path "bench" "ABeta3D" "main" "42"
# Returns: Output/logs/bench/abeta3d/benchmark_abeta3d_main_42.log
make_benchmark_log_path() {
    local category="$1"
    local task="$2"
    local exp_name="$3"
    local seed="$4"
    
    local bench_name="${task,,}"
    local log_dir="${PROJECT_ROOT:-$(pwd)}/Output/logs/${category}/${bench_name}"
    mkdir -p "$log_dir"
    
    local log_file="${log_dir}/benchmark_${bench_name}_${exp_name}_${seed}.log"
    echo "$log_file"
}

# Append CSV with basic lock (mkdir-based) to avoid concurrent corruption
append_csv_with_lock() {
    local csv_path="$1"
    local line="$2"
    local lock_dir="${csv_path}.lock"
    while ! mkdir "$lock_dir" 2>/dev/null; do
        sleep 1
    done
    mkdir -p "$(dirname "$csv_path")"
    if [[ ! -f "$csv_path" ]]; then
        echo "ModelName,Task,JobID,ExitCode,StartTime,EndTime,Status,CompletionTime" > "$csv_path"
    fi
    echo "$line" >> "$csv_path"
    rmdir "$lock_dir"
}

# Check if command exists
check_command() {
    local command="$1"
    local error_msg="${2:-"Required command '$command' not found"}"

    if ! command -v "$command" &> /dev/null; then
        error_exit "$error_msg"
    fi
    success_msg "$command is available"
}

# Check if file exists
check_file() {
    local file_path="$1"
    local description="${2:-"File"}"
    local should_exist="${3:-true}"

    if [[ "$should_exist" == "true" ]]; then
        if [[ ! -f "$file_path" ]]; then
            error_exit "$description not found: $file_path"
        fi
        success_msg "$description exists: $(basename "$file_path")"
    else
        if [[ -f "$file_path" ]]; then
            warning_msg "$description already exists: $file_path"
        fi
    fi
}

# Check if directory exists and create if needed
ensure_directory() {
    local dir_path="$1"
    local description="${2:-"Directory"}"

    if [[ ! -d "$dir_path" ]]; then
        if mkdir -p "$dir_path"; then
            success_msg "Created $description: $dir_path"
        else
            error_exit "Failed to create $description: $dir_path"
        fi
    else
        success_msg "$description exists: $dir_path"
    fi
}

# Check Slurm environment
check_slurm_environment() {
    if [[ -n "$SLURM_JOB_ID" ]]; then
        log "INFO" "Running under Slurm Job ID: $SLURM_JOB_ID"
        log "INFO" "Node: $SLURMD_NODENAME"
        # Calculate total allocated CPUs
        local allocated_cpus
        if [[ -n "$SLURM_CPUS_ON_NODE" ]]; then
            allocated_cpus="$SLURM_CPUS_ON_NODE"
        elif [[ -n "$SLURM_CPUS_PER_TASK" && -n "$SLURM_NTASKS" ]]; then
            allocated_cpus=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
        else
            allocated_cpus="$SLURM_CPUS_PER_TASK"
        fi
        log "INFO" "CPUs allocated: $allocated_cpus"
        log "INFO" "Memory allocated: $SLURM_MEM_PER_NODE MB"
    else
        log "INFO" "Not running under Slurm (direct execution)"
        log "INFO" "Node: $(hostname)"
        log "INFO" "CPUs available: $(nproc)"
        log "INFO" "Memory available: $(free -m | awk 'NR==2{printf "%.0f", $2}') MB"
    fi
}

# Check MPI environment and validate resource allocation
check_mpi_environment() {
    local required_mpi_processes="$1"
    local log_file="${2:-/dev/null}"
    
    log_with_file "INFO" "Checking MPI environment..." "$log_file"
    
    # Check if mpiexec is available
    local mpiexec_path
    if command -v mpiexec &> /dev/null; then
        mpiexec_path=$(command -v mpiexec)
        log_with_file "INFO" "mpiexec is available at: $mpiexec_path" "$log_file"
    else
        error_exit "mpiexec not found - MPI is required for parallel simulation"
    fi
    
    # Check if nrniv is available
    if ! command -v nrniv &> /dev/null; then
        error_exit "nrniv not found - NEURON is required for simulation"
    fi
    log_with_file "INFO" "nrniv is available" "$log_file"
    
    # Test MPI with a simple command
    log_with_file "INFO" "Testing MPI with $required_mpi_processes processes..." "$log_file"
    
    # In Slurm environment, be more lenient with MPI testing
    # Test with 1 process first (most basic test)
    if mpiexec -n 1 hostname &> /dev/null; then
        log_with_file "INFO" "MPI basic test successful with 1 process" "$log_file"
        
        # Try with 2 processes if we need more than 1
        if [[ "$required_mpi_processes" -gt 1 ]]; then
            if mpiexec -n 2 hostname &> /dev/null; then
                log_with_file "INFO" "MPI test successful with 2 processes" "$log_file"
                
                # Try with the full number if we need more than 2
                if [[ "$required_mpi_processes" -gt 2 ]]; then
                    if mpiexec -n "$required_mpi_processes" hostname &> /dev/null; then
                        log_with_file "INFO" "MPI test successful with $required_mpi_processes processes" "$log_file"
                    else
                        log_with_file "WARNING" "MPI test failed with $required_mpi_processes processes, but basic MPI works" "$log_file"
                        log_with_file "INFO" "Continuing with simulation - MPI will be tested during execution" "$log_file"
                    fi
                fi
            else
                log_with_file "WARNING" "MPI test failed with 2 processes, but basic MPI works" "$log_file"
                log_with_file "INFO" "Continuing with simulation - MPI will be tested during execution" "$log_file"
            fi
        fi
    else
        error_exit "MPI basic test failed with 1 process - check MPI installation"
    fi
    
    # Check available CPU cores
    local available_cores
    if [[ -n "$SLURM_CPUS_ON_NODE" ]]; then
        # Use total cores allocated to this job on this node
        available_cores="$SLURM_CPUS_ON_NODE"
    elif [[ -n "$SLURM_CPUS_PER_TASK" && -n "$SLURM_NTASKS" ]]; then
        # Calculate total cores: tasks × cores_per_task
        available_cores=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
    else
        available_cores=$(nproc)
    fi
    
    log_with_file "INFO" "Available CPU cores: $available_cores" "$log_file"
    log_with_file "INFO" "Required MPI processes: $required_mpi_processes" "$log_file"
    
    # Validate resource allocation
    if [[ "$required_mpi_processes" -gt "$available_cores" ]]; then
        error_exit "Insufficient CPU cores: need $required_mpi_processes, have $available_cores"
    fi
    
    log_with_file "INFO" "MPI environment validation passed" "$log_file"
}

# =============================================================================
# Configuration Functions
# =============================================================================

# Validate YAML configuration file
validate_yaml_config() {
    local config_file="$1"
    local log_file="$2"

    check_file "$config_file" "Configuration file"

    log_with_file "INFO" "Validating configuration file..." "$log_file"

    python3 - "$config_file" <<'PY'
import sys, yaml
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    yaml.safe_load(f)
print("Configuration file is valid")
PY
    if [[ $? -ne 0 ]]; then
        error_exit "Configuration file validation failed"
    fi
}

# Update configuration for cluster resources
update_config_for_cluster() {
    local config_file="$1"
    local nthreads="$2"
    local mpi_processes="$3"
    local log_file="$4"

    log_with_file "INFO" "Updating configuration for cluster resources..." "$log_file"

    if python3 -c "
import yaml
import sys

try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)

    # Update threads and MPI settings
    if 'simulation' not in config:
        config['simulation'] = {}
    if 'computation' not in config:
        config['computation'] = {}

    config['simulation']['nthread'] = $nthreads
    config['computation']['use_mpi'] = True
    config['computation']['mpi_processes'] = $mpi_processes

    with open('$config_file', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f'Updated configuration: {$nthreads} threads, {$mpi_processes} MPI processes')
    sys.exit(0)
except Exception as e:
    print(f'Failed to update config: {e}')
    sys.exit(1)
"; then
        success_msg "Configuration updated for cluster"
    else
        error_exit "Failed to update configuration"
    fi
}

# =============================================================================
# Performance Optimization Functions
# =============================================================================

# Set optimal environment variables for neuron simulations
set_neuron_environment() {
    local omp_threads="$1"
    local rxd_threads="$2"

    export OMP_NUM_THREADS="$omp_threads"
    export RXD_THREADS="$rxd_threads"
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export NEURON_NUM_THREADS="$omp_threads"
    export NEURON_RXD_THREADS="$rxd_threads"

    # Modern OpenMP settings
    export OMP_PROC_BIND=spread
    export OMP_PLACES=cores
    export OMP_DYNAMIC=true
    export OMP_NESTED=true

    # Memory optimization
    export MALLOC_TRIM_THRESHOLD_=0
    export MALLOC_MMAP_THRESHOLD_=131072
    export MALLOC_MMAP_MAX_=1048576

    log "INFO" "Environment optimized for neuron simulation:"
    log "INFO" "  OMP_NUM_THREADS: $omp_threads"
    log "INFO" "  RXD_THREADS: $rxd_threads"
    log "INFO" "  NEURON threads: $omp_threads"
    log "INFO" "  Memory optimization: enabled"
}

# =============================================================================
# Process Management Functions
# =============================================================================

# Monitor background process
monitor_process() {
    local pid="$1"
    local process_name="$2"
    local log_file="$3"
    local check_interval="${4:-300}"  # 5 minutes default

    log_with_file "INFO" "Starting $process_name monitoring (PID: $pid)" "$log_file"

    while kill -0 "$pid" 2>/dev/null; do
        log_with_file "INFO" "$process_name (PID: $pid) is still running..." "$log_file"
        sleep "$check_interval"
    done

    wait "$pid" 2>/dev/null
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        success_msg "$process_name completed successfully"
    else
        error_exit "$process_name failed with exit code $exit_code"
    fi

    return $exit_code
}

# =============================================================================
# Python/NEURON Validation Functions
# =============================================================================

# Check NEURON installation
check_neuron_installation() {
    local log_file="$1"
    local simulation_type="${2:-unknown}"

    log_with_file "INFO" "Checking NEURON installation..." "$log_file"

    # Check Python NEURON module
    python3 - <<'PY'
from neuron import h  # noqa: F401
print('NEURON Python module is available')
PY
    if [[ $? -ne 0 ]]; then
        error_exit "NEURON Python module not available"
    fi

    # Check nrniv command
    check_command "nrniv" "NEURON compiler (nrniv) not found"

    # Check if mechanisms need compilation
    # SD simulations use RxD framework and don't require .mod files
    if [[ "$simulation_type" == "SD" || "$simulation_type" == "sd" ]]; then
        log_with_file "INFO" "SD simulation detected - using RxD framework, no mod files required" "$log_file"
        log_with_file "INFO" "NEURON installation check completed for SD simulation" "$log_file"
    elif [[ -d "mods" && -n "$(find mods -name "*.mod" 2>/dev/null)" ]]; then
        log_with_file "INFO" "Found mod files: $(find mods -name "*.mod" | wc -l) files" "$log_file"

        if [[ -f "x86_64/libnrnmech.so" ]]; then
            success_msg "Mechanisms already compiled"
        else
            log_with_file "INFO" "Compiling NEURON mechanisms..." "$log_file"
            if nrnivmodl mods >> "$log_file" 2>&1; then
                success_msg "NEURON mechanisms compiled successfully"
            else
                error_exit "Failed to compile NEURON mechanisms"
            fi
        fi
    else
        log_with_file "WARNING" "No mod files found in mods directory" "$log_file"
    fi
}

# =============================================================================
# Utility Functions
# =============================================================================

# Get timestamp for file naming
get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# Format bytes to human readable
format_bytes() {
    local bytes="$1"
    local units=("B" "KB" "MB" "GB" "TB")

    local i=0
    while [[ $bytes -gt 1024 && $i -lt 4 ]]; do
        bytes=$((bytes / 1024))
        ((i++))
    done

    echo "$bytes${units[i]}"
}

# Check available disk space
check_disk_space() {
    local dir="$1"
    local min_space_gb="${2:-10}"

    local available_space
    available_space=$(df "$dir" | awk 'NR==2 {print int($4/1024/1024)}')

    log "INFO" "Available disk space in $(dirname "$dir"): ${available_space}GB"

    if [[ $available_space -lt $min_space_gb ]]; then
        warning_msg "Low disk space: ${available_space}GB available, minimum ${min_space_gb}GB recommended"
    fi
}
