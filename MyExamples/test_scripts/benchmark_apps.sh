#!/bin/bash

# Benchmark Script: Run all apps multiple times across parameter combinations
# Detects system resources and benchmarks StarPU + torcpy apps with NUMA saturation
# Records minimum execution time from 5 runs per configuration
# Tracks failed parameter combinations

set -u

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
ORANGE='\033[0;33m'
NC='\033[0m' # No Color

# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Directories
TORCPY_APPS_DIR="$PROJECT_ROOT/torcpy_examples/apps"
STARPUPY_APPS_DIR="$PROJECT_ROOT/starpupy_examples/apps"
RESULTS_DIR="$PROJECT_ROOT/test_results"

# Runtime resource detection
AVAILABLE_CORES=1
LOGICAL_CORES=1
PHYSICAL_CORES=1
CORES_PER_SOCKET=1
NUMA_NODES=0
NUMA_AVAILABLE=0   # hardware capability (set by initialize_resources)
NUMA_ENABLED=0     # actual execution mode (set by user prompt)
MPI_PROCESS_LIMIT=1

# Benchmark parameters
RUNS_PER_CONFIG=5
TIMEOUT_PER_RUN=200  # timeout per run

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# SYSTEM DETECTION
# ============================================================================

initialize_resources() {
    print_header "Detecting System Resources"
    
    # Detect logical cores
    LOGICAL_CORES=$(nproc 2>/dev/null)
    if [ -z "$LOGICAL_CORES" ]; then
        LOGICAL_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null)
    fi
    if [ -z "$LOGICAL_CORES" ] || [ "$LOGICAL_CORES" -lt 1 ] 2>/dev/null; then
        LOGICAL_CORES=1
    fi

    # Detect physical cores and cores per socket
    PHYSICAL_CORES=$LOGICAL_CORES
    CORES_PER_SOCKET=$LOGICAL_CORES
    if command -v lscpu >/dev/null 2>&1; then
        local physical_cores cores_per_socket_val
        read -r physical_cores cores_per_socket_val < <(lscpu 2>/dev/null | awk -F: '
            /^Core\(s\) per socket:/ {gsub(/^[ \t]+/, "", $2); cps=$2}
            /^Socket\(s\):/          {gsub(/^[ \t]+/, "", $2); sockets=$2}
            END {
                if (cps > 0 && sockets > 0) {
                    print cps * sockets, cps
                }
            }')
        if [ -n "$physical_cores" ] && [ "$physical_cores" -gt 0 ] 2>/dev/null; then
            PHYSICAL_CORES=$physical_cores
        fi
        if [ -n "$cores_per_socket_val" ] && [ "$cores_per_socket_val" -gt 0 ] 2>/dev/null; then
            CORES_PER_SOCKET=$cores_per_socket_val
        fi
    fi

    if [ -z "$PHYSICAL_CORES" ] || [ "$PHYSICAL_CORES" -lt 1 ] 2>/dev/null; then
        PHYSICAL_CORES=1
    fi

    AVAILABLE_CORES=$PHYSICAL_CORES

    # Detect NUMA nodes
    NUMA_NODES=0
    if command -v lscpu >/dev/null 2>&1; then
        local lscpu_nodes
        lscpu_nodes=$(lscpu 2>/dev/null | awk -F: '/^NUMA node\(s\):/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')
        if [ -n "$lscpu_nodes" ] && [ "$lscpu_nodes" -gt 0 ] 2>/dev/null; then
            NUMA_NODES=$lscpu_nodes
        fi
    fi

    if [ "$NUMA_NODES" -le 1 ] 2>/dev/null && command -v numactl >/dev/null 2>&1; then
        local numactl_nodes
        numactl_nodes=$(numactl --hardware 2>/dev/null | awk '/available:/ {print $2; exit}')
        if [ -n "$numactl_nodes" ] && [ "$numactl_nodes" -gt 0 ] 2>/dev/null; then
            NUMA_NODES=$numactl_nodes
        fi
    fi

    if [ "$NUMA_NODES" -gt 1 ] 2>/dev/null && command -v numactl >/dev/null 2>&1; then
        NUMA_AVAILABLE=1
    else
        NUMA_AVAILABLE=0
    fi
    NUMA_ENABLED=0  # will be set by prompt_numa_choice after this function

    # Detect MPI process limit
    detect_mpi_process_limit

    # Display detected resources
    print_success "Logical cores: $LOGICAL_CORES"
    print_success "Physical cores: $PHYSICAL_CORES"
    print_success "Cores per socket: $CORES_PER_SOCKET"
    if [ "$NUMA_AVAILABLE" -eq 1 ]; then
        print_success "NUMA: detected ($NUMA_NODES nodes) — will prompt for execution mode"
    else
        print_success "NUMA: not detected"
    fi
    print_success "MPI process limit: $MPI_PROCESS_LIMIT"
    echo ""
}

# Ask the user whether to run in NUMA-aware or standard mode.
# Called only when NUMA hardware is detected (NUMA_AVAILABLE=1).
# Sets NUMA_ENABLED=1 for NUMA-aware, NUMA_ENABLED=0 for standard.
prompt_numa_choice() {
    if [ "$NUMA_AVAILABLE" -eq 0 ]; then
        NUMA_ENABLED=0
        return
    fi

    echo ""
    print_header "NUMA Execution Mode"
    echo -e "${YELLOW}NUMA hardware detected: $NUMA_NODES nodes${NC}"
    echo ""
    echo -e "  ${GREEN}[1]${NC} NUMA-aware execution"
    echo -e "      torcpy  → mpirun --map-by socket --bind-to core"
    echo -e "      StarPU  → STARPU_USE_NUMA=1, STARPU_SCHED=dmda, STARPU_WORKERS_GETBIND=1"
    echo -e "      Canonical config: $NUMA_NODES MPI rank(s) × $CORES_PER_SOCKET worker(s)/rank"
    echo ""
    echo -e "  ${BLUE}[2]${NC} Standard execution (non-NUMA)"
    echo -e "      torcpy  → plain mpirun -n <procs>"
    echo -e "      StarPU  → no NUMA env vars, default scheduler"
    echo ""

    local choice=""
    while true; do
        read -r -p "Select execution mode [1/2]: " choice
        case "$choice" in
            1)
                NUMA_ENABLED=1
                print_success "Running in NUMA-aware mode."
                break
                ;;
            2)
                NUMA_ENABLED=0
                print_info "Running in standard (non-NUMA) mode."
                break
                ;;
            *)
                print_warning "Invalid choice. Please enter 1 or 2."
                ;;
        esac
    done
    echo ""
}

detect_mpi_process_limit() {
    MPI_PROCESS_LIMIT=$PHYSICAL_CORES

    if ! command -v mpirun >/dev/null 2>&1; then
        return
    fi

    local p=1
    local last_ok=0

    while [ "$p" -le "$PHYSICAL_CORES" ]; do
        if timeout 5 mpirun -n "$p" /bin/true >/dev/null 2>&1; then
            last_ok=$p
            p=$((p * 2))
        else
            break
        fi
    done

    if [ "$last_ok" -ge 1 ] 2>/dev/null; then
        MPI_PROCESS_LIMIT=$last_ok
    else
        MPI_PROCESS_LIMIT=1
    fi
}

# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

# Extract execution time from output
extract_execution_time() {
    local output="$1"
    
    # Try to find "Elapsed time" pattern
    local time_line=$(echo "$output" | grep -i "elapsed time" | tail -1)
    if [ -n "$time_line" ]; then
        # Extract numeric value (handles formats like "3.1234 seconds" or "3.1234s")
        local time_val=$(echo "$time_line" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [ -n "$time_val" ]; then
            echo "$time_val"
            return 0
        fi
    fi
    
    return 1
}

# Run a single starpupy app with given worker count
run_starpupy_single() {
    local app_path="$1"
    local workers="$2"
    local timeout="$3"
    
    local output
    local exit_code
    
    # Build StarPU environment based on chosen execution mode.
    # NUMA-aware mode adds:
    #   STARPU_USE_NUMA=1          -> maps memory nodes to physical NUMA nodes via hwloc,
    #                                 enabling DMDA to calculate cross-socket transfer costs.
    #   STARPU_LIMIT_CPU_NUMA_MEM  -> caps memory per NUMA node (MB) to prevent OOM kills.
    #   STARPU_WORKERS_GETBIND=1   -> respects CPU masks from the MPI launcher, preventing
    #                                 core oversubscription inside an mpirun environment.
    #   STARPU_SCHED=dmda          -> Data-aware Multiple-implementation scheduler; uses the
    #                                 NUMA topology to place tasks on the earliest-completion unit.
    if [ "$NUMA_ENABLED" -eq 1 ]; then
        output=$(timeout "$timeout" env \
            STARPU_NCPU="$workers" \
            STARPU_NCUDA=0 \
            STARPU_NOPENCL=0 \
            STARPU_USE_NUMA=1 \
            STARPU_LIMIT_CPU_NUMA_MEM=32000 \
            STARPU_WORKERS_GETBIND=1 \
            STARPU_SCHED=dmda \
            python3 "$app_path" 2>&1)
    else
        # Standard mode: only worker-count variables, no NUMA topology or dmda scheduler.
        output=$(timeout "$timeout" env \
            STARPU_NCPU="$workers" \
            STARPU_NCUDA=0 \
            STARPU_NOPENCL=0 \
            python3 "$app_path" 2>&1)
    fi
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        # Timeout
        return 124
    elif [ $exit_code -ne 0 ]; then
        # Error
        return 1
    fi
    
    # Try to extract time
    local exec_time
    if exec_time=$(extract_execution_time "$output"); then
        echo "$exec_time"
        return 0
    fi
    
    return 1
}

# Run a single torcpy app with given process and worker counts
run_torcpy_single() {
    local app_path="$1"
    local processes="$2"
    local workers="$3"
    local timeout="$4"
    
    local output
    local exit_code
    local mpirun_cmd
    
    # Build mpirun command with strict NUMA affinity flags if NUMA is available.
    # --map-by socket  -> distributes MPI ranks across physical CPU sockets so each rank
    #                     owns an isolated NUMA memory domain (Rank 0 = Socket 0, etc.).
    # --bind-to core   -> pins every worker thread to a unique physical core, preventing
    #                     OS migration and guaranteeing L1/L2 cache and local NUMA RAM affinity.
    mpirun_cmd="mpirun -n $processes"
    if [ "$NUMA_ENABLED" -eq 1 ]; then
        mpirun_cmd="$mpirun_cmd --map-by socket --bind-to core"
    fi

    # Execute with environment variables (torcpy+MPI handles NUMA-aware distribution)
    output=$(timeout "$timeout" env TORCPY_WORKERS="$workers" $mpirun_cmd python3 "$app_path" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        # Timeout
        return 124
    elif [ $exit_code -ne 0 ]; then
        # Error
        return 1
    fi
    
    # Try to extract time
    local exec_time
    if exec_time=$(extract_execution_time "$output"); then
        echo "$exec_time"
        return 0
    fi
    
    return 1
}

# Run multiple trials and track results
run_trials_starpupy() {
    local app_path="$1"
    local app_name="$2"
    local workers="$3"
    
    local min_time=""
    local success_count=0
    local fail_count=0
    local timeout_count=0
    local run_num
    local trial_output=""
    local numa_info=""
    
    # Add NUMA info if enabled
    if [ "$NUMA_ENABLED" -eq 1 ]; then
        numa_info=" (NUMA-aware)"
    fi
    
    for ((run_num=1; run_num<=RUNS_PER_CONFIG; run_num++)); do
        local result
        result=$(run_starpupy_single "$app_path" "$workers" "$TIMEOUT_PER_RUN" 2>/dev/null)
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            ((success_count++))
            trial_output="${trial_output}✓($result)"
            
            # Update minimum time (safe numeric comparison)
            if [ -z "$min_time" ]; then
                min_time="$result"
            else
                min_time=$(awk -v a="$min_time" -v b="$result" 'BEGIN {print (a < b ? a : b)}')
            fi
        elif [ $exit_code -eq 124 ]; then
            ((timeout_count++))
            trial_output="${trial_output}⏱"
        else
            ((fail_count++))
            trial_output="${trial_output}✗"
        fi
    done
    
    echo "  Running $RUNS_PER_CONFIG trials with $workers workers${numa_info}: $trial_output" >&2
    
    printf "%d|%d|%d|%s\n" "$success_count" "$timeout_count" "$fail_count" "$min_time"
}

run_trials_torcpy() {
    local app_path="$1"
    local app_name="$2"
    local processes="$3"
    local workers="$4"
    
    local min_time=""
    local success_count=0
    local fail_count=0
    local timeout_count=0
    local run_num
    local trial_output=""
    local numa_info=""
    
    # Add NUMA info if enabled
    if [ "$NUMA_ENABLED" -eq 1 ]; then
        numa_info=" (NUMA-aware)"
    fi
    
    for ((run_num=1; run_num<=RUNS_PER_CONFIG; run_num++)); do
        local result
        result=$(run_torcpy_single "$app_path" "$processes" "$workers" "$TIMEOUT_PER_RUN" 2>/dev/null)
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            ((success_count++))
            trial_output="${trial_output}✓($result)"
            
            # Update minimum time (safe numeric comparison)
            if [ -z "$min_time" ]; then
                min_time="$result"
            else
                min_time=$(awk -v a="$min_time" -v b="$result" 'BEGIN {print (a < b ? a : b)}')
            fi
        elif [ $exit_code -eq 124 ]; then
            ((timeout_count++))
            trial_output="${trial_output}⏱"
        else
            ((fail_count++))
            trial_output="${trial_output}✗"
        fi
    done
    
    echo "  Running $RUNS_PER_CONFIG trials with $processes procs × $workers workers${numa_info}: $trial_output" >&2
    
    printf "%d|%d|%d|%s\n" "$success_count" "$timeout_count" "$fail_count" "$min_time"
}

# ============================================================================
# BENCHMARK ORCHESTRATION
# ============================================================================

benchmark_starpupy_apps() {
    print_header "Benchmarking StarPU Python Apps"
    
    local results_file="$RESULTS_DIR/benchmark_starpupy_$(date +%Y%m%d_%H%M%S).csv"
    local failed_configs=()
    
    # CSV Header
    {
        echo "timestamp,app_name,workers,success_runs,timeout_runs,failed_runs,min_time_seconds,status,numa_mode"
    } > "$results_file"
    
    local app_path
    for app_path in "$STARPUPY_APPS_DIR"/app*.py; do
        local app_name=$(basename "$app_path" .py)
        
        print_info "Benchmarking: $app_name"
        
        # Test worker counts: 1, 2, 4, 8, ...
        local workers=1
        while [ "$workers" -le "$PHYSICAL_CORES" ]; do
            local result_line
            result_line=$(run_trials_starpupy "$app_path" "$app_name" "$workers")
            
            # Parse result safely
            local success_count=$(echo "$result_line" | cut -d'|' -f1 | tr -d ' ')
            local timeout_count=$(echo "$result_line" | cut -d'|' -f2 | tr -d ' ')
            local fail_count=$(echo "$result_line" | cut -d'|' -f3 | tr -d ' ')
            local min_time=$(echo "$result_line" | cut -d'|' -f4 | tr -d ' ')
            
            # Ensure numeric values
            success_count=${success_count:-0}
            timeout_count=${timeout_count:-0}
            fail_count=${fail_count:-0}
            
            local status="OK"
            if (( success_count == 0 )); then
                status="FAILED_ALL_5"
                failed_configs+=("$app_name:${workers}workers")
                print_warning "Config FAILED_ALL_5: $app_name with $workers workers"
            elif [ "$min_time" = "" ] || [ "$min_time" = "N/A" ]; then
                status="NO_TIME"
                min_time="N/A"
            fi
            
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            local numa_mode_label="standard"
            if [ "$NUMA_ENABLED" -eq 1 ]; then numa_mode_label="numa_aware"; fi
            echo "$timestamp,$app_name,$workers,$success_count,$timeout_count,$fail_count,$min_time,$status,$numa_mode_label" >> "$results_file"
            
            workers=$((workers * 2))
        done
        
        echo ""
    done
    
    print_success "StarPU results saved to: $results_file"
    
    if [ ${#failed_configs[@]} -gt 0 ]; then
        print_warning "Failed configurations (0/5 runs succeeded):"
        printf '  - %s\n' "${failed_configs[@]}"
    fi
    
    echo ""
    return 0
}

benchmark_torcpy_apps() {
    print_header "Benchmarking torcpy Apps"
    
    local results_file="$RESULTS_DIR/benchmark_torcpy_$(date +%Y%m%d_%H%M%S).csv"
    local failed_configs=()
    
    # CSV Header
    {
        echo "timestamp,app_name,processes,workers,success_runs,timeout_runs,failed_runs,min_time_seconds,status,numa_mode"
    } > "$results_file"
    
    local app_path
    for app_path in "$TORCPY_APPS_DIR"/app*.py; do
        local app_name=$(basename "$app_path" .py)
        
        print_info "Benchmarking: $app_name"
        
        # Per the PDF, the NUMA-optimal torcpy configuration is:
        #   processes = number of NUMA nodes (one MPI rank per socket)
        #   workers   = cores per socket     (one worker thread per physical core)
        # The sweep below still tests smaller combinations for comparison, but the
        # upper bound for processes is capped at NUMA_NODES (not PHYSICAL_CORES),
        # since spawning more ranks than sockets breaks the NUMA isolation model.
        local max_processes=$NUMA_NODES
        if [ "$NUMA_ENABLED" -eq 0 ]; then
            max_processes=$PHYSICAL_CORES
        fi
        if [ "$MPI_PROCESS_LIMIT" -lt "$max_processes" ]; then
            max_processes=$MPI_PROCESS_LIMIT
        fi
        
        local processes=1
        while [ "$processes" -le "$max_processes" ]; do
            local workers=1
            while [ "$workers" -le $((PHYSICAL_CORES / processes)) ]; do
                local result_line
                result_line=$(run_trials_torcpy "$app_path" "$app_name" "$processes" "$workers")
                
                # Parse result safely
                local success_count=$(echo "$result_line" | cut -d'|' -f1 | tr -d ' ')
                local timeout_count=$(echo "$result_line" | cut -d'|' -f2 | tr -d ' ')
                local fail_count=$(echo "$result_line" | cut -d'|' -f3 | tr -d ' ')
                local min_time=$(echo "$result_line" | cut -d'|' -f4 | tr -d ' ')
                
                # Ensure numeric values
                success_count=${success_count:-0}
                timeout_count=${timeout_count:-0}
                fail_count=${fail_count:-0}
                
                local status="OK"
                if (( success_count == 0 )); then
                    status="FAILED_ALL_5"
                    failed_configs+=("$app_name:${processes}p×${workers}w")
                    print_warning "Config FAILED_ALL_5: $app_name with $processes processes × $workers workers"
                elif [ "$min_time" = "" ] || [ "$min_time" = "N/A" ]; then
                    status="NO_TIME"
                    min_time="N/A"
                fi
                
                local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
                local numa_mode_label="standard"
                if [ "$NUMA_ENABLED" -eq 1 ]; then numa_mode_label="numa_aware"; fi
                echo "$timestamp,$app_name,$processes,$workers,$success_count,$timeout_count,$fail_count,$min_time,$status,$numa_mode_label" >> "$results_file"
                
                workers=$((workers * 2))
            done
            
            processes=$((processes * 2))
        done
        
        echo ""
    done
    
    print_success "torcpy results saved to: $results_file"
    
    if [ ${#failed_configs[@]} -gt 0 ]; then
        print_warning "Failed configurations (0/5 runs succeeded):"
        printf '  - %s\n' "${failed_configs[@]}"
    fi
    
    echo ""
    return 0
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    print_header "App Benchmarking Suite"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Results directory: $RESULTS_DIR"
    echo "Runs per configuration: $RUNS_PER_CONFIG"
    echo "Timeout per run: ${TIMEOUT_PER_RUN}s"
    echo ""
    
    # System detection
    initialize_resources
    
    # Prompt user for NUMA execution mode (only if NUMA hardware is available)
    prompt_numa_choice
    
    # Show chosen NUMA execution mode
    if [ "$NUMA_ENABLED" -eq 1 ]; then
        print_success "NUMA-aware execution: ENABLED"
        print_info "  torcpy  → mpirun --map-by socket --bind-to core"
        print_info "  StarPU  → STARPU_USE_NUMA=1 STARPU_SCHED=dmda STARPU_WORKERS_GETBIND=1"
        print_info "  Canonical config: $NUMA_NODES MPI rank(s) x $CORES_PER_SOCKET worker(s)/rank"
    else
        print_info "NUMA-aware execution: DISABLED (running in standard mode)"
    fi
    echo ""
    
    # Verify app directories exist
    if [ ! -d "$STARPUPY_APPS_DIR" ]; then
        print_error "StarPU apps directory not found: $STARPUPY_APPS_DIR"
        exit 1
    fi
    
    if [ ! -d "$TORCPY_APPS_DIR" ]; then
        print_error "torcpy apps directory not found: $TORCPY_APPS_DIR"
        exit 1
    fi
    
    # Count apps
    local starpupy_count=$(find "$STARPUPY_APPS_DIR" -maxdepth 1 -name "app*.py" -type f | wc -l)
    local torcpy_count=$(find "$TORCPY_APPS_DIR" -maxdepth 1 -name "app*.py" -type f | wc -l)
    
    print_info "Found $starpupy_count StarPU apps and $torcpy_count torcpy apps"
    echo ""
    
    # Run benchmarks
    benchmark_starpupy_apps
    benchmark_torcpy_apps
    
    # Summary
    print_header "Benchmark Complete"
    print_success "All results saved to: $RESULTS_DIR"
    ls -lh "$RESULTS_DIR"/benchmark_*.csv 2>/dev/null | tail -5
    echo ""
}

# Run main function
main "$@"
