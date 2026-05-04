#!/bin/bash

# Test Runner Script for torcpy vs starpupy Examples
# Usage: ./run_tests.sh [example_name] [config]
# Example: ./run_tests.sh ex00_torcpy_masterworker baseline

set -u

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directories
TORCPY_DIR="/home/george-tsavos/CEID/HPC_Lab/Thesis/MyExamples/torcpy_examples"
STARPUPY_DIR="/home/george-tsavos/CEID/HPC_Lab/Thesis/MyExamples/startpupy_examples"
RESULTS_DIR="/home/george-tsavos/CEID/HPC_Lab/Thesis/MyExamples/test_results"

# Runtime resource detection
AVAILABLE_CORES=1
LOGICAL_CORES=1
PHYSICAL_CORES=1
NUMA_NODES=0
NUMA_ENABLED=0
MPI_PROCESS_LIMIT=1

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Function to print section headers
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Detect the usable physical CPU cores and whether the machine exposes NUMA.
initialize_resources() {
    LOGICAL_CORES=$(nproc 2>/dev/null)
    if [ -z "$LOGICAL_CORES" ]; then
        LOGICAL_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null)
    fi
    if [ -z "$LOGICAL_CORES" ] || [ "$LOGICAL_CORES" -lt 1 ] 2>/dev/null; then
        LOGICAL_CORES=1
    fi

    PHYSICAL_CORES=$LOGICAL_CORES
    if command -v lscpu >/dev/null 2>&1; then
        local physical_cores
        physical_cores=$(lscpu 2>/dev/null | awk -F: '
            /^Core\(s\) per socket:/ {gsub(/^[ \t]+/, "", $2); cores_per_socket=$2}
            /^Socket\(s\):/ {gsub(/^[ \t]+/, "", $2); sockets=$2}
            END {
                if (cores_per_socket > 0 && sockets > 0) {
                    print cores_per_socket * sockets
                }
            }')
        if [ -n "$physical_cores" ] && [ "$physical_cores" -gt 0 ] 2>/dev/null; then
            PHYSICAL_CORES=$physical_cores
        fi
    fi

    if [ -z "$PHYSICAL_CORES" ] || [ "$PHYSICAL_CORES" -lt 1 ] 2>/dev/null; then
        PHYSICAL_CORES=1
    fi

    AVAILABLE_CORES=$PHYSICAL_CORES

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
        NUMA_ENABLED=1
    else
        NUMA_ENABLED=0
    fi
}

# Detect how many MPI ranks can be launched without oversubscription.
detect_mpi_process_limit() {
    MPI_PROCESS_LIMIT=$PHYSICAL_CORES

    if ! command -v mpirun >/dev/null 2>&1; then
        return
    fi

    local p=1
    local last_ok=0

    while [ "$p" -le "$PHYSICAL_CORES" ]; do
        if mpirun -n "$p" /bin/true >/dev/null 2>&1; then
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

# Resolve an example file from the new folder layout.
resolve_example_path() {
    local root=$1
    local example=$2
    local example_name

    example_name=$(basename "$example")

    local candidates=(
        "$example"
        "$root/$example"
        "$root/apps/$example_name"
        "$root/examples/$example_name"
    )

    for candidate in "${candidates[@]}"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

# Function to run torcpy example
run_torcpy_test() {
    local example=$1
    local processes=$2
    local workers=$3
    local description=$4
    
    print_header "Running: $description"
    
    local example_path
    if ! example_path=$(resolve_example_path "$TORCPY_DIR" "$example"); then
        print_error "Could not find torcpy example: $example"
        return 1
    fi

    cd "$TORCPY_DIR"
    
    # Build command
    local cmd="mpirun -n $processes"
    if [ "$NUMA_ENABLED" -eq 1 ]; then
        cmd="$cmd --bind-to numa"
    fi
    if [ "$workers" -gt 1 ]; then
        cmd="TORCPY_WORKERS=$workers $cmd"
    fi
    cmd="$cmd python3 $example_path"
    
    echo "Command: $cmd"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Run and capture output
    local output_file="$RESULTS_DIR/${example%.py}_${processes}p_${workers}w_$(date +%s).log"
    
    if eval "$cmd" > "$output_file" 2>&1; then
        print_success "Execution completed"
        
        # Extract timing info
        if grep -q "Elapsed time" "$output_file"; then
            local time_info=$(grep "Elapsed time" "$output_file" | tail -1)
            print_success "Timing: $time_info"
        fi
        
        echo "Output saved to: $output_file"
        return 0
    else
        print_error "Execution failed"
        echo "Output saved to: $output_file"
        return 1
    fi
}

# Function to run starpupy example
run_starpupy_test() {
    local example=$1
    local workers=$2
    local description=$3
    
    print_header "Running: $description"
    
    local example_path
    if ! example_path=$(resolve_example_path "$STARPUPY_DIR" "$example"); then
        print_error "Could not find starpupy example: $example"
        return 1
    fi

    cd "$STARPUPY_DIR"
    
    # Build command
    local cmd="python3 $example_path"
    local starpu_env="STARPU_NCPU=$workers STARPU_NCUDA=0 STARPU_NOPENCL=0"
    cmd="$starpu_env $cmd"
    
    echo "Command: $cmd"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Run and capture output
    local output_file="$RESULTS_DIR/${example%.py}_${workers}w_$(date +%s).log"
    
    if eval "$cmd" > "$output_file" 2>&1; then
        print_success "Execution completed"
        
        # Extract timing info
        if grep -q "Elapsed time" "$output_file"; then
            local time_info=$(grep "Elapsed time" "$output_file" | tail -1)
            print_success "Timing: $time_info"
        fi
        
        echo "Output saved to: $output_file"
        return 0
    else
        print_error "Execution failed"
        echo "Output saved to: $output_file"
        return 1
    fi
}

# Function to run a complete test suite
run_full_test() {
    local example=$1
    local max_cores=$PHYSICAL_CORES
    
    print_header "Full Test Suite for $example"
    
    if [[ $example == *"torcpy"* ]]; then
        print_header "Testing torcpy example: $example"

        local max_processes=$max_cores
        if [ "$MPI_PROCESS_LIMIT" -lt "$max_processes" ]; then
            max_processes=$MPI_PROCESS_LIMIT
        fi

        local processes=1
        while [ "$processes" -le "$max_processes" ]; do
            local workers=1
            while [ "$workers" -le $((max_cores / processes)) ]; do
                run_torcpy_test "$example" "$processes" "$workers" "${processes} process(es), ${workers} worker(s)"
                workers=$((workers * 2))
            done
            processes=$((processes * 2))
        done
        
    elif [[ $example == *"starpupy"* ]]; then
        print_header "Testing starpupy example: $example"

        local workers=1
        while [ "$workers" -le "$max_cores" ]; do
            run_starpupy_test "$example" "$workers" "${workers} worker(s)"
            workers=$((workers * 2))
        done
    fi
}

# Function to run all torcpy examples and apps
run_all_torcpy_tests() {
    print_header "Running All torcpy Examples"

    for f in "$TORCPY_DIR"/apps/*.py "$TORCPY_DIR"/examples/*.py; do
        run_full_test "$(basename "$f")"
    done
}

# Function to run all starpupy examples and apps
run_all_starpupy_tests() {
    print_header "Running All starpupy Examples"

    for f in "$STARPUPY_DIR"/apps/*.py "$STARPUPY_DIR"/examples/*.py; do
        run_full_test "$(basename "$f")"
    done
}

# Function to compare results
compare_results() {
    print_header "Comparison Results"
    
    echo "Results directory: $RESULTS_DIR"
    echo ""
    echo "Available log files:"
    ls -lh "$RESULTS_DIR" | tail -10
}

# Function to show system info
show_system_info() {
    print_header "System Information"
    
    echo "Operating System:"
    uname -a
    echo ""
    
    echo "Python Version:"
    python3 --version
    echo ""
    
    echo "CPU Information:"
    echo "Physical cores: $PHYSICAL_CORES"
    echo "Logical cores: $LOGICAL_CORES"
    if [ "$NUMA_ENABLED" -eq 1 ]; then
        echo "NUMA: enabled (${NUMA_NODES} nodes)"
    else
        echo "NUMA: not detected"
    fi
    echo ""
    
    echo "MPI Information:"
    mpirun --version 2>/dev/null || echo "MPI not found"
    echo "MPI process limit (detected): $MPI_PROCESS_LIMIT"
    echo ""
    
    echo "torcpy Installation:"
    python3 -c "import torcpy; print('✓ torcpy installed')" 2>/dev/null || echo "✗ torcpy not installed"
    echo ""
    
    echo "StarPU Installation:"
    python3 -c "from starpu import starpu; print('✓ StarPU Python installed')" 2>/dev/null || echo "✗ StarPU Python not installed"
}

# Main script logic
main() {
    initialize_resources
    detect_mpi_process_limit

    if [ $# -eq 0 ]; then
        # Show help if no arguments
        echo "Usage: $0 [command] [arguments]"
        echo ""
        echo "Commands:"
        echo "  system              - Show system information"
        echo "  test <example>      - Run full test suite for example"
        echo "  test-all            - Run all examples"
        echo "  test-all-torcpy     - Run all torcpy examples and apps"
        echo "  test-all-starpupy   - Run all starpupy examples and apps"
        echo "  torcpy <example>    - Test specific torcpy example"
        echo "  starpupy <example>  - Test specific starpupy example"
        echo "  compare             - Show comparison results"
        echo ""
        echo "Examples:"
        echo "  $0 system"
        echo "  $0 test examples/ex00_torcpy_masterworker.py"
        echo "  $0 torcpy examples/ex00_torcpy_masterworker.py"
        echo "  $0 starpupy examples/ex00_starpupy_masterworker.py"
        echo "  $0 test-all"
        echo "  $0 test-all-torcpy"
        echo "  $0 test-all-starpupy"
        exit 0
    fi
    
    case "$1" in
        system)
            show_system_info
            ;;
        test)
            if [ $# -lt 2 ]; then
                print_error "Please specify an example"
                exit 1
            fi
            run_full_test "$2"
            ;;
        test-all)
            print_header "Running All Examples"

            run_all_torcpy_tests
            run_all_starpupy_tests

            compare_results
            ;;
        test-all-torcpy)
            run_all_torcpy_tests
            compare_results
            ;;
        test-all-starpupy)
            run_all_starpupy_tests
            
            compare_results
            ;;
        torcpy)
            if [ $# -lt 2 ]; then
                print_error "Please specify an example"
                exit 1
            fi
            run_full_test "$2"
            ;;
        starpupy)
            if [ $# -lt 2 ]; then
                print_error "Please specify an example"
                exit 1
            fi
            run_full_test "$2"
            ;;
        compare)
            compare_results
            ;;
        *)
            print_error "Unknown command: $1"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
