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

# Directories
TORCPY_DIR="/home/george-tsavos/CEID/HPC_Lab/Thesis/MyExamples/torcpy_examples"
STARPUPY_DIR="/home/george-tsavos/CEID/HPC_Lab/Thesis/MyExamples/startpupy_examples"
RESULTS_DIR="/home/george-tsavos/CEID/HPC_Lab/Thesis/MyExamples/test_results"

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

# Function to run torcpy example
run_torcpy_test() {
    local example=$1
    local processes=$2
    local workers=$3
    local description=$4
    
    print_header "Running: $description"
    
    cd "$TORCPY_DIR"
    
    # Build command
    local cmd="mpirun -n $processes"
    if [ "$workers" -gt 1 ]; then
        cmd="$cmd -x TORCPY_WORKERS=$workers"
    fi
    cmd="$cmd python3 $example"
    
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
    
    cd "$STARPUPY_DIR"
    
    # Build command
    local cmd="python3 $example"
    if [ "$workers" -eq 0 ]; then
        cmd="STARPU_NWORKERS=0 $cmd"
    elif [ "$workers" -gt 1 ]; then
        cmd="STARPU_NWORKERS=$workers $cmd"
    fi
    
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
    
    print_header "Full Test Suite for $example"
    
    if [[ $example == *"torcpy"* ]]; then
        print_header "Testing torcpy example: $example"
        
        # Baseline
        run_torcpy_test "$example" 1 1 "Baseline (1 process, 1 worker)"
        
        # Distributed
        run_torcpy_test "$example" 2 1 "Distributed (2 processes, 1 worker each)"
        
        # Multi-threaded
        run_torcpy_test "$example" 1 2 "Multi-threaded (1 process, 2 workers)"
        
        # Hybrid
        run_torcpy_test "$example" 2 2 "Hybrid (2 processes, 2 workers each)"
        
    elif [[ $example == *"starpupy"* ]]; then
        print_header "Testing starpupy example: $example"
        
        # Sequential (GIL measurement)
        run_starpupy_test "$example" 0 "Sequential (STARPU_NWORKERS=0)"
        
        # Single worker
        run_starpupy_test "$example" 1 "Single worker (STARPU_NWORKERS=1)"
        
        # Multi-worker
        run_starpupy_test "$example" 2 "Multi-worker (STARPU_NWORKERS=2)"
    fi
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
    echo "Cores: $(nproc)"
    echo ""
    
    echo "MPI Information:"
    mpirun --version 2>/dev/null || echo "MPI not found"
    echo ""
    
    echo "torcpy Installation:"
    python3 -c "import torcpy; print('✓ torcpy installed')" 2>/dev/null || echo "✗ torcpy not installed"
    echo ""
    
    echo "StarPU Installation:"
    python3 -c "from starpu import starpu; print('✓ StarPU Python installed')" 2>/dev/null || echo "✗ StarPU Python not installed"
}

# Main script logic
main() {
    if [ $# -eq 0 ]; then
        # Show help if no arguments
        echo "Usage: $0 [command] [arguments]"
        echo ""
        echo "Commands:"
        echo "  system              - Show system information"
        echo "  test <example>      - Run full test suite for example"
        echo "  test-all            - Run all examples"
        echo "  torcpy <example>    - Test specific torcpy example"
        echo "  starpupy <example>  - Test specific starpupy example"
        echo "  compare             - Show comparison results"
        echo ""
        echo "Examples:"
        echo "  $0 system"
        echo "  $0 test ex00_torcpy_masterworker.py"
        echo "  $0 torcpy ex00_torcpy_masterworker.py"
        echo "  $0 starpupy ex00_starpupy_masterworker.py"
        echo "  $0 test-all"
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
            
            # Test all torcpy examples
            for f in "$TORCPY_DIR"/ex*.py; do
                run_full_test "$(basename "$f")"
            done
            
            # Test all starpupy examples
            for f in "$STARPUPY_DIR"/ex*.py; do
                run_full_test "$(basename "$f")"
            done
            
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
