#!/usr/bin/env bash
# Construction-stage timing. See docs/benchmarks.md.
set -euo pipefail
if [[ $# -ne 1 || ${1:-} == --help ]]; then
    printf '%s\n' 'Usage: run_bench_segmenting.sh <raw-key-file|number-of-generated-keys>' \
        'Environment: BUILD_DIR, RESULTS_DIR, THREADS, B_VALUES, F_VALUES,' \
        'BUILD_JOBS, NUMA_NODE (optional), TIMEOUT_SECONDS (optional), PLOT=1.'
    [[ ${1:-} == --help ]] && exit 0
    exit 1
fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dataset=$1
build_dir=${BUILD_DIR:-"$root/build"}
results_dir=${RESULTS_DIR:-"$root/results"}
read -r -a threads <<< "${THREADS:-1 2 4}"
read -r -a blocks <<< "${B_VALUES:-64}"
read -r -a fingers <<< "${F_VALUES:-2 4 8}"
prefix=()
if [[ -n ${TIMEOUT_SECONDS:-} ]]; then
    command -v timeout >/dev/null || { echo 'TIMEOUT_SECONDS requires timeout' >&2; exit 1; }
    prefix=(timeout "${TIMEOUT_SECONDS}s")
fi
if [[ -n ${NUMA_NODE:-} ]]; then
    command -v numactl >/dev/null || { echo 'NUMA_NODE requires numactl' >&2; exit 1; }
    prefix+=(numactl "--cpunodebind=$NUMA_NODE" "--membind=$NUMA_NODE")
fi
target=bench_segmenting_lat0_typeNONE_b0_f0
cmake -S "$root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DFH_INDEX_BUILD_BENCHMARKS=ON
cmake --build "$build_dir" --target "$target" -j "${BUILD_JOBS:-4}"
mkdir -p "$results_dir"
result_file="$results_dir/segmenting_$(date +%Y%m%d_%H%M%S)_$$.csv"
printf '%s\n' 'dataset,n,B,F,threads,total_ms,build_maximal_ms,segment_selection_ms,segment_adjustment_ms,generate_fgds_ms,generate_finger_keys_ms,compute_finger_positions_ms' > "$result_file"
for thread in "${threads[@]}"; do
    for block in "${blocks[@]}"; do
        for finger in "${fingers[@]}"; do
            output=$("${prefix[@]}" env "PARLAY_NUM_THREADS=$thread" \
                "$build_dir/$target" "$dataset" "$block" "$finger")
            printf '%s\n' "$output" >> "$result_file"
        done
    done
done
printf 'Results: %s\n' "$result_file"
if [[ ${PLOT:-0} == 1 ]]; then
    MPLBACKEND=Agg python3 "$root/scripts/plot_segmenting.py" "$result_file"
fi
