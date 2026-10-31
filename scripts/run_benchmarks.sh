#!/usr/bin/env bash
# Portable lookup benchmark driver. See docs/benchmarks.md.
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    printf '%s\n' 'Usage: run_benchmarks.sh [GEN|raw-key-file]' \
        'Environment: BUILD_DIR, RESULTS_DIR, PARLAY_NUM_THREADS, NUMA_NODE,' \
        'LOAD_LIMIT, NUM_ROUNDS, NUM_QUERIES, SEED, TYPES, BLOCK_SIZES, FINGER_SPARSITIES.'
    exit 0
fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${BUILD_DIR:-"$root/build"}
results_dir=${RESULTS_DIR:-"$root/results/lookup_$(date +%Y%m%d_%H%M%S)"}
dataset=${1:-GEN}
threads=${PARLAY_NUM_THREADS:-4}
read -r -a types <<< "${TYPES:-RO RW}"
read -r -a blocks <<< "${BLOCK_SIZES:-32 40 48}"
read -r -a fingers <<< "${FINGER_SPARSITIES:-2 4}"
prefix=()
if [[ -n ${NUMA_NODE:-} ]]; then
    command -v numactl >/dev/null || { echo 'NUMA_NODE requires numactl' >&2; exit 1; }
    prefix=(numactl "--cpunodebind=$NUMA_NODE" "--membind=$NUMA_NODE")
fi
if [[ $dataset != GEN && ! -f $dataset ]]; then
    echo "Dataset not found: $dataset" >&2
    exit 1
fi
# Check every target before starting a potentially long run.
for type in "${types[@]}"; do
    for block in "${blocks[@]}"; do
        for finger in "${fingers[@]}"; do
            binary="$build_dir/bench_fh_index_lat0_type${type}_b${block}_f${finger}"
            [[ -x $binary ]] || { echo "Build the requested benchmark first: $binary" >&2; exit 1; }
        done
    done
done
mkdir -p "$results_dir"
for type in "${types[@]}"; do
    for block in "${blocks[@]}"; do
        for finger in "${fingers[@]}"; do
            binary="$build_dir/bench_fh_index_lat0_type${type}_b${block}_f${finger}"
            logfile="$results_dir/${type}_b${block}_f${finger}.log"
            command=("${prefix[@]}" env "PARLAY_NUM_THREADS=$threads" "$binary"
                "$dataset" "${LOAD_LIMIT:-100000}" "${NUM_ROUNDS:-3}"
                "${NUM_QUERIES:-100000}" 0 0 0 "${SEED:-42}" find)
            printf '%q ' "${command[@]}" | tee "$logfile"
            printf '\n' | tee -a "$logfile"
            "${command[@]}" 2>&1 | tee -a "$logfile"
        done
    done
done
printf 'Logs: %s\n' "$results_dir"
