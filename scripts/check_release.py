#!/usr/bin/env python3
"""Check a built source release using small generated inputs, not paper timings."""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import socket
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--pgm-cxx", default="clang++")
    args = parser.parse_args()
    require(args.threads > 0 and args.jobs > 0, "threads and jobs must be positive")
    build = args.build_dir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or build / f"release-checks-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PARLAY_NUM_THREADS=str(args.threads), MPLBACKEND="Agg")
    for name in ("NUMA_NODE", "TIMEOUT_SECONDS", "PLOT"):
        env.pop(name, None)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    checks = []

    def run(name, command, *, cwd=ROOT, expected=0, extra_env=None):
        command = list(map(str, command))
        result = subprocess.run(command, cwd=cwd, env={**env, **(extra_env or {})},
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=600)
        log = output / f"{len(checks):02d}-{name.replace(' ', '-')}.log"
        log.write_text(result.stdout)
        record = {"name": name, "command": command, "cwd": str(cwd),
                  "exit_code": result.returncode, "expected_exit_code": expected,
                  "log": log.name}
        with (output / "commands.jsonl").open("a") as target:
            target.write(json.dumps(record) + "\n")
        require(result.returncode == expected,
                f"{name} failed: expected exit {expected}, got {result.returncode}; see {log}")
        checks.append(record)
        print(f"PASS {name}", flush=True)
        return result.stdout

    for folder in (ROOT / "scripts", ROOT / "experiments"):
        for source in folder.rglob("*.py"):
            relative = source.relative_to(folder)
            if any(part == "build" or part.startswith("build-") for part in relative.parts):
                continue
            compile(source.read_bytes(), str(source), "exec")
    for source in sorted((ROOT / "scripts").glob("*.sh")):
        run(source.stem + " syntax", ["bash", "-n", source])
        run(source.stem + " help", [source, "--help"])
    run("CTest", ["ctest", "--test-dir", build, "--output-on-failure", "-j1"])
    run("RO example", [build / "example_ro"])
    run("RW example", [build / "example_rw"])

    fixtures = output / "fixtures"
    fixtures.mkdir()
    keys = [i * i + 7 * i + 2 for i in range(1000)]
    for name, values in (("noise.bin", keys), ("pgm.bin", keys + [(1 << 64) - 1])):
        (fixtures / name).write_bytes(struct.pack("<Q", len(values)) +
                                     struct.pack("<" + "Q" * len(values), *values))
    (fixtures / "bad.bin").write_bytes(struct.pack("<QQQ", 1000, 1, 2))

    # Raw-key utilities must accept unsorted input, preserve duplicates unless
    # requested, and reject malformed data or failed writes.
    sorter = build / "sort_dataset_lat1_typeNONE_b0_f0"
    raw = fixtures / "raw-unsorted.bin"
    raw.write_bytes(struct.pack("<4Q", 3, 1, 1, 2))
    sorted_raw = fixtures / "raw-sorted.bin"
    run("Sort raw keys", [sorter, raw, sorted_raw])
    require(sorted_raw.read_bytes() == struct.pack("<4Q", 1, 1, 2, 3),
            "sorter did not preserve duplicate keys")
    run("Deduplicate raw keys", [sorter, raw, sorted_raw, "deduplicate"])
    require(sorted_raw.read_bytes() == struct.pack("<3Q", 1, 2, 3),
            "sorter did not remove duplicates")
    for name, payload in (("singleton", struct.pack("<Q", 7)), ("empty", b"")):
        source = fixtures / f"raw-{name}.bin"
        target = fixtures / f"raw-{name}-sorted.bin"
        source.write_bytes(payload)
        run(f"Sort {name} input", [sorter, source, target, "deduplicate"])
        require(target.read_bytes() == payload, f"sorter changed {name} input")
    truncated_raw = fixtures / "raw-truncated.bin"
    truncated_raw.write_bytes(struct.pack("<Q", 1) + b"x")
    run("Reject truncated raw input", [sorter, truncated_raw, sorted_raw], expected=1)
    run("Reject failed raw output", [sorter, raw, "/dev/full"], expected=1)

    partition = build / "experiments/partition_count/bench_partition_count"
    partition_out = output / "partition"
    partition_out.mkdir()
    run("Partition SOSD input", [partition, fixtures / "pgm.bin", 2000, 64, 128, 4],
        cwd=partition_out)
    row = read_rows(partition_out / "results/result.csv")[-1]
    require(int(row["n"]) == 1000, "partition input must omit the count header and sentinel")
    require(int(row["pgm_partitions"]) > 0 and int(row["fh_partitions"]) > 0,
            "partition counts must be positive")
    message = run("Reject truncated SOSD", [partition, fixtures / "bad.bin", 2000, 64, 128, 4],
                  cwd=partition_out, expected=1)
    require("header does not match" in message, "expected explicit SOSD length error")
    run("Partition GEN driver", [sys.executable, ROOT / "experiments/partition_count/run_benchmark.py",
        "bench", "--datasets", "GEN", "--load-limit", 1000, "--out-dir", output / "partition-gen",
        "--build-dir", output / "partition-build", "--build-jobs", args.jobs, "--compiler", args.cxx])
    require(int(read_rows(output / "partition-gen/results/result.csv")[0]["n"]) == 1000,
            "generated partition test must preserve the requested key count")

    run("Noise analysis and plots", [sys.executable, ROOT / "experiments/noise_accumulation/noise_accumulation.py",
        "--data-dir", fixtures, "--datasets", "noise.bin", "--analysis", 0, 1000, "--bins", 10,
        "--chunk-size", 200, "--out-dir", output / "noise", "--contact-sheet"])
    require(int(read_rows(output / "noise/summary_bins10.csv")[0]["analysis_n_keys"]) == 1000,
            "noise analysis used the wrong rank range")
    require((output / "noise/contact_sheet.png").is_file(), "noise contact sheet missing")

    run("PGM analysis and plots", [sys.executable, ROOT / "experiments/pgm_statistics/pgm_statistics.py",
        "--data-dir", fixtures, "--datasets", "pgm.bin", "--analysis", 0, 1001, "--num-queries", 1000,
        "--warmup-queries", 1000, "--runs", 1, "--epsilons", 2, 64, "--out-dir", output / "pgm",
        "--build-dir", output / "pgm-build", "--compiler", args.pgm_cxx, "--contact-sheet"])
    for row in read_rows(output / "pgm/summary.csv"):
        require(int(row["pgm_indexed_keys"]) == 1000 and int(row["excluded_sentinel_keys"]) == 1,
                "PGM sentinel accounting is incorrect")
    require((output / "pgm/contact_sheet.png").is_file(), "PGM contact sheet missing")
    run("Partition plot", [sys.executable, ROOT / "experiments/partition_count/run_benchmark.py",
                          "plot", "--out-dir", partition_out])

    benchmark_env = {"BUILD_DIR": str(build), "BLOCK_SIZES": "40", "FINGER_SPARSITIES": "2",
                     "LOAD_LIMIT": "1000", "NUM_QUERIES": "1000", "NUM_ROUNDS": "2",
                     "TYPES": "RO RW"}
    run("Lookup script", [ROOT / "scripts/run_benchmarks.sh", "GEN"],
        extra_env={**benchmark_env, "RESULTS_DIR": str(output / "lookup")})
    run("RO script", [ROOT / "scripts/run_benchmarks_ro.sh", "GEN"],
        extra_env={**benchmark_env, "RESULTS_DIR": str(output / "lookup-ro")})
    run("Construction script and plots", [ROOT / "scripts/run_bench_segmenting.sh", "1000"],
        extra_env={"BUILD_DIR": str(build), "RESULTS_DIR": str(output / "segmenting"),
                   "THREADS": str(args.threads), "B_VALUES": "64", "F_VALUES": "2", "PLOT": "1",
                   "BUILD_JOBS": str(args.jobs)})
    segment = next((output / "segmenting").glob("*.csv"))
    require(int(read_rows(segment)[0]["n"]) == 1000, "construction script output is incorrect")

    # Failure accounting must be visible without changing the source CSV.
    valid_row = read_rows(segment)[0]
    mixed_rows = [valid_row]
    for status in ("ERROR", "TIMEOUT", "not-a-time"):
        mixed_rows.append({**valid_row, "total_ms": status})
    mixed_csv = segment.parent / "mixed-failures.csv"
    failed_csv = segment.parent / "all-failures.csv"
    for target, records in ((mixed_csv, mixed_rows), (failed_csv, mixed_rows[1:])):
        with target.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(valid_row))
            writer.writeheader()
            writer.writerows(records)
    original_csv = mixed_csv.read_bytes()
    message = run("Construction failure reporting",
                  [sys.executable, ROOT / "scripts/plot_segmenting.py", mixed_csv])
    require("1/4 usable rows; excluded ERROR=1, TIMEOUT=1, invalid=1" in message,
            "construction plot did not account for excluded rows")
    require(mixed_csv.read_bytes() == original_csv, "plotting modified the input CSV")
    message = run("Construction all-failed reporting",
                  [sys.executable, ROOT / "scripts/plot_segmenting.py", failed_csv], expected=1)
    require("0/3 usable rows; excluded ERROR=1, TIMEOUT=1, invalid=1" in message,
            "all-failed construction input was not reported")

    consumer = output / "consumer"
    consumer.mkdir()
    (consumer / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.16)\nproject(consumer LANGUAGES CXX)\n'
        f'add_subdirectory("{ROOT}" fh_index)\n'
        f'add_executable(consumer "{ROOT}/examples/rw.cpp")\n'
        'target_link_libraries(consumer PRIVATE fh_index::fh_index)\n')
    run("Consumer configure", ["cmake", "-S", consumer, "-B", consumer / "build",
                               "-DCMAKE_BUILD_TYPE=", f"-DCMAKE_CXX_COMPILER={args.cxx}"])
    cache = (consumer / "build/CMakeCache.txt").read_text()
    require("CMAKE_BUILD_TYPE:STRING=\n" in cache,
            "FH-index changed the parent project's build configuration")
    for option in ("BENCHMARKS", "TESTS", "EXAMPLES", "UTILITIES", "EXPERIMENTS"):
        require(f"FH_INDEX_BUILD_{option}:BOOL=OFF" in cache,
                f"consumer unexpectedly enables FH_INDEX_BUILD_{option}")
    run("Consumer build", ["cmake", "--build", consumer / "build", "-j", args.jobs])
    run("Consumer run", [consumer / "build/consumer"])
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True)
    report = {"host": socket.gethostname(), "source_commit": revision.stdout.strip(),
              "python": platform.python_version(), "threads": args.threads,
              "packages": {name: version(name) for name in ("numpy", "matplotlib", "pandas")},
              "checks": checks, "scope": "Small correctness and integration checks; not paper timings"}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"All {len(checks)} checks passed. Report: {output / 'verification.json'}")


if __name__ == "__main__":
    main()
