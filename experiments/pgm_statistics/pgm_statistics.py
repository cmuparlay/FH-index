#!/usr/bin/env python3
"""Run and plot PGM-index statistics on real SOSD-format datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_BUILD_DIR = SCRIPT_DIR / "build"
PGM_INDEX_DIR = PROJECT_DIR / "third_party" / "PGM-index"

DATASETS_200M = [
    "uniformrandom_200M_uint64.bin",
    "books_200M_uint64.bin",
    "covid_200M_uint64.bin",
    "fb_200M_uint64.bin",
    "genome_200M_uint64.bin",
    "history_200M_uint64.bin",
    "libio_200M_uint64.bin",
    "osm_200M_uint64.bin",
    "planet_200M_uint64.bin",
    "stack_200M_uint64.bin",
    "wise_200M_uint64.bin",
]

DATASET_DISPLAY_NAMES = {
    "uniformrandom": "Uniform Random",
    "books": "Books",
    "covid": "COVID",
    "fb": "FB",
    "genome": "Genome",
    "history": "History",
    "libio": "Libio",
    "osm": "OSM",
    "planet": "Planet",
    "stack": "Stack",
    "wise": "Wise",
}

SUPPORTED_EPSILONS = (2, 4, 8, 16, 32, 64, 128, 256, 512)

COMMON_FIELDS = [
    "dataset",
    "source_path",
    "source_n_keys",
    "analysis_start",
    "analysis_end",
    "indexed_keys",
    "pgm_indexed_keys",
    "excluded_sentinel_keys",
    "num_queries",
    "runs",
    "seed",
    "warmup_queries",
    "pgm_commit",
]

RAW_FIELDS = COMMON_FIELDS + [
    "epsilon",
    "group",
    "forward_partition_ns",
    "forward_total_ns",
    "reverse_total_ns",
    "reverse_partition_ns",
    "num_segments",
    "height",
    "memory_bytes",
]

SUMMARY_FIELDS = COMMON_FIELDS + [
    "epsilon",
    "total_time_ns",
    "total_stddev_ns",
    "find_partition_time_ns",
    "find_partition_stddev_ns",
    "last_mile_time_ns",
    "last_mile_stddev_ns",
    "num_segments",
    "height",
    "memory_bytes",
]


def parse_count(value: str) -> int:
    match = re.fullmatch(r"([0-9][0-9_]*)\s*([KMBkmb]?)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid count {value!r}; use an integer or a K/M/B suffix"
        )
    number = int(match.group(1).replace("_", ""))
    scale = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    return number * scale[match.group(2).lower()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PGM-index on real datasets using successful queries "
            "sampled uniformly by rank from the indexed keys."
        )
    )
    parser.add_argument(
        "--analysis",
        type=parse_count,
        nargs=2,
        required=True,
        metavar=("START", "END"),
        help="Zero-based half-open rank range; bounds may use K/M/B suffixes.",
    )
    parser.add_argument(
        "--data-dir", default="datasets"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--datasets", nargs="*", default=DATASETS_200M)
    parser.add_argument("--num-queries", type=parse_count, default=100_000_000)
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of forward/reverse counterbalanced measurement groups.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup-queries",
        type=parse_count,
        default=None,
        help="Queries per warmup batch (default: the full query count).",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=int,
        default=list(SUPPORTED_EPSILONS),
    )
    parser.add_argument(
        "--mode", choices=("all", "benchmark", "plot"), default="all"
    )
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument(
        "--compiler",
        default="clang++",
        help="C++ compiler used for the benchmark build (default: clang++).",
    )
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument(
        "--contact-sheet",
        nargs="?",
        const="contact_sheet.png",
        default=None,
        metavar="FILENAME",
        help="Also assemble all generated PNGs into one contact sheet.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip datasets whose raw and summary CSV files already exist.",
    )
    args = parser.parse_args()

    start, end = args.analysis
    if start >= end:
        parser.error("--analysis must satisfy START < END")
    if not args.datasets:
        parser.error("--datasets must contain at least one dataset")
    if args.num_queries <= 0:
        parser.error("--num-queries must be positive")
    if args.warmup_queries is None:
        args.warmup_queries = args.num_queries
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.warmup_queries < 0:
        parser.error("--warmup-queries must be nonnegative")
    if len(set(args.epsilons)) != len(args.epsilons):
        parser.error("--epsilons must not contain duplicates")
    unsupported = sorted(set(args.epsilons) - set(SUPPORTED_EPSILONS))
    if unsupported:
        parser.error(
            "unsupported epsilon(s): "
            + ", ".join(map(str, unsupported))
            + "; supported values are "
            + ", ".join(map(str, SUPPORTED_EPSILONS))
        )
    if args.mode == "benchmark" and args.contact_sheet:
        parser.error("--contact-sheet requires --mode all or --mode plot")
    return args


def dataset_stem(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name
    for suffix in ("_200M_uint64.bin", "_800M_uint64.bin", ".bin"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def dataset_display_name(name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def resolve_dataset(data_dir: Path, dataset_name: str) -> Path:
    path = Path(dataset_name)
    return path if path.is_absolute() else data_dir / path


def inspect_dataset(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"{path} is too small to contain a uint64 length header")
        (n_keys,) = struct.unpack("<Q", header)
        expected_size = 8 + n_keys * 8
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"{path} size mismatch: header expects {expected_size} bytes, "
                f"file contains {actual_size}"
            )

        excluded_sentinel_keys = 0
        max_u64 = (1 << 64) - 1
        position = n_keys
        while position > 0:
            take = min(4096, position)
            position -= take
            file.seek(8 + position * 8)
            values = struct.unpack(f"<{take}Q", file.read(take * 8))
            for value in reversed(values):
                if value == max_u64:
                    excluded_sentinel_keys += 1
                else:
                    return n_keys, excluded_sentinel_keys
        return n_keys, excluded_sentinel_keys


def run_checked(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, **kwargs)


def compiler_path(compiler: str) -> str:
    resolved = shutil.which(compiler)
    if resolved is None:
        raise RuntimeError(f"C++ compiler not found: {compiler}")
    return resolved


def build_benchmark(build_dir: Path, compiler: str) -> Path:
    resolved_compiler = compiler_path(compiler)
    build_dir.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "cmake",
            "-S",
            str(SCRIPT_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_COMPILER={resolved_compiler}",
        ]
    )
    run_checked(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            str(os.cpu_count() or 1),
        ]
    )
    benchmark = build_dir / "pgm_benchmark"
    if not benchmark.is_file():
        raise RuntimeError(f"benchmark executable was not created: {benchmark}")
    return benchmark


def git_output(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def sample_stddev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_epsilon: dict[int, list[dict[str, Any]]] = {}
    for row in raw_rows:
        by_epsilon.setdefault(int(row["epsilon"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for epsilon in sorted(by_epsilon):
        rows = sorted(by_epsilon[epsilon], key=lambda row: int(row["group"]))
        expected_runs = int(rows[0]["runs"])
        if len(rows) != expected_runs:
            raise ValueError(
                f"epsilon {epsilon} has {len(rows)} measurement groups; "
                f"expected {expected_runs}"
            )

        partition_samples = [
            (float(row["forward_partition_ns"]) +
             float(row["reverse_partition_ns"]))
            / 2.0
            for row in rows
        ]
        total_samples = [
            (float(row["forward_total_ns"]) + float(row["reverse_total_ns"]))
            / 2.0
            for row in rows
        ]
        last_mile_samples = [
            total - partition
            for total, partition in zip(total_samples, partition_samples)
        ]
        first = rows[0]
        summaries.append(
            {
                **{field: first[field] for field in COMMON_FIELDS},
                "epsilon": epsilon,
                "total_time_ns": statistics.mean(total_samples),
                "total_stddev_ns": sample_stddev(total_samples),
                "find_partition_time_ns": statistics.mean(partition_samples),
                "find_partition_stddev_ns": sample_stddev(partition_samples),
                "last_mile_time_ns": statistics.mean(last_mile_samples),
                "last_mile_stddev_ns": sample_stddev(last_mile_samples),
                "num_segments": int(first["num_segments"]),
                "height": int(first["height"]),
                "memory_bytes": int(first["memory_bytes"]),
            }
        )
    return summaries


def run_dataset(
    benchmark: Path,
    dataset_path: Path,
    source_n_keys: int,
    excluded_sentinel_keys: int,
    args: argparse.Namespace,
    raw_path: Path,
    summary_path: Path,
    pgm_commit: str,
) -> list[dict[str, Any]]:
    start, end = args.analysis
    name = dataset_stem(dataset_path)
    if end > source_n_keys:
        raise ValueError(
            f"rank range {start}:{end} exceeds dataset {name} "
            f"with {source_n_keys} keys"
        )

    command = [
        str(benchmark),
        str(dataset_path),
        str(start),
        str(end),
        str(args.num_queries),
        str(args.runs),
        str(args.seed),
        str(args.warmup_queries),
        ",".join(map(str, args.epsilons)),
    ]
    print("$ " + " ".join(command), flush=True)
    backend_path = raw_path.with_suffix(".backend.csv.partial")
    backend_path.parent.mkdir(parents=True, exist_ok=True)
    with backend_path.open("w") as backend_file:
        subprocess.run(command, check=True, text=True, stdout=backend_file)

    backend_rows = read_csv(backend_path)
    expected_rows = len(args.epsilons) * args.runs
    if len(backend_rows) != expected_rows:
        raise ValueError(
            f"backend emitted {len(backend_rows)} rows for {name}; "
            f"expected {expected_rows}"
        )

    common = {
        "dataset": name,
        "source_path": str(dataset_path.resolve()),
        "source_n_keys": source_n_keys,
        "analysis_start": start,
        "analysis_end": end,
        "indexed_keys": end - start,
        "pgm_indexed_keys": end - start - excluded_sentinel_keys,
        "excluded_sentinel_keys": excluded_sentinel_keys,
        "num_queries": args.num_queries,
        "runs": args.runs,
        "seed": args.seed,
        "warmup_queries": args.warmup_queries,
        "pgm_commit": pgm_commit,
    }
    raw_rows = [{**common, **row} for row in backend_rows]
    summaries = summarize_rows(raw_rows)
    write_csv(raw_path, raw_rows, RAW_FIELDS)
    write_csv(summary_path, summaries, SUMMARY_FIELDS)
    backend_path.unlink()
    print(f"Saved raw measurements: {raw_path}", flush=True)
    print(f"Saved dataset summary: {summary_path}", flush=True)
    return summaries


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    compiler: str,
    datasets: list[dict[str, Any]],
) -> None:
    compiler_result = subprocess.run(
        [compiler, "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    cpu_affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else "unavailable"
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_affinity": cpu_affinity,
        "compiler": compiler,
        "compiler_version": compiler_result.stdout.splitlines()[0],
        "pgm_commit": git_output(PGM_INDEX_DIR, "rev-parse", "HEAD"),
        "pgm_remote": git_output(PGM_INDEX_DIR, "remote", "get-url", "origin"),
        "analysis": list(args.analysis),
        "num_queries": args.num_queries,
        "runs": args.runs,
        "run_definition": (
            "Each run is one partition-then-total measurement followed by one "
            "total-then-partition measurement; the two orders are averaged."
        ),
        "query_definition": (
            "100% successful lookups sampled uniformly by rank with replacement "
            "from all PGM-indexed keys in the analysis range. Values equal to "
            "UINT64_MAX are excluded because upstream PGM-index reserves that "
            "value as an internal sentinel."
        ),
        "seed": args.seed,
        "warmup_queries": args.warmup_queries,
        "epsilons": args.epsilons,
        "datasets": datasets,
        "source_sha256": {
            "pgm_statistics.py": sha256_file(Path(__file__)),
            "pgm_benchmark.cpp": sha256_file(SCRIPT_DIR / "pgm_benchmark.cpp"),
            "CMakeLists.txt": sha256_file(SCRIPT_DIR / "CMakeLists.txt"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    print(f"Saved run metadata: {path}", flush=True)


def plot_summary(
    summary_path: Path,
    figures_dir: Path,
    save_pdf: bool,
) -> Path:
    try:
        import matplotlib
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "plotting requires matplotlib and numpy; install them or run "
            "with --mode benchmark"
        ) from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["hatch.linewidth"] = 0.5

    rows = read_csv(summary_path)
    rows.sort(key=lambda row: int(row["epsilon"]), reverse=True)
    if not rows:
        raise ValueError(f"summary contains no rows: {summary_path}")

    epsilons = np.asarray([int(row["epsilon"]) for row in rows])
    total_time = np.asarray([float(row["total_time_ns"]) for row in rows])
    total_stddev = np.asarray([float(row["total_stddev_ns"]) for row in rows])
    partition_time = np.asarray(
        [float(row["find_partition_time_ns"]) for row in rows]
    )
    last_mile_time = np.asarray([float(row["last_mile_time_ns"]) for row in rows])
    num_segments = np.asarray([int(row["num_segments"]) for row in rows])
    dataset = rows[0]["dataset"]

    if np.any(last_mile_time < 0):
        print(
            f"Warning: {dataset} has a negative total-minus-partition estimate; "
            "the plot preserves the measured value.",
            file=sys.stderr,
        )

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, left_axis = plt.subplots(figsize=(3.5, 2.55))
    right_axis = left_axis.twinx()
    x = np.arange(len(epsilons))

    left_axis.bar(
        x,
        partition_time,
        0.6,
        label="Find Partition",
        color="#EE854A",
        edgecolor="#222222",
        linewidth=0,
        hatch="///",
        alpha=0.9,
    )
    left_axis.bar(
        x,
        last_mile_time,
        0.6,
        bottom=partition_time,
        label="Last-Mile Search",
        color="#4878D0",
        edgecolor="#222222",
        linewidth=0,
        hatch="...",
        alpha=0.9,
    )
    left_axis.errorbar(
        x,
        total_time,
        yerr=total_stddev,
        fmt="none",
        ecolor="#333333",
        elinewidth=0.7,
        capsize=1.8,
        capthick=0.7,
        zorder=3,
    )
    right_axis.plot(
        x,
        num_segments,
        color="#6ACC64",
        linewidth=1.5,
        marker="o",
        markersize=4,
        label="Number of Partitions",
    )

    left_axis.set_xticks(x)
    left_axis.set_xticklabels(epsilons, fontsize=9)
    left_axis.set_xlabel(
        "Last-Mile Search Bound ($\\epsilon$)\n" + dataset_display_name(dataset),
        fontsize=10,
    )
    left_axis.set_ylabel("Lookup Time (ns)", fontsize=10)
    left_axis.tick_params(axis="y", labelsize=9)
    left_axis.grid(True, alpha=0.3, axis="y")

    right_axis.set_yscale("log")
    right_axis.set_ylabel("Number of Partitions", fontsize=10)
    right_axis.tick_params(axis="y", labelsize=9)

    left_handles, left_labels = left_axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        fontsize=9,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
    )

    figure.tight_layout()
    figure.subplots_adjust(top=0.85, left=0.05, right=0.95)
    png_path = figures_dir / f"{dataset}.png"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        figure.savefig(
            figures_dir / f"{dataset}.pdf", dpi=300, bbox_inches="tight"
        )
    plt.close(figure)
    print(f"Saved figure: {png_path}", flush=True)
    return png_path


def create_contact_sheet(figures: list[Path], output_path: Path) -> None:
    try:
        import matplotlib
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "contact-sheet generation requires matplotlib and numpy"
        ) from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not figures:
        raise ValueError("cannot create a contact sheet without figures")
    columns = min(3, len(figures))
    rows = math.ceil(len(figures) / columns)
    sheet, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.8 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes_array.flat:
        axis.axis("off")
    for axis, figure_path in zip(axes_array.flat, figures):
        axis.imshow(plt.imread(figure_path))
    sheet.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.savefig(output_path, dpi=100, facecolor="white")
    plt.close(sheet)
    print(f"Saved contact sheet: {output_path}", flush=True)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    summary_dir = out_dir / "summary"
    figures_dir = (
        Path(args.figures_dir) if args.figures_dir else out_dir / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [dataset_stem(name) for name in args.datasets]
    summaries: list[dict[str, Any]] = []

    if args.mode in ("all", "benchmark"):
        data_dir = Path(args.data_dir)
        dataset_records: list[dict[str, Any]] = []
        for dataset_name in args.datasets:
            path = resolve_dataset(data_dir, dataset_name)
            n_keys, source_sentinel_keys = inspect_dataset(path)
            if args.analysis[1] > n_keys:
                raise ValueError(
                    f"rank range {args.analysis[0]}:{args.analysis[1]} exceeds "
                    f"dataset {dataset_stem(path)} with {n_keys} keys"
                )
            excluded_sentinel_keys = (
                source_sentinel_keys if args.analysis[1] == n_keys else 0
            )
            if args.analysis[1] - args.analysis[0] <= excluded_sentinel_keys:
                raise ValueError(
                    f"analysis range for {dataset_stem(path)} contains only "
                    "PGM's reserved UINT64_MAX sentinel"
                )
            dataset_records.append(
                {
                    "name": dataset_stem(path),
                    "path": str(path.resolve()),
                    "source_n_keys": n_keys,
                    "pgm_indexed_keys": (
                        args.analysis[1]
                        - args.analysis[0]
                        - excluded_sentinel_keys
                    ),
                    "excluded_sentinel_keys": excluded_sentinel_keys,
                }
            )

        build_dir = Path(args.build_dir)
        benchmark = build_benchmark(build_dir, args.compiler)
        compiler = compiler_path(args.compiler)
        pgm_commit = git_output(PGM_INDEX_DIR, "rev-parse", "HEAD")
        write_metadata(
            out_dir / "run_metadata.json", args, compiler, dataset_records
        )

        for dataset_name, record in zip(args.datasets, dataset_records):
            path = resolve_dataset(data_dir, dataset_name)
            name = record["name"]
            raw_path = raw_dir / f"{name}.csv"
            summary_path = summary_dir / f"{name}.csv"
            if args.resume and raw_path.is_file() and summary_path.is_file():
                print(f"Skipping completed dataset: {name}", flush=True)
                summaries.extend(read_csv(summary_path))
                continue
            print(
                f"[{path.name}] n={record['source_n_keys']:,}, "
                f"rank_range={args.analysis[0]}:{args.analysis[1]}, "
                f"queries={args.num_queries:,}",
                flush=True,
            )
            summaries.extend(
                run_dataset(
                    benchmark,
                    path,
                    int(record["source_n_keys"]),
                    int(record["excluded_sentinel_keys"]),
                    args,
                    raw_path,
                    summary_path,
                    pgm_commit,
                )
            )
        write_csv(out_dir / "summary.csv", summaries, SUMMARY_FIELDS)
        print(f"Saved combined summary: {out_dir / 'summary.csv'}", flush=True)

    if args.mode in ("all", "plot"):
        generated_figures: list[Path] = []
        for name in dataset_names:
            summary_path = summary_dir / f"{name}.csv"
            if not summary_path.is_file():
                raise FileNotFoundError(f"summary not found: {summary_path}")
            generated_figures.append(
                plot_summary(summary_path, figures_dir, args.save_pdf)
            )
        if args.contact_sheet:
            create_contact_sheet(
                generated_figures, out_dir / args.contact_sheet
            )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
