#!/usr/bin/env python3
"""Generate real-data noise-accumulation figures for an explicit rank range.

Example:
    python3 noise_accumulation.py \
        --analysis 99M 101M \
        --out-dir results/median_2m_20260626 \
        --bins 100 --save-pdf

The rank range is zero-based and half-open. Bounds may use decimal K/M/B
suffixes, so ``99M 101M`` means ranks ``[99_000_000, 101_000_000)``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


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

# ACM's two-column ``sigconf`` layout has a roughly 3.33-inch column. Generate
# figures at their final print size so LaTeX does not shrink nominal 10-point
# Matplotlib text to an unreadable size.
PAPER_COLUMN_WIDTH_IN = 3.33
PAPER_FIGURE_HEIGHT_IN = 1.90
PAPER_X_LABEL_SIZE_PT = 7.0
PAPER_TICK_SIZE_PT = 6.0
PAPER_LEGEND_SIZE_PT = 6.0


@dataclass
class WindowDiagnostic:
    dataset: str
    source_n_keys: int
    window_start: int
    window_end: int
    window_size: int
    window_quantile: float
    min_key: int
    max_key: int
    key_span: int
    bins: int
    slope: float
    intercept: float
    expected_keys_per_bin: float
    max_frequency: float
    p95_abs_error: float
    max_abs_error: float
    p95_abs_bin_deviation: float
    max_abs_bin_deviation: float
    p95_error_to_bin_deviation: float
    count_span_to_expected: float


def parse_rank_bound(value: str) -> int:
    match = re.fullmatch(r"([0-9][0-9_]*)\s*([KMBkmb]?)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid rank bound {value!r}; use an integer or a K/M/B suffix"
        )
    number = int(match.group(1).replace("_", ""))
    scale = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    return number * scale[match.group(2).lower()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        type=parse_rank_bound,
        nargs=2,
        required=True,
        metavar=("START", "END"),
        help="Zero-based half-open rank range; bounds may use K/M/B suffixes.",
    )
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=5_000_000)
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument(
        "--contact-sheet",
        nargs="?",
        const="contact_sheet.png",
        default=None,
        metavar="FILENAME",
        help="Also assemble all generated PNGs into one contact sheet.",
    )
    parser.add_argument("--datasets", nargs="*", default=DATASETS_200M)
    args = parser.parse_args()

    start, end = args.analysis
    if end <= start:
        parser.error("--analysis must satisfy START < END")
    if args.bins <= 0:
        parser.error("--bins must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if not args.datasets:
        parser.error("--datasets must contain at least one dataset")
    return args


def dataset_stem(path: Path) -> str:
    name = path.name
    for suffix in ("_200M_uint64.bin", "_800M_uint64.bin", ".bin"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def dataset_display_name(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset, dataset.replace("_", " ").title())


def format_key_count_tick(value: float, _position: float | None = None) -> str:
    if value == 0:
        return "0"
    absolute_value = abs(value)
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if absolute_value >= scale:
            return f"{value / scale:.1f}{suffix}"
    return f"{value:.1f}"


def read_keys(path: Path) -> np.memmap:
    with path.open("rb") as file:
        header = file.read(8)
    if len(header) != 8:
        raise ValueError(f"{path} is too small to contain a uint64 length header")
    n = int(np.frombuffer(header, dtype="<u8")[0])
    return np.memmap(path, dtype="<u8", mode="r", offset=8, shape=(n,))


def resolve_dataset(data_dir: Path, dataset_name: str) -> Path:
    dataset = Path(dataset_name)
    return dataset if dataset.is_absolute() else data_dir / dataset


def iter_window_chunks(start: int, end: int, chunk_size: int) -> Iterable[slice]:
    for chunk_start in range(start, end, chunk_size):
        yield slice(chunk_start, min(chunk_start + chunk_size, end))


def fit_ols(
    keys: np.memmap,
    start: int,
    end: int,
    min_key: int,
    span: int,
    chunk_size: int,
) -> tuple[float, float]:
    """Fit y = slope*x + intercept to every key/rank pair in the range."""
    window_n = end - start
    sum_x = 0.0
    sum_x2 = 0.0
    sum_xy = 0.0

    for chunk in iter_window_chunks(start, end, chunk_size):
        key_chunk = keys[chunk].astype(np.float64, copy=False)
        x = (key_chunk - float(min_key)) / float(span)
        y = np.arange(chunk.start - start, chunk.stop - start, dtype=np.float64)
        sum_x += float(np.sum(x))
        sum_x2 += float(np.sum(x * x))
        sum_xy += float(np.sum(x * y))

    sum_y = window_n * (window_n - 1) / 2.0
    mean_x = sum_x / window_n
    mean_y = sum_y / window_n
    var_x = sum_x2 - window_n * mean_x * mean_x
    cov_xy = sum_xy - window_n * mean_x * mean_y
    if var_x == 0.0:
        return 0.0, mean_y
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def make_edges(min_key: int, max_key: int, bins: int) -> np.ndarray:
    span = max_key - min_key
    edges = [min_key + int(round(span * i / bins)) for i in range(bins + 1)]
    edges[0] = min_key
    edges[-1] = max_key
    for index in range(1, len(edges)):
        if edges[index] < edges[index - 1]:
            edges[index] = edges[index - 1]
    return np.asarray(edges, dtype=np.uint64)


def predecessor_ranks(keys: np.ndarray | np.memmap, edges: np.ndarray) -> np.ndarray:
    ranks = np.searchsorted(keys, edges, side="right").astype(np.float64) - 1.0
    return np.clip(ranks, 0.0, len(keys) - 1.0)


def evaluate_range(
    dataset_name: str,
    keys: np.memmap,
    start: int,
    end: int,
    bins: int,
    chunk_size: int,
) -> tuple[WindowDiagnostic, dict[str, np.ndarray | float]]:
    local_keys = keys[start:end]
    window_n = end - start
    min_key = int(local_keys[0])
    max_key = int(local_keys[-1])
    span = max_key - min_key
    if window_n <= bins or span <= 0:
        raise ValueError("rank range has too few keys or zero key span")

    slope, intercept = fit_ols(keys, start, end, min_key, span, chunk_size)
    edges = make_edges(min_key, max_key, bins)
    x = (edges.astype(np.float64) - float(min_key)) / float(span)
    ranks = predecessor_ranks(local_keys, edges)
    prediction = slope * x + intercept
    error = ranks - prediction

    # Count the keys crossed between consecutive predecessor queries. Using
    # rank differences gives the bins the same (left, right] convention as the
    # plotted predecessor error; the minimum key is the initial condition.
    bin_counts = np.diff(ranks)
    expected_per_bin = slope * np.diff(x)
    expected = float(np.mean(expected_per_bin))
    bin_deviation = bin_counts - expected_per_bin
    prefix_deviation = np.insert(np.cumsum(bin_deviation), 0, 0.0)
    prefix_deviation += error[0] - prefix_deviation[0]

    p95_abs_error = float(np.percentile(np.abs(error), 95))
    max_abs_error = float(np.max(np.abs(error)))
    p95_abs_bin_deviation = float(np.percentile(np.abs(bin_deviation), 95))
    max_abs_bin_deviation = float(np.max(np.abs(bin_deviation)))
    amplification = (
        p95_abs_error / p95_abs_bin_deviation
        if p95_abs_bin_deviation
        else float("inf")
    )
    count_span = max(max_abs_error, max_abs_bin_deviation, 1.0) * 1.05

    diagnostic = WindowDiagnostic(
        dataset=dataset_name,
        source_n_keys=len(keys),
        window_start=start,
        window_end=end,
        window_size=window_n,
        window_quantile=((start + end) / 2.0) / len(keys),
        min_key=min_key,
        max_key=max_key,
        key_span=span,
        bins=bins,
        slope=slope,
        intercept=intercept,
        expected_keys_per_bin=float(expected),
        max_frequency=float(np.max(bin_counts)),
        p95_abs_error=p95_abs_error,
        max_abs_error=max_abs_error,
        p95_abs_bin_deviation=p95_abs_bin_deviation,
        max_abs_bin_deviation=max_abs_bin_deviation,
        p95_error_to_bin_deviation=amplification,
        count_span_to_expected=(
            count_span / abs(expected) if expected else float("inf")
        ),
    )
    arrays: dict[str, np.ndarray | float] = {
        "x": x,
        "error": error,
        "bin_counts": bin_counts,
        "expected_per_bin": expected_per_bin,
        "prefix_deviation": prefix_deviation,
        "count_span": count_span,
    }
    return diagnostic, arrays


def plot_diagnostic(
    diagnostic: WindowDiagnostic,
    arrays: dict[str, np.ndarray | float],
    out_dir: Path,
    save_pdf: bool,
) -> Path:
    x = np.asarray(arrays["x"])
    error = np.asarray(arrays["error"])
    bin_counts = np.asarray(arrays["bin_counts"])
    expected_per_bin = np.asarray(arrays["expected_per_bin"])
    count_span = float(arrays["count_span"])
    is_full_range = (
        diagnostic.window_start == 0
        and diagnostic.window_end == diagnostic.source_n_keys
    )

    common_ticks = np.linspace(-count_span, count_span, 5)
    figure, error_axis = plt.subplots(
        figsize=(PAPER_COLUMN_WIDTH_IN, PAPER_FIGURE_HEIGHT_IN), dpi=300
    )
    domain_label = (
        "Normalized Key Domain" if is_full_range else "Local Normalized Key Domain"
    )
    error_axis.set_xlabel(
        f"{dataset_display_name(diagnostic.dataset)}: {domain_label}",
        fontsize=PAPER_X_LABEL_SIZE_PT,
    )
    error_axis.tick_params(axis="both", labelsize=PAPER_TICK_SIZE_PT)
    error_axis.grid(True, alpha=0.3)

    widths = np.diff(x)
    deviations = bin_counts - expected_per_bin
    positive = deviations >= 0
    negative = ~positive
    positive_bars = error_axis.bar(
        x[:-1][positive],
        deviations[positive],
        width=widths[positive],
        color="#2ca02c",
        alpha=0.68,
        label="More Keys than Predicted",
        align="edge",
    )
    negative_bars = error_axis.bar(
        x[:-1][negative],
        deviations[negative],
        width=widths[negative],
        color="#d62728",
        alpha=0.68,
        label="Fewer Keys than Predicted",
        align="edge",
    )

    (error_line,) = error_axis.plot(
        x,
        error,
        linewidth=1.0,
        color="blue",
        label="Linear Regression Prediction Error",
    )
    error_axis.set_ylim(-count_span, count_span)
    error_axis.set_yticks(common_ticks)
    error_axis.yaxis.set_major_formatter(FuncFormatter(format_key_count_tick))
    legend_options = {
        "fontsize": PAPER_LEGEND_SIZE_PT,
        "framealpha": 0.8,
        "borderpad": 0.3,
        "labelspacing": 0.3,
        "handlelength": 1.5,
        "handletextpad": 0.4,
    }
    error_axis.legend(
        [error_line, positive_bars, negative_bars],
        [
            error_line.get_label(),
            positive_bars.get_label(),
            negative_bars.get_label(),
        ],
        loc="upper right",
        **legend_options,
    )

    if is_full_range:
        filename = f"{diagnostic.dataset}_histogram_error_bins{diagnostic.bins}"
    else:
        filename = f"{diagnostic.dataset}_local_noise_bins{diagnostic.bins}"
    left_margin = 0.11
    figure.subplots_adjust(left=left_margin, right=0.98, bottom=0.23, top=0.96)

    # Keep the axes compact, but expand the left margin when a wide formatted
    # tick (for example, -248.1K) would otherwise be clipped by the canvas.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    visible_tick_labels = [
        label for label in error_axis.get_yticklabels() if label.get_visible()
    ]
    if visible_tick_labels:
        min_tick_x = min(
            label.get_window_extent(renderer=renderer).x0
            for label in visible_tick_labels
        )
        minimum_canvas_padding_px = 4.0
        if min_tick_x < minimum_canvas_padding_px:
            left_margin += (
                minimum_canvas_padding_px - min_tick_x
            ) / figure.bbox.width
            figure.subplots_adjust(left=left_margin)

    png_path = out_dir / f"{filename}.png"
    figure.canvas.draw()
    figure.savefig(png_path, dpi=300)
    if save_pdf:
        figure.savefig(out_dir / f"{filename}.pdf")
    plt.close(figure)
    return png_path


def diagnostic_row(
    diagnostic: WindowDiagnostic,
    corr_error_prefix_deviation: float,
    figure_png: str | None = None,
) -> dict[str, str | int | float]:
    is_full_range = (
        diagnostic.window_start == 0
        and diagnostic.window_end == diagnostic.source_n_keys
    )
    row: dict[str, str | int | float] = {
        "analysis": "full-range" if is_full_range else "rank-range",
        "dataset": diagnostic.dataset,
        "source_n_keys": diagnostic.source_n_keys,
        "analysis_n_keys": diagnostic.window_size,
        "rank_range": f"{diagnostic.window_start}:{diagnostic.window_end}",
        "window_quantile": "full" if is_full_range else diagnostic.window_quantile,
        "min_key": diagnostic.min_key,
        "max_key": diagnostic.max_key,
        "key_span": diagnostic.key_span,
        "bins": diagnostic.bins,
        "slope": diagnostic.slope,
        "intercept": diagnostic.intercept,
        "expected_keys_per_bin": diagnostic.expected_keys_per_bin,
        "corr_error_prefix_deviation": corr_error_prefix_deviation,
        "p95_abs_ols_error": diagnostic.p95_abs_error,
        "max_abs_ols_error": diagnostic.max_abs_error,
        "p95_abs_bin_deviation": diagnostic.p95_abs_bin_deviation,
        "max_abs_bin_deviation": diagnostic.max_abs_bin_deviation,
        "p95_error_to_bin_deviation": diagnostic.p95_error_to_bin_deviation,
        "max_keys_per_bin": diagnostic.max_frequency,
        "count_span_to_expected": diagnostic.count_span_to_expected,
    }
    if figure_png is not None:
        row["figure_png"] = figure_png
    return row


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_contact_sheet(
    figures: list[tuple[str, Path]],
    output_path: Path,
) -> None:
    columns = 2
    rows = math.ceil(len(figures) / columns)
    sheet, axes = plt.subplots(rows, columns, figsize=(14.8, 4.15 * rows), dpi=100)
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes_array.flat:
        axis.axis("off")
    for axis, (_, figure_path) in zip(axes_array.flat, figures):
        axis.imshow(plt.imread(figure_path))
    sheet.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.savefig(output_path, dpi=100, facecolor="white")
    plt.close(sheet)


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir) if args.figures_dir else out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    start, end = args.analysis

    summary_rows: list[dict[str, str | int | float]] = []
    window_rows: list[dict[str, str | int | float]] = []
    generated_figures: list[tuple[str, Path]] = []

    for dataset_name in args.datasets:
        dataset = resolve_dataset(data_dir, dataset_name)
        keys = read_keys(dataset)
        name = dataset_stem(dataset)
        n = len(keys)
        if end > n:
            raise ValueError(
                f"rank range {start}:{end} exceeds dataset {name} with {n} keys"
            )

        print(f"[{dataset.name}] n={n:,}, rank_range={start}:{end}")
        diagnostic, arrays = evaluate_range(
            name, keys, start, end, args.bins, args.chunk_size
        )
        error = np.asarray(arrays["error"])
        prefix_deviation = np.asarray(arrays["prefix_deviation"])
        corr = (
            float(np.corrcoef(error, prefix_deviation)[0, 1])
            if len(error) > 1
            else float("nan")
        )
        png_path = plot_diagnostic(diagnostic, arrays, figures_dir, args.save_pdf)
        row = diagnostic_row(diagnostic, corr, str(png_path))
        summary_rows.append(row)
        window_rows.append({**row, "valid": 1, "reason": ""})
        generated_figures.append((name, png_path))

    summary_path = out_dir / f"summary_bins{args.bins}.csv"
    window_stats_path = out_dir / f"window_stats_bins{args.bins}.csv"
    write_csv(summary_path, summary_rows)
    write_csv(window_stats_path, window_rows)
    print(f"Saved summary: {summary_path}")
    print(f"Saved window stats: {window_stats_path}")
    print(f"Saved figures: {figures_dir}")

    if args.contact_sheet:
        contact_sheet = out_dir / args.contact_sheet
        create_contact_sheet(generated_figures, contact_sheet)
        print(f"Saved contact sheet: {contact_sheet}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
