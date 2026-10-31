#!/usr/bin/env python3
"""Render the paper's partition-count figure from an explicit measured CSV."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["uniformrandom", "stack", "libio", "wise", "history", "covid",
         "books", "planet", "fb", "genome", "osm"]
LABELS = ["Uniform\nrandom", "Stack", "Libio", "WISE", "History", "COVID",
          "Books", "Planet", "FB", "Genome", "OSM"]


def load_rows(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    by_name = {}
    for raw in rows:
        name = Path(raw["dataset"]).name.removesuffix("_200M_uint64.bin")
        if name in by_name:
            raise ValueError(f"Duplicate dataset: {name}")
        row = {"dataset": name}
        row.update({field: int(raw[field]) for field in
                    ("n", "epsilon", "B", "F", "pgm_partitions", "fh_partitions")})
        if (row["epsilon"], row["B"], row["F"]) != (64, 128, 4):
            raise ValueError(f"Unexpected paper parameters for {name}")
        if min(row["n"], row["pgm_partitions"], row["fh_partitions"]) <= 0:
            raise ValueError(f"Nonpositive count for {name}")
        by_name[name] = row
    if set(by_name) != set(ORDER):
        raise ValueError("The figure requires Uniform and all ten real datasets")
    return [by_name[name] for name in ORDER]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.input)
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 2.2))
    ax.bar(x - width / 2, [r["pgm_partitions"] for r in rows], width,
           label="Position-based", color="#1f77b4", edgecolor="black",
           linewidth=0, hatch="///")
    ax.bar(x + width / 2, [r["fh_partitions"] for r in rows], width,
           label="Finger-hinted", color="#ff7f0e", edgecolor="black",
           linewidth=0, hatch="---")
    ax.set_yscale("log")
    ax.set_ylabel("Number of partitions")
    ax.set_xticks(x, LABELS)
    ax.tick_params(axis="x", labelsize=7.5, pad=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", ncol=2)
    for position, row in zip(x, rows):
        count = row["fh_partitions"]
        if count <= 3:
            ax.text(position + width / 2, count * 1.45, str(count),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps({
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "matplotlib": matplotlib.__version__, "numpy": np.__version__,
            "figure_size_inches": [6.8, 2.2], "rows": rows,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
