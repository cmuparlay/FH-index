#!/usr/bin/env python3
"""
Plot partition count comparison between PGM (CDF Approximation) and FH-index (Finger-Hinted Modeling)

Usage:
    python plot_partition_count.py [results/result.csv]

CSV format:
    dataset,n,epsilon,B,F,pgm_partitions,fh_partitions
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, 'results', 'result.csv')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

def load_data(filepath):
    """Load and validate benchmark rows from CSV."""
    required_cols = ['dataset', 'n', 'pgm_partitions', 'fh_partitions']
    with open(filepath, newline='') as source:
        reader = csv.DictReader(source)
        missing = [column for column in required_cols
                   if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        rows = list(reader)
    for row in rows:
        for column in ('n', 'epsilon', 'B', 'F',
                       'pgm_partitions', 'fh_partitions'):
            if row.get(column, ''):
                row[column] = int(row[column])
    return rows

DATASET_ORDER = [
    'uniformrandom', 'stack', 'libio', 'wise', 'history',
    'covid', 'books', 'planet', 'fb', 'genome', 'osm',
]

DATASET_LABELS = {
    'uniformrandom': 'uniform\nrandom',
}

def dataset_key(name):
    """Extract short dataset key from full path/filename"""
    name = os.path.basename(name)
    # Remove size/type suffixes like _200M_uint64.bin
    import re
    name = re.sub(r'_\d+[MKB]_uint\d+\.bin$', '', name)
    return name.lower()

def format_dataset_name(name):
    """Format dataset name for display"""
    key = dataset_key(name)
    return DATASET_LABELS.get(key, key)

def write_summary(rows, output_dir):
    """Write the portable, plot-ordered summary used by the artifact."""
    path = os.path.join(output_dir, 'summary.csv')
    fieldnames = [
        'dataset', 'n', 'epsilon', 'B', 'F',
        'pgm_partitions', 'fh_partitions', 'pgm_to_fh_ratio',
    ]
    with open(path, 'w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'dataset': dataset_key(row['dataset']),
                'n': row['n'],
                'epsilon': row.get('epsilon', ''),
                'B': row.get('B', ''),
                'F': row.get('F', ''),
                'pgm_partitions': row['pgm_partitions'],
                'fh_partitions': row['fh_partitions'],
                'pgm_to_fh_ratio':
                    f"{row['pgm_partitions'] / row['fh_partitions']:.5f}",
            })
    print(f"Saved: {path}")

def plot_comparison(rows, output_dir=RESULTS_DIR):
    """Generate comparison plot"""
    order_map = {k: i for i, k in enumerate(DATASET_ORDER)}
    rows = sorted(rows, key=lambda row: order_map.get(
        dataset_key(row['dataset']), len(DATASET_ORDER)))

    # Format dataset names
    datasets = [format_dataset_name(row['dataset']) for row in rows]
    pgm_counts = [row['pgm_partitions'] for row in rows]
    fh_counts = [row['fh_partitions'] for row in rows]

    # Create figure (single-column width for double-column paper: 3.5in)
    fig, ax = plt.subplots(figsize=(3.5, 1.3), dpi=150)

    x = np.arange(len(datasets))
    width = 0.35

    # Bars
    bars1 = ax.bar(x - width/2, pgm_counts, width,
                   label='Position-Based', color='#1f77b4', zorder=3)
    bars2 = ax.bar(x + width/2, fh_counts, width,
                   label='Finger-Hinted', color='#ff7f0e', zorder=3)

    # Log scale for y-axis
    ax.set_yscale('log')

    # Dynamic y-axis limits
    max_val = max(max(pgm_counts), max(fh_counts))
    ax.set_ylim(0.5, max_val * 2)

    # Labels
    ax.set_ylabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=5.5)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(loc='upper left', fontsize=6, ncol=2)

    # Add value labels for small values (1-10)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height <= 10:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=6, fontweight='bold')

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save to results directory
    os.makedirs(output_dir, exist_ok=True)
    write_summary(rows, output_dir)
    output_prefix = os.path.join(output_dir, 'partition_count_comparison')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_prefix}.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_prefix}.pdf and {output_prefix}.png")

    # Print summary
    print("\n=== Summary ===")
    for row in rows:
        dataset = format_dataset_name(row['dataset']).replace('\n', ' ')
        pgm = row['pgm_partitions']
        fh = row['fh_partitions']
        ratio = pgm / fh if fh > 0 else float('inf')
        print(f"{dataset}: n={row['n']}, epsilon={row.get('epsilon') or 'N/A'}, "
              f"B={row.get('B') or 'N/A'}, F={row.get('F') or 'N/A'}, "
              f"Position-based={pgm}, Finger-hinted={fh}, Ratio={ratio:.0f}x")

if __name__ == '__main__':
    # Determine CSV file path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = DEFAULT_CSV

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        print(f"Run benchmark first: ./build/bench_partition_count GEN 200000000")
        sys.exit(1)

    print(f"Loading data from: {csv_path}")
    rows = load_data(csv_path)
    print(f"Loaded {len(rows)} records")

    plot_comparison(rows)
