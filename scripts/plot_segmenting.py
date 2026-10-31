#!/usr/bin/env python3
"""
Plot segmenting benchmark results.

Usage: python3 plot_segmenting.py <result.csv>

Generates:
1. Thread scalability plot (X=threads, Y=time, different lines for B/F)
2. B/F parameter comparison plots
3. Stage breakdown stacked bar chart

Excluded ERROR, TIMEOUT, and non-numeric timing rows are reported in the
console and on every generated figure. The input CSV is never modified.
"""

import sys
import os
from pathlib import Path

# Check for required dependencies
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f'ERROR: Missing required Python package: {e.name}')
    print('Install with: pip3 install pandas matplotlib numpy')
    sys.exit(1)

# Stage columns for breakdown
STAGE_COLS = [
    'build_maximal_ms',
    'segment_selection_ms',
    'segment_adjustment_ms',
    'generate_fgds_ms',
    'generate_finger_keys_ms',
    'compute_finger_positions_ms'
]

STAGE_LABELS = [
    'BuildMaximalCovering',
    'SegmentSelection',
    'SegmentAdjustment',
    'GenerateFGDs',
    'GenerateFingerKeys',
    'ComputeFingerPositions'
]

# Colors for stages
STAGE_COLORS = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']


def load_data(csv_path):
    """Load timings and account for every excluded input row."""
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    error_rows = int((df['total_ms'] == 'ERROR').sum())
    timeout_rows = int((df['total_ms'] == 'TIMEOUT').sum())
    df = df[~df['total_ms'].isin(['ERROR', 'TIMEOUT'])].copy()

    # Convert to numeric
    for col in ['n', 'B', 'F', 'threads', 'total_ms'] + STAGE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_rows = len(df)
    df = df.dropna(subset=['total_ms'])
    invalid_rows = numeric_rows - len(df)
    coverage_note = (
        f'Input CSV: {len(df)}/{total_rows} usable rows; '
        f'excluded ERROR={error_rows}, TIMEOUT={timeout_rows}, invalid={invalid_rows}'
    )
    return df, coverage_note


def save_figure(fig, output_dir, prefix, suffix, coverage_note):
    fig.text(0.5, 0.01, coverage_note, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for fmt in ['png', 'pdf']:
        filepath = output_dir / f'{prefix}_{suffix}.{fmt}'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'  Saved: {filepath}')
    plt.close(fig)


def plot_thread_scalability(df, output_dir, prefix, coverage_note):
    """Plot thread scalability: X=threads, Y=time, lines for each B/F combination."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get unique values
    threads = sorted(df['threads'].unique())
    b_values = sorted(df['B'].unique())
    f_values = sorted(df['F'].unique())

    # Left plot: Fixed F, vary B
    ax = axes[0]
    f_fixed = f_values[len(f_values)//2]  # Middle F value
    df_f = df[df['F'] == f_fixed]

    for b in b_values:
        df_b = df_f[df_f['B'] == b].groupby('threads')['total_ms'].mean()
        if not df_b.empty:
            ax.plot(df_b.index, df_b.values, 'o-', label=f'B={b}', linewidth=2, markersize=6)

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Total Time (ms)')
    ax.set_title(f'Thread Scalability (F={f_fixed})')
    ax.set_xticks(threads)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Right plot: Fixed B, vary F
    ax = axes[1]
    b_fixed = b_values[len(b_values)//2]  # Middle B value
    df_b = df[df['B'] == b_fixed]

    for f in f_values:
        df_f = df_b[df_b['F'] == f].groupby('threads')['total_ms'].mean()
        if not df_f.empty:
            ax.plot(df_f.index, df_f.values, 's-', label=f'F={f}', linewidth=2, markersize=6)

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Total Time (ms)')
    ax.set_title(f'Thread Scalability (B={b_fixed})')
    ax.set_xticks(threads)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    save_figure(fig, output_dir, prefix, 'thread_scalability', coverage_note)


def plot_bf_comparison(df, output_dir, prefix, coverage_note):
    """Plot B/F parameter comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get unique values
    threads = sorted(df['threads'].unique())
    b_values = sorted(df['B'].unique())
    f_values = sorted(df['F'].unique())

    # Use max threads for comparison
    max_threads = max(threads)
    df_max = df[df['threads'] == max_threads]

    # Left plot: X=B, different lines for F
    ax = axes[0]
    for f in f_values:
        df_f = df_max[df_max['F'] == f].groupby('B')['total_ms'].mean()
        if not df_f.empty:
            ax.plot(df_f.index, df_f.values, 'o-', label=f'F={f}', linewidth=2, markersize=8)

    ax.set_xlabel('Block Size (B)')
    ax.set_ylabel('Total Time (ms)')
    ax.set_title(f'Effect of B (threads={max_threads})')
    ax.set_xticks(b_values)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right plot: X=F, different lines for B
    ax = axes[1]
    for b in b_values:
        df_b = df_max[df_max['B'] == b].groupby('F')['total_ms'].mean()
        if not df_b.empty:
            ax.plot(df_b.index, df_b.values, 's-', label=f'B={b}', linewidth=2, markersize=8)

    ax.set_xlabel('Finger Sparsity (F)')
    ax.set_ylabel('Total Time (ms)')
    ax.set_title(f'Effect of F (threads={max_threads})')
    ax.set_xticks(f_values)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, prefix, 'bf_comparison', coverage_note)


def plot_stage_breakdown(df, output_dir, prefix, coverage_note):
    """Plot stage breakdown as stacked bar chart."""
    # Get unique values
    threads = sorted(df['threads'].unique())
    b_values = sorted(df['B'].unique())
    f_values = sorted(df['F'].unique())

    # Use middle B and F values
    b_fixed = b_values[len(b_values)//2]
    f_fixed = f_values[len(f_values)//2]

    df_filtered = df[(df['B'] == b_fixed) & (df['F'] == f_fixed)]

    if df_filtered.empty:
        print('  WARNING: No data for stage breakdown plot')
        return

    # Group by threads
    grouped = df_filtered.groupby('threads')[STAGE_COLS].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(grouped))
    width = 0.6

    bottom = np.zeros(len(grouped))
    for i, (col, label) in enumerate(zip(STAGE_COLS, STAGE_LABELS)):
        values = grouped[col].values
        ax.bar(x, values, width, label=label, bottom=bottom, color=STAGE_COLORS[i])
        bottom += values

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Stage Breakdown (B={b_fixed}, F={f_fixed})')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in grouped.index])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    save_figure(fig, output_dir, prefix, 'stage_breakdown', coverage_note)


def plot_stage_percentage(df, output_dir, prefix, coverage_note):
    """Plot stage percentage breakdown."""
    # Get unique values
    threads = sorted(df['threads'].unique())
    b_values = sorted(df['B'].unique())
    f_values = sorted(df['F'].unique())

    # Use max threads, middle B and F
    max_threads = max(threads)
    b_fixed = b_values[len(b_values)//2]
    f_fixed = f_values[len(f_values)//2]

    df_filtered = df[(df['threads'] == max_threads) &
                     (df['B'] == b_fixed) & (df['F'] == f_fixed)]

    if df_filtered.empty:
        print('  WARNING: No data for stage percentage plot')
        return

    # Calculate average times for each stage
    stage_times = [df_filtered[col].mean() for col in STAGE_COLS]
    total = sum(stage_times)

    if total == 0:
        return

    percentages = [t / total * 100 for t in stage_times]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bar chart
    y = np.arange(len(STAGE_LABELS))
    bars = ax.barh(y, percentages, color=STAGE_COLORS)

    # Add percentage labels
    for bar, pct, time in zip(bars, percentages, stage_times):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% ({time:.2f}ms)', va='center', fontsize=9)

    ax.set_xlabel('Percentage of Total Time')
    ax.set_ylabel('Stage')
    ax.set_title(f'Stage Time Distribution (threads={max_threads}, B={b_fixed}, F={f_fixed})')
    ax.set_yticks(y)
    ax.set_yticklabels(STAGE_LABELS)
    ax.set_xlim(0, max(percentages) * 1.3)
    ax.grid(True, alpha=0.3, axis='x')

    save_figure(fig, output_dir, prefix, 'stage_percentage', coverage_note)


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 plot_segmenting.py <result.csv>')
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f'ERROR: File not found: {csv_path}')
        sys.exit(1)

    # Output directory
    output_dir = csv_path.parent.parent / 'plot'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefix from filename
    prefix = csv_path.stem  # e.g., segmenting_uniform_1000000_20260107_143000

    print(f'Loading data from: {csv_path}')
    df, coverage_note = load_data(csv_path)
    print(coverage_note)

    if df.empty:
        print('ERROR: No valid data found')
        sys.exit(1)

    print(f'Loaded {len(df)} valid data points')
    print(f'  n values: {sorted(df["n"].unique())}')
    print(f'  B values: {sorted(df["B"].unique())}')
    print(f'  F values: {sorted(df["F"].unique())}')
    print(f'  Thread counts: {sorted(df["threads"].unique())}')

    print(f'\nGenerating plots to: {output_dir}')

    # Generate all plots
    print('\n1. Thread scalability plots...')
    plot_thread_scalability(df, output_dir, prefix, coverage_note)

    print('\n2. B/F comparison plots...')
    plot_bf_comparison(df, output_dir, prefix, coverage_note)

    print('\n3. Stage breakdown plots...')
    plot_stage_breakdown(df, output_dir, prefix, coverage_note)

    print('\n4. Stage percentage plots...')
    plot_stage_percentage(df, output_dir, prefix, coverage_note)

    print('\nDone!')


if __name__ == '__main__':
    main()
