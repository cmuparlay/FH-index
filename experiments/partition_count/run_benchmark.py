#!/usr/bin/env python3
"""Build, run, and plot the paper's partition-count experiment."""

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BUILD_DIR = SCRIPT_DIR / 'build'
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = Path('datasets')

DATASETS = [
    'uniformrandom_200M_uint64.bin',
    'books_200M_uint64.bin',
    'covid_200M_uint64.bin',
    'fb_200M_uint64.bin',
    'genome_200M_uint64.bin',
    'history_200M_uint64.bin',
    'libio_200M_uint64.bin',
    'osm_200M_uint64.bin',
    'planet_200M_uint64.bin',
    'stack_200M_uint64.bin',
    'wise_200M_uint64.bin',
]


def run(command, cwd=None, dry_run=False):
    print('$', ' '.join(str(value) for value in command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def build(args):
    if args.clean_build and args.build_dir.exists():
        unsafe = {Path('/'), Path.home(), SCRIPT_DIR, SCRIPT_DIR.parent}
        if args.build_dir in unsafe:
            raise ValueError(f'refusing to remove unsafe path: {args.build_dir}')
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    run([
        'cmake', '-S', str(SCRIPT_DIR), '-B', str(args.build_dir),
        '-DCMAKE_BUILD_TYPE=Release',
        f'-DCMAKE_CXX_COMPILER={args.compiler}',
        f'-DFH_INDEX_SOURCE_DIR={args.fh_index_source_dir}',
        f'-DPGM_INDEX_DIR={args.pgm_index_dir}',
    ], dry_run=args.dry_run)
    run([
        'cmake', '--build', str(args.build_dir), '--parallel',
        str(args.build_jobs), '--target', 'bench_partition_count',
    ], dry_run=args.dry_run)


def completed_datasets(csv_path):
    if not csv_path.is_file():
        return set()
    with csv_path.open(newline='') as source:
        return {Path(row['dataset']).name for row in csv.DictReader(source)}


def benchmark(args):
    build(args)
    results_dir = args.out_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'result.csv'
    if csv_path.exists() and not args.resume:
        raise FileExistsError(
            f'{csv_path} already exists; use --resume or a new --out-dir')
    completed = completed_datasets(csv_path) if args.resume else set()
    executable = args.build_dir / 'bench_partition_count'
    for dataset in args.datasets:
        if Path(dataset).name in completed:
            print(f'SKIP (cached): {dataset}')
            continue
        path = Path(dataset)
        if dataset != 'GEN' and not path.is_absolute():
            path = args.data_dir / path
        if dataset != 'GEN' and not args.dry_run and not path.is_file():
            raise FileNotFoundError(path)
        run([
            str(executable), str(path), str(args.load_limit),
            str(args.epsilon), str(args.bucket_bound),
            str(args.finger_sparsity),
        ], cwd=args.out_dir, dry_run=args.dry_run)
    print(f'Results: {csv_path}')
    return csv_path


def plot(csv_path, out_dir):
    from plot_partition_count import load_data, plot_comparison
    plot_comparison(load_data(csv_path), out_dir / 'results')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=('all', 'bench', 'plot'),
                        default='all')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument('--compiler', default=os.environ.get('CXX', 'g++'))
    parser.add_argument('--build-jobs', type=int, default=min(os.cpu_count() or 4, 4))
    parser.add_argument(
        '--fh-index-source-dir', type=Path,
        default=REPO_ROOT)
    parser.add_argument(
        '--pgm-index-dir', type=Path,
        default=REPO_ROOT / 'third_party' / 'PGM-index')
    parser.add_argument('--load-limit', type=int, default=200_000_000)
    parser.add_argument('--epsilon', type=int, default=64)
    parser.add_argument('--bucket-bound', type=int, default=128)
    parser.add_argument('--finger-sparsity', type=int, default=4)
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--clean-build', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.build_dir = args.build_dir.resolve()
    args.fh_index_source_dir = args.fh_index_source_dir.resolve()
    args.pgm_index_dir = args.pgm_index_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / 'results' / 'result.csv'
    if args.mode in ('all', 'bench'):
        csv_path = benchmark(args)
    if args.mode in ('all', 'plot') and not args.dry_run:
        plot(csv_path, args.out_dir)


if __name__ == '__main__':
    main()
