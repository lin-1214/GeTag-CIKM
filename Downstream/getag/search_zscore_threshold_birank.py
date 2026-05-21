"""
Binary search for optimal z-score threshold using BiRank performance.

Starts with L=-2, mid=0, R=2 and narrows the search range each iteration
based on which region yields the best userbased NDCG@10/val score.

Tag files are written to a temporary search directory and cleaned up as
thresholds are discarded.  Only the winning tag file is copied to the
real tags/ directory at the end.

Usage:
    python getag/search_zscore_threshold_birank.py \
        --dataset games \
        --base_tag native \
        --classified_csv data/classified/games/native.csv

Run from GeTag root directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import pandas as pd
from pathlib import Path
from math import inf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Binary search for optimal z-score threshold using BiRank')
parser.add_argument('--dataset', type=str, required=True, choices=['food', 'games', 'yelp'])
parser.add_argument('--base_tag', type=str, required=True, choices=['native', 'basetag', 'betags', 'betags_top20'])
parser.add_argument('--classified_csv', type=str, required=True,
                    help='Path to classified CSV (same as gen_getag.py)')
parser.add_argument('--output_dir', type=str, default='tags',
                    help='Final output directory for the winning tag file (default: tags/)')
parser.add_argument('--search_dir', type=str, default='tags/_search',
                    help='Temporary directory for tag files generated during search '
                         '(default: tags/_search/)')
parser.add_argument('--tolerance', type=float, default=0.1,
                    help='Stop when search range < tolerance (default: 0.1)')
parser.add_argument('--max_iter', type=int, default=8,
                    help='Maximum number of search iterations (default: 8)')
parser.add_argument('--result_file', type=str, default=None,
                    help='Optional path to write search result as JSON '
                         '(best_threshold, tag_name, score)')
parser.add_argument('--fast_grid', action='store_true',
                    help='Use a reduced hyperparameter grid during threshold search '
                         '(32 user-based configs instead of 162). '
                         'Recommended for large datasets like yelp.')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def fmt(t: float) -> str:
    return f'z{t:g}'


def tag_name(threshold: float) -> str:
    return f'getag_{args.base_tag}_{fmt(threshold)}'


def search_tag_file(threshold: float) -> Path:
    return Path(args.search_dir) / args.dataset / f'{tag_name(threshold)}.json'


def final_tag_file(threshold: float) -> Path:
    return Path(args.output_dir) / args.dataset / f'{tag_name(threshold)}.json'


DATASET_KS = [1, 5, 10, 20]


def result_csv(threshold: float) -> Path:
    return (Path('results/birank') / args.dataset /
            f'birank_results_userbased_{tag_name(threshold)}.csv')


def result_csv_i(threshold: float) -> Path:
    return (Path('results/birank') / args.dataset /
            f'birank_results_itembased_{tag_name(threshold)}.csv')


def get_test_row(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    best = df.iloc[df['ndcg@10/100/val'].argmax()]
    row = {}
    for k in DATASET_KS:
        row[f'hr@{k}']   = float(best.get(f'hr@{k}/100/test',   float('nan')))
        row[f'ndcg@{k}'] = float(best.get(f'ndcg@{k}/100/test', float('nan')))
    return row


# ---------------------------------------------------------------------------
# Generation / evaluation helpers
# ---------------------------------------------------------------------------

def generate_tags(threshold: float) -> bool:
    tf = search_tag_file(threshold)
    if tf.exists():
        print(f'    [cached] {tf}')
        return True
    cmd = [
        sys.executable, 'getag/gen_getag.py',
        '--dataset', args.dataset,
        '--base_tag', args.base_tag,
        '--classified_csv', args.classified_csv,
        '--output_dir', args.search_dir,
        '--zscore_threshold', str(threshold),
    ]
    print(f'    Generating tags  ({fmt(threshold)})...')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'    ERROR in gen_getag.py:\n{proc.stderr[-800:]}')
        return False
    return True


CACHE_BASE = Path('downstream/birank/cache/predictor_cache')
# Maps threshold → list of cache subdirs created by that BiRank run
birank_cache_dirs: dict = {}


def run_birank(threshold: float) -> bool:
    csv = result_csv(threshold)
    if csv.exists():
        print(f'    [cached] {csv}')
        return True
    # BiRank retrieval.py reads corpus from tags/<dataset>/<tag_name>.json,
    # so temporarily copy from search_dir into output_dir.
    src = search_tag_file(threshold)
    dst = final_tag_file(threshold)
    dst.parent.mkdir(parents=True, exist_ok=True)
    copied = False
    if not dst.exists():
        shutil.copy2(src, dst)
        copied = True

    # Clear all BiRank cache before each run to prevent corrupt files from a
    # previous failed run propagating to the next threshold evaluation.
    for cache_subdir in ['datasets_cache', 'predictor_cache', 'locks', 'corpus_logs']:
        p = Path('downstream/birank/cache') / cache_subdir
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    # Snapshot cache dirs before running so we can track what gets created
    before = set(CACHE_BASE.iterdir()) if CACHE_BASE.exists() else set()

    cmd = [
        sys.executable, 'downstream/birank/retrieval.py',
        '--dataset', args.dataset,
        '--tag_name', tag_name(threshold),
        '--verbose',
    ]
    if args.fast_grid:
        cmd.append('--fast_grid')
    log_dir = Path('logs/birank_debug') / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{tag_name(threshold)}.log'

    print(f'    Running BiRank   ({fmt(threshold)})...  log → {log_path}')
    with open(log_path, 'w') as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=log_f, text=True)

    if copied and dst.exists():
        dst.unlink()

    # Record any new cache subdirs created by this run
    after = set(CACHE_BASE.iterdir()) if CACHE_BASE.exists() else set()
    birank_cache_dirs[threshold] = list(after - before)

    if proc.returncode != 0:
        print(f'    ERROR in retrieval.py (exit={proc.returncode}) — full log: {log_path}')
        log_lines = log_path.read_text().splitlines()
        print('\n'.join(log_lines[-50:]))
        return False
    return True


def get_score(threshold: float) -> float | None:
    csv = result_csv(threshold)
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return float(df['ndcg@10/100/val'].max())


def evaluate(threshold: float) -> float | None:
    if not generate_tags(threshold):
        return None
    if not run_birank(threshold):
        return None
    return get_score(threshold)


def ensure_scores(thresholds: list, scores: dict) -> None:
    for t in thresholds:
        if t not in scores:
            print(f'\n  Evaluating  {fmt(t)}')
            scores[t] = evaluate(t)
            s = scores[t]
            print(f'  → NDCG@10/val = {f"{s:.4f}" if s is not None else "FAILED"}')


def discard(threshold: float, active: set) -> None:
    """Delete search-dir tag file, BiRank result CSVs, and BiRank cache for a discarded threshold."""
    if threshold in active:
        return

    tf = search_tag_file(threshold)
    if tf.exists():
        tf.unlink()
        print(f'  [cleanup] Removed {tf}')

    csv = result_csv(threshold)
    if csv.exists():
        csv.unlink()
        print(f'  [cleanup] Removed {csv}')

    csv_i = result_csv_i(threshold)
    if csv_i.exists():
        csv_i.unlink()
        print(f'  [cleanup] Removed {csv_i}')

    for cache_dir in birank_cache_dirs.pop(threshold, []):
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
            print(f'  [cleanup] Removed cache {cache_dir}')


def print_test(label: str, metrics: dict) -> None:
    if not metrics:
        print(f'  {label}: (no data)')
        return
    print(f'  {label}:')
    for k in DATASET_KS:
        hr   = metrics.get(f'hr@{k}',   float('nan'))
        ndcg = metrics.get(f'ndcg@{k}', float('nan'))
        print(f'    @{k:>2}  HR={hr:.4f}  NDCG={ndcg:.4f}')


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

print('=' * 70)
print('BINARY SEARCH FOR OPTIMAL Z-SCORE THRESHOLD  (BiRank)')
print('=' * 70)
print(f'  Dataset:        {args.dataset}')
print(f'  Base tag:       {args.base_tag}')
print(f'  Classified CSV: {args.classified_csv}')
print(f'  Search dir:     {args.search_dir}/')
print(f'  Output dir:     {args.output_dir}/')
print(f'  Tolerance:      {args.tolerance}')
print(f'  Max iterations: {args.max_iter}')
print(f'  Fast grid:      {args.fast_grid}  ({"32 user / 4 item configs" if args.fast_grid else "162 user / 16 item configs"})')
print('=' * 70)

Path(args.search_dir, args.dataset).mkdir(parents=True, exist_ok=True)

L, R = -2.0, 2.0
mid = (L + R) / 2

scores: dict = {}

pbar = tqdm(range(1, args.max_iter + 1), desc='Searching', unit='iter')
for iteration in pbar:
    pbar.set_postfix(L=fmt(L), mid=fmt(mid), R=fmt(R), range=f'{R-L:.2f}')
    print(f'\n{"─" * 70}')
    print(f'Iteration {iteration}  |  L={fmt(L)}  mid={fmt(mid)}  R={fmt(R)}  '
          f'(range={R - L:.2f})')
    print('─' * 70)

    ensure_scores([L, mid, R], scores)

    sL = scores[L] if scores.get(L) is not None else -inf
    sM = scores[mid] if scores.get(mid) is not None else -inf
    sR = scores[R] if scores.get(R) is not None else -inf

    best_s = max(sL, sM, sR)
    print()
    for t, s in [(L, sL), (mid, sM), (R, sR)]:
        flag = ' ← best' if s == best_s else ''
        print(f'  {fmt(t):>8}   NDCG@10/val = {s:.4f}{flag}')

    if R - L <= args.tolerance:
        print(f'\n  ✓ Converged: range {R - L:.2f} ≤ tolerance {args.tolerance}')
        pbar.close()
        break

    old_L, old_mid, old_R = L, mid, R

    if sL >= sR:
        R   = old_mid
        mid = (L + R) / 2
        dropped = {old_R}
        print(f'\n  Decision: L ({fmt(old_L)}={sL:.4f}) ≥ R ({fmt(old_R)}={sR:.4f}) → '
              f'keep [{fmt(L)}, {fmt(R)}],  new mid = {fmt(mid)}')
    else:
        L   = old_mid
        mid = (L + R) / 2
        dropped = {old_L}
        print(f'\n  Decision: R ({fmt(old_R)}={sR:.4f}) > L ({fmt(old_L)}={sL:.4f}) → '
              f'keep [{fmt(L)}, {fmt(R)}],  new mid = {fmt(mid)}')

    active = {L, mid, R}
    for t in dropped:
        discard(t, active)

else:
    pbar.close()
    print(f'\n  Reached max_iter = {args.max_iter}')

# ---------------------------------------------------------------------------
# Final report + copy winner to real output dir
# ---------------------------------------------------------------------------
valid_scores = {t: s for t, s in scores.items() if s is not None}
if not valid_scores:
    print('\nERROR: All threshold evaluations failed — no valid scores collected.')
    sys.exit(1)

best_threshold = max(valid_scores, key=lambda t: valid_scores[t])
best_score = valid_scores[best_threshold]

print('\n' + '=' * 70)
print('SEARCH COMPLETE — ALL EVALUATED THRESHOLDS')
print('=' * 70)
print(f'  {"Threshold":>10}   {"NDCG@10/val":>12}')
print(f'  {"─" * 10}   {"─" * 12}')
for t in sorted(valid_scores):
    flag = '  ← BEST' if t == best_threshold else ''
    print(f'  {fmt(t):>10}   {valid_scores[t]:>12.4f}{flag}')

src = search_tag_file(best_threshold)
dst = final_tag_file(best_threshold)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print(f'\n  ✓ Copied winner to: {dst}')

for t in list(scores):
    if t != best_threshold:
        discard(t, set())
search_dataset_dir = Path(args.search_dir) / args.dataset
try:
    search_dataset_dir.rmdir()
    print(f'  ✓ Removed empty search dir: {search_dataset_dir}')
except OSError:
    pass

print()
print(f'  Optimal threshold : {fmt(best_threshold)}  ({best_threshold:g})')
print(f'  Best NDCG@10/val  : {best_score:.4f}')
print(f'  Final tag file    : {dst}')
print('=' * 70)

if args.result_file:
    result = {
        'dataset': args.dataset,
        'base_tag': args.base_tag,
        'best_threshold': best_threshold,
        'tag_name': tag_name(best_threshold),
        'score': best_score,
        'test': {
            'userbased': get_test_row(result_csv(best_threshold)),
            'itembased': get_test_row(result_csv_i(best_threshold)),
        },
    }
    Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  ✓ Result written to: {args.result_file}')

    print('\n── Test Performance ──')
    print_test('User-based', result['test']['userbased'])
    print_test('Item-based', result['test']['itembased'])
