"""
Binary search for optimal z-score threshold using BM25 performance.

Starts with L=-2, mid=0, R=2 and narrows the search range each iteration
based on which region yields the best userbased NDCG@10/val score.

Tag files are written to a temporary search directory and cleaned up as
thresholds are discarded.  Only the winning tag file is copied to the
real tags/ directory at the end.

Usage:
    python getag/search_zscore_threshold.py \
        --dataset games \
        --base_tag native \
        --classified_csv data/classified/games_native.csv

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
parser = argparse.ArgumentParser(description='Binary search for optimal z-score threshold')
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
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Path helpers  (all tag files go into the search_dir during the search)
# ---------------------------------------------------------------------------

def fmt(t: float) -> str:
    """Format threshold as 'z{value}', matching gen_getag.py convention."""
    return f'z{t:g}'


def tag_name(threshold: float) -> str:
    return f'getag_{args.base_tag}_{fmt(threshold)}'


def search_tag_file(threshold: float) -> Path:
    """Tag file path inside the temporary search directory."""
    return Path(args.search_dir) / args.dataset / f'{tag_name(threshold)}.json'


def final_tag_file(threshold: float) -> Path:
    """Tag file path in the real output directory."""
    return Path(args.output_dir) / args.dataset / f'{tag_name(threshold)}.json'


DATASET_KS = {
    'food':  [1, 3, 5, 10, 20],
    'games': [1, 5, 10, 20],
    'yelp':  [1, 5, 10, 20],
}


def result_csv(threshold: float) -> Path:
    return (Path('results/bm25') / args.dataset /
            f'retrieval_results_v2_userbased_bm25_{tag_name(threshold)}.csv')


def result_csv_i(threshold: float) -> Path:
    return (Path('results/bm25') / args.dataset /
            f'retrieval_results_v2_itembased_bm25_{tag_name(threshold)}.csv')


def get_test_row(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    best = df.iloc[df['ndcg@10/100/val'].argmax()]
    row = {}
    for k in DATASET_KS[args.dataset]:
        row[f'hr@{k}']   = float(best.get(f'hr@{k}/100/test',   float('nan')))
        row[f'ndcg@{k}'] = float(best.get(f'ndcg@{k}/100/test', float('nan')))
    return row


def print_test(label: str, metrics: dict) -> None:
    if not metrics:
        print(f'  {label}: (no data)')
        return
    print(f'  {label}:')
    for k in DATASET_KS[args.dataset]:
        hr   = metrics.get(f'hr@{k}',   float('nan'))
        ndcg = metrics.get(f'ndcg@{k}', float('nan'))
        print(f'    @{k:>2}  HR={hr:.4f}  NDCG={ndcg:.4f}')


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
        '--output_dir', args.search_dir,   # write into search_dir
        '--zscore_threshold', str(threshold),
    ]
    print(f'    Generating tags  ({fmt(threshold)})...')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'    ERROR in gen_getag.py:\n{proc.stderr[-800:]}')
        return False
    return True


def run_bm25(threshold: float) -> bool:
    csv = result_csv(threshold)
    if csv.exists():
        print(f'    [cached] {csv}')
        return True
    # BM25 retrieval.py reads the corpus from tags/<dataset>/<tag_name>.json,
    # so we temporarily symlink (or copy) from search_dir into output_dir.
    src = search_tag_file(threshold)
    dst = final_tag_file(threshold)
    dst.parent.mkdir(parents=True, exist_ok=True)
    copied = False
    if not dst.exists():
        shutil.copy2(src, dst)
        copied = True

    cmd = [
        sys.executable, 'downstream/bm25/retrieval.py',
        '--dataset', args.dataset,
        '--tag_name', tag_name(threshold),
    ]
    print(f'    Running BM25     ({fmt(threshold)})...')
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Remove the temporary copy from output_dir (unless it was already there)
    if copied and dst.exists():
        dst.unlink()

    if proc.returncode != 0:
        print(f'    ERROR in retrieval.py:\n{proc.stderr[-800:]}')
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
    if not run_bm25(threshold):
        return None
    return get_score(threshold)


def ensure_scores(thresholds: list[float], scores: dict) -> None:
    """Evaluate any thresholds not yet in the scores cache."""
    for t in thresholds:
        if t not in scores:
            print(f'\n  Evaluating  {fmt(t)}')
            scores[t] = evaluate(t)
            s = scores[t]
            print(f'  → NDCG@10/val = {f"{s:.4f}" if s is not None else "FAILED"}')


def discard(threshold: float, active: set[float]) -> None:
    """Delete search-dir tag file and BM25 result CSV for a discarded threshold."""
    if threshold in active:
        return  # still in use

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


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

print('=' * 70)
print('BINARY SEARCH FOR OPTIMAL Z-SCORE THRESHOLD')
print('=' * 70)
print(f'  Dataset:        {args.dataset}')
print(f'  Base tag:       {args.base_tag}')
print(f'  Classified CSV: {args.classified_csv}')
print(f'  Search dir:     {args.search_dir}/')
print(f'  Output dir:     {args.output_dir}/')
print(f'  Tolerance:      {args.tolerance}')
print(f'  Max iterations: {args.max_iter}')
print('=' * 70)

# Create search directory
Path(args.search_dir, args.dataset).mkdir(parents=True, exist_ok=True)

L, R = -2.0, 2.0
mid = (L + R) / 2   # = 0.0

scores: dict[float, float | None] = {}

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

    # Convergence check before updating bounds
    if R - L <= args.tolerance:
        print(f'\n  ✓ Converged: range {R - L:.2f} ≤ tolerance {args.tolerance}')
        pbar.close()
        break

    # Decide new bounds by comparing the two endpoints (L vs R).
    # Keep the half whose endpoint scored higher; mid re-centers on the new range.
    old_L, old_mid, old_R = L, mid, R

    if sL >= sR:
        # Left endpoint is better → keep left half [L, mid]
        R   = old_mid
        mid = (L + R) / 2
        dropped = {old_R}
        print(f'\n  Decision: L ({fmt(old_L)}={sL:.4f}) > R ({fmt(old_R)}={sR:.4f}) → '
              f'keep [{fmt(L)}, {fmt(R)}],  new mid = {fmt(mid)}')
    else:
        # Right endpoint is better → keep right half [mid, R]
        L   = old_mid
        mid = (L + R) / 2
        dropped = {old_L}
        print(f'\n  Decision: R ({fmt(old_R)}={sR:.4f}) > L ({fmt(old_L)}={sL:.4f}) → '
              f'keep [{fmt(L)}, {fmt(R)}],  new mid = {fmt(mid)}')

    # Clean up files for discarded thresholds
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

# Copy winner into the real output directory
src = search_tag_file(best_threshold)
dst = final_tag_file(best_threshold)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print(f'\n  ✓ Copied winner to: {dst}')

# Remove remaining search-dir files (all except the winner, which we already copied)
active_after = {L, mid, R}
for t in list(scores):
    if t != best_threshold:
        discard(t, set())  # force-discard everything else
# Clean up the empty search dataset subdirectory if possible
search_dataset_dir = Path(args.search_dir) / args.dataset
try:
    search_dataset_dir.rmdir()   # only succeeds if empty
    print(f'  ✓ Removed empty search dir: {search_dataset_dir}')
except OSError:
    pass  # not empty — leave it

print()
print(f'  Optimal threshold : {fmt(best_threshold)}  ({best_threshold:g})')
print(f'  Best NDCG@10/val  : {best_score:.4f}')
print(f'  Final tag file    : {dst}')
print('=' * 70)
print(f'\nTo regenerate the winning tags explicitly:')
print(f'  python getag/gen_getag.py \\')
print(f'    --dataset {args.dataset} \\')
print(f'    --base_tag {args.base_tag} \\')
print(f'    --classified_csv {args.classified_csv} \\')
print(f'    --zscore_threshold {best_threshold:g}')

# Write result file if requested
if args.result_file:
    result = {
        'dataset': args.dataset,
        'base_tag': args.base_tag,
        'best_threshold': best_threshold,
        'tag_name': tag_name(best_threshold),
        'score': best_score,
        'all_scores': {fmt(t): s for t, s in sorted(valid_scores.items())},
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
