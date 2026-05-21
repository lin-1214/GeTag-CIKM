"""
Binary search for optimal z-score threshold using UniSRec validation performance.

Starts with L=-2, mid=0, R=2 and narrows the search range each iteration
based on which region yields the best UniSRec best_valid_score (NDCG@10).

Tag files are written to a temporary search directory and cleaned up as
thresholds are discarded.  Preprocessed UniSRec data for discarded thresholds
is also deleted to save disk space.  Only the winning tag file is copied to
the real tags/ directory at the end.

Usage:
    python getag/search_zscore_threshold_unisrec.py \\
        --dataset games \\
        --base_tag native \\
        --classified_csv data/classified/games/native.csv \\
        --checkpoint checkpoints/UniSRec-FHCKM-300.pth \\
        --device cuda

Run from GeTag root directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from math import inf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Binary search for optimal z-score threshold using UniSRec'
)
parser.add_argument('--dataset', type=str, required=True, choices=['food', 'games', 'yelp'])
parser.add_argument('--base_tag', type=str, required=True, choices=['native', 'basetag', 'betags', 'betags_top20'])
parser.add_argument('--classified_csv', type=str, required=True,
                    help='Path to classified CSV (same as gen_getag.py)')
parser.add_argument('--checkpoint', type=str, default='checkpoints/UniSRec-FHCKM-300.pth',
                    help='Path to pre-trained UniSRec checkpoint')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device for UniSRec fine-tuning (cuda or cpu)')
parser.add_argument('--output_dir', type=str, default='tags',
                    help='Final output directory for the winning tag file (default: tags/)')
parser.add_argument('--search_dir', type=str, default='tags/_search',
                    help='Temporary directory for tag files generated during search '
                         '(default: tags/_search/)')
parser.add_argument('--data_path', type=str, default='data/preprocessed/UniSRec',
                    help='Base path for UniSRec preprocessed data '
                         '(default: data/preprocessed/UniSRec)')
parser.add_argument('--result_search_dir', type=str, default='results/unisrec_search',
                    help='Directory for intermediate validation result JSONs '
                         '(default: results/unisrec_search/)')
parser.add_argument('--tolerance', type=float, default=0.1,
                    help='Stop when search range < tolerance (default: 0.1)')
parser.add_argument('--max_iter', type=int, default=8,
                    help='Maximum number of search iterations (default: 8)')
parser.add_argument('--result_file', type=str, default=None,
                    help='Optional path to write search result as JSON '
                         '(best_threshold, tag_name, score)')
args = parser.parse_args()

# PLM model: food uses Chinese BERT, others use English BERT
PLM_NAME = 'hfl/chinese-bert-wwm-ext' if args.dataset == 'food' else 'bert-base-uncased'

# ---------------------------------------------------------------------------
# Path helpers  (tag files go into search_dir during the search)
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


def preprocess_dir(threshold: float) -> Path:
    """User-based preprocessed data directory."""
    return Path(args.data_path) / f'{args.dataset}_{tag_name(threshold)}'


def preprocess_dir_i(threshold: float) -> Path:
    """Item-based preprocessed data directory."""
    return Path(args.data_path) / f'{args.dataset}_{tag_name(threshold)}_i'


def result_json(threshold: float) -> Path:
    """Validation result JSON written by finetune.py."""
    return (Path(args.result_search_dir) / args.dataset /
            f'{tag_name(threshold)}_valid.json')


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


def run_preprocess(threshold: float) -> bool:
    """Run dataset-specific preprocessing script (generates .inter + .feat1CLS)."""
    pd_dir = preprocess_dir(threshold)
    dataset_name = f'{args.dataset}_{tag_name(threshold)}'
    train_inter = pd_dir / f'{dataset_name}.train.inter'
    if train_inter.exists():
        print(f'    [cached] preprocess {dataset_name}')
        return True

    # Preprocess script reads the tag file from tags/{dataset}/,
    # so temporarily copy from search_dir.
    src = search_tag_file(threshold)
    dst = final_tag_file(threshold)
    dst.parent.mkdir(parents=True, exist_ok=True)
    copied = False
    if not dst.exists():
        shutil.copy2(src, dst)
        copied = True

    preprocess_script = f'scripts/preprocess_{args.dataset}_for_unisrec.py'
    cmd = [
        sys.executable, preprocess_script,
        '--dataset_name', dataset_name,
        '--tags_file', str(dst),
        '--plm_name', PLM_NAME,
    ]
    print(f'    Preprocessing {dataset_name}...')
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if copied and dst.exists():
        dst.unlink()

    if proc.returncode != 0:
        print(f'    ERROR in preprocessing:\n{proc.stderr[-800:]}')
        return False
    return True


def run_convert(threshold: float) -> bool:
    """Convert user-based dataset to item-based format."""
    dataset_name = f'{args.dataset}_{tag_name(threshold)}'
    pd_dir_i = preprocess_dir_i(threshold)
    train_inter_i = pd_dir_i / f'{dataset_name}_i.train.inter'
    if train_inter_i.exists():
        print(f'    [cached] item-based {dataset_name}_i')
        return True

    cmd = [
        sys.executable, 'downstream/UniSRec/convert_to_item_based.py',
        '--dataset', str(preprocess_dir(threshold)),
    ]
    print(f'    Converting to item-based...')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'    ERROR in convert_to_item_based:\n{proc.stderr[-800:]}')
        return False
    return True


def run_finetune(threshold: float) -> bool:
    """Fine-tune UniSRec (user-based) and write validation score to result_json."""
    rj = result_json(threshold)
    if rj.exists():
        print(f'    [cached] finetune result {rj}')
        return True

    rj.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = f'{args.dataset}_{tag_name(threshold)}'
    cmd = [
        sys.executable, 'downstream/UniSRec/finetune.py',
        '--dataset', dataset_name,
        '--checkpoint', args.checkpoint,
        '--device', args.device,
        '--result_file', str(rj),
    ]
    print(f'    Fine-tuning UniSRec ({dataset_name})...')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'    ERROR in finetune.py:\n{proc.stderr[-800:]}')
        return False
    return True


def get_score(threshold: float) -> float | None:
    rj = result_json(threshold)
    if not rj.exists():
        return None
    with open(rj) as f:
        d = json.load(f)
    score = d.get('best_valid_score')
    return float(score) if score is not None else None


def evaluate(threshold: float) -> float | None:
    if not generate_tags(threshold):
        return None
    if not run_preprocess(threshold):
        return None
    if not run_convert(threshold):
        return None
    if not run_finetune(threshold):
        return None
    return get_score(threshold)


def ensure_scores(thresholds: list, scores: dict) -> None:
    """Evaluate any thresholds not yet in the scores cache."""
    for t in thresholds:
        if t not in scores:
            print(f'\n  Evaluating  {fmt(t)}')
            scores[t] = evaluate(t)
            s = scores[t]
            print(f'  → best_valid_score = {f"{s:.4f}" if s is not None else "FAILED"}')


def discard(threshold: float, active: set) -> None:
    """Delete search-dir tag file, result JSON, and preprocessed data for a discarded threshold."""
    if threshold in active:
        return

    tf = search_tag_file(threshold)
    if tf.exists():
        tf.unlink()
        print(f'  [cleanup] Removed {tf}')

    rj = result_json(threshold)
    if rj.exists():
        rj.unlink()
        print(f'  [cleanup] Removed {rj}')

    # Remove preprocessed data directories to save disk space
    for d in [preprocess_dir(threshold), preprocess_dir_i(threshold)]:
        if d.exists():
            shutil.rmtree(d)
            print(f'  [cleanup] Removed {d}')


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

print('=' * 70)
print('BINARY SEARCH FOR OPTIMAL Z-SCORE THRESHOLD  (UniSRec)')
print('=' * 70)
print(f'  Dataset:        {args.dataset}')
print(f'  Base tag:       {args.base_tag}')
print(f'  Classified CSV: {args.classified_csv}')
print(f'  Checkpoint:     {args.checkpoint}')
print(f'  Device:         {args.device}')
print(f'  PLM:            {PLM_NAME}')
print(f'  Search dir:     {args.search_dir}/')
print(f'  Output dir:     {args.output_dir}/')
print(f'  Tolerance:      {args.tolerance}')
print(f'  Max iterations: {args.max_iter}')
print('=' * 70)

# Create search directory
Path(args.search_dir, args.dataset).mkdir(parents=True, exist_ok=True)

L, R = -2.0, 2.0
mid = (L + R) / 2   # = 0.0

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
        print(f'  {fmt(t):>8}   best_valid_score = {s:.4f}{flag}')

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
        print(f'\n  Decision: L ({fmt(old_L)}={sL:.4f}) ≥ R ({fmt(old_R)}={sR:.4f}) → '
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
print(f'  {"Threshold":>10}   {"best_valid_score":>16}')
print(f'  {"─" * 10}   {"─" * 16}')
for t in sorted(valid_scores):
    flag = '  ← BEST' if t == best_threshold else ''
    print(f'  {fmt(t):>10}   {valid_scores[t]:>16.4f}{flag}')

# Copy winner into the real output directory
src = search_tag_file(best_threshold)
dst = final_tag_file(best_threshold)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print(f'\n  ✓ Copied winner to: {dst}')

# Remove remaining search-dir files (force-discard everything except the winner)
for t in list(scores):
    if t != best_threshold:
        discard(t, set())
# Clean up empty search dataset subdirectory
search_dataset_dir = Path(args.search_dir) / args.dataset
try:
    search_dataset_dir.rmdir()
    print(f'  ✓ Removed empty search dir: {search_dataset_dir}')
except OSError:
    pass

print()
print(f'  Optimal threshold   : {fmt(best_threshold)}  ({best_threshold:g})')
print(f'  Best valid score    : {best_score:.4f}')
print(f'  Final tag file      : {dst}')
print('=' * 70)
print(f'\nTo regenerate the winning tags explicitly:')
print(f'  python getag/gen_getag.py \\')
print(f'    --dataset {args.dataset} \\')
print(f'    --base_tag {args.base_tag} \\')
print(f'    --classified_csv {args.classified_csv} \\')
print(f'    --zscore_threshold {best_threshold:g}')

# Write result file if requested
if args.result_file:
    # Read test_result written by finetune.py for the winning threshold
    test_result = {}
    rj = result_json(best_threshold)
    if rj.exists():
        with open(rj) as f:
            d = json.load(f)
        test_result = d.get('test_result', {})

    result = {
        'dataset': args.dataset,
        'base_tag': args.base_tag,
        'best_threshold': best_threshold,
        'tag_name': tag_name(best_threshold),
        'score': best_score,
        'test': test_result,
    }
    Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  ✓ Result written to: {args.result_file}')

    if result['test']:
        print('\n── Test Performance ──')
        for metric, value in result['test'].items():
            print(f'  {metric}: {value:.4f}')
