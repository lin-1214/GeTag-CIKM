"""
Evaluation Phase: Run Tag Generation and Retrieval Evaluation

This script runs both:
1. gen_getag.py - Generate GETags from classified session data
2. retrieval_v2.py - Run retrieval evaluation using generated tags

Now parameterized to evaluate a specific classified CSV and emit a summary JSON.
"""

import subprocess
import sys
import os
import shutil
import json
import argparse
from config import Config

CONFIG = Config()
DOMAIN = getattr(CONFIG, "DOMAIN", "food").lower()

# Set up the environment with PYTHONPATH
def setup_environment():
    """Set up PYTHONPATH to include tmp directory"""
    env = os.environ.copy()

    # Get the current directory (src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(current_dir, 'tmp')

    # Add tmp directory to PYTHONPATH
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{tmp_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = tmp_dir

    return env

def run_script(cmd, description, env):
    """Run a command list and handle errors"""
    print("=" * 80)
    print(f"Running: {description}")
    print("=" * 80)
    print("Command:", ' '.join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            env=env
        )
        print(f"\n✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {description}")
        print(f"Error code: {e.returncode}")
        return False

def clean_cache_once():
    """Clean cache directories to ensure fresh results"""
    print("=" * 80)
    print("CLEANING CACHE")
    print("=" * 80)

    # Clean experiment cache directory (remove completely)
    if os.path.exists('exps_v2'):
        try:
            shutil.rmtree('exps_v2')
            print(f"✓ Cleaned exps_v2/")
        except Exception as e:
            print(f"⚠ Warning: Could not clean exps_v2: {e}")
    else:
        print(f"  exps_v2/ does not exist (skipping)")

    # For dataset cache, just recreate the directory structure
    dataset_cache = os.path.join('dataset', 'movielens', 'ml-1m') if DOMAIN == 'movie' else 'dataset/i3fresh'
    # Dataset directories contain canonical cache files checked into the repo.
    # Removing them would break retrieval (missing smap.json/base_tags.json, etc.),
    # so we keep them intact for both food and movie domains.
    if os.path.exists(dataset_cache):
        print(f"Skipping dataset cache cleanup to preserve {dataset_cache}/")
    else:
        os.makedirs(dataset_cache, exist_ok=True)
        print(f"✓ Created {dataset_cache}/")

    print("=" * 80 + "\n")


def write_summary(tag_name: str, results_dir: str, eval_source: str):
    """Aggregate results CSVs and write a concise summary JSON for selection.

    eval_source: 'user' | 'item' | 'auto'
    Uses ndcg@10/100/test as the selection metric.
    """
    import pandas as pd
    metric_col = 'ndcg@10/100/test'
    summary = {
        'tag_name': tag_name,
        'metric': metric_col,
        'eval_source': eval_source,
        'results': {},
        'best': {}
    }
    user_path = os.path.join(results_dir, f'retrieval_results_v2_userbased_bm25_{tag_name}.csv')
    item_path = os.path.join(results_dir, f'retrieval_results_v2_itembased_bm25_{tag_name}.csv')

    def extract_best(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return None
            if metric_col not in df.columns:
                return None
            idx = df[metric_col].argmax()
            row = df.iloc[idx]
            return {
                metric_col: float(row[metric_col]),
                'csv': os.path.basename(csv_path)
            }
        except Exception as e:
            print(f"Warning: failed to read {csv_path}: {e}")
            return None

    best_user = extract_best(user_path)
    best_item = extract_best(item_path)
    summary['results']['userbased'] = best_user
    summary['results']['itembased'] = best_item

    # Choose according to eval_source
    chosen = None
    if eval_source == 'user':
        chosen = best_user
    elif eval_source == 'item':
        chosen = best_item
    else:  # auto fallback
        chosen = best_user or best_item

    if chosen is None:
        summary['best'] = {'score': 0.0, 'metric': metric_col, 'source': None}
    else:
        summary['best'] = {
            'score': chosen.get(metric_col, 0.0),
            'metric': metric_col,
            'source': chosen['csv']
        }

    out_path = os.path.join(results_dir, f'summary_{tag_name}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Wrote summary to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Evaluation Phase Orchestrator')
    parser.add_argument('--classified_csv', type=str, required=True, help='Path to classified CSV to evaluate')
    parser.add_argument('--tag_name', type=str, required=True, help='Tag JSON base name (used for outputs)')
    parser.add_argument('--eval_source', type=str, default='auto', choices=['user', 'item', 'auto'], help='Which evaluation source to use for selection')
    parser.add_argument('--clean_cache', action='store_true', help='Clean caches before running')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("EVALUATION PHASE: TAG GENERATION + RETRIEVAL EVALUATION")
    print("=" * 80)

    env = setup_environment()
    print(f"✓ PYTHONPATH set to: {env['PYTHONPATH']}\n")

    if args.clean_cache:
        clean_cache_once()

    # Phase 1: Generate tags for this classified CSV
    success = run_script(
        [sys.executable, 'gen_getag.py', '--classified_csv', args.classified_csv, '--tag_name', args.tag_name],
        f"Tag Generation for {args.classified_csv} -> {args.tag_name}", env
    )
    if not success:
        print("\n✗ Evaluation phase failed at tag generation step")
        sys.exit(1)

    # Phase 2: Retrieval evaluation using generated tags
    success = run_script(
        [sys.executable, 'retrieval_v2.py', '--tag_name', args.tag_name],
        f"Retrieval Evaluation for tag {args.tag_name}", env
    )
    if not success:
        print("\n✗ Evaluation phase failed at retrieval evaluation step")
        sys.exit(1)

    # Summary
    results_dir = '../results/retrieval_v2'
    write_summary(args.tag_name, results_dir, args.eval_source)

    print("=" * 80)
    print("EVALUATION PHASE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
