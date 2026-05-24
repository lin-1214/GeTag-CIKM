"""
Unified BiRank Retrieval evaluation for all datasets

BiRank is a graph-based ranking algorithm that considers both item-user and user-item relationships.
Runs with grid search over alpha and beta hyperparameters.

Usage:
    python downstream/birank/retrieval.py --dataset food --tag_name getag_native
    python downstream/birank/retrieval.py --dataset games --tag_name getag_native --verbose
    python downstream/birank/retrieval.py --dataset yelp --tag_name getag_native --verbose

Notes:
    - Use --verbose flag for large datasets (e.g., yelp) to monitor progress
    - Large datasets (>100k items) will show a warning with estimated memory usage
    - BiRank precomputation is cached in downstream/birank/cache for faster reruns
"""

import os
import sys
from itertools import product
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bm25'))
from bm25_core.configs import *
import pandas as pd
from tqdm import tqdm
import argparse
import json
import time

# Dataset configuration mapping
DATASET_CONFIG = {
    'food': {
        'internal_name': 'food',
        'display_name': 'Food',
        'ks': [1, 5, 10, 20]
    },
    'games': {
        'internal_name': 'amazon',
        'display_name': 'Games',
        'ks': [1, 5, 10, 20]
    },
    'yelp': {
        'internal_name': 'yelp',
        'display_name': 'Yelp',
        'ks': [1, 5, 10, 20]
    }
}

# CLI
parser = argparse.ArgumentParser(description='BiRank retrieval evaluation')
parser.add_argument('--dataset', type=str, required=True, choices=['food', 'games', 'yelp'],
                    help='Dataset to evaluate (food, games, or yelp)')
parser.add_argument('--tag_name', type=str, default='getag_native',
                    help='Tag JSON base name (without .json)')
parser.add_argument('--verbose', action='store_true',
                    help='Enable verbose output to monitor progress (recommended for large datasets)')
parser.add_argument('--no_cache', action='store_true',
                    help='Disable BiRank precomputation cache (useful during threshold search)')
parser.add_argument('--fast_grid', action='store_true',
                    help='Use a reduced hyperparameter grid for faster threshold search '
                         '(32 user-based configs instead of 162, 4 item-based instead of 16)')
args = parser.parse_args()

# Get dataset configuration
dataset_info = DATASET_CONFIG[args.dataset]
DATASET = args.dataset
TAG_NAME = args.tag_name
CORPUS_PATH = f'tags/{DATASET}/{TAG_NAME}.json'
INTERNAL_NAME = dataset_info['internal_name']
DISPLAY_NAME = dataset_info['display_name']
KS = dataset_info['ks']
VERBOSE = args.verbose


def best_row(df: pd.DataFrame):
    return df.iloc[df['ndcg@10/100/val'].argmax()]


def itembased_birank_grid(dataset_config: DatasetConfigT):
    """Grid search for item-based BiRank retrieval"""
    if args.fast_grid:
        ALPHA = [0.5, 0.9]
        BETA = [0.5, 0.9]
    else:
        ALPHA = [0.5, 0.75, 0.85, 0.9]
        BETA = [0.5, 0.75, 0.85, 0.9]

    total = len(ALPHA) * len(BETA)
    if VERBOSE:
        print(f"Running {total} item-based BiRank experiments with {len(ALPHA)} alpha x {len(BETA)} beta values...")

    for i, (alpha, beta) in enumerate(tqdm(product(ALPHA, BETA), desc='BiRank IR', total=total)):
        if VERBOSE:
            print(f"\n[Experiment {i+1}/{total}] alpha={alpha:.2f}, beta={beta:.2f}")
            start_time = time.time()

        predictor_config = BiRankIRPredictorConfig(
            alpha=alpha,
            beta=beta
        )

        corpus_config = CorpusConfig(
            name=CORPUS_PATH,
            unique=False,
        )

        config = ExperimentConfig(
            predictor_config=predictor_config,
            corpus_config=corpus_config,
            dataset_config=dataset_config,
            ks=KS,
        )

        results = config.eval(cache_dir=None if args.no_cache else 'downstream/birank/cache', verbose=VERBOSE)
        results['config'] = ExperimentConfig.model_validate(
            config.model_dump()
        )

        if VERBOSE:
            elapsed = time.time() - start_time
            ndcg_val = results.get('ndcg@10/100/val', float('nan'))
            if not pd.isna(ndcg_val):
                print(f"  ✓ Completed in {elapsed:.1f}s - NDCG@10(val)={ndcg_val:.4f}")
            else:
                print(f"  ✓ Completed in {elapsed:.1f}s")

        yield results


def userbased_birank_grid(dataset_config: DatasetConfigT):
    """Grid search for user-based BiRank recommendation"""
    if args.fast_grid:
        ALPHA = [0.75, 0.9]
        BETA = [0.75, 0.9]
        MAX_SEQ_LEN = [None, 20]
        SEQ_WEIGHTS = [None, 'log2']
        QUERY_TYPE = ['sum', 'union']
    else:
        ALPHA = [0.75, 0.85, 0.9]
        BETA = [0.75, 0.85, 0.9]
        MAX_SEQ_LEN = [None, 15, 20]
        SEQ_WEIGHTS = [None, 'linear', 'log2']
        QUERY_TYPE = ['sum', 'union']

    total = len(ALPHA) * len(BETA) * len(MAX_SEQ_LEN) * len(SEQ_WEIGHTS) * len(QUERY_TYPE)
    if VERBOSE:
        print(f"Running {total} user-based BiRank experiments...")
        print(f"  Grid: {len(ALPHA)} alpha x {len(BETA)} beta x {len(MAX_SEQ_LEN)} max_seq_len x {len(SEQ_WEIGHTS)} seq_weights x {len(QUERY_TYPE)} query_type")

    for i, (alpha, beta, max_seq_len, seq_weights, query_type) in enumerate(
        tqdm(
            product(ALPHA, BETA, MAX_SEQ_LEN, SEQ_WEIGHTS, QUERY_TYPE),
            desc='BiRank Rec',
            total=total
        )
    ):
        if VERBOSE:
            print(f"\n[Experiment {i+1}/{total}] alpha={alpha:.2f}, beta={beta:.2f}, "
                  f"max_seq_len={max_seq_len}, seq_weights={seq_weights}, query_type={query_type}")
            start_time = time.time()

        predictor_config = BiRankRecPredictorConfig(
            alpha=alpha,
            beta=beta,
            query_config=QueryFnConfig(
                query_type=query_type,
                seq_weights=seq_weights,
                max_seq_len=max_seq_len,
            )
        )

        corpus_config = CorpusConfig(
            name=CORPUS_PATH,
            unique=False,
        )

        config = ExperimentConfig(
            predictor_config=predictor_config,
            corpus_config=corpus_config,
            dataset_config=dataset_config,
            ks=KS,
        )

        results = config.eval(cache_dir=None if args.no_cache else 'downstream/birank/cache', verbose=VERBOSE)
        results['config'] = ExperimentConfig.model_validate(
            config.model_dump()
        )

        if VERBOSE:
            elapsed = time.time() - start_time
            ndcg_val = results.get('ndcg@10/100/val', float('nan'))
            if not pd.isna(ndcg_val):
                print(f"  ✓ Completed in {elapsed:.1f}s - NDCG@10(val)={ndcg_val:.4f}")
            else:
                print(f"  ✓ Completed in {elapsed:.1f}s")

        yield results


if __name__ == '__main__':

    print("=" * 80)
    print(f"BIRANK RETRIEVAL - {DISPLAY_NAME} Dataset - Using {TAG_NAME}")
    print("=" * 80)

    # Check corpus size and provide warnings for large datasets
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, 'r') as f:
            corpus_data = json.load(f)
            corpus_size = len(corpus_data)
            print(f"\nCorpus size: {corpus_size:,} items")

            if corpus_size > 100000:
                print(f"⚠ WARNING: Large corpus detected ({corpus_size:,} items)")
                print(f"  - BiRank will create {corpus_size}x{corpus_size} matrices ({corpus_size**2/1e9:.1f}B elements)")
                print(f"  - This may take a VERY long time or run out of memory")
                print(f"  - Estimated memory: ~{corpus_size**2 * 4 / 1e9:.1f} GB (float32)")
                print(f"  - Consider using --verbose to monitor progress")
                if not VERBOSE:
                    print(f"  - Run with --verbose flag to see detailed progress")
                    response = input("\nContinue anyway? [y/N]: ")
                    if response.lower() != 'y':
                        print("Aborted.")
                        sys.exit(0)
    else:
        print(f"⚠ WARNING: Corpus file not found at {CORPUS_PATH}")

    # User-based dataset config (for recommendation task)
    user_dataset_config = UserItemLastItemPredictionDatasetConfig(
        name=f'{INTERNAL_NAME}.leave_one_last_item.last',
        repeatable=True,
        min_in_seq_len=3,
    )

    # Item-based dataset config (for retrieval task)
    item_dataset_config = UserItemLastItemRetrievalDatasetConfig(
        name=f'{INTERNAL_NAME}.leave_one_last_item.lastir',
        repeatable=True,
    )

    # Create results directory
    results_dir = f'results/birank/{DATASET}'
    os.makedirs(results_dir, exist_ok=True)

    print("\n1. User-based BiRank Recommendation")
    print("-" * 80)
    dfs_user_birank = list(userbased_birank_grid(user_dataset_config))
    df_user_birank = pd.DataFrame(dfs_user_birank)
    user_results_path = f'{results_dir}/birank_results_userbased_{TAG_NAME}.csv'
    df_user_birank.to_csv(user_results_path, index=False)
    best = df_user_birank.iloc[df_user_birank['ndcg@10/100/val'].argmax()]

    print("\nBest configuration (based on NDCG@10 validation):")
    print(f"Validation: NDCG@10={best['ndcg@10/100/val']:.4f}")
    print("\nTest Results (all metrics):")
    for k in KS:
        hr = best[f'hr@{k}/100/test']
        ndcg = best[f'ndcg@{k}/100/test']
        print(f"  @{k:2d}: HR={hr:.4f}, NDCG={ndcg:.4f}")

    print("\n2. Item-based BiRank Retrieval")
    print("-" * 80)
    dfs_item_birank = list(itembased_birank_grid(item_dataset_config))
    df_item_birank = pd.DataFrame(dfs_item_birank)
    item_results_path = f'{results_dir}/birank_results_itembased_{TAG_NAME}.csv'
    df_item_birank.to_csv(item_results_path, index=False)
    best = df_item_birank.iloc[df_item_birank['ndcg@10/100/val'].argmax()]

    print("\nBest configuration (based on NDCG@10 validation):")
    print(f"Validation: NDCG@10={best['ndcg@10/100/val']:.4f}")
    print("\nTest Results (all metrics):")
    for k in KS:
        hr = best[f'hr@{k}/100/test']
        ndcg = best[f'ndcg@{k}/100/test']
        print(f"  @{k:2d}: HR={hr:.4f}, NDCG={ndcg:.4f}")

    print("\n" + "=" * 80)
    print("BiRank retrieval completed! Results saved to CSV files.")
    print("Files:")
    print(f"  - {user_results_path}")
    print(f"  - {item_results_path}")
    print(f"\nCorpus: {CORPUS_PATH}")
    print(f"Dataset: {DISPLAY_NAME}")
    print("=" * 80)
