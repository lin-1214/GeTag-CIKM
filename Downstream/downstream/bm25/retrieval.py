"""
Unified BM25 Retrieval evaluation for all datasets
"""

import os
import sys
from itertools import product
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from bm25_core.configs import *
import pandas as pd
from tqdm import tqdm
import argparse

# Dataset configuration mapping
DATASET_CONFIG = {
    'food': {
        'internal_name': 'food',
        'display_name': 'Food',
        'ks': [1, 3, 5, 10, 20]
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
parser = argparse.ArgumentParser(description='BM25 Retrieval evaluation for datasets')
parser.add_argument('--dataset', type=str, required=True, choices=['food', 'games', 'yelp'],
                    help='Dataset to evaluate (food, games, or yelp)')
parser.add_argument('--tag_name', type=str, default='getag_native',
                    help='Tag JSON base name (without .json)')
args = parser.parse_args()

# Get dataset configuration
dataset_info = DATASET_CONFIG[args.dataset]
DATASET = args.dataset
TAG_NAME = args.tag_name
CORPUS_PATH = f'tags/{DATASET}/{TAG_NAME}.json'  # Path relative to GeTag root
INTERNAL_NAME = dataset_info['internal_name']
DISPLAY_NAME = dataset_info['display_name']
KS = dataset_info['ks']


def best_row(df: pd.DataFrame):
    return df.iloc[df['ndcg@10/100/val'].argmax()]


def itembased_bm25_grid(dataset_config: DatasetConfigT):
    K1 = [0, 0.5, 1.0, 1.5, 2.0]
    B = [0, 0.25, 0.5, 0.75]
    for k1, b in tqdm(product(K1, B), desc='BM25 IR'):
        predictor_config = BM25IRPredictorConfig(k1=k1, b=b, alpha=0.)

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
        results = config.eval(cache_dir='downstream/bm25/cache', verbose=False)
        results['config'] = ExperimentConfig.model_validate(
            config.model_dump()
        )
        yield results


def userbased_bm25_grid(dataset_config: DatasetConfigT):
    K1 = [0.5]
    B = [0.25]
    MAX_SEQ_LEN = [None, 15, 20]
    SEQ_WEIGHTS = [None, 'linear', 'log2', 'exp']
    for k1, b, max_seq_len, seq_weights in tqdm(
        product(K1, B, MAX_SEQ_LEN, SEQ_WEIGHTS), desc='BM25 Rec'
    ):
        predictor_config = BM25RecPredictorConfig(
            k1=k1, b=b, alpha=0., query_config=QueryFnConfig(
                query_type='sum',
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
        results = config.eval(cache_dir='downstream/bm25/cache', verbose=False)
        results['config'] = ExperimentConfig.model_validate(
            config.model_dump()
        )
        yield results


if __name__ == '__main__':

    print("=" * 80)
    print(f"BM25 RETRIEVAL - {DISPLAY_NAME} Dataset - Using {TAG_NAME}")
    print("=" * 80)

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
    results_dir = f'results/bm25/{DATASET}'
    os.makedirs(results_dir, exist_ok=True)

    print("\n1. User-based BM25 Recommendation")
    print("-" * 80)
    dfs_user_bm25 = list(userbased_bm25_grid(user_dataset_config))
    df_user_bm25 = pd.DataFrame(dfs_user_bm25)
    user_results_path = f'{results_dir}/retrieval_results_v2_userbased_bm25_{TAG_NAME}.csv'
    df_user_bm25.to_csv(user_results_path, index=False)
    best = df_user_bm25.iloc[df_user_bm25['ndcg@10/100/val'].argmax()]

    print("\nBest configuration (based on NDCG@10 validation):")
    print(f"Validation: NDCG@10={best['ndcg@10/100/val']:.4f}")
    print("\nTest Results (all metrics):")
    for k in KS:
        hr = best[f'hr@{k}/100/test']
        ndcg = best[f'ndcg@{k}/100/test']
        print(f"  @{k:2d}: HR={hr:.4f}, NDCG={ndcg:.4f}")

    print("\n2. Item-based BM25 Retrieval")
    print("-" * 80)
    dfs_item_bm25 = list(itembased_bm25_grid(item_dataset_config))
    df_item_bm25 = pd.DataFrame(dfs_item_bm25)
    item_results_path = f'{results_dir}/retrieval_results_v2_itembased_bm25_{TAG_NAME}.csv'
    df_item_bm25.to_csv(item_results_path, index=False)
    best = df_item_bm25.iloc[df_item_bm25['ndcg@10/100/val'].argmax()]

    print("\nBest configuration (based on NDCG@10 validation):")
    print(f"Validation: NDCG@10={best['ndcg@10/100/val']:.4f}")
    print("\nTest Results (all metrics):")
    for k in KS:
        hr = best[f'hr@{k}/100/test']
        ndcg = best[f'ndcg@{k}/100/test']
        print(f"  @{k:2d}: HR={hr:.4f}, NDCG={ndcg:.4f}")

    print("\n" + "=" * 80)
    print("BM25 Retrieval completed! Results saved to CSV files.")
    print("Files:")
    print(f"  - {user_results_path}")
    print(f"  - {item_results_path}")
    print(f"\nCorpus: {CORPUS_PATH}")
    print(f"Dataset: {DISPLAY_NAME}")
    print("=" * 80)
