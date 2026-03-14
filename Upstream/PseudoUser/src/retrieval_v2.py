"""
Retrieval evaluation 
"""

import os
from itertools import product
from downstream.configs import *
import pandas as pd
from tqdm import tqdm
import argparse
from config import Config

# CLI
parser = argparse.ArgumentParser(description='Retrieval evaluation for generated tags')
parser.add_argument('--tag_name', type=str, default='getags_zscore', help='Tag JSON base name under ../json/tags (without .json)')
args = parser.parse_args()

# Dataset configuration
config = Config()
DOMAIN = getattr(config, "DOMAIN", "food").lower()
TAG_NAME = args.tag_name
CORPUS_PATH = f'../json/tags/{TAG_NAME}.json'  # Using generated getags from gen_getag.py


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
            ks=[5, 10, 20],
        )
        results = config.eval(cache_dir='exps_v2', verbose=False)
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
            ks=[5, 10, 20],
        )
        results = config.eval(cache_dir='exps_v2', verbose=False)
        results['config'] = ExperimentConfig.model_validate(
            config.model_dump()
        )
        yield results


if __name__ == '__main__':

    print("=" * 80)
    print(f"RETRIEVAL V2 - Using {TAG_NAME}")
    print("=" * 80)

    # User-based dataset config (for recommendation task)
    if DOMAIN == "movie":
        user_dataset_name = 'movielens.leave_one_last_item.last'
        item_dataset_name = 'movielens.leave_one_last_item.lastir'
    else:
        user_dataset_name = 'i3fresh.leave_one_last_item.last'
        item_dataset_name = 'i3fresh.leave_one_last_item.lastir'

    user_dataset_config = UserItemLastItemPredictionDatasetConfig(
        name=user_dataset_name,
        repeatable=(DOMAIN != "movie"),
        min_in_seq_len=3,
    )

    # Item-based dataset config (for retrieval task)
    item_dataset_config = UserItemLastItemRetrievalDatasetConfig(
        name=item_dataset_name,
        repeatable=(DOMAIN != "movie"),
    )

    # Create results directory
    results_dir = '../results/retrieval_v2'
    os.makedirs(results_dir, exist_ok=True)

    print("\n1. User-based BM25 Recommendation")
    print("-" * 80)
    dfs_user_bm25 = list(userbased_bm25_grid(user_dataset_config))
    df_user_bm25 = pd.DataFrame(dfs_user_bm25)
    user_results_path = f'{results_dir}/retrieval_results_v2_userbased_bm25_{TAG_NAME}.csv'
    df_user_bm25.to_csv(user_results_path, index=False)
    best = df_user_bm25.iloc[df_user_bm25['ndcg@10/100/val'].argmax()]
    print(f"Best NDCG@10 (val): {best['ndcg@10/100/val']:.4f}, HR@10 (test): {best['hr@10/100/test']:.4f}, NDCG@10 (test): {best['ndcg@10/100/test']:.4f}")

    print("\n2. Item-based BM25 Retrieval")
    print("-" * 80)
    dfs_item_bm25 = list(itembased_bm25_grid(item_dataset_config))
    df_item_bm25 = pd.DataFrame(dfs_item_bm25)
    item_results_path = f'{results_dir}/retrieval_results_v2_itembased_bm25_{TAG_NAME}.csv'
    df_item_bm25.to_csv(item_results_path, index=False)
    best = df_item_bm25.iloc[df_item_bm25['ndcg@10/100/val'].argmax()]
    print(f"Best NDCG@10 (val): {best['ndcg@10/100/val']:.4f}, HR@10 (test): {best['hr@10/100/test']:.4f}, NDCG@10 (test): {best['ndcg@10/100/test']:.4f}")

    print("\n" + "=" * 80)
    print("V2 Retrieval completed! Results saved to CSV files.")
    print("Files:")
    print(f"  - {user_results_path}")
    print(f"  - {item_results_path}")
    print("\nDataset used:")
    print(f"  - Corpus: {CORPUS_PATH}")
    print("=" * 80)
