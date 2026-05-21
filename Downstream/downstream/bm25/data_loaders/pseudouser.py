"""
PseudoUser food commerce dataset loader for BeTag retrieval evaluation.
"""

from typing import Literal, Optional
import os
import pandas as pd
import numpy as np
from functools import lru_cache
import json
import ast
import re

__all__ = [
    'load_df',
    'inters_df',
    'all_pids',
]

_CACHE_DIR = 'dataset/pseudouser'
# Food raw session CSV (lives in the Upstream stage). Override with the FOOD_RAW_CSV env var.
_DATA_PATH = os.getenv(
    'FOOD_RAW_CSV',
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../../../Upstream/PseudoUser/data/food/food_commerce_data_cleaned_v3.csv',
    ),
)


@lru_cache(maxsize=1)
def load_df(cache_dir: str = _CACHE_DIR, auto_split: bool = True) -> pd.DataFrame:
    """
    Load the food commerce session data.

    Returns DataFrame with columns:
    - session_id: User session ID
    - loaded_pids: List of product names/IDs in the session
    """
    cache_file = os.path.join(cache_dir, 'processed_sessions.pkl')

    # Check if cached
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        df = pd.read_pickle(cache_file)
    else:
        print(f"Processing raw data from {_DATA_PATH}...")
        # Load raw CSV
        raw_df = pd.read_csv(_DATA_PATH, low_memory=False)

        # Process each row (session) to extract product sequences
        sessions = []
        for idx, row in raw_df.iterrows():
            products = []
            for col in raw_df.columns:
                if pd.notna(row[col]):
                    # Extract product name from event string
                    # Format: [..., 'event_type', 'product_name']
                    event_str = str(row[col])
                    matches = re.findall(r"'([^']*)'", event_str)
                    if len(matches) >= 2:
                        product_name = matches[-1]  # Last quoted string is product name
                        products.append(product_name)

            if products:  # Only keep sessions with at least one product
                sessions.append({
                    'session_id': idx,
                    'loaded_pids': products
                })

        df = pd.DataFrame(sessions)
        df = df.set_index('session_id')

        # Filter sessions with at least 3 products (for meaningful evaluation)
        df = df[df['loaded_pids'].map(len) >= 3]

        # Cache the processed data
        os.makedirs(cache_dir, exist_ok=True)
        df.to_pickle(cache_file)
        print(f"Cached processed data to {cache_file}")

    print(f"Loaded {len(df)} sessions")

    if not auto_split:
        return df

    # Split by session index (70% train, 20% val, 10% test)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)

    return {
        'train': df.iloc[:train_end],
        'val': df.iloc[train_end:val_end],
        'test': df.iloc[val_end:],
    }


def inters_df(
    cache_dir: str = _CACHE_DIR,
    split: Optional[Literal['train', 'val', 'test', 'all']] = None,
    use_iid: bool = False,
):
    """
    Get interaction data for specific split.

    Returns DataFrame with 'loaded_pids' column containing product sequences.
    """
    df = load_df(cache_dir, auto_split=False)

    if split is None:
        return {
            spt: inters_df(cache_dir, split=spt, use_iid=use_iid)
            for spt in ['train', 'val', 'test']
        }

    if split == 'all':
        assert use_iid is False
        return df

    # Get split indices
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)

    if split == 'train':
        split_df = df.iloc[:train_end]
        # For training: use all but last 2 items in sequence
        return split_df['loaded_pids'].map(lambda seq: seq[:-2]).to_frame()
    elif split == 'val':
        split_df = df.iloc[train_end:val_end]
        # For validation: predict second-to-last item
        return split_df['loaded_pids'].map(lambda seq: seq[-2:-1]).to_frame()
    else:  # test
        split_df = df.iloc[val_end:]
        # For test: predict last item
        return split_df['loaded_pids'].map(lambda seq: seq[-1:]).to_frame()


def all_pids(cache_dir: str = _CACHE_DIR):
    """
    Get all unique product IDs/names in the dataset.
    """
    df = load_df(cache_dir, auto_split=False)
    all_products = set()
    for products in df['loaded_pids']:
        all_products.update(products)
    return sorted(list(all_products))


# Create variant helper for compatibility with BeTag
class _VariantHelper:
    def __init__(self):
        return

    def __getattr__(self, __name: str):
        fn = globals().get(__name, None)
        assert fn is not None, f'data: {__name} for pseudouser does not exist'
        return fn


leave_one_last_item = _VariantHelper()
loli = leave_one_last_item
last = leave_one_last_item
