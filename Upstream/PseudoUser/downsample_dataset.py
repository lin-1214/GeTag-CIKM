"""
Downsample dataset to match target user count

This creates a smaller dataset by randomly sampling users (in-place modification).
"""

import pandas as pd
import numpy as np
import os
import shutil
import argparse


def downsample_dataset(dataset_path, target_users, seed=42):
    """
    Downsample dataset to target number of users (in-place modification)

    Args:
        dataset_path: Path to the dataset directory
        target_users: Target number of users
        seed: Random seed for reproducibility
    """
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_path.rstrip('/'))

    print("="*80)
    print("DOWNSAMPLING DATASET")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"Target users: {target_users}")
    print(f"Random seed: {seed}")
    print("="*80)

    # Define file paths
    train_file = os.path.join(dataset_path, f'{dataset_name}.train.inter')
    valid_file = os.path.join(dataset_path, f'{dataset_name}.valid.inter')
    test_file = os.path.join(dataset_path, f'{dataset_name}.test.inter')
    feat_file = os.path.join(dataset_path, f'{dataset_name}.feat1CLS')

    # Read datasets
    print("\nLoading datasets...")
    train = pd.read_csv(train_file, sep='\t')
    valid = pd.read_csv(valid_file, sep='\t')
    test = pd.read_csv(test_file, sep='\t')

    # Get all unique users
    all_users = train['user_id:token'].unique()
    print(f"  Original users: {len(all_users)}")
    print(f"  Original train samples: {len(train)}")
    print(f"  Original valid samples: {len(valid)}")
    print(f"  Original test samples: {len(test)}")

    # Sample users
    print(f"\nSampling {target_users} users...")
    np.random.seed(seed)
    sampled_users = np.random.choice(
        all_users,
        size=min(target_users, len(all_users)),
        replace=False
    )
    sampled_users = set(sampled_users)

    # Filter all splits
    print("Filtering datasets...")
    train_sampled = train[train['user_id:token'].isin(sampled_users)]
    valid_sampled = valid[valid['user_id:token'].isin(sampled_users)]
    test_sampled = test[test['user_id:token'].isin(sampled_users)]

    print(f"  Sampled users: {len(sampled_users)}")
    print(f"  Sampled train samples: {len(train_sampled)}")
    print(f"  Sampled valid samples: {len(valid_sampled)}")
    print(f"  Sampled test samples: {len(test_sampled)}")

    # Save filtered data (overwrite original files)
    print("\nSaving filtered datasets...")
    train_sampled.to_csv(train_file, sep='\t', index=False)
    valid_sampled.to_csv(valid_file, sep='\t', index=False)
    test_sampled.to_csv(test_file, sep='\t', index=False)

    # Feature file remains the same (no need to modify)
    print(f"Feature file unchanged: {dataset_name}.feat1CLS")

    print("\n" + "="*80)
    print("DOWNSAMPLING COMPLETE!")
    print("="*80)
    print(f"Files updated:")
    print(f"  - {dataset_name}.train.inter ({len(train_sampled)} samples)")
    print(f"  - {dataset_name}.valid.inter ({len(valid_sampled)} samples)")
    print(f"  - {dataset_name}.test.inter ({len(test_sampled)} samples)")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downsample dataset to match target user count')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory (e.g., data/unisrec/i3fresh_basetag_zh)'
    )
    parser.add_argument(
        '--target_users',
        type=int,
        default=13374,
        help='Target number of users (default: 13374 to match old i3fresh)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    downsample_dataset(
        dataset_path=args.dataset,
        target_users=args.target_users,
        seed=args.seed
    )
