import pandas as pd
import os
import argparse
import shutil


def keep_last_item(df):
    """Keep only the last item in the item_id_list column"""
    # Get the column name for item_id_list (second column)
    item_list_col = df.columns[1]
    df[item_list_col] = df[item_list_col].apply(lambda x: str(x).split()[-1] if isinstance(x, str) else x)
    return df


def convert_to_item_based(dataset_path):
    """
    Convert user-based dataset to item-based format.

    Args:
        dataset_path: Path to the dataset directory (e.g., 'data/unisrec/i3fresh_basetag_zh')
    """
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_path.rstrip('/'))
    parent_dir = os.path.dirname(dataset_path)

    # Define paths for input files
    train_file = os.path.join(dataset_path, f'{dataset_name}.train.inter')
    valid_file = os.path.join(dataset_path, f'{dataset_name}.valid.inter')
    test_file = os.path.join(dataset_path, f'{dataset_name}.test.inter')
    feat_file = os.path.join(dataset_path, f'{dataset_name}.feat1CLS')

    # Check if input files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not os.path.exists(valid_file):
        raise FileNotFoundError(f"Valid file not found: {valid_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    print(f"Converting dataset: {dataset_name}")
    print(f"  Reading files from: {dataset_path}")

    # Read the files into dataframes (skip header row)
    df_train = pd.read_csv(train_file, delimiter='\t', header=0)
    df_valid = pd.read_csv(valid_file, delimiter='\t', header=0)
    df_test = pd.read_csv(test_file, delimiter='\t', header=0)

    print(f"  Original sizes - Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}")

    # Apply the transformation to keep only the last item
    df_train_item = keep_last_item(df_train.copy())
    df_valid_item = keep_last_item(df_valid.copy())
    df_test_item = keep_last_item(df_test.copy())

    # Create output directory with _i suffix
    item_dataset_name = dataset_name + "_i"
    item_dataset_path = os.path.join(parent_dir, item_dataset_name)

    if not os.path.exists(item_dataset_path):
        os.makedirs(item_dataset_path)
        print(f"  Created directory: {item_dataset_path}")

    # Copy the feature file if it exists
    if os.path.exists(feat_file):
        output_feat_file = os.path.join(item_dataset_path, f'{item_dataset_name}.feat1CLS')
        shutil.copyfile(feat_file, output_feat_file)
        print(f"  Copied feature file: {os.path.basename(feat_file)}")
    else:
        print(f"  Warning: Feature file not found: {feat_file}")

    # Save the item-based versions (CORRECTED: preserving train/valid/test splits)
    train_output = os.path.join(item_dataset_path, f'{item_dataset_name}.train.inter')
    valid_output = os.path.join(item_dataset_path, f'{item_dataset_name}.valid.inter')
    test_output = os.path.join(item_dataset_path, f'{item_dataset_name}.test.inter')

    df_train_item.to_csv(train_output, index=False, header=True, sep='\t')
    df_valid_item.to_csv(valid_output, index=False, header=True, sep='\t')
    df_test_item.to_csv(test_output, index=False, header=True, sep='\t')

    print(f"  Saved item-based files:")
    print(f"    - {os.path.basename(train_output)}")
    print(f"    - {os.path.basename(valid_output)}")
    print(f"    - {os.path.basename(test_output)}")
    print(f"  ✓ Conversion complete!\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Convert user-based dataset to item-based format')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory (e.g., PseudoUser/data/unisrec/i3fresh_basetag_zh)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_to_item_based(args.dataset)
