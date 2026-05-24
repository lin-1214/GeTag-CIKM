"""
Preprocess food data for UniSRec next-item prediction

This script converts:
- food_commerce_data_cleaned_v3.csv (session data)
- keywords_v2.json (item tags)

Into UniSRec format:
- food.train.inter
- food.valid.inter
- food.test.inter
- food.feat1CLS (BERT embeddings)
"""

import argparse
import csv
import ast
import collections
import json
import os
import sys
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_interactions_from_csv(csv_file, product_name_to_id_file):
    """
    Load interactions from food_commerce_data_cleaned_v3.csv

    Returns:
        users: set of user IDs
        items: set of item IDs
        inters: list of (user_id, item_id, rating=1, timestamp)
    """
    print("Loading product name to ID mapping...")
    with open(product_name_to_id_file, 'r', encoding='utf-8') as f:
        name_to_id = json.load(f)

    print(f"Loading interactions from {csv_file}...")
    df = pd.read_csv(csv_file, low_memory=False)

    users, items, inters = set(), set(), []
    skipped_products = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sessions"):
        # Extract all events from this session/row
        events = []
        session_id = None

        for col in row.index:
            cell_value = row[col]
            if pd.isna(cell_value) or cell_value == '':
                continue

            # The cell contains string representation of a list
            # Format: [session_id, datetime.datetime(...), 'event_type', 'product_name']
            cell_str = str(cell_value)

            try:
                # Extract numeric values (session_id is the first number)
                numbers = re.findall(r'\d+', cell_str)
                if numbers and session_id is None:
                    session_id = numbers[0]

                # Find all quoted strings in the cell
                # Pattern to match quoted strings (both single and double quotes)
                pattern = r"'([^']*)'|\"([^\"]*)\""
                matches = re.findall(pattern, cell_str)
                # Flatten the matches (regex returns tuples)
                strings = [m[0] if m[0] else m[1] for m in matches]

                # The last string should be the product name, second to last is event_type
                if len(strings) >= 2:
                    event_type = strings[-2]
                    product_name = strings[-1]

                    if product_name and product_name != '':
                        events.append((session_id, event_type, product_name))
            except Exception as e:
                # If parsing fails, skip this cell
                continue

        if not events or session_id is None:
            continue

        # Use session_id as user_id
        user_id = str(session_id)

        # Process each event in chronological order (column order = chronological)
        for event_idx, (sid, event_type, product_name) in enumerate(events):
            # Convert product name to ID
            if product_name not in name_to_id:
                if product_name not in skipped_products:
                    skipped_products.add(product_name)
                continue

            item_id = str(name_to_id[product_name])

            # Use event index as timestamp (maintains chronological order within session)
            ts = event_idx + 1

            users.add(user_id)
            items.add(item_id)
            inters.append((user_id, item_id, 1.0, ts))

    print(f"\nSkipped {len(skipped_products)} unknown products")
    print(f"Loaded {len(users)} users, {len(items)} items, {len(inters)} interactions")

    return users, items, inters


def filter_inters_k_core(inters, user_k=5, item_k=5):
    """K-core filtering to ensure data quality"""
    print(f"\nApplying k-core filtering (user_k={user_k}, item_k={item_k})...")
    print(f"Initial interactions: {len(inters)}")

    new_inters = []
    epoch = 0

    while True:
        user2count = collections.defaultdict(int)
        item2count = collections.defaultdict(int)

        for user, item, rating, ts in inters:
            user2count[user] += 1
            item2count[item] += 1

        # Filter users and items
        valid_users = {u for u, cnt in user2count.items() if cnt >= user_k}
        valid_items = {i for i, cnt in item2count.items() if cnt >= item_k}

        filtered_users = len(user2count) - len(valid_users)
        filtered_items = len(item2count) - len(valid_items)

        if filtered_users == 0 and filtered_items == 0:
            break

        # Filter interactions
        new_inters = []
        for user, item, rating, ts in inters:
            if user in valid_users and item in valid_items:
                new_inters.append((user, item, rating, ts))

        epoch += 1
        print(f"  Epoch {epoch}: {len(new_inters)} inters, "
              f"{len(valid_users)} users, {len(valid_items)} items")

        inters = new_inters

    return inters


def make_inters_chronological(inters):
    """Sort interactions chronologically for each user"""
    print("\nSorting interactions chronologically...")
    user2inters = collections.defaultdict(list)

    for user, item, rating, ts in inters:
        user2inters[user].append((user, item, rating, ts))

    sorted_inters = []
    for user in user2inters:
        user_inters = sorted(user2inters[user], key=lambda x: x[3])  # Sort by timestamp
        sorted_inters.extend(user_inters)

    return sorted_inters


def convert_inters_to_indexed(inters):
    """Convert string IDs to integer indices"""
    print("\nConverting to indexed format...")
    user2items = collections.defaultdict(list)
    user2index, item2index = {}, {}

    for user, item, rating, ts in inters:
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)

        user2items[user2index[user]].append(item2index[item])

    print(f"  {len(user2index)} unique users, {len(item2index)} unique items")
    return user2items, user2index, item2index


def split_train_valid_test(user2items):
    """Split into train/valid/test using leave-one-out"""
    print("\nSplitting train/valid/test...")
    train_inters, valid_inters, test_inters = {}, {}, {}

    for u_index in range(len(user2items)):
        items = user2items[u_index]

        # Leave last 2 items for validation and test
        train_inters[u_index] = [str(i) for i in items[:-2]]
        valid_inters[u_index] = [str(items[-2])]
        test_inters[u_index] = [str(items[-1])]

    train_count = sum(len(v) for v in train_inters.values())
    print(f"  Train: {train_count} sequences")
    print(f"  Valid: {len(valid_inters)} sequences")
    print(f"  Test: {len(test_inters)} sequences")

    return train_inters, valid_inters, test_inters


def generate_item_texts(item2index, keywords_file):
    """Generate text for each item by concatenating keywords"""
    print(f"\nLoading keywords from {keywords_file}...")
    with open(keywords_file, 'r', encoding='utf-8') as f:
        keywords = json.load(f)

    # Create item_text_list in index order
    item_text_list = []
    index2item = {v: k for k, v in item2index.items()}

    for idx in range(len(item2index)):
        item_id = index2item[idx]

        if item_id in keywords:
            tags = keywords[item_id]
            if isinstance(tags, list):
                text = ' '.join(tags) + '.'
            else:
                text = str(tags) + '.'
        else:
            text = 'unknown item.'

        item_text_list.append(text)

    print(f"  Generated text for {len(item_text_list)} items")
    return item_text_list


def generate_bert_embeddings(item_text_list, plm_name='bert-base-uncased', device='cpu', batch_size=4):
    """Generate BERT embeddings for item texts"""
    print(f"\nGenerating BERT embeddings using {plm_name}...")

    tokenizer = AutoTokenizer.from_pretrained(plm_name)
    model = AutoModel.from_pretrained(plm_name)
    model = model.to(device)
    model.eval()

    embeddings = []
    start = 0

    with torch.no_grad():
        while start < len(item_text_list):
            sentences = item_text_list[start:start + batch_size]

            encoded = tokenizer(
                sentences,
                padding=True,
                max_length=512,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            outputs = model(**encoded)
            # Use CLS token embedding
            cls_output = outputs.last_hidden_state[:, 0, :].detach().cpu()
            embeddings.append(cls_output)

            start += batch_size
            if start % 100 == 0:
                print(f"  Processed {start}/{len(item_text_list)} items...")

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(f"  Embeddings shape: {embeddings.shape}")

    return embeddings


def save_to_recbole_format(output_dir, dataset_name, train_data, valid_data, test_data):
    """Save data in RecBole format"""
    print(f"\nSaving to RecBole format in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    uid_list = list(train_data.keys())
    uid_list.sort()

    # Save train.inter with sliding window
    train_file = os.path.join(output_dir, f'{dataset_name}.train.inter')
    with open(train_file, 'w') as f:
        f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]  # Max 50 items in history
                f.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    # Save valid.inter
    valid_file = os.path.join(output_dir, f'{dataset_name}.valid.inter')
    with open(valid_file, 'w') as f:
        f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            f.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    # Save test.inter
    test_file = os.path.join(output_dir, f'{dataset_name}.test.inter')
    with open(test_file, 'w') as f:
        f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            f.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    print(f"  Saved: {train_file}")
    print(f"  Saved: {valid_file}")
    print(f"  Saved: {test_file}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess food data for UniSRec with different tag files')
    parser.add_argument('--csv_file', type=str, default='data/food_commerce_data_cleaned_v3.csv',
                        help='Input CSV file with session data')
    parser.add_argument('--product_mapping', type=str, default='json/product_name_to_id.json',
                        help='Product name to ID mapping')
    parser.add_argument('--keywords_file', type=str, default='json/tags/keywords_v2.json',
                        help='Tag file: keywords_v2.json, base_tags_v2.json, or getags_zscore.json')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated based on tag file if not specified)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset name (auto-generated based on tag file if not specified)')
    parser.add_argument('--user_k', type=int, default=5, help='User k-core threshold')
    parser.add_argument('--item_k', type=int, default=5, help='Item k-core threshold')
    parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    args = parser.parse_args()

    # Auto-generate dataset name and output dir based on keywords file
    if args.dataset_name is None or args.output_dir is None:
        tag_file_name = os.path.basename(args.keywords_file).replace('.json', '')

        # Map tag file names to dataset suffixes
        if 'keywords' in tag_file_name:
            tag_suffix = 'native'
        elif 'base_tags' in tag_file_name:
            tag_suffix = 'basetag'
        elif 'getags_zscore' in tag_file_name or 'getag' in tag_file_name:
            tag_suffix = 'getag'
        else:
            tag_suffix = 'custom'

        if args.dataset_name is None:
            args.dataset_name = f'food_{tag_suffix}'
        if args.output_dir is None:
            args.output_dir = f'data/unisrec/{args.dataset_name}'

    print("="*80)
    print("PREPROCESSING FOOD DATA FOR UNISREC")
    print("="*80)
    print(f"Tag file: {args.keywords_file}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Step 1: Load interactions
    users, items, inters = load_interactions_from_csv(args.csv_file, args.product_mapping)

    # Step 2: Filter by k-core
    inters = filter_inters_k_core(inters, args.user_k, args.item_k)

    # Step 3: Sort chronologically
    inters = make_inters_chronological(inters)

    # Step 4: Convert to indices
    user2items, user2index, item2index = convert_inters_to_indexed(inters)

    # Step 5: Split train/valid/test
    train_data, valid_data, test_data = split_train_valid_test(user2items)

    # Step 6: Generate item texts from keywords
    item_texts = generate_item_texts(item2index, args.keywords_file)

    # Step 7: Generate BERT embeddings
    embeddings = generate_bert_embeddings(item_texts, args.plm_name, args.device)

    # Step 8: Save to RecBole format
    save_to_recbole_format(args.output_dir, args.dataset_name, train_data, valid_data, test_data)

    # Step 9: Save embeddings
    emb_file = os.path.join(args.output_dir, f'{args.dataset_name}.feat1CLS')
    embeddings.tofile(emb_file)
    print(f"  Saved: {emb_file}")

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Files created:")
    print(f"  - {args.dataset_name}.train.inter")
    print(f"  - {args.dataset_name}.valid.inter")
    print(f"  - {args.dataset_name}.test.inter")
    print(f"  - {args.dataset_name}.feat1CLS")
    print(f"\nDataset statistics:")
    print(f"  Users: {len(user2index)}")
    print(f"  Items: {len(item2index)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print("="*80)


if __name__ == '__main__':
    main()
