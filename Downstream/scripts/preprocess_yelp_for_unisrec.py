"""
Preprocess Yelp data for UniSRec next-item prediction

This script converts:
- yelp_sessions_evaluation.csv (session data with business IDs)
- yelp_native.json or yelp_basetag.json (item tags)

Into UniSRec format:
- yelp_*.train.inter
- yelp_*.valid.inter
- yelp_*.test.inter
- yelp_*.feat1CLS (BERT embeddings)
"""

import argparse
import csv
import collections
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_interactions_from_csv(csv_file, business_id_to_item_id, valid_item_ids=None):
    """
    Load interactions from CSV format where each row is a session with business IDs

    Args:
        csv_file: CSV file where each row is a session with business IDs
        business_id_to_item_id: Mapping from business_id to numeric item_id
        valid_item_ids: Set of valid item IDs to filter against (from tags file)

    Returns:
        users: set of user IDs
        items: set of item IDs
        inters: list of (user_id, item_id, rating=1, timestamp)
    """
    print(f"Loading sequences from {csv_file}...")

    users, items, inters = set(), set(), []
    skipped_items = 0
    unknown_business_ids = 0

    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for session_idx, row in enumerate(tqdm(csv_reader, desc="Processing sessions")):
            # Each row is a session, each cell is a business ID
            user_id = str(session_idx)

            # Extract business IDs and convert to item IDs
            item_ids = []
            for business_id in row:
                business_id = business_id.strip()
                if not business_id:
                    continue

                # Convert business_id to item_id (numeric)
                if business_id not in business_id_to_item_id:
                    unknown_business_ids += 1
                    continue

                item_id = business_id_to_item_id[business_id]

                # Only include items that have tags
                if valid_item_ids is None or item_id in valid_item_ids:
                    item_ids.append(item_id)
                else:
                    skipped_items += 1

            # Only keep sessions with at least 2 items
            if len(item_ids) < 2:
                continue

            # Add interactions with timestamps
            for event_idx, item_id in enumerate(item_ids):
                ts = event_idx + 1
                users.add(user_id)
                items.add(item_id)
                inters.append((user_id, item_id, 1.0, ts))

    if unknown_business_ids > 0:
        print(f"  Skipped {unknown_business_ids} unknown business IDs")
    if skipped_items > 0:
        print(f"  Skipped {skipped_items} items not in tags file")

    print(f"  Loaded {len(users)} users with sequences")
    print(f"  {len(items)} unique items")

    # Print sequence length distribution
    user_seq_lengths = collections.defaultdict(int)
    for user_id in users:
        user_inters = [inter for inter in inters if inter[0] == user_id]
        seq_len = len(user_inters)
        user_seq_lengths[seq_len] += 1

    print(f"  Sequences by length:")
    for length in sorted(user_seq_lengths.keys())[:20]:
        print(f"    {length} items: {user_seq_lengths[length]} users")
    if len(user_seq_lengths) > 20:
        print(f"    ... ({len(user_seq_lengths)} different lengths total)")

    return users, items, inters


def filter_inters_k_core(inters, user_k=5, item_k=5):
    """K-core filtering to ensure data quality"""
    if user_k == 0 and item_k == 0:
        print(f"\nK-core filtering disabled (user_k=0, item_k=0)")
        return inters

    print(f"\nApplying k-core filtering (user_k={user_k}, item_k={item_k})...")
    print(f"Initial interactions: {len(inters)}")

    while True:
        # Count interactions per user and item
        user_counts = collections.defaultdict(int)
        item_counts = collections.defaultdict(int)
        for user_id, item_id, rating, ts in inters:
            user_counts[user_id] += 1
            item_counts[item_id] += 1

        # Filter out users and items below threshold
        filtered_inters = []
        for user_id, item_id, rating, ts in inters:
            if user_counts[user_id] >= user_k and item_counts[item_id] >= item_k:
                filtered_inters.append((user_id, item_id, rating, ts))

        # If no change, we're done
        if len(filtered_inters) == len(inters):
            break

        inters = filtered_inters

    print(f"  After k-core: {len(inters)} interactions")
    return inters


def convert_to_indices(users, items, inters):
    """Convert user and item IDs to sequential indices (0, 1, 2, ...)"""
    print("\nConverting to indices...")

    # Create sequential mapping for users
    user2index = {user_id: idx for idx, user_id in enumerate(sorted(users))}

    # Create sequential mapping for items (like i3fresh does)
    # This creates dense indices 0, 1, 2, ... N-1
    item2index = {item_id: idx for idx, item_id in enumerate(sorted(items, key=lambda x: int(x)))}

    indexed_inters = []
    for user_id, item_id, rating, ts in inters:
        user_idx = user2index[user_id]
        item_idx = item2index[item_id]
        indexed_inters.append((user_idx, item_idx, rating, ts))

    return user2index, item2index, indexed_inters


def split_train_valid_test(user2index, inters):
    """Split into train/valid/test using leave-one-last-item"""
    print("\nSplitting train/valid/test...")

    # Group by user and sort by timestamp
    user_inters = collections.defaultdict(list)
    for user_idx, item_idx, rating, ts in inters:
        user_inters[user_idx].append((item_idx, ts))

    # Sort by timestamp for each user
    for user_idx in user_inters:
        user_inters[user_idx].sort(key=lambda x: x[1])

    train_inters, valid_inters, test_inters = {}, {}, {}
    skipped_users = 0

    for user_idx in range(len(user2index)):
        if user_idx not in user_inters:
            skipped_users += 1
            continue

        items = [str(item_idx) for item_idx, ts in user_inters[user_idx]]

        # Handle sequences of different lengths
        if len(items) == 2:
            # For 2-item sequences: first item for train, second for test, skip validation
            train_inters[user_idx] = [items[0]]
            valid_inters[user_idx] = []
            test_inters[user_idx] = [items[1]]
        else:
            # For 3+ items: leave last 2 items for validation and test
            train_inters[user_idx] = items[:-2]
            valid_inters[user_idx] = [items[-2]]
            test_inters[user_idx] = [items[-1]]

    train_count = sum(len(v) for v in train_inters.values())
    print(f"  Train: {train_count} interactions from {len(train_inters)} users")
    print(f"  Valid: {len(valid_inters)} sequences")
    print(f"  Test: {len(test_inters)} sequences")
    if skipped_users > 0:
        print(f"  Skipped: {skipped_users} users")

    return train_inters, valid_inters, test_inters


def generate_item_texts(item2index, tags_file):
    """Generate text for each item by concatenating tags"""
    print(f"\nLoading tags from {tags_file}...")
    with open(tags_file, 'r', encoding='utf-8') as f:
        tags_data = json.load(f)

    # Create reverse mapping: sequential index -> original smap ID
    # item2index is {smap_id: sequential_idx}, so reverse it
    index2item = {idx: item_id for item_id, idx in item2index.items()}

    # Create item_text_list in sequential order (0 to N-1)
    # Tags file has original smap IDs as keys: {"0": [...], "123": [...], ...}
    item_text_list = []

    for idx in range(len(item2index)):
        # Get original smap ID for this sequential index
        original_item_id = index2item[idx]

        # Lookup tags using original smap ID
        if original_item_id in tags_data:
            tags = tags_data[original_item_id]
            if isinstance(tags, list):
                text = ' '.join(tags) + '.'
            else:
                text = str(tags) + '.'
        else:
            text = 'unknown item.'

        item_text_list.append(text)

    print(f"  Generated text for {len(item_text_list)} items")

    # Show some examples
    print(f"\nExample item texts:")
    for i in range(min(5, len(item_text_list))):
        print(f"  Item {i}: {item_text_list[i][:80]}...")

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
            # Sliding window requires at least 1 item (will skip if seq_len <= 1)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]  # Max 50 items in history
                f.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    # Save valid.inter
    valid_file = os.path.join(output_dir, f'{dataset_name}.valid.inter')
    with open(valid_file, 'w') as f:
        f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            # Skip users with empty validation (2-item sessions)
            if len(valid_data[uid]) == 0:
                continue
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
    parser = argparse.ArgumentParser(description='Preprocess Yelp data for UniSRec')
    parser.add_argument('--csv_file', type=str,
                        default='data/raw/yelp/evaluation.csv',
                        help='Input CSV file with sessions')
    parser.add_argument('--smap_file', type=str,
                        default='data/preprocessed/yelp/bm25/smap.json',
                        help='SMAP file (business_id to index mapping)')
    parser.add_argument('--tags_file', type=str, default='tags/yelp/native.json',
                        help='Tag file: native.json, getag_native.json, basetag.json, etc.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated based on tag file if not specified)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset name (auto-generated based on tag file if not specified)')
    parser.add_argument('--user_k', type=int, default=0,
                        help='K-core filtering: minimum interactions per user (0 to disable)')
    parser.add_argument('--item_k', type=int, default=0,
                        help='K-core filtering: minimum interactions per item (0 to disable)')
    parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for BERT encoding')
    args = parser.parse_args()

    # Auto-generate dataset name and output dir based on tags file
    if args.dataset_name is None or args.output_dir is None:
        # Use the tag filename directly as suffix (e.g., native, getag_native, basetag, getag_basetag)
        tag_suffix = os.path.basename(args.tags_file).replace('.json', '')

        if args.dataset_name is None:
            args.dataset_name = f'yelp_{tag_suffix}'
        if args.output_dir is None:
            args.output_dir = f'data/preprocessed/UniSRec/{args.dataset_name}'

    print("="*80)
    print("PREPROCESSING YELP DATA FOR UNISREC")
    print("="*80)
    print(f"CSV file: {args.csv_file}")
    print(f"SMAP file: {args.smap_file}")
    print(f"Tags file: {args.tags_file}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"K-core filtering: user_k={args.user_k}, item_k={args.item_k}")
    print(f"PLM: {args.plm_name}")
    print(f"Device: {args.device}")
    print("="*80)

    # Load smap to get business_id to item_id mapping
    print(f"\nLoading business_id mapping from {args.smap_file}...")
    with open(args.smap_file, 'r', encoding='utf-8') as f:
        smap = json.load(f)
    # smap is {"0": "business_id1", "1": "business_id2", ...}
    # Create reverse mapping: business_id -> item_id (numeric string)
    business_id_to_item_id = {business_id: item_id for item_id, business_id in smap.items()}
    print(f"  Loaded mapping for {len(business_id_to_item_id)} business IDs")

    # Load tags to get valid item IDs
    with open(args.tags_file, 'r', encoding='utf-8') as f:
        tags_data = json.load(f)
    valid_item_ids = set(tags_data.keys())

    # Step 1: Load interactions from CSV
    users, items, inters = load_interactions_from_csv(args.csv_file, business_id_to_item_id, valid_item_ids)

    # Step 2: K-core filtering
    inters = filter_inters_k_core(inters, args.user_k, args.item_k)

    # Recalculate users and items after filtering
    users = set(user_id for user_id, _, _, _ in inters)
    items = set(item_id for _, item_id, _, _ in inters)

    # Step 3: Convert to indices
    user2index, item2index, indexed_inters = convert_to_indices(users, items, inters)

    # Step 4: Split train/valid/test
    train_data, valid_data, test_data = split_train_valid_test(user2index, indexed_inters)

    # Step 5: Generate item texts from tags
    item_texts = generate_item_texts(item2index, args.tags_file)

    # Step 6: Generate BERT embeddings
    embeddings = generate_bert_embeddings(
        item_texts, args.plm_name, args.device, args.batch_size
    )

    # Step 7: Save to RecBole format
    save_to_recbole_format(args.output_dir, args.dataset_name, train_data, valid_data, test_data)

    # Step 8: Save embeddings
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
