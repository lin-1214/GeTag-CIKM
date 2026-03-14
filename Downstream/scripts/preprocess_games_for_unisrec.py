"""
Preprocess Amazon data for UniSRec next-item prediction

This script converts:
- sequences.json (user purchase sequences)
- smap.json (ASIN to index mapping)
- amazon_native.json or amazon_basetag.json (item tags)

Into UniSRec format:
- amazon_*.train.inter
- amazon_*.valid.inter
- amazon_*.test.inter
- amazon_*.feat1CLS (BERT embeddings)
"""

import argparse
import json
import os
import collections
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_sequences(sequences_file, smap_file):
    """
    Load sequences from sequences.json and convert ASINs to integer indices

    Returns:
        user2items: dict mapping user_index to list of item_indices
        user2index: dict mapping original user_id to user_index
        item2index: dict mapping ASIN to item_index
    """
    print(f"Loading sequences from {sequences_file}...")
    with open(sequences_file, 'r', encoding='utf-8') as f:
        sequences = json.load(f)

    print(f"Loading ASIN mapping from {smap_file}...")
    with open(smap_file, 'r', encoding='utf-8') as f:
        smap = json.load(f)

    # Create ASIN to index mapping
    # smap is {"0": "ASIN1", "1": "ASIN2", ...}
    asin_to_idx = {asin: int(idx) for idx, asin in smap.items()}

    # Create user index mapping
    user2index = {user_id: idx for idx, user_id in enumerate(sequences.keys())}

    # Convert sequences to integer indices
    user2items = {}
    for user_id, asins in sequences.items():
        user_idx = user2index[user_id]
        # Convert ASINs to indices
        item_indices = [asin_to_idx[asin] for asin in asins if asin in asin_to_idx]
        if len(item_indices) >= 3:  # Need at least 3 items for train/val/test split
            user2items[user_idx] = item_indices

    print(f"  Loaded {len(user2items)} users with sequences")
    print(f"  {len(asin_to_idx)} unique items")

    return user2items, user2index, asin_to_idx


def split_train_valid_test(user2items):
    """Split into train/valid/test using leave-one-last-item"""
    print("\nSplitting train/valid/test...")
    train_inters, valid_inters, test_inters = {}, {}, {}

    for u_index in range(len(user2items)):
        if u_index not in user2items:
            continue

        items = user2items[u_index]

        # Leave last 2 items for validation and test
        train_inters[u_index] = [str(i) for i in items[:-2]]
        valid_inters[u_index] = [str(items[-2])]
        test_inters[u_index] = [str(items[-1])]

    train_count = sum(len(v) for v in train_inters.values())
    print(f"  Train: {train_count} interactions from {len(train_inters)} users")
    print(f"  Valid: {len(valid_inters)} sequences")
    print(f"  Test: {len(test_inters)} sequences")

    return train_inters, valid_inters, test_inters


def generate_item_texts(item2index, tags_file):
    """Generate text for each item by concatenating tags"""
    print(f"\nLoading tags from {tags_file}...")
    with open(tags_file, 'r', encoding='utf-8') as f:
        tags_data = json.load(f)

    # Create item_text_list in index order
    item_text_list = []

    # Tags file has integer string keys: {"0": [...], "1": [...], ...}
    for idx in range(len(item2index)):
        idx_key = str(idx)

        if idx_key in tags_data:
            tags = tags_data[idx_key]
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
    parser = argparse.ArgumentParser(description='Preprocess games (Amazon) data for UniSRec')
    parser.add_argument('--sequences_file', type=str,
                        default='data/preprocessed/games/bm25/sequences.json',
                        help='Input sequences.json file')
    parser.add_argument('--smap_file', type=str,
                        default='data/preprocessed/games/bm25/smap.json',
                        help='Input smap.json file (ASIN to index mapping)')
    parser.add_argument('--tags_file', type=str, default='tags/games/native.json',
                        help='Tag file: native.json, getag_native.json, basetag.json, etc.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated based on tag file if not specified)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset name (auto-generated based on tag file if not specified)')
    parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for BERT encoding')
    args = parser.parse_args()

    # Auto-generate dataset name and output dir based on tags file
    if args.dataset_name is None or args.output_dir is None:
        # Use the tag filename directly as suffix (e.g., native, getag_native, basetag, getag_basetag)
        tag_suffix = os.path.basename(args.tags_file).replace('.json', '')

        if args.dataset_name is None:
            args.dataset_name = f'games_{tag_suffix}'
        if args.output_dir is None:
            args.output_dir = f'data/preprocessed/UniSRec/{args.dataset_name}'

    print("="*80)
    print("PREPROCESSING AMAZON DATA FOR UNISREC")
    print("="*80)
    print(f"Sequences file: {args.sequences_file}")
    print(f"SMAP file: {args.smap_file}")
    print(f"Tags file: {args.tags_file}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"PLM: {args.plm_name}")
    print(f"Device: {args.device}")
    print("="*80)

    # Step 1: Load sequences
    user2items, user2index, item2index = load_sequences(args.sequences_file, args.smap_file)

    # Step 2: Split train/valid/test
    train_data, valid_data, test_data = split_train_valid_test(user2items)

    # Step 3: Generate item texts from tags
    item_texts = generate_item_texts(item2index, args.tags_file)

    # Step 4: Generate BERT embeddings
    embeddings = generate_bert_embeddings(
        item_texts, args.plm_name, args.device, args.batch_size
    )

    # Step 5: Save to RecBole format
    save_to_recbole_format(args.output_dir, args.dataset_name, train_data, valid_data, test_data)

    # Step 6: Save embeddings
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
