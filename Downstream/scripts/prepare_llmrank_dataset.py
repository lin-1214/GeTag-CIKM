"""
Universal data preparation script for LLMRank
Supports: food, games, yelp datasets with native, basetag, getag_native, getag_basetag tags

Run from GeTag root directory.
"""

import json
import argparse
import os
from pathlib import Path


def load_data(base_dir, dataset_name, tag_name):
    """Load data files for any dataset/tag combination"""

    # Load product/item name to ID mapping
    if dataset_name == 'food':
        mapping_file = 'data/mappings/food_name_to_id.json'
        eval_file = 'data/raw/food/evaluation.csv'
        title_mapping_file = None  # Food uses product names directly
    elif dataset_name == 'games':
        mapping_file = 'data/preprocessed/games/bm25/smap.json'  # index → ASIN
        eval_file = 'data/raw/games/evaluation.csv'
        title_mapping_file = None  # Games uses AS INs
    elif dataset_name == 'yelp':
        mapping_file = 'data/preprocessed/yelp/bm25/smap.json'  # index → Business_id
        eval_file = 'data/raw/yelp/evaluation.csv'
        title_mapping_file = None  # Yelp uses business IDs
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load mappings
    with open(os.path.join(base_dir, mapping_file), 'r', encoding='utf-8') as f:
        name_to_id = json.load(f)

    # For games/yelp: smap.json is index → name, but we need name → index for evaluation.csv lookup
    if dataset_name in ['games', 'yelp']:
        name_to_id = {v: k for k, v in name_to_id.items()}

    # Load title mapping for Amazon (ASIN → readable title)
    asin_to_title = None
    if title_mapping_file:
        with open(os.path.join(base_dir, title_mapping_file), 'r', encoding='utf-8') as f:
            asin_to_title = json.load(f)

    # Load tags
    tag_file = f'tags/{dataset_name}/{tag_name}.json'
    with open(os.path.join(base_dir, tag_file), 'r', encoding='utf-8') as f:
        id_to_tags = json.load(f)

    # Load evaluation data
    with open(os.path.join(base_dir, eval_file), 'r', encoding='utf-8') as f:
        eval_sessions = [line.strip().split(',') for line in f]

    return name_to_id, id_to_tags, eval_sessions, asin_to_title


def create_item_file(name_to_id, id_to_tags, output_path, asin_to_title=None, max_tags=None):
    """Create RecBole .item file

    Args:
        asin_to_title: Optional mapping from ASIN to readable product title (for Amazon)
        max_tags: Maximum number of NATIVE tags per item (None = no limit).
                  ALL preference tags are always included.
    """
    print(f"Creating item file: {output_path}")

    # Create reverse mapping (id to name/ASIN)
    id_to_name = {str(v): k for k, v in name_to_id.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("item_id:token\titem_name:token_seq\ttags:token_seq\n")

        # Write each item
        for item_id in sorted(id_to_tags.keys(), key=lambda x: int(x)):
            asin_or_name = id_to_name.get(item_id, f"Item_{item_id}")

            # For Amazon, convert ASIN to readable title
            if asin_to_title:
                item_name = asin_to_title.get(asin_or_name, asin_or_name)
            else:
                item_name = asin_or_name

            tags = id_to_tags[item_id]

            # Separate preference tags from native tags
            preference_tags = [t for t in tags if '偏好' in t or ' Preference' in t or 'preference' in t.lower()]
            native_tags = [t for t in tags if t not in preference_tags]

            # Apply max_tags limit ONLY to native tags, keep ALL preference tags
            if max_tags is not None:
                limited_native_tags = native_tags[:max_tags]
            else:
                limited_native_tags = native_tags

            # Put ALL preference tags first, then limited native tags
            reordered_tags = preference_tags + limited_native_tags
            tags_str = ' '.join(reordered_tags)

            f.write(f"{item_id}\t{item_name}\t{tags_str}\n")

    print(f"Created item file with {len(id_to_tags)} items")
    if max_tags is not None:
        print(f"  ✓ Using ALL preference tags + max {max_tags} native tags per item")
    else:
        print(f"  ✓ Using all tags per item (no limit)")
    if asin_to_title:
        print(f"  ✓ Using product titles instead of ASINs for better LLM understanding")


def create_inter_file(sessions, name_to_id, output_path, split_name='train', min_interactions=2):
    """Create RecBole .inter file with individual interactions (not sessions)

    Args:
        min_interactions: Minimum number of interactions per user (default 2 for train/test split)
    """
    print(f"Creating {split_name} interaction file: {output_path}")

    interactions = []
    skipped_sessions = 0
    user_interaction_count = {}
    current_user_id = 0  # Contiguous user ID counter

    for session_idx, session in enumerate(sessions):
        # Convert item names to IDs
        try:
            item_ids = [str(name_to_id[item_name]) for item_name in session]

            # Need at least min_interactions for the split to work
            if len(item_ids) >= min_interactions:
                # Create one interaction per item in the session
                for item_position, item_id in enumerate(item_ids):
                    interactions.append({
                        'user_id': current_user_id,  # Use contiguous user ID
                        'item_id': item_id,
                        'timestamp': item_position
                    })
                user_interaction_count[current_user_id] = len(item_ids)
                current_user_id += 1  # Increment for next valid user
            else:
                skipped_sessions += 1
        except KeyError:
            # Skip sessions with items not in mapping
            skipped_sessions += 1
            continue

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header (standard RecBole format for sequential data)
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")

        # Write each interaction
        for interaction in interactions:
            user_id = interaction['user_id']
            item_id = interaction['item_id']
            timestamp = interaction['timestamp']
            f.write(f"{user_id}\t{item_id}\t{timestamp}\n")

    # Get list of valid user IDs that made it into the dataset
    valid_user_ids = sorted(set(i['user_id'] for i in interactions))

    print(f"Created {split_name} file with {len(interactions)} interactions from {len(valid_user_ids)} sessions")
    print(f"Skipped {skipped_sessions} sessions (< {min_interactions} interactions or missing items)")
    return len(interactions), valid_user_ids


def create_dataset_config(output_path):
    """Create dataset configuration YAML file"""
    config_content = """load_col:
    inter: [user_id, item_id, timestamp]
eval_args:
    split: {'LS': 'test_only'}
    order: TO
    group_by: user
    mode: full
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"Created config file: {output_path}")


def create_random_candidates(output_path, valid_user_ids, num_items, num_candidates=100):
    """Create random candidate items file for LLMRank"""
    import random

    print(f"Generating {num_candidates} random candidates for {len(valid_user_ids)} users from {num_items} items...")

    # Item IDs start from 1 (0 is PAD)
    all_items = list(range(1, num_items + 1))

    with open(output_path, 'w', encoding='utf-8') as f:
        for user_id in valid_user_ids:
            # Randomly sample candidates for this user
            candidates = random.sample(all_items, min(num_candidates, len(all_items)))
            # Write as: user_id\titem1 item2 item3 ...
            f.write(f"{user_id}\t{' '.join(map(str, candidates))}\n")

    print(f"Created candidate file: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare LLMRank dataset for any dataset/tag combination')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['food', 'games', 'yelp'],
                       help='Dataset name')
    parser.add_argument('--tag', type=str, required=True,
                       choices=['native', 'basetag', 'betag', 'getag_native', 'getag_basetag', 'getag_betag'],
                       help='Tag variant')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train/test split ratio (default: 0.8)')
    parser.add_argument('--num_sessions', type=int, default=None,
                       help='Limit number of sessions (for testing, e.g., 100)')
    parser.add_argument('--max_tags', type=int, default=None,
                       help='Maximum NATIVE tags per item (default: None = no limit, use all tags). ALL preference tags are always included.')

    args = parser.parse_args()

    # Set paths
    base_dir = Path(__file__).parent.parent
    dataset_tag_name = f"{args.dataset}_{args.tag}"
    output_dir = base_dir / f'downstream/LLMRank/llmrank/dataset/{dataset_tag_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Preparing LLMRank Dataset: {dataset_tag_name}")
    print("=" * 80)

    # Load data
    print(f"\n[1/3] Loading data for {args.dataset} with {args.tag} tags...")
    try:
        name_to_id, id_to_tags, eval_sessions, asin_to_title = load_data(base_dir, args.dataset, args.tag)

        # Limit sessions if specified
        if args.num_sessions:
            eval_sessions = eval_sessions[:args.num_sessions]
            print(f"  - Limited to {args.num_sessions} sessions for testing")

        print(f"  - Loaded {len(name_to_id)} item mappings")
        print(f"  - Loaded {len(id_to_tags)} items with tags")
        print(f"  - Loaded {len(eval_sessions)} evaluation sessions")
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        print(f"Please ensure the following files exist:")
        print(f"  - tags/{args.dataset}/{args.tag}.json")
        print(f"  - data/raw/{args.dataset}/evaluation.csv")
        if args.dataset == 'food':
            print(f"  - data/mappings/food_name_to_id.json")
        else:
            print(f"  - data/preprocessed/{args.dataset}/bm25/smap.json")
        return

    # Create item file
    print(f"\n[2/3] Creating item file...")
    item_path = output_dir / f'{dataset_tag_name}.item'
    create_item_file(name_to_id, id_to_tags, item_path, asin_to_title, max_tags=args.max_tags)

    # Create interaction file (RecBole will handle splitting)
    print(f"\n[3/3] Creating interaction file...")
    inter_path = output_dir / f'{dataset_tag_name}.inter'
    n_interactions, valid_user_ids = create_inter_file(eval_sessions, name_to_id, inter_path, 'all')

    # Create dataset config file
    config_path = base_dir / f'downstream/LLMRank/llmrank/props/{dataset_tag_name}.yaml'
    create_dataset_config(config_path)

    # Create candidate file for random sampling
    print(f"\n[4/4] Creating random candidate file...")
    candidate_path = output_dir / f'{dataset_tag_name}.random'
    num_items = len(id_to_tags)
    create_random_candidates(candidate_path, valid_user_ids, num_items, num_candidates=100)

    print("\n" + "=" * 80)
    print("Data preparation completed!")
    print("=" * 80)
    print(f"Dataset: {dataset_tag_name}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {item_path.name}: {len(id_to_tags)} items")
    print(f"  - {inter_path.name}: {n_interactions} interactions from {len(valid_user_ids)} sessions")
    print(f"  - Config: downstream/LLMRank/llmrank/props/{dataset_tag_name}.yaml")
    print()
    print("Note: RecBole will automatically split the data according to props/overall.yaml")
    print()
    print("Ready to run:")
    print(f"  cd downstream/LLMRank/llmrank && python evaluate.py -m Rank -d {dataset_tag_name}")
    print()


if __name__ == '__main__':
    main()
