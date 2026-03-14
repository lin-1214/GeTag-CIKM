"""
Generate GeTag tags from classified session data

Supports: food, games, yelp datasets
Run from GeTag root directory

Usage:
    python getag/gen_getag.py --dataset food --base_tag native --classified_csv data/classified/food_native.csv
    python getag/gen_getag.py --dataset games --base_tag basetag --classified_csv data/classified/games_basetag.csv
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import ast
import os
import re
import argparse
from itertools import combinations
from pathlib import Path

# CLI arguments
parser = argparse.ArgumentParser(description='Generate GETags from classified CSV')
parser.add_argument('--dataset', type=str, required=True, choices=['food', 'games', 'yelp'],
                    help='Dataset name')
parser.add_argument('--base_tag', type=str, required=True, choices=['native', 'basetag', 'betags'],
                    help='Base tag type to use as input')
parser.add_argument('--classified_csv', type=str, required=True,
                    help='Path to classified CSV file')
parser.add_argument('--output_dir', type=str, default='tags',
                    help='Output directory for generated tags (default: tags/)')
parser.add_argument('--no_zscore_filter', action='store_true',
                    help='Disable z-score filtering (include all tags regardless of z-score)')
parser.add_argument('--zscore_threshold', type=float, default=None,
                    help='Custom z-score threshold (e.g., -2, -1, 0, 1, 2). Overrides dataset default.')
args = parser.parse_args()

# Dataset-specific configurations
DATASET_CONFIG = {
    'food': {
        'mapping_file': 'data/mappings/food_name_to_id.json',
        'use_smap': False,  # Uses name-to-id mapping
        'level_tags': ["偏好"],  # Chinese
        'z_score_thresholds': [2.0],
        'enable_composite_tags': True,
        'min_session_support': 40,
        'max_composite_patterns': 3,
        'max_pattern_size': 3,
    },
    'games': {
        'mapping_file': 'data/preprocessed/games/bm25/smap.json',
        'use_smap': True,  # Uses index-to-ASIN mapping
        'level_tags': [" Preference"],  # English
        'z_score_thresholds': [0.0],
        'enable_composite_tags': True,
        'min_session_support': 20,
        'max_composite_patterns': 10,
        'max_pattern_size': 3,
    },
    'yelp': {
        'mapping_file': 'data/preprocessed/yelp/bm25/smap.json',
        'use_smap': True,  # Uses index-to-business_id mapping
        'level_tags': [" Preference"],  # English
        'z_score_thresholds': [0.0],
        'enable_composite_tags': False,  # Disabled for yelp
        'min_session_support': 20,
        'max_composite_patterns': 10,
        'max_pattern_size': 3,
    },
}

# Get dataset config
config = DATASET_CONFIG[args.dataset]

# Set up paths
base_dir = Path('.')
CLASSIFIED_DATA_PATH = args.classified_csv
MAPPING_PATH = base_dir / config['mapping_file']
BASE_TAGS_PATH = base_dir / f'tags/{args.dataset}/{args.base_tag}.json'

# Add suffix for nozscore variant or custom threshold
if args.no_zscore_filter:
    output_suffix = f'getag_{args.base_tag}_nozscore'
elif args.zscore_threshold is not None:
    # Format threshold: z1.0, z-1.5, z0.0, etc.
    thresh_str = f'z{args.zscore_threshold:g}'  # :g removes trailing zeros
    output_suffix = f'getag_{args.base_tag}_{thresh_str}'
else:
    output_suffix = f'getag_{args.base_tag}'
OUTPUT_TAG_PATH = base_dir / args.output_dir / args.dataset / f'{output_suffix}.json'
OUTPUT_ITEM_GROUP_MAPPING_PATH = base_dir / args.output_dir / args.dataset / f'item_group_mapping_{output_suffix}.json'
OUTPUT_GROUP_TAG_FREQ_PATH = base_dir / args.output_dir / args.dataset / f'group_tag_frequency_{output_suffix}.json'

# Create output directories
OUTPUT_TAG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Dataset-specific settings from config
STRONG_PERCENTILE = 60
LEVEL_TAGS = config['level_tags']
# Determine z-score threshold: custom > no_filter > dataset default
if args.no_zscore_filter:
    Z_SCORE_THRESHOLDS = [float('-inf')]
elif args.zscore_threshold is not None:
    Z_SCORE_THRESHOLDS = [args.zscore_threshold]
else:
    Z_SCORE_THRESHOLDS = config['z_score_thresholds']
ENABLE_COMPOSITE_TAGS = config['enable_composite_tags']
MIN_SESSION_SUPPORT = config['min_session_support']
MAX_COMPOSITE_PATTERNS = config['max_composite_patterns']
MAX_PATTERN_SIZE = config['max_pattern_size']

print("=" * 80)
print(f"GENERATE GETAGS FROM CLASSIFIED {args.dataset.upper()} DATA")
print("=" * 80)
print(f"Dataset: {args.dataset}")
print(f"Base tags: {args.base_tag}")
if args.no_zscore_filter:
    print(f"Z-score filtering: DISABLED")
elif args.zscore_threshold is not None:
    print(f"Z-score threshold: {args.zscore_threshold} (custom)")
else:
    print(f"Z-score threshold: {config['z_score_thresholds'][0]} (dataset default)")
print(f"Classified CSV: {CLASSIFIED_DATA_PATH}")
print(f"Output tag file: {OUTPUT_TAG_PATH}")
print("=" * 80)

# Step 1: Load item mapping (name-to-id or smap)
if config['use_smap']:
    print(f"\nLoading {args.dataset} smap (index → item ID)...")
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        smap = json.load(f)  # {"0": "ASIN/business_id", "1": ...}
    # Create reverse mapping: item_id → index
    name_to_id = {item_id: idx for idx, item_id in smap.items()}
    print(f"✓ Loaded {len(smap)} item mappings")
else:
    print(f"\nLoading {args.dataset} name-to-ID mapping...")
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        name_to_id = json.load(f)
    print(f"✓ Loaded {len(name_to_id)} product mappings")

# Step 2: Load base tags
print(f"\nLoading base tags from {args.base_tag}...")
with open(BASE_TAGS_PATH, 'r', encoding='utf-8') as f:
    base_tags_v2 = json.load(f)
print(f"✓ Loaded base tags for {len(base_tags_v2)} items")

# Step 3: Read classified data and build item-group mapping
print("\nReading classified data...")
# Read with low_memory=False to prevent mixed type inference
df = pd.read_csv(CLASSIFIED_DATA_PATH, low_memory=False)
print(f"✓ Loaded {len(df)} sessions")

# Step 4: Extract tags and product IDs from each session
print("\nBuilding item-group tag mapping...")
item_group_counts = defaultdict(lambda: defaultdict(int))  # item_id -> {tag: count}
group_tag_freq = defaultdict(int)  # tag -> total_frequency

for idx, row in df.iterrows():
    if idx % 100 == 0:
        print(f"  Processing session {idx}/{len(df)}...", end='\r')

    # Parse the Class column (contains tags)
    tags_str = row['Class']
    try:
        tags = ast.literal_eval(tags_str)
        if not isinstance(tags, list):
            tags = []
    except:
        tags = []

    # Extract product IDs from the session events
    product_ids = set()
    for col in row.index:
        if col == 'Class':
            continue
        cell_value = row[col]
        if pd.isna(cell_value):
            continue

        # Cell contains product name directly
        product_name = str(cell_value).strip()
        if product_name == '' or product_name == 'nan':
            continue

        # Convert product name to ID
        if product_name in name_to_id:
            product_id = str(name_to_id[product_name])
            product_ids.add(product_id)

    # Count each tag for each product ID in this session
    for product_id in product_ids:
        for tag in tags:
            item_group_counts[product_id][tag] += 1
            group_tag_freq[tag] += 1

print(f"\n  Finished processing {len(df)} sessions")

# Convert to regular dict for JSON serialization
item_group_mapping = {k: dict(v) for k, v in item_group_counts.items()}
group_tag_frequency = dict(group_tag_freq)

print(f"✓ Processed {len(item_group_mapping)} unique products")
print(f"✓ Found {len(group_tag_frequency)} unique group tags")

# Step 4.5: Filter sparse group tags using first quartile (Q1)
print("\n" + "=" * 80)
print("FILTERING SPARSE GROUP TAGS (SESSION COUNT THRESHOLD)")
print("=" * 80)

frequencies = np.array(list(group_tag_frequency.values()))

# Use absolute threshold: total_sessions / 20
lower_bound = len(df) / 20
upper_bound = np.percentile(frequencies, STRONG_PERCENTILE)

print(f"\nAbsolute threshold-based filtering:")
print(f"  Total sessions: {len(df)}")
print(f"  Minimum threshold (sessions/20): {lower_bound:.1f}")
print(f"  P60 (60th percentile): {upper_bound:.1f}")
print(f"  Tags below {lower_bound:.1f} sessions are considered too sparse")
print(f"  Tags above P60 are considered strong")

# Identify sparse tags to filter
sparse_tags = {tag: freq for tag, freq in group_tag_frequency.items() if freq < lower_bound}
reliable_tags = {tag: freq for tag, freq in group_tag_frequency.items() if freq >= lower_bound}
strong_tags = {tag: freq for tag, freq in group_tag_frequency.items() if freq >= upper_bound}

print(f"\nFiltering results:")
print(f"  Sparse tags (to remove): {len(sparse_tags)}")
for tag, freq in sorted(sparse_tags.items(), key=lambda x: x[1]):
    print(f"    - {tag}: {freq} sessions")
print(f"  Reliable tags (to keep): {len(reliable_tags)}")

# Filter item_group_mapping to only include reliable tags
filtered_mapping = {}
for item_id, group_counts in item_group_mapping.items():
    filtered_counts = {
        tag: count for tag, count in group_counts.items()
        if tag in reliable_tags
    }
    if filtered_counts:  # Only keep items that have at least one reliable tag
        filtered_mapping[item_id] = filtered_counts

item_group_mapping = filtered_mapping
group_tag_frequency = reliable_tags

print(f"\n✓ After filtering:")
print(f"  Items with reliable group tags: {len(item_group_mapping)}")
print(f"  Reliable group tags: {len(group_tag_frequency)}")
print("=" * 80)

# Step 5: Save item_group_mapping and group_tag_frequency
print(f"\nSaving item-group mapping to {OUTPUT_ITEM_GROUP_MAPPING_PATH}...")
os.makedirs(os.path.dirname(OUTPUT_ITEM_GROUP_MAPPING_PATH), exist_ok=True)
with open(OUTPUT_ITEM_GROUP_MAPPING_PATH, 'w', encoding='utf-8') as f:
    json.dump(item_group_mapping, f, ensure_ascii=False, indent=2)

print(f"Saving group tag frequency to {OUTPUT_GROUP_TAG_FREQ_PATH}...")
with open(OUTPUT_GROUP_TAG_FREQ_PATH, 'w', encoding='utf-8') as f:
    json.dump(group_tag_frequency, f, ensure_ascii=False, indent=2)

# Step 6: Generate getags using z-score (similar to analyze_and_generate_zscore_tags.py)
print("\n" + "=" * 80)
print("GENERATING GETAGS WITH Z-SCORE")
print("=" * 80)

# Organize data by group tag
print("\nOrganizing data by group tag...")
group_data = defaultdict(list)  # group_tag -> [counts]

for item_id, group_counts in item_group_mapping.items():
    for group_tag, count in group_counts.items():
        if group_tag in group_tag_frequency:
            group_data[group_tag].append(count)

# Calculate statistics for each group tag
print("\nCalculating statistics for each group tag...")
group_stats = {}
for group_tag, counts in group_data.items():
    counts_array = np.array(counts)
    group_stats[group_tag] = {
        'mean': np.mean(counts_array),
        'std': np.std(counts_array),
        'min': np.min(counts_array),
        'max': np.max(counts_array),
        'median': np.median(counts_array),
        'count': len(counts_array)
    }

# Print statistics
print("\n" + "=" * 80)
print("GROUP TAG STATISTICS")
print("=" * 80)
print(f"{'Group Tag':<40} {'Mean':<8} {'Std':<8} {'Min':<6} {'Max':<6} {'Items':<6}")
print("-" * 80)
for group_tag in sorted(group_stats.keys(), key=lambda x: group_stats[x]['mean'], reverse=True):
    stats = group_stats[group_tag]
    print(f"{group_tag:<40} {stats['mean']:>7.2f} {stats['std']:>7.2f} {stats['min']:>5.0f} {stats['max']:>5.0f} {stats['count']:>5.0f}")

# Define z-score level function
def get_level_from_zscore(z):
    """
    Determine level based on z-score dynamically

    Returns the appropriate level tag based on Z_SCORE_THRESHOLDS and LEVEL_TAGS
    """
    # Below the first threshold, don't add
    if z < Z_SCORE_THRESHOLDS[0]:
        return None

    # Find the appropriate level
    for i in range(len(Z_SCORE_THRESHOLDS) - 1):
        if z < Z_SCORE_THRESHOLDS[i + 1]:
            return LEVEL_TAGS[i]

    # If z is greater than or equal to the last threshold
    return LEVEL_TAGS[-1]

# Calculate z-scores and generate tags with top-K selection
print("\nCalculating z-scores and generating tags with top-K selection...")

# Configuration for top-K selection
MAX_TAGS_PER_ITEM = 6  # Maximum tags per item
FREQUENCY_PENALTY_WEIGHT = 0.3  # Weight for frequency adjustment

# First, calculate item frequencies (total co-occurrence count)
item_frequencies = {}
for item_id, group_counts in item_group_mapping.items():
    item_frequencies[item_id] = sum(group_counts.values())

# Calculate frequency statistics for adaptive scoring
freq_values = np.array(list(item_frequencies.values()))
freq_median = np.median(freq_values)
freq_q75 = np.percentile(freq_values, 75)
freq_q25 = np.percentile(freq_values, 25)

print(f"Item frequency statistics:")
print(f"  Median: {freq_median:.1f}")
print(f"  Q1 (25th): {freq_q25:.1f}")
print(f"  Q3 (75th): {freq_q75:.1f}")
print(f"Maximum tags per item: {MAX_TAGS_PER_ITEM}")

group_enhanced_tags = defaultdict(list)  # item_id -> [enhanced_tags]
items_processed = 0
items_with_enhanced_tags = 0
total_enhanced_tags = 0
level_counts = defaultdict(int)

for item_id, group_counts in item_group_mapping.items():
    items_processed += 1

    # Calculate frequency-aware scores for all tags
    tag_scores = []  # List of (group_tag, z_score, adjusted_score, level)

    for group_tag, count in group_counts.items():
        if group_tag not in group_stats:
            continue

        # Calculate z-score
        mean = group_stats[group_tag]['mean']
        std = group_stats[group_tag]['std']

        if std == 0:  # Avoid division by zero
            z = 0
        else:
            z = (count - mean) / std

        # Get level
        level = get_level_from_zscore(z)

        if level is not None:
            # Calculate frequency adjustment factor
            # Popular items (high frequency) get penalty, rare items get bonus
            item_freq = item_frequencies[item_id]

            if item_freq > freq_q75:
                # High frequency: increase threshold (penalty)
                freq_adjustment = -(item_freq - freq_q75) / (freq_q75 - freq_median) * FREQUENCY_PENALTY_WEIGHT
            elif item_freq < freq_q25:
                # Low frequency: decrease threshold (bonus)
                freq_adjustment = (freq_q25 - item_freq) / (freq_median - freq_q25) * FREQUENCY_PENALTY_WEIGHT
            else:
                # Medium frequency: no adjustment
                freq_adjustment = 0

            # Adjusted score: higher is better
            # For high-freq items, need higher z-score to compensate for penalty
            adjusted_score = z + freq_adjustment

            tag_scores.append((group_tag, z, adjusted_score, level))

    # Sort by adjusted score and select top K
    tag_scores.sort(key=lambda x: x[2], reverse=True)
    selected_tags = tag_scores[:MAX_TAGS_PER_ITEM]

    # Add selected tags
    for group_tag, z, adj_score, level in selected_tags:
        enhanced_tag = f"{group_tag}{level}"
        group_enhanced_tags[item_id].append(enhanced_tag)
        total_enhanced_tags += 1
        level_counts[level] += 1

    if group_enhanced_tags[item_id]:
        items_with_enhanced_tags += 1

print(f"✓ Processed {items_processed} items")
print(f"✓ {items_with_enhanced_tags} items have group-enhanced tags")
print(f"✓ Generated {total_enhanced_tags} total enhanced tags (no suffix)")

# Step 6.5: Generate composite tags from co-occurring patterns
if ENABLE_COMPOSITE_TAGS:
    print("\n" + "=" * 80)
    print("GENERATING COMPOSITE TAGS FROM CO-OCCURRING PATTERNS")
    print("=" * 80)

    # Extract frequent tag pairs from sessions
    print(f"\nAnalyzing session-level co-occurrence patterns...")
    session_pair_counts = Counter()
    session_tag_counts = Counter()

    for idx, row in df.iterrows():
        tags_str = row['Class']
        try:
            tags = ast.literal_eval(tags_str)
            if not isinstance(tags, list):
                tags = []
        except:
            tags = []

        # Filter to only reliable tags
        tags = [t for t in tags if t in group_tag_frequency]

        # Count individual tags
        for tag in tags:
            session_tag_counts[tag] += 1

        # Count patterns of different sizes (2, 3, up to MAX_PATTERN_SIZE)
        for size in range(2, min(MAX_PATTERN_SIZE + 1, len(tags) + 1)):
            for pattern in combinations(sorted(tags), size):
                session_pair_counts[pattern] += 1

    # Select frequent patterns
    all_frequent_patterns = [pattern for pattern, count in session_pair_counts.items()
                             if count >= MIN_SESSION_SUPPORT]

    # Sort by frequency and select top K
    sorted_patterns = sorted(session_pair_counts.items(), key=lambda x: -x[1])
    top_frequent_patterns = [pattern for pattern, count in sorted_patterns
                             if count >= MIN_SESSION_SUPPORT][:MAX_COMPOSITE_PATTERNS]

    frequent_pairs = top_frequent_patterns

    print(f"✓ Found {len(all_frequent_patterns)} frequent patterns (support ≥ {MIN_SESSION_SUPPORT})")
    print(f"✓ Selected top {len(frequent_pairs)} most frequent patterns")
    print(f"\nTop {len(frequent_pairs)} frequent patterns:")
    for pattern, count in sorted_patterns[:len(frequent_pairs)]:
        support = count / len(df) * 100
        pattern_str = " + ".join(pattern)
        print(f"  {count:3d} sessions ({support:5.2f}%) [{len(pattern)} tags]: {pattern_str}")

    # Count item-pattern co-occurrences
    print(f"\nCounting item-pattern co-occurrences...")
    item_pattern_counts = defaultdict(lambda: defaultdict(int))

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing session {idx}/{len(df)}...", end='\r')

        # Parse session tags
        tags_str = row['Class']
        try:
            session_tags = ast.literal_eval(tags_str)
            if not isinstance(session_tags, list):
                session_tags = []
        except:
            session_tags = []

        session_tags = [t for t in session_tags if t in group_tag_frequency]

        # Extract product IDs from session
        product_ids = set()
        for col in row.index:
            if col == 'Class':
                continue
            cell_value = row[col]
            if pd.isna(cell_value) or cell_value == '':
                continue

            # Try plain product name first (for new CSV format)
            product_name = str(cell_value).strip()
            if product_name and product_name in name_to_id:
                product_id = str(name_to_id[product_name])
                product_ids.add(product_id)
                continue

            # Fall back to regex parsing (for old CSV format with quotes)
            try:
                pattern = r"'([^']*)'|\"([^\"]*)\""
                matches = re.findall(pattern, str(cell_value))
                strings = [m[0] if m[0] else m[1] for m in matches]

                if len(strings) >= 2:
                    product_name = strings[-1]
                    if product_name and product_name in name_to_id:
                        product_id = str(name_to_id[product_name])
                        product_ids.add(product_id)
            except:
                continue

        # Check which frequent patterns appear in this session
        for pattern in frequent_pairs:
            if all(tag in session_tags for tag in pattern):
                # This session exhibits the pattern
                for product_id in product_ids:
                    item_pattern_counts[product_id][pattern] += 1

    print(f"\n  Finished processing {len(df)} sessions")

    # Debug: Show item_pattern_counts statistics
    print(f"\n  DEBUG: Items with pattern associations: {len(item_pattern_counts)}")
    print(f"  DEBUG: Total items in catalog: {len(item_group_mapping)}")

    if frequent_pairs:
        pattern = frequent_pairs[0]
        counts = [item_pattern_counts[item_id][pattern] for item_id in item_pattern_counts.keys()]
        print(f"\n  DEBUG: Statistics for first pattern {pattern}:")
        print(f"    Items in item_pattern_counts: {len(counts)}")
        print(f"    Items with non-zero count: {sum(1 for c in counts if c > 0)}")
        print(f"    Max count: {max(counts) if counts else 0}")
        print(f"    Mean: {np.mean(counts) if counts else 0:.2f}")
        print(f"    Std: {np.std(counts) if counts else 0:.2f}")

        # Show distribution of counts
        count_dist = Counter(counts)
        print(f"    Count distribution (top 10):")
        for count, freq in sorted(count_dist.items(), reverse=True)[:10]:
            print(f"      {count} items appeared {freq} times")

    # Calculate z-scores for composite patterns
    print(f"\nCalculating z-scores for composite patterns...")
    composite_enhanced_tags = defaultdict(list)
    composite_tag_stats = {}

    for pattern in frequent_pairs:
        counts = []
        for item_id in item_pattern_counts.keys():
            counts.append(item_pattern_counts[item_id][pattern])

        if len(counts) > 0:
            counts_array = np.array(counts)
            mean = np.mean(counts_array)
            std = np.std(counts_array)

            composite_tag_stats[pattern] = {
                'mean': mean,
                'std': std,
                'min': np.min(counts_array),
                'max': np.max(counts_array),
                'count': len(counts_array)
            }

    # Generate composite tags with z-scores
    composite_tags_added = 0
    items_with_composite = 0
    composite_level_counts = defaultdict(int)

    # Debug: Track z-score distribution
    all_zscores = []

    for item_id in item_pattern_counts.keys():
        item_has_composite = False

        for pattern in frequent_pairs:
            if pattern not in composite_tag_stats:
                continue

            count = item_pattern_counts[item_id][pattern]
            mean = composite_tag_stats[pattern]['mean']
            std = composite_tag_stats[pattern]['std']

            if std == 0:
                z = 0
            else:
                z = (count - mean) / std

            all_zscores.append(z)

            # Get level using same function
            level = get_level_from_zscore(z)

            if level is not None:
                composite_tag = f"{'+'.join(pattern)}{level}"
                composite_enhanced_tags[item_id].append(composite_tag)
                composite_tags_added += 1
                composite_level_counts[level] += 1
                item_has_composite = True

        if item_has_composite:
            items_with_composite += 1

    # Debug: Show z-score statistics
    if all_zscores:
        print(f"\n  DEBUG: Z-score distribution:")
        print(f"    Total z-scores calculated: {len(all_zscores)}")
        print(f"    Min z-score: {min(all_zscores):.2f}")
        print(f"    Max z-score: {max(all_zscores):.2f}")
        print(f"    Mean z-score: {np.mean(all_zscores):.2f}")
        print(f"    Z-scores >= 0.0: {sum(1 for z in all_zscores if z >= 0.0)}")
        print(f"    Z-scores >= 1.0: {sum(1 for z in all_zscores if z >= 1.0)}")
        print(f"    Z-scores >= 2.0: {sum(1 for z in all_zscores if z >= 2.0)}")
        print(f"    Current threshold: {Z_SCORE_THRESHOLDS[0]}")

    print(f"✓ Generated {composite_tags_added} composite tags")
    print(f"✓ {items_with_composite} items have composite tags")
    print(f"\nComposite tag level distribution:")
    for level in LEVEL_TAGS:
        count = composite_level_counts[level]
        percentage = (count / composite_tags_added * 100) if composite_tags_added > 0 else 0
        print(f"  {level}: {count} tags ({percentage:.1f}%)")

    # Merge composite tags with group tags
    print(f"\nMerging composite tags with group tags...")
    for item_id, comp_tags in composite_enhanced_tags.items():
        if item_id in group_enhanced_tags:
            group_enhanced_tags[item_id].extend(comp_tags)
        else:
            group_enhanced_tags[item_id] = comp_tags

    total_enhanced_tags += composite_tags_added
    print(f"✓ Total tags after adding composites: {total_enhanced_tags}")

# Combine with base tags
print(f"\nCombining with {args.base_tag} tags...")
getags_zscore = {}

for item_id, base_tags in base_tags_v2.items():
    all_tags = list(base_tags) if isinstance(base_tags, list) else []

    if item_id in group_enhanced_tags:
        all_tags.extend(group_enhanced_tags[item_id])

    getags_zscore[item_id] = all_tags

print(f"✓ Combined tags for {len(getags_zscore)} items")

# Save getags_zscore.json
print(f"\nSaving to {OUTPUT_TAG_PATH}...")
OUTPUT_TAG_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_TAG_PATH, 'w', encoding='utf-8') as f:
    json.dump(getags_zscore, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Base tags source: {args.base_tag}")
print(f"Items in {args.base_tag} tags: {len(base_tags_v2)}")
print(f"Items with z-score enhanced tags: {items_with_enhanced_tags}")
print(f"Total z-score enhanced tags: {total_enhanced_tags}")
print(f"\nFiltering statistics:")
print(f"  Lower bound (Q1): {lower_bound:.1f}")
print(f"  Upper bound (P60): {upper_bound:.1f}")
print(f"  Sparse tags removed: {len(sparse_tags)}")
print(f"  Reliable tags retained: {len(group_tag_frequency)}")
print(f"  Strong tags identified: {len(strong_tags)}")
if len(sparse_tags) > 0:
    removed_tag_list = [tag for tag, freq in sorted(sparse_tags.items(), key=lambda x: x[1])]
    print(f"\nRemoved sparse tags: {removed_tag_list}")
if len(strong_tags) > 0:
    strong_tag_list = [tag for tag, freq in sorted(strong_tags.items(), key=lambda x: -x[1])]
    print(f"Strong tags: {strong_tag_list}")
print(f"\nOutput files:")
print(f"  1. {OUTPUT_ITEM_GROUP_MAPPING_PATH}")
print(f"  2. {OUTPUT_GROUP_TAG_FREQ_PATH}")
print(f"  3. {OUTPUT_TAG_PATH}")
print("=" * 80)
print("\n✓ GeTag generation completed successfully!")
