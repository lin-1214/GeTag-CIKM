"""
Generate getags for the dataset, refer to the content in ../BETag-master
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import ast
import os
import re
import matplotlib.pyplot as plt
import argparse
from itertools import combinations
from config import Config

# Configure matplotlib to display Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STHeiti', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# CLI arguments
parser = argparse.ArgumentParser(description='Generate GETags from classified CSV')
parser.add_argument('--classified_csv', type=str, default='../label_data/classified_data_0.csv', help='Path to classified CSV')
parser.add_argument('--tag_name', type=str, default='getags_zscore', help='Base name for output tag JSON (without extension)')
args = parser.parse_args()

config = Config()
DOMAIN = getattr(config, "DOMAIN", "food").lower()

# Paths
CLASSIFIED_DATA_PATH = args.classified_csv
if DOMAIN == "movie":
    NAME_TO_ID_PATH = '../json/movie_name_to_id.json'
    BASE_TAGS_PATH = '../json/tags/movie_base_tags.json'
    NATIVE_TAGS_PATH = '../json/tags/movie_native_tags.json'
else:
    NAME_TO_ID_PATH = '../json/product_name_to_id.json'
    #BASE_TAGS_PATH = '../json/tags/base_tags_v2.json'
    #NATIVE_TAGS_PATH = '../json/tags/keywords_v2.json'
    BASE_TAGS_PATH = '../json/tags/keywords_v2.json'

OUTPUT_ITEM_GROUP_MAPPING_PATH = '../json/item_group_mapping.json'
OUTPUT_GROUP_TAG_FREQ_PATH = '../json/group_tag_frequency.json'
OUTPUT_GETAGS_ZSCORE_PATH = f"../json/tags/{args.tag_name}.json"
PLOT_DIR = '../statistics'

LEVEL_TAGS = ["偏好"]
Z_SCORE_THRESHOLDS = [0.0]

# Sparse/Strong tag filtering thresholds
SPARSE_PERCENTILE = 25  # Q1 - Remove bottom 25% sparse tags
STRONG_PERCENTILE = 60  # P60 - Identify top 40% strong tags

# Composite tag settings
ENABLE_COMPOSITE_TAGS = False
MIN_SESSION_SUPPORT = 20  # Minimum sessions for frequent pattern

print("=" * 80)
print("GENERATE GETAGS FROM CLASSIFIED DATA")
print("=" * 80)
print(f"Using classified CSV: {CLASSIFIED_DATA_PATH}")
print(f"Output tag file: {OUTPUT_GETAGS_ZSCORE_PATH}")

# Step 1: Load product name to ID mapping
print("\nLoading name to ID mapping...")
with open(NAME_TO_ID_PATH, 'r', encoding='utf-8') as f:
    name_to_id = json.load(f)
print(f"✓ Loaded {len(name_to_id)} name mappings")

# Step 2: Load base tags
print("\nLoading base tags...")
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

    # Extract entity IDs (product/movie) from the session events
    product_ids = set()
    for col in row.index:
        if col == 'Class':
            continue
        cell_value = row[col]
        if pd.isna(cell_value):
            continue
        cell_str = str(cell_value).strip()
        if not cell_str:
            continue

        candidates = []

        # Existing quote-based extraction covers tuples like ('ts', 'act', 'product')
        pattern = r"'([^']*)'|\"([^\"]*)\""
        matches = re.findall(pattern, cell_str)
        strings = [m[0] if m[0] else m[1] for m in matches]

        if DOMAIN == "movie":
            if strings:
                title = strings[-1].strip()
                if title:
                    candidates.append(title)
            else:
                # Try literal eval for tuple/list formats
                try:
                    parsed = ast.literal_eval(cell_str)
                    if isinstance(parsed, (list, tuple)) and len(parsed) >= 1:
                        maybe_title = str(parsed[-1]).strip()
                        if maybe_title:
                            candidates.append(maybe_title)
                except Exception:
                    pass
        else:
            if strings:
                product_name = strings[-1].strip()
                if product_name:
                    candidates.append(product_name)

        # Fallback: treat the whole cell as a product name (handles plain CSV values)
        candidates.append(cell_str)

        # Further split on common delimiters in case multiple names share a cell
        extra_candidates = []
        for cand in candidates:
            for sep in ['|', ',', ';', '、']:
                if sep in cand:
                    extra_candidates.extend([part.strip() for part in cand.split(sep) if part.strip()])
        candidates.extend(extra_candidates)

        # Deduplicate while preserving order
        seen = set()
        deduped_candidates = []
        for cand in candidates:
            if cand and cand not in seen:
                seen.add(cand)
                deduped_candidates.append(cand)

        # Map candidates to product IDs
        for cand in deduped_candidates:
            if cand in name_to_id:
                product_ids.add(str(name_to_id[cand]))
            else:
                normalized = cand.replace('"', '').replace("'", "").strip()
                if normalized and normalized in name_to_id:
                    product_ids.add(str(name_to_id[normalized]))

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
print("FILTERING SPARSE GROUP TAGS (Q1 THRESHOLD)")
print("=" * 80)

frequencies = np.array(list(group_tag_frequency.values()))

print("\nPercentile-based filtering:")
if frequencies.size == 0:
    print("  No group tag frequencies found; skipping percentile computation and falling back to base tags.")
    lower_bound = 0.0
    upper_bound = 0.0
    sparse_tags = {}
    reliable_tags = {}
    strong_tags = {}
else:
    # Use percentile thresholds for filtering
    lower_bound = np.percentile(frequencies, SPARSE_PERCENTILE)
    upper_bound = np.percentile(frequencies, STRONG_PERCENTILE)

    print(f"  Q1 (25th percentile): {lower_bound:.1f}")
    print(f"  P60 (60th percentile): {upper_bound:.1f}")
    print("  Tags below Q1 are considered too sparse")
    print("  Tags above P60 are considered strong")

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

# Step 5.5: Save strong_tags and sparse_tags for prompt generation
print(f"\nSaving strong tags and sparse tags for prompt generation...")
OUTPUT_STRONG_TAGS_PATH = f"../json/tags/{args.tag_name}_strong_tags.json"
OUTPUT_SPARSE_TAGS_PATH = f"../json/tags/{args.tag_name}_sparse_tags.json"

# Convert to sorted lists with metadata
strong_tags_output = {
    "tags": sorted(strong_tags.items(), key=lambda x: -x[1]),  # Sort by frequency descending
    "threshold": f"P{STRONG_PERCENTILE} (>= {upper_bound:.1f})",
    "count": len(strong_tags)
}
sparse_tags_output = {
    "tags": sorted(sparse_tags.items(), key=lambda x: x[1]),  # Sort by frequency ascending
    "threshold": f"Q1 (< {lower_bound:.1f})",
    "count": len(sparse_tags)
}

with open(OUTPUT_STRONG_TAGS_PATH, 'w', encoding='utf-8') as f:
    json.dump(strong_tags_output, f, ensure_ascii=False, indent=2)

with open(OUTPUT_SPARSE_TAGS_PATH, 'w', encoding='utf-8') as f:
    json.dump(sparse_tags_output, f, ensure_ascii=False, indent=2)

print(f"✓ Saved {len(strong_tags)} strong tags to {OUTPUT_STRONG_TAGS_PATH}")
print(f"✓ Saved {len(sparse_tags)} sparse tags to {OUTPUT_SPARSE_TAGS_PATH}")

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

# Calculate z-scores and generate tags
print("\nCalculating z-scores and generating tags...")
group_enhanced_tags = defaultdict(list)  # item_id -> [enhanced_tags]

items_processed = 0
items_with_enhanced_tags = 0
total_enhanced_tags = 0
level_counts = defaultdict(int)

for item_id, group_counts in item_group_mapping.items():
    items_processed += 1

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

        if level is not None:  # Changed to handle empty string as valid level
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

        # Count pairs
        if len(tags) >= 2:
            for tag1, tag2 in combinations(sorted(tags), 2):
                session_pair_counts[(tag1, tag2)] += 1

    # Select frequent pairs
    frequent_pairs = [(tag1, tag2) for (tag1, tag2), count in session_pair_counts.items()
                      if count >= MIN_SESSION_SUPPORT]

    print(f"✓ Found {len(frequent_pairs)} frequent tag pairs (support ≥ {MIN_SESSION_SUPPORT})")
    print(f"\nTop 10 frequent pairs:")
    for (tag1, tag2), count in sorted(session_pair_counts.items(), key=lambda x: -x[1])[:10]:
        if (tag1, tag2) in frequent_pairs:
            support = count / len(df) * 100
            print(f"  {count:3d} sessions ({support:5.2f}%): {tag1} + {tag2}")

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
        for tag1, tag2 in frequent_pairs:
            if tag1 in session_tags and tag2 in session_tags:
                # This session exhibits the pattern
                for product_id in product_ids:
                    item_pattern_counts[product_id][(tag1, tag2)] += 1

    print(f"\n  Finished processing {len(df)} sessions")

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

            # Get level using same function
            level = get_level_from_zscore(z)

            if level is not None:
                tag1, tag2 = pattern
                composite_tag = f"{tag1}+{tag2}{level}"
                composite_enhanced_tags[item_id].append(composite_tag)
                composite_tags_added += 1
                composite_level_counts[level] += 1
                item_has_composite = True

        if item_has_composite:
            items_with_composite += 1

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

# Generate z-score distribution plots
print(f"\n" + "=" * 80)
print("GENERATING Z-SCORE DISTRIBUTION PLOTS")
print("=" * 80)
print(f"\nCreating plots directory: {PLOT_DIR}")
os.makedirs(PLOT_DIR, exist_ok=True)

# Calculate z-score data for plotting
print("\nCalculating z-scores for plotting...")
zscore_data = {}  # item_id -> {group_tag: {'count': x, 'zscore': y}}

for item_id, group_counts in item_group_mapping.items():
    zscore_data[item_id] = {}
    for group_tag, count in group_counts.items():
        if group_tag not in group_stats:
            continue

        # Calculate z-score
        mean = group_stats[group_tag]['mean']
        std = group_stats[group_tag]['std']

        if std == 0:
            z = 0
        else:
            z = (count - mean) / std

        zscore_data[item_id][group_tag] = {
            'count': count,
            'zscore': z
        }

# Organize z-scores by group tag for plotting
group_zscore_stats = {}
for group_tag in group_data.keys():
    zscores = []
    for item_id, group_zscores in zscore_data.items():
        if group_tag in group_zscores:
            zscores.append(group_zscores[group_tag]['zscore'])

    if zscores:
        zscores_array = np.array(zscores)
        group_zscore_stats[group_tag] = {
            'mean': np.mean(zscores_array),
            'std': np.std(zscores_array),
            'min': np.min(zscores_array),
            'max': np.max(zscores_array),
            'median': np.median(zscores_array),
            'p25': np.percentile(zscores_array, 25),
            'p75': np.percentile(zscores_array, 75),
            'count': len(zscores_array)
        }

print(f"✓ Calculated z-score statistics for {len(group_zscore_stats)} group tags")

# Create subplot grid for z-score distributions
print("\nCreating z-score distribution plots...")
n_tags = len(group_data)
plot_path = None
if n_tags == 0:
    print("No group tag data available; skipping z-score distribution plot generation.")
else:
    n_cols = 4
    n_rows = (n_tags + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, group_tag in enumerate(sorted(group_data.keys())):
        ax = axes[idx]

        # Extract z-scores for this group tag
        zscores = []
        for item_id, group_zscores in zscore_data.items():
            if group_tag in group_zscores:
                zscores.append(group_zscores[group_tag]['zscore'])

        if zscores:
            # Plot histogram
            ax.hist(zscores, bins=30, edgecolor='black', alpha=0.7)

            stats = group_zscore_stats[group_tag]

            # Add median line
            ax.axvline(stats['median'], color='green', linestyle='--', linewidth=1.5,
                       label=f"Median: {stats['median']:.2f}")

            # Add threshold lines dynamically
            colors = ['red', 'orange', 'yellow', 'blue', 'purple', 'brown']
            for i, (threshold, level) in enumerate(zip(Z_SCORE_THRESHOLDS, LEVEL_TAGS)):
                color = colors[i % len(colors)]
                ax.axvline(threshold, color=color, linestyle=':', linewidth=1, alpha=0.5,
                          label=f'{threshold}: {level}')

            ax.set_title(f"{group_tag}\n(n={stats['count']:.0f})", fontsize=10)
            ax.set_xlabel('Z-Score', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.legend(fontsize=7)

            # Set x-axis ticks with 1.0 intervals
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(1.0))

            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(group_data), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plot_path = f'{PLOT_DIR}/zscore_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved z-score distribution plot to {plot_path}")
    plt.close()

# Combine with base_tags_v2
print("\nCombining with base_tags_v2...")
getags_zscore = {}

for item_id, base_tags in base_tags_v2.items():
    all_tags = list(base_tags) if isinstance(base_tags, list) else []

    if item_id in group_enhanced_tags:
        all_tags.extend(group_enhanced_tags[item_id])

    getags_zscore[item_id] = all_tags

print(f"✓ Combined tags for {len(getags_zscore)} items")

# Save getags_zscore.json
print(f"\nSaving to {OUTPUT_GETAGS_ZSCORE_PATH}...")
os.makedirs(os.path.dirname(OUTPUT_GETAGS_ZSCORE_PATH), exist_ok=True)
with open(OUTPUT_GETAGS_ZSCORE_PATH, 'w', encoding='utf-8') as f:
    json.dump(getags_zscore, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Items in base_tags: {len(base_tags_v2)}")
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
print(f"  3. {OUTPUT_GETAGS_ZSCORE_PATH}")
if plot_path:
    print(f"  4. {plot_path}")
else:
    print("  4. (plot skipped)")
OUTPUT_STRONG_TAGS_PATH = f"../json/tags/{args.tag_name}_strong_tags.json"
OUTPUT_SPARSE_TAGS_PATH = f"../json/tags/{args.tag_name}_sparse_tags.json"
print(f"  5. {OUTPUT_STRONG_TAGS_PATH}")
print(f"  6. {OUTPUT_SPARSE_TAGS_PATH}")
print("\nNote: Group tags are used as-is without suffix for better semantic preservation")
print("=" * 80)
print("\n✓ getags generation completed successfully!")
