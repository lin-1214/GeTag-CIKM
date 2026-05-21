#!/usr/bin/env python3
"""
Z-Score Ablation Study - All-in-One Script

Combines:
1. GeTag generation with different z-score thresholds
2. BM25 downstream evaluation
3. Results summarization
4. Google Sheets formatting

Usage:
    python zscore_ablation.py generate [--dataset DATASET] [--base_tag TAG]
    python zscore_ablation.py summarize
    python zscore_ablation.py format
    python zscore_ablation.py all [--dataset DATASET] [--base_tag TAG]
"""

import argparse
import subprocess
import os
import re
import sys
from pathlib import Path

# Try to import pandas (needed for summarize/format)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Configuration
GETAG_ROOT = Path(__file__).parent.parent
THRESHOLDS = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
DATASETS = ["food", "games", "yelp"]
BASE_TAGS = ["native", "basetag", "betags"]
ABLATION_DIR = GETAG_ROOT / "results" / "bm25" / "ablation"
SUMMARY_DIR = ABLATION_DIR / "summary"


# ============================================================================
# PART 1: GeTag Generation
# ============================================================================

def generate_getags(datasets=None, base_tags=None):
    """Generate GeTags with different z-score thresholds."""
    datasets = datasets or DATASETS
    base_tags = base_tags or BASE_TAGS
    
    print("=" * 60)
    print("GeTag Z-Score Threshold Ablation - Generation")
    print("=" * 60)
    print(f"Datasets: {datasets}")
    print(f"Tag types: {base_tags}")
    print(f"Thresholds: {THRESHOLDS}")
    print()
    
    total_runs = len(datasets) * len(base_tags) * len(THRESHOLDS)
    current_run = 0
    
    os.chdir(GETAG_ROOT)
    
    for dataset in datasets:
        for base_tag in base_tags:
            csv_path = GETAG_ROOT / "data" / "classified" / f"{dataset}" / f"{base_tag}.csv"
            
            if not csv_path.exists():
                print(f"WARNING: {csv_path} not found, skipping {dataset}/{base_tag}...")
                continue
            
            for threshold in THRESHOLDS:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Dataset: {dataset} | Tag: {base_tag} | Threshold: {threshold}")
                print("-" * 40)
                
                cmd = [
                    sys.executable, "getag/gen_getag.py",
                    "--dataset", dataset,
                    "--base_tag", base_tag,
                    "--classified_csv", str(csv_path),
                    "--zscore_threshold", str(threshold)
                ]
                
                result = subprocess.run(cmd, capture_output=False)
                if result.returncode == 0:
                    print(f"✓ Completed: tags/{dataset}/getag_{base_tag}_z{threshold}.json")
                else:
                    print(f"✗ Failed!")
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


# ============================================================================
# PART 2: Results Summarization
# ============================================================================

def extract_config_info(config_str):
    """Extract seq_weights and max_seq_len from config string."""
    sw_match = re.search(r"seq_weights='?([^',\)]+)'?", config_str)
    seq_weights = sw_match.group(1) if sw_match else "None"
    if seq_weights == "None":
        seq_weights = "none"
    
    msl_match = re.search(r"max_seq_len=(\w+)", config_str)
    max_seq_len = msl_match.group(1) if msl_match else "None"
    if max_seq_len == "None":
        max_seq_len = "∞"
    
    return seq_weights, max_seq_len


def process_ablation_file(filepath):
    """Process a single ablation CSV file."""
    df = pd.read_csv(filepath)
    
    results = []
    for _, row in df.iterrows():
        zscore = row['zscore_threshold']
        tag_name = row['tag_name']
        config = row['config']
        
        seq_weights, max_seq_len = extract_config_info(config)
        
        # Both user-based and item-based use 100 negative samples for evaluation
        # This matches how retrieval.py selects best config (by ndcg@10/100/val)
        hr10_test = row.get('hr@10/100/test', None)
        ndcg10_test = row.get('ndcg@10/100/test', None)
        
        results.append({
            'zscore_threshold': zscore,
            'tag_name': tag_name.strip(),
            'seq_weights': seq_weights,
            'max_seq_len': max_seq_len,
            'HR@10': hr10_test,
            'NDCG@10': ndcg10_test
        })
    
    return pd.DataFrame(results)


def summarize_results():
    """Summarize ablation results from BM25 experiments."""
    if not HAS_PANDAS:
        print("ERROR: pandas is required. Install with: pip install pandas")
        return
    
    print("=" * 60)
    print("Summarizing Ablation Results")
    print("=" * 60)
    
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    
    all_files = list(ABLATION_DIR.glob("*_ablation.csv"))
    print(f"Found {len(all_files)} ablation files\n")
    
    if len(all_files) == 0:
        print("No ablation files found. Run BM25 experiments first.")
        return
    
    all_summaries = []
    
    for filepath in sorted(all_files):
        print(f"Processing: {filepath.name}")
        
        parts = filepath.stem.replace("_ablation", "").split("_")
        dataset = parts[0]
        base_tag = parts[1]
        mode = parts[2]
        
        df = process_ablation_file(filepath)
        
        # Create summary (best config per threshold)
        summary = df.loc[df.groupby('zscore_threshold')['HR@10'].idxmax()]
        summary = summary.sort_values('zscore_threshold')
        summary['dataset'] = dataset
        summary['base_tag'] = base_tag
        summary['mode'] = mode
        all_summaries.append(summary)
        
        # Print concise table
        print(f"\n  === {dataset.upper()} / {base_tag} / {mode} ===")
        print(f"  {'Threshold':<12} {'HR@10':<10} {'NDCG@10':<10}")
        print(f"  {'-'*34}")
        for _, row in summary.iterrows():
            print(f"  {row['zscore_threshold']:<12} {row['HR@10']:.4f}     {row['NDCG@10']:.4f}")
        print()
    
    # Create combined summary
    combined = pd.concat(all_summaries, ignore_index=True)
    combined_output = SUMMARY_DIR / "all_best_results.csv"
    combined.to_csv(combined_output, index=False)
    
    print(f"\nSaved: {combined_output}")


# ============================================================================
# PART 3: Google Sheets Formatting
# ============================================================================

def format_for_sheets(csv_path=None):
    """Format results for Google Sheets (tab-separated output)."""
    if not HAS_PANDAS:
        print("ERROR: pandas is required. Install with: pip install pandas")
        return
    
    csv_path = csv_path or SUMMARY_DIR / "all_best_results.csv"
    
    if not Path(csv_path).exists():
        print(f"ERROR: {csv_path} not found. Run 'summarize' first.")
        return
    
    df = pd.read_csv(csv_path)
    df['HR@10'] = df['HR@10'].round(4)
    df['NDCG@10'] = df['NDCG@10'].round(4)
    
    thresholds = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # ========== TABLE 1: HR@10 ==========
    print("=" * 70)
    print("HR@10 TABLE (Copy & Paste to Google Sheets)")
    print("=" * 70)
    print()
    
    header = ["Dataset", "Base Tag", "Mode"] + [f"z={t}" for t in thresholds]
    print("\t".join(header))
    
    for dataset in ['food', 'games', 'yelp']:
        for base_tag in ['basetag', 'native']:
            for mode in ['itembased', 'userbased']:
                row = [dataset, base_tag, mode]
                for t in thresholds:
                    val = df[(df['dataset'] == dataset) & 
                            (df['base_tag'] == base_tag) & 
                            (df['mode'] == mode) & 
                            (df['zscore_threshold'] == t)]['HR@10'].values
                    row.append(f"{val[0]:.4f}" if len(val) > 0 else "-")
                print("\t".join(row))
    
    # ========== TABLE 2: NDCG@10 ==========
    print()
    print("=" * 70)
    print("NDCG@10 TABLE (Copy & Paste to Google Sheets)")
    print("=" * 70)
    print()
    
    print("\t".join(header))
    
    for dataset in ['food', 'games', 'yelp']:
        for base_tag in ['basetag', 'native']:
            for mode in ['itembased', 'userbased']:
                row = [dataset, base_tag, mode]
                for t in thresholds:
                    val = df[(df['dataset'] == dataset) & 
                            (df['base_tag'] == base_tag) & 
                            (df['mode'] == mode) & 
                            (df['zscore_threshold'] == t)]['NDCG@10'].values
                    row.append(f"{val[0]:.4f}" if len(val) > 0 else "-")
                print("\t".join(row))
    
    # ========== TABLE 3: Best per config ==========
    print()
    print("=" * 70)
    print("BEST THRESHOLD PER CONFIG (Copy & Paste to Google Sheets)")
    print("=" * 70)
    print()
    
    summary_header = ["Dataset", "Base Tag", "Mode", "Best Z", "HR@10", "NDCG@10"]
    print("\t".join(summary_header))
    
    for dataset in ['food', 'games', 'yelp']:
        for base_tag in ['basetag', 'native']:
            for mode in ['itembased', 'userbased']:
                subset = df[(df['dataset'] == dataset) & 
                           (df['base_tag'] == base_tag) & 
                           (df['mode'] == mode)]
                if len(subset) == 0:
                    continue
                best_row = subset.loc[subset['HR@10'].idxmax()]
                row = [
                    dataset, base_tag, mode,
                    f"{best_row['zscore_threshold']:.1f}",
                    f"{best_row['HR@10']:.4f}",
                    f"{best_row['NDCG@10']:.4f}"
                ]
                print("\t".join(row))
    
    # ========== TABLE 4: Compact per mode ==========
    print()
    print("=" * 70)
    print("COMPACT FORMAT (HR@10 with Best column)")
    print("=" * 70)
    
    for mode in ['itembased', 'userbased']:
        print(f"\n--- {mode.upper()} ---")
        header2 = ["Dataset", "Base Tag"] + [f"z={t}" for t in thresholds] + ["Best"]
        print("\t".join(header2))
        
        for dataset in ['food', 'games', 'yelp']:
            for base_tag in ['basetag', 'native']:
                row = [dataset, base_tag]
                values = []
                for t in thresholds:
                    val = df[(df['dataset'] == dataset) & 
                            (df['base_tag'] == base_tag) & 
                            (df['mode'] == mode) & 
                            (df['zscore_threshold'] == t)]['HR@10'].values
                    if len(val) > 0:
                        values.append(val[0])
                        row.append(f"{val[0]:.4f}")
                    else:
                        values.append(0)
                        row.append("-")
                
                if values:
                    best_idx = values.index(max(values))
                    row.append(f"z={thresholds[best_idx]}")
                else:
                    row.append("-")
                print("\t".join(row))


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Z-Score Ablation Study - All-in-One Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zscore_ablation.py generate                    # Generate all GeTags
  python zscore_ablation.py generate --dataset food     # Generate for food only
  python zscore_ablation.py summarize                   # Summarize BM25 results
  python zscore_ablation.py format                      # Format for Google Sheets
  python zscore_ablation.py format > output.tsv         # Save to file
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate GeTags with different thresholds')
    gen_parser.add_argument('--dataset', choices=DATASETS, help='Single dataset to process')
    gen_parser.add_argument('--base_tag', choices=BASE_TAGS, help='Single tag type to process')
    
    # Summarize command
    subparsers.add_parser('summarize', help='Summarize BM25 ablation results')
    
    # Format command
    fmt_parser = subparsers.add_parser('format', help='Format results for Google Sheets')
    fmt_parser.add_argument('--csv', help='Path to CSV file (default: all_best_results.csv)')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run generate + summarize + format')
    all_parser.add_argument('--dataset', choices=DATASETS, help='Single dataset to process')
    all_parser.add_argument('--base_tag', choices=BASE_TAGS, help='Single tag type to process')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'generate':
        datasets = [args.dataset] if args.dataset else None
        base_tags = [args.base_tag] if args.base_tag else None
        generate_getags(datasets, base_tags)
    
    elif args.command == 'summarize':
        summarize_results()
    
    elif args.command == 'format':
        format_for_sheets(args.csv)
    
    elif args.command == 'all':
        datasets = [args.dataset] if args.dataset else None
        base_tags = [args.base_tag] if args.base_tag else None
        generate_getags(datasets, base_tags)
        print("\n\n")
        summarize_results()
        print("\n\n")
        format_for_sheets()


if __name__ == "__main__":
    main()
