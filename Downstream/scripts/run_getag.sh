#!/bin/bash
# Generate GeTag for all dataset + tag combinations
# Total: 6 runs (3 datasets × 2 tag types)
#
# Usage:
#   ./run_getag.sh              # With z-score filtering (default)
#   ./run_getag.sh --no-zscore  # Without z-score filtering

set -e  # Exit on error

cd "$(dirname "$0")/.."  # Navigate to GeTag root

# Parse arguments
NO_ZSCORE_FLAG=""
OUTPUT_SUFFIX=""
if [[ "$1" == "--no-zscore" ]]; then
    NO_ZSCORE_FLAG="--no_zscore_filter"
    OUTPUT_SUFFIX="_nozscore"
    echo "========================================"
    echo "Running GeTag (NO z-score filtering)"
    echo "========================================"
else
    echo "========================================"
    echo "Running GeTag (WITH z-score filtering)"
    echo "========================================"
    echo ""
    echo "Z-score thresholds (from gen_getag.py):"
    echo "  food:  2.0"
    echo "  games: 0.0"
    echo "  yelp:  0.0"
fi

DATASETS=("food" "games" "yelp")
BASE_TAGS=("native" "basetag")

for dataset in "${DATASETS[@]}"; do
    for base_tag in "${BASE_TAGS[@]}"; do
        csv_path="data/classified/${dataset}_${base_tag}.csv"
        
        echo ""
        echo "----------------------------------------"
        echo "Dataset: $dataset | Base tag: $base_tag"
        echo "CSV: $csv_path"
        echo "----------------------------------------"
        
        # Check if CSV exists
        if [ ! -f "$csv_path" ]; then
            echo "WARNING: $csv_path not found, skipping..."
            continue
        fi
        
        python getag/gen_getag.py \
            --dataset "$dataset" \
            --base_tag "$base_tag" \
            --classified_csv "$csv_path" \
            $NO_ZSCORE_FLAG
        
        echo "✓ Completed: ${dataset}_${base_tag}${OUTPUT_SUFFIX}"
    done
done

echo ""
echo "========================================"
echo "All GeTag runs completed!"
echo "========================================"
echo "Output files in tags/{dataset}/getag_{base_tag}${OUTPUT_SUFFIX}.json"
