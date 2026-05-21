#!/bin/bash
# Run BM25 evaluation for all z-score threshold variants of GeTag.
# Aggregates results into a single CSV per (dataset, base_tag) combination.
#
# Usage:
#   ./scripts/run_bm25_zscore_ablation.sh                          # All combinations
#   ./scripts/run_bm25_zscore_ablation.sh food                     # Single dataset
#   ./scripts/run_bm25_zscore_ablation.sh food native              # Single combo

set -e  # Exit on error

cd "$(dirname "$0")/.."  # Navigate to GeTag root

# Parse arguments
FILTER_DATASET="${1:-}"
FILTER_TAG="${2:-}"

# Z-score thresholds to test
THRESHOLDS=(-2 -1.5 -1 -0.5 0 0.5 1 1.5 2)

# Datasets and base tags
DATASETS=("food" "games" "yelp")
BASE_TAGS=("native" "basetag")

# Apply filters if specified
if [ -n "$FILTER_DATASET" ]; then
    DATASETS=("$FILTER_DATASET")
fi
if [ -n "$FILTER_TAG" ]; then
    BASE_TAGS=("$FILTER_TAG")
fi

# Output directory
OUTPUT_DIR="results/bm25/ablation"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "BM25 Z-Score Ablation Study"
echo "========================================"
echo "Datasets: ${DATASETS[*]}"
echo "Base tags: ${BASE_TAGS[*]}"
echo "Thresholds: ${THRESHOLDS[*]}"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    for base_tag in "${BASE_TAGS[@]}"; do
        echo ""
        echo "============================================================"
        echo "Processing: ${dataset} / ${base_tag}"
        echo "============================================================"
        
        # Temp files for collecting results
        USER_TEMP=$(mktemp)
        ITEM_TEMP=$(mktemp)
        HEADER_SAVED_USER=false
        HEADER_SAVED_ITEM=false
        
        for threshold in "${THRESHOLDS[@]}"; do
            # Generate tag name: z1, z-1.5, z0, etc.
            # Remove trailing zeros and handle negative
            thresh_str=$(printf "z%g" "$threshold")
            tag_name="getag_${base_tag}_${thresh_str}"
            
            echo ""
            echo "[Threshold: $threshold] Tag: $tag_name"
            
            # Check if tag file exists
            tag_file="tags/${dataset}/${tag_name}.json"
            if [ ! -f "$tag_file" ]; then
                echo "  SKIP: $tag_file not found"
                continue
            fi
            
            # Run BM25 retrieval
            echo "  Running retrieval..."
            python3 downstream/bm25/retrieval.py \
                --dataset "$dataset" \
                --tag_name "$tag_name"
            
            # Paths to generated CSVs
            user_csv="results/bm25/${dataset}/retrieval_results_v2_userbased_bm25_${tag_name}.csv"
            item_csv="results/bm25/${dataset}/retrieval_results_v2_itembased_bm25_${tag_name}.csv"
            
            # Append to temp files with threshold info
            if [ -f "$user_csv" ]; then
                if [ "$HEADER_SAVED_USER" = false ]; then
                    # First file: add header with new columns
                    echo "zscore_threshold,tag_name,$(head -1 "$user_csv")" > "$USER_TEMP"
                    HEADER_SAVED_USER=true
                fi
                # Append data rows with threshold and tag_name
                tail -n +2 "$user_csv" | while IFS= read -r line; do
                    echo "${threshold},${tag_name},${line}" >> "$USER_TEMP"
                done
                echo "  ✓ User-based results collected"
            fi
            
            if [ -f "$item_csv" ]; then
                if [ "$HEADER_SAVED_ITEM" = false ]; then
                    echo "zscore_threshold,tag_name,$(head -1 "$item_csv")" > "$ITEM_TEMP"
                    HEADER_SAVED_ITEM=true
                fi
                tail -n +2 "$item_csv" | while IFS= read -r line; do
                    echo "${threshold},${tag_name},${line}" >> "$ITEM_TEMP"
                done
                echo "  ✓ Item-based results collected"
            fi
        done
        
        # Save aggregated results
        if [ -s "$USER_TEMP" ]; then
            output_file="${OUTPUT_DIR}/${dataset}_${base_tag}_userbased_ablation.csv"
            mv "$USER_TEMP" "$output_file"
            echo ""
            echo "✓ Saved: $output_file"
        else
            rm -f "$USER_TEMP"
        fi
        
        if [ -s "$ITEM_TEMP" ]; then
            output_file="${OUTPUT_DIR}/${dataset}_${base_tag}_itembased_ablation.csv"
            mv "$ITEM_TEMP" "$output_file"
            echo "✓ Saved: $output_file"
        else
            rm -f "$ITEM_TEMP"
        fi
    done
done

echo ""
echo "========================================"
echo "Ablation study complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "(no CSV files generated)"
