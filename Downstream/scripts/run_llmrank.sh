#!/bin/bash
# LLMRank Experiment Runner
# Usage: ./scripts/run_llmrank.sh <dataset> <tag> [max_tags]
# Example: ./scripts/run_llmrank.sh food betag 20

set -e  # Exit on error

# Check arguments
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <dataset> <tag> [max_tags]"
    echo ""
    echo "Datasets: food, games, yelp"
    echo "Tags: native, basetag, betag, getag_native, getag_basetag, getag_betag"
    echo "max_tags: Optional limit on NATIVE tags per item (ALL preference tags always included)"
    echo ""
    echo "Examples:"
    echo "  $0 food betag          # Use all tags"
    echo "  $0 food betag 20       # Limit to 20 native tags + ALL preference tags"
    exit 1
fi

DATASET=$1
TAG=$2
MAX_TAGS=$3
DATASET_TAG="${DATASET}_${TAG}"

echo "=========================================="
echo "LLMRank Experiment: ${DATASET_TAG}"
echo "=========================================="
echo ""

# Step 1: Prepare data
echo "[Step 1/2] Preparing data..."
if [ -n "$MAX_TAGS" ]; then
    echo "Using max_tags: $MAX_TAGS (NATIVE tags only, ALL preference tags included)"
    python3 scripts/prepare_llmrank_dataset.py --dataset $DATASET --tag $TAG --max_tags $MAX_TAGS
else
    echo "Using all tags (no limit)"
    python3 scripts/prepare_llmrank_dataset.py --dataset $DATASET --tag $TAG
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Data preparation failed!"
    exit 1
fi

echo ""
echo "[Step 2/2] Running LLMRank evaluation..."
echo ""

# Step 2: Run evaluation (in subshell to preserve working directory)
(cd downstream/LLMRank/llmrank && python3 evaluate.py -m Rank -d $DATASET_TAG)

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results saved in: downstream/LLMRank/llmrank/log/"
echo "=========================================="
