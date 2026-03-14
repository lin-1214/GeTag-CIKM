#!/bin/bash
# LLMRank Experiment Runner
# Usage: ./scripts/run_llmrank.sh <dataset> <tag>
# Example: ./scripts/run_llmrank.sh food native

set -e  # Exit on error

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset> <tag>"
    echo ""
    echo "Datasets: food, games, yelp"
    echo "Tags: native, basetag, getag_native, getag_basetag"
    echo ""
    echo "Example: $0 food getag_native"
    exit 1
fi

DATASET=$1
TAG=$2
DATASET_TAG="${DATASET}_${TAG}"

echo "=========================================="
echo "LLMRank Experiment: ${DATASET_TAG}"
echo "=========================================="
echo ""

# Step 1: Prepare data
echo "[Step 1/2] Preparing data..."
python3 scripts/prepare_llmrank_dataset.py --dataset $DATASET --tag $TAG

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
