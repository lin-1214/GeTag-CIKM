#!/bin/bash
# Prepare UniSRec dataset: preprocess + convert to item-based format
#
# Usage:
#   ./scripts/prepare_unisrec_dataset.sh <dataset> <tag_type> [--device DEVICE]
#
# Examples:
#   ./scripts/prepare_unisrec_dataset.sh food native
#   ./scripts/prepare_unisrec_dataset.sh games getag_native --device cuda
#   ./scripts/prepare_unisrec_dataset.sh yelp basetag

set -e  # Exit on error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <tag_type> [--device DEVICE]"
    echo ""
    echo "Arguments:"
    echo "  dataset   : food, games, or yelp"
    echo "  tag_type  : native, getag_native, basetag, or getag_basetag"
    echo "  --device  : (optional) cpu or cuda (default: cpu)"
    echo ""
    echo "Examples:"
    echo "  $0 food native"
    echo "  $0 games getag_native --device cuda"
    echo "  $0 yelp basetag"
    exit 1
fi

DATASET=$1
TAG_TYPE=$2
DEVICE="cpu"

# Parse optional device argument
shift 2
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate dataset
if [[ "$DATASET" != "food" && "$DATASET" != "games" && "$DATASET" != "yelp" ]]; then
    echo "Error: dataset must be 'food', 'games', or 'yelp'"
    exit 1
fi

# Validate tag type
if [[ "$TAG_TYPE" != "native" && "$TAG_TYPE" != "getag_native" && "$TAG_TYPE" != "basetag" && "$TAG_TYPE" != "getag_basetag" ]]; then
    echo "Error: tag_type must be 'native', 'getag_native', 'basetag', or 'getag_basetag'"
    exit 1
fi

# Determine dataset name (output directory name)
DATASET_NAME="${DATASET}_${TAG_TYPE}"
OUTPUT_DIR="data/preprocessed/UniSRec/${DATASET_NAME}"

echo "=========================================="
echo "PREPARING UNISREC DATASET"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Tag type: $TAG_TYPE"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Step 1: Run preprocessing
echo "Step 1/2: Running preprocessing..."
echo "---"
python3 scripts/preprocess_${DATASET}_for_unisrec.py \
    --tags_file tags/${DATASET}/${TAG_TYPE}.json \
    --device $DEVICE

if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi

echo ""
echo "Step 2/2: Converting to item-based format..."
echo "---"
python3 downstream/UniSRec/convert_to_item_based.py \
    --dataset $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "Error: Conversion to item-based format failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "DATASET PREPARATION COMPLETE!"
echo "=========================================="
echo "User-based dataset: $OUTPUT_DIR"
echo "Item-based dataset: ${OUTPUT_DIR}_i"
echo ""
echo "Files created:"
echo "  - ${DATASET_NAME}.train.inter"
echo "  - ${DATASET_NAME}.valid.inter"
echo "  - ${DATASET_NAME}.test.inter"
echo "  - ${DATASET_NAME}.feat1CLS"
echo ""
echo "Next step: Fine-tune the model"
echo "  python3 downstream/UniSRec/finetune.py \\"
echo "    --dataset ${DATASET_NAME}_i \\"
echo "    --checkpoint checkpoints/UniSRec-FHCKM-300.pth \\"
echo "    --device $DEVICE"
echo "=========================================="
