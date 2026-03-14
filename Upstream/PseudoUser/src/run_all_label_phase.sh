#!/bin/bash
# Run label_phase.py for all dataset + tag combinations
# Generates classified CSV files for GeTag
#
# Required combinations (5 total):
#   - food + base (food_native already exists)
#   - amazon + native  (amazon = games in GeTag)
#   - amazon + base  
#   - yelp + native
#   - yelp + base

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# GeTag classified data directory
GETAG_CLASSIFIED_DIR="/data2/b11902154/GeTag/data/classified"

echo "========================================"
echo "Running Label Phase for All Combinations"
echo "========================================"
echo "Working directory: $(pwd)"
echo "GeTag output directory: $GETAG_CLASSIFIED_DIR"
echo ""

# Create GeTag classified directory if not exists
mkdir -p "$GETAG_CLASSIFIED_DIR"

# Define combinations to run
# Format: "domain:tag:getag_dataset:getag_tag"
# Mapping: amazon -> games, base -> basetag
COMBINATIONS=(
    "food:base:food:basetag"
    "amazon:native:games:native"
    "amazon:base:games:basetag"
    "yelp:native:yelp:native"
    "yelp:base:yelp:basetag"
)

# Optional: Add food:native if you want to regenerate
# COMBINATIONS+=("food:native:food:native")

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r domain tag getag_dataset getag_tag <<< "$combo"
    
    echo ""
    echo "========================================"
    echo "Processing: ${domain} + ${tag}"
    echo "  -> GeTag: ${getag_dataset}_${getag_tag}"
    echo "========================================"
    
    # Set environment variables
    export DATA_DOMAIN="$domain"
    export INCLUDE_TAG="$tag"
    
    echo "DATA_DOMAIN=$DATA_DOMAIN"
    echo "INCLUDE_TAG=$INCLUDE_TAG"
    echo ""
    
    # Run label_phase.py
    python label_phase.py
    
    # The output file will be: ../label_data/classified_data_{domain}_{tag}_0.csv
    OUTPUT_FILE="../label_data/classified_data_${domain}_${tag}_0.csv"
    GETAG_FILE="${GETAG_CLASSIFIED_DIR}/${getag_dataset}_${getag_tag}.csv"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✓ Generated: $OUTPUT_FILE"
        # Copy to GeTag with correct naming
        cp "$OUTPUT_FILE" "$GETAG_FILE"
        echo "✓ Copied to: $GETAG_FILE"
    else
        echo "WARNING: Expected output not found: $OUTPUT_FILE"
    fi
    
    echo ""
done

echo "========================================"
echo "All Label Phase runs completed!"
echo "========================================"
echo ""
echo "Generated files in PseudoUser:"
ls -la ../label_data/classified_data_*.csv 2>/dev/null || echo "No classified files found"
echo ""
echo "Generated files in GeTag:"
ls -la "$GETAG_CLASSIFIED_DIR"/*.csv 2>/dev/null || echo "No classified files found"
