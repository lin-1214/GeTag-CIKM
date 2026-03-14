#!/bin/bash

# Rsync GeTag project to remote server
# Excludes cache, results, PULLRS, and other non-essential files

REMOTE_USER="fintest"
REMOTE_HOST="140.112.31.189"
REMOTE_PATH="/data1/exp/"
LOCAL_PATH="."

echo "=========================================="
echo "Syncing GeTag to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "=========================================="
echo ""

# Check if we're in the GeTag directory
if [ ! -f "setup.py" ] || [ ! -d "getag" ]; then
    echo "Error: Please run this script from the GeTag root directory"
    exit 1
fi

# Perform rsync with exclusions matching .gitignore
rsync -avz --progress \
    --exclude 'PULLRS/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    --exclude '*.pyd' \
    --exclude '.Python' \
    --exclude 'venv/' \
    --exclude 'env/' \
    --exclude 'ENV/' \
    --exclude '.vscode/' \
    --exclude '.idea/' \
    --exclude '*.swp' \
    --exclude '*.swo' \
    --exclude '.DS_Store' \
    --exclude '.ipynb_checkpoints' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude 'downstream/bm25/cache/' \
    --exclude 'downstream/birank/cache/' \
    --exclude 'downstream/UniSRec/cache/' \
    --exclude 'downstream/LLMRank/llmrank/dataset/' \
    --exclude 'downstream/LLMRank/llmrank/log/' \
    --exclude 'downstream/LLMRank/llmrank/log_tensorboard/' \
    --exclude 'downstream/LLMRank/llmrank/saved/' \
    --exclude 'results/' \
    --exclude 'output/' \
    --exclude 'data/preprocessed/UniSRec/' \
    --exclude '*.log' \
    --exclude 'logs/' \
    --exclude 'log/' \
    --exclude '.git/' \
    --exclude '*.tmp' \
    --exclude '*.bak' \
    "${LOCAL_PATH}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/GeTag/"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Sync completed successfully!"
    echo "Remote location: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/GeTag/"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Sync failed with error code $?"
    echo "=========================================="
    exit 1
fi
