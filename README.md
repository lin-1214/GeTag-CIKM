# GeTag: Mining Collective Intent via Session-Group Semantics

GeTag generates semantically enriched item tags by mining collective user intent from session-group patterns, using LLM-derived group semantics and z-score filtering to identify statistically significant item-tag associations.

## Installation

```bash
cd GeTag
pip install -r requirements.txt
```

## Data

Download the following files and place them under the `GeTag/` directory:

| File | Contents | Link |
|---|---|---|
| `data.zip` | Classified CSVs, raw data, ID mappings, preprocessed files | [Google Drive](https://drive.google.com/file/d/1MUARRWQ3tEhMJ5aBw01MNiSAt2dFdTq5/view?usp=sharing) |
| `tags.zip` | Base item tags (native, basetag, betags) for all datasets | [Google Drive](https://drive.google.com/file/d/1OSeI6uW_tiouFNQuY_ItbNdtpBiyTn2Q/view?usp=sharing) |

> **Note:** The Food dataset is derived from a private data source and is not included in the download above. If you need access for research purposes, please contact the authors.

After downloading, unzip both archives inside the `GeTag/` directory:

```bash
cd GeTag
unzip data.zip
unzip tags.zip
```

Expected directory structure after download:
```
GeTag/
├── tags/
│   ├── food/          # native.json, basetag.json, betags.json (contact authors for access)
│   ├── games/
│   └── yelp/
├── data/
│   ├── classified/    # LLM-classified session CSVs (input to GeTag)
│   ├── raw/           # Raw evaluation/labeling data
│   ├── mappings/      # ID mappings
│   └── preprocessed/  # Preprocessed data for downstream tasks
└── checkpoints/
    └── UniSRec-FHCKM-300.pth
```

## Project Structure

```
GeTag/
├── getag/                     # Core GeTag implementation
│   ├── gen_getag.py           # Tag generation
│   ├── search_zscore_threshold.py        # BM25 threshold search
│   ├── search_zscore_threshold_birank.py # BiRank threshold search
│   └── search_zscore_threshold_unisrec.py# UniSRec threshold search
├── downstream/
│   ├── bm25/                  # BM25 retrieval
│   ├── birank/                # BiRank retrieval
│   ├── UniSRec/               # Sequential recommendation
│   └── LLMRank/               # LLM-based ranking
├── scripts/                   # Preprocessing and visualization scripts
└── checkpoints/               # Pre-trained model checkpoints
```

## Step 1: Generate GeTags

GeTag enhances item tags by computing z-scores for item-tag associations within session groups and filtering by a threshold. The optimal threshold is found via binary search using downstream validation performance.

```bash
# Generate tags at a specific threshold (e.g. z = -0.375 for games/native)
python getag/gen_getag.py \
    --dataset games \
    --base_tag native \
    --classified_csv data/classified/games/native.csv \
    --zscore_threshold -0.375
```

**Parameters:**
- `--dataset`: `food`, `games`, or `yelp`
- `--base_tag`: `native`, `basetag`, or `betags`
- `--classified_csv`: Path to classified session data CSV
- `--zscore_threshold`: Z-score cutoff (lower = more tags included)

**Output:** `tags/{dataset}/getag_{base_tag}_z{threshold}.json`

### Find the Optimal Threshold

Use binary search over the z-score threshold guided by downstream validation performance:

```bash
# BM25-guided threshold search
python getag/search_zscore_threshold.py \
    --dataset games \
    --base_tag native \
    --classified_csv data/classified/games/native.csv \
    --result_file results/threshold_search/games_native.json

# BiRank-guided threshold search
python getag/search_zscore_threshold_birank.py \
    --dataset games \
    --base_tag native \
    --classified_csv data/classified/games/native.csv \
    --result_file results/threshold_search_birank/games_native.json

# UniSRec-guided threshold search
python getag/search_zscore_threshold_unisrec.py \
    --dataset games \
    --base_tag native \
    --classified_csv data/classified/games/native.csv \
    --checkpoint checkpoints/UniSRec-FHCKM-300.pth \
    --result_file results/threshold_search_unisrec/games_native.json
```

## Step 2: Downstream Evaluation

### BM25 Retrieval

```bash
python downstream/bm25/retrieval.py \
    --dataset games \
    --tag_name getag_native_z-0.375
```

### BiRank Retrieval

```bash
python downstream/birank/retrieval.py \
    --dataset games \
    --tag_name getag_native_z-0.375 \
    --verbose
```

### UniSRec Sequential Recommendation

First preprocess the dataset for UniSRec:

```bash
# Generate .inter and BERT embeddings
python scripts/preprocess_games_for_unisrec.py \
    --dataset_name games_getag_native_z-0.375 \
    --tags_file tags/games/getag_native_z-0.375.json

# Convert to item-based format
python downstream/UniSRec/convert_to_item_based.py \
    --dataset data/preprocessed/UniSRec/games_getag_native_z-0.375

# Fine-tune
python downstream/UniSRec/finetune.py \
    --dataset games_getag_native_z-0.375_i \
    --checkpoint checkpoints/UniSRec-FHCKM-300.pth \
    --device cuda
```

Or use the preparation script:

```bash
./scripts/prepare_unisrec_dataset.sh games getag_native_z-0.375
```

### LLMRank

```bash
# Configure OpenRouter API key
cp downstream/LLMRank/llmrank/openai_api.yaml.example \
   downstream/LLMRank/llmrank/openai_api.yaml
# Edit openai_api.yaml and add your API key

# Prepare LLMRank dataset
python scripts/prepare_llmrank_dataset.py --dataset games

# Run LLMRank
python downstream/LLMRank/llmrank/eval.py --dataset Games-6k
```

## Results

All results are saved to `results/`:
- `results/bm25/{dataset}/` — BM25 retrieval CSVs
- `results/birank/{dataset}/` — BiRank retrieval CSVs
- `results/threshold_search/` — Best threshold per (dataset, tag) for BM25
- `results/threshold_search_birank/` — Best threshold for BiRank
- `results/threshold_search_unisrec/` — Best threshold for UniSRec
- UniSRec fine-tune logs: `output/logs/`

## Datasets

| Dataset | Domain | Language |
|---|---|---|
| Food | Food commerce | Chinese |
| Games | Amazon Video Games | English |
| Yelp | Yelp businesses | English |

## Tag Types

| Tag | Description |
|---|---|
| `native` | Original item tags |
| `basetag` | TagGPT-generated tags |
| `betags` | BeTags-generated tags |
| `getag_{base}_z{t}` | GeTag-enhanced tags at threshold `t` |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (recommended for UniSRec fine-tuning)
- See `requirements.txt` for full list
