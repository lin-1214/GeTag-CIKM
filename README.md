# GeTag: Mining Collective Intent via Session-Group Semantics

GeTag generates semantically enriched item tags by mining collective user intent from session-group patterns, using LLM-derived group semantics and z-score filtering to identify statistically significant item-tag associations.

## Pipeline Overview

GeTag is a two-stage pipeline:

| Stage | Directory | Role |
|---|---|---|
| **Upstream** | `Upstream/` | LLM-based session labeling. Classifies user sessions into persona labels and emits classified-session CSVs. |
| **Downstream** | `Downstream/` | GeTag tag generation (z-score filtering) + evaluation (BM25 / BiRank / UniSRec / LLMRank). |

```
Upstream/PseudoUser  ──(run_all_label_phase.sh, LLM via vLLM)──►  classified CSVs
        │
        └──► Downstream/data/classified/{dataset}_{tag}.csv
                  │
                  └──► getag/gen_getag.py ──► enhanced tags ──► downstream/ evaluation
```

The upstream's classified CSVs are the input to the Downstream tag-generation step.

## Project Structure

```
GeTag-pipeline/
├── Upstream/                      # Stage 1: session labeling
│   └── PseudoUser/
│       ├── src/                   # label_phase.py, config.py, prompt.py, utils.py, run_all_label_phase.sh
│       ├── data/                  # raw session data + item tags per dataset
│       └── json/                  # id↔name mappings, tag dictionaries
└── Downstream/                    # Stage 2: GeTag generation + evaluation
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

## Installation

A single dependency set covers both stages:

```bash
pip install -r requirements.txt
```

## Stage 1 — Upstream: Session Labeling

The upstream classifies user sessions into persona labels via an LLM served through a vLLM OpenAI-compatible endpoint. Its output — classified-session CSVs — is the input to GeTag generation in Stage 2.

**Required inputs** (shipped under `Upstream/PseudoUser/`):
- `data/{food,amazon,yelp}/*_labeling.csv` — raw user sessions per domain
- `json/tags/{domain}_{native,basetag,betags}.json` — item-tag dictionaries used to build tag-aware context
- `json/*_id_to_name.json` / `json/*_name_to_id.json` — item ID ↔ name mappings

### Run labeling for all dataset × tag combinations

Labeling requires a vLLM (or any OpenAI-compatible) endpoint. Point the env vars at your endpoint and run:

```bash
cd Upstream/PseudoUser/src

# Defaults: http://localhost:1357/v1, Qwen/Qwen3-4B, key=DUMMY
export VLLM_BASE_URL=http://localhost:1357/v1
export VLLM_MODEL=Qwen/Qwen3-8B
export VLLM_API_KEY=your-key        # optional

bash run_all_label_phase.sh
```

This runs all six combinations — `food`, `amazon` (→ games), `yelp` × `native` / `base` — writing `classified_data_{domain}_{tag}_0.csv` and copying each into the downstream classified-data directory as `Downstream/data/classified/{dataset}_{tag}.csv` (e.g. `food_native.csv`).

### How labeling works

Labeling proceeds in two LLM stages, with no predefined label space:

1. **Cluster generation** — a sampled subset of sessions (every `SAMPLE_STEP`-th session) is shown to the LLM, which proposes `T_CLUSTERS` persona-style category labels (e.g. *"Fine Dining Enthusiast"*, *"Budget Conscious Shopper"*).
2. **Session classification** — every session is then assigned one or more of those labels (multi-label, processed in batches).

Each session is enriched with **tag-aware context**: the top tags of its items (`native` category tags or `base` TagGPT-generated tags) are appended to the prompt, so the LLM grounds its labels in domain semantics rather than raw item names.

Key parameters live in [`Upstream/PseudoUser/src/config.py`](Upstream/PseudoUser/src/config.py); several can be overridden via environment variables:

| Parameter | Env var | Default | Meaning |
|---|---|---|---|
| `T_CLUSTERS` | `T_CLUSTERS` | 20 | number of persona labels to discover |
| `SAMPLE_STEP` | `SAMPLE_STEP` | 25 | session sampling interval for clustering |
| `SAMPLE_SIZE` | `SAMPLE_SIZE` | 4900 | session window to classify |
| `INCLUDE_TAG` | — | `base` / `native` | tag type used for context (set per run) |

## Data

Download the following files and place them under the `Downstream/` directory:

| File | Contents | Link |
|---|---|---|
| `data.zip` | Classified CSVs, raw data, ID mappings, preprocessed files | [Google Drive](https://drive.google.com/file/d/1MUARRWQ3tEhMJ5aBw01MNiSAt2dFdTq5/view?usp=sharing) |
| `tags.zip` | Base item tags (native, basetag, betags) for all datasets | [Google Drive](https://drive.google.com/file/d/1OSeI6uW_tiouFNQuY_ItbNdtpBiyTn2Q/view?usp=sharing) |

> **Note:** The Food dataset is derived from a private data source and is not included in the download above. If you need access for research purposes, please contact the authors.

After downloading, unzip both archives inside the `Downstream/` directory:

```bash
cd Downstream
unzip data.zip
unzip tags.zip
```

Expected directory structure after download:
```
Downstream/
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

## Stage 2 — Downstream: Tag Generation & Evaluation

> Run all commands in this section from inside `Downstream/` (`cd Downstream`).

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
