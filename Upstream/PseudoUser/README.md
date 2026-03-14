# PseudoUser: Label Phase for GeTag

This directory contains the session labeling system used to generate classified user session data for the GeTag downstream pipeline.

## Quick Start

```bash
cd src
bash ./src/run_all_label_phase.sh
```

That's it. The script runs `label_phase.py` for all required dataset × tag combinations and copies the outputs to the GeTag classified data directory.

---

## What It Does

`run_all_label_phase.sh` iterates over 5 combinations:

| Domain  | Tag    | GeTag Output File          |
|---------|--------|----------------------------|
| food    | base   | `food_basetag.csv`         |
| amazon  | native | `games_native.csv`         |
| amazon  | base   | `games_basetag.csv`        |
| yelp    | native | `yelp_native.csv`          |
| yelp    | base   | `yelp_basetag.csv`         |

For each combination it:
1. Sets `DATA_DOMAIN` and `INCLUDE_TAG` environment variables
2. Runs `label_phase.py`, which uses an LLM (Qwen3-4B via vLLM) to:
   - **Stage 1**: Sample sessions and ask the LLM to generate ~20 persona cluster labels
   - **Stage 2**: Classify all sessions into those clusters (multi-label, batch of 4)
3. Outputs `../label_data/classified_data_{domain}_{tag}_0.csv`
4. Copies to GeTag as `{dataset}_{tag}.csv`

---

## Prerequisites

### Environment

```bash
pip install -r src/requirements.txt
```

Key dependencies: `vllm`, `transformers`, `openai`, `pandas`, `numpy`, `tqdm`

The LLM (Qwen3-4B) is loaded **locally** — no API key needed. Requires a GPU with ~8GB+ VRAM.

### Required Data Files

All paths are relative to the `PseudoUser/` directory:

**Session data** (input):
```
data/food/food_commerce_data_labeling.csv
data/amazon/amazon_sessions_labeling.csv
data/yelp/yelp_sessions_labeling.csv
```

**Item tag mappings** (used to enrich prompts with semantic context):
```
json/tags/food_basetag.json
json/tags/food_native.json
json/tags/food_betags.json
json/tags/amazon_basetag.json
json/tags/amazon_native.json
json/tags/amazon_betags.json
json/tags/amazon_mapping.json
json/tags/yelp_basetag.json
json/tags/yelp_native.json
json/tags/yelp_betags.json
```

**ID/name mappings**:
```
json/product_name_to_id.json   (food)
json/product_id_to_name.json   (food)
json/amazon_title_to_asin.json (amazon/games)
json/amazon_asin_to_title.json (amazon/games)
json/yelp_name_to_id.json      (yelp)
json/yelp_id_to_name.json      (yelp)
```

### Output

Classified CSV files are written to `label_data/` and then copied to GeTag:
```
label_data/classified_data_food_base_0.csv
label_data/classified_data_amazon_native_0.csv
label_data/classified_data_amazon_base_0.csv
label_data/classified_data_yelp_native_0.csv
label_data/classified_data_yelp_base_0.csv
```

---

## Configuration

Edit `src/config.py` to change:
- `MODEL_NAME` — default: `Qwen/Qwen3-4B`
- `T_CLUSTERS` — number of persona clusters to generate (default: 20)
- `BATCH_SIZE` — sessions per classification batch (default: 4)
- `SAMPLE_STEP` — sampling interval for Stage 1 (default: every 25 sessions)
- `MULTI_LABEL` / `MAX_LABELS_PER_SESSION` — multi-label settings

---

## Directory Structure

```
PseudoUser/
├── src/                          # Active scripts
│   ├── run_all_label_phase.sh    # Entry point
│   ├── label_phase.py            # Main labeling logic
│   ├── config.py                 # Configuration
│   ├── utils.py                  # LLM client + data loading
│   ├── prompt.py                 # Prompt templates
│   ├── gen_cllm_prompt.py        # Prompt file generation
│   ├── requirements.txt          # Python dependencies
│   └── json/                     # Runtime state (iteration metadata)
│
├── data/                         # Input session data
│   ├── food/
│   ├── amazon/
│   └── yelp/
│
├── json/                         # ID mappings and tag files
│   ├── tags/
│   └── *.json
│
└── label_data/                   # Output classified CSVs
```

### Files NOT Used in Current Workflow

The following are leftovers from earlier research iterations and are **safe to delete** if you want to clean up:

**In `src/`: (Some of them are already deleted)**
- `finetune.py`, `finetune_i3fresh.py` — UniSRec fine-tuning (not used)
- `model.py`, `models/` — model definitions (not used)
- `train.py` — training script (not used)
- `predict.py`, `predict_phase.py` — prediction phase (not used)
- `evaluation_phase.py` — evaluation (not used)
- `retrieval_v2.py` — retrieval experiment (not used)
- `multibeam.py`, `resample_beams.py` — multi-beam ensemble (not used)
- `gen_getag.py` — GeTag tag generation (belongs in downstream)
- `run_sasrec_baseline.py`, `run.sh`, `test_phase1.sh` — old run scripts
- `plot_metrics.py`, `convert_to_item_based.py` — utility scripts (not used)
- `libs/` — library code for old models
- `dataset/` — preprocessed datasets for old experiments
- `exps_v2/` — experiment logs and caches
- `tmp/` — temporary files
- `PHASE1_IMPLEMENTATION.md`, `PHASE1_SUMMARY.md` — old design docs

**In `PseudoUser/`:**
- `downsample_dataset.py`, `preprocess_i3fresh_for_unisrec.py` — one-time preprocessing
- `label_data_mb_ama/`, `label_data_mb_food/`, `label_data_mb_yelp/` — multi-beam experiment outputs
- `label_data_tmp/` — temporary outputs
- `results/` — old retrieval experiment results
- `data/checkpoints/` — old UniSRec model checkpoint
- `data/food/pkl/`, `data/food/*_filtered.csv`, `data/food/*_evaluation.csv` — intermediate files
- `data/amazon/amazon_sessions_filtered.csv`, `data/amazon/amazon_sessions_evaluation.csv`
- `data/yelp/session_yelp.csv`, `data/yelp/yelp_sessions_evaluation.csv`
- `data/movie/`, `data/unisrec/` — unused datasets
- `json/tags/*.backup*` — backup files
- `json/tags/movie_*.json`, `json/tags/base_tags_v2.json`, `json/tags/keywords_v2.json`, `json/tags/old_full_keyword.json` — unused tag files
- `json/movie_*.json`, `json/filtered_product_list.json` — unused mappings
