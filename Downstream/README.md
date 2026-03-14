# GeTag

GeTag: Tagging with Collective Intent via LLM-Derived Semantics from User Session Groups

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GeTag.git
cd GeTag

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Project Structure

```
GeTag/
├── tags/                      # Generated tag files
│   ├── food/                  # Food dataset tags
│   ├── games/                 # Games dataset tags
│   └── yelp/                  # Yelp dataset tags
├── data/
│   ├── raw/                   # Raw evaluation data
│   ├── preprocessed/          # Preprocessed data for downstream tasks
│   │   ├── {dataset}/bm25/    # BM25 preprocessed data
│   │   └── UniSRec/           # UniSRec preprocessed data (large, gitignored)
│   └── mappings/              # ID mappings
├── downstream/                # Downstream evaluation tasks
│   ├── bm25/                  # BM25 retrieval
│   ├── birank/                # BiRank retrieval
│   ├── UniSRec/               # Sequential recommendation
│   └── LLMRank/               # LLM-based ranking
├── scripts/                   # Helper scripts
├── checkpoints/               # Pre-trained models
└── getag/                     # Core GeTag implementation

```

## Generating GeTags

GeTag generates enhanced tags from classified session data using z-score analysis and composite pattern mining.

### Generate GeTags for a Dataset

```bash
# Generate GeTags for food dataset from native tags
python getag/gen_getag.py \
  --dataset food \
  --base_tag native \
  --classified_csv data/classified/food_native.csv

# Generate GeTags for games dataset from basetag
python getag/gen_getag.py \
  --dataset games \
  --base_tag basetag \
  --classified_csv data/classified/games_basetag.csv

# Generate GeTags for yelp dataset
python getag/gen_getag.py \
  --dataset yelp \
  --base_tag native \
  --classified_csv data/classified/yelp_native.csv
```

**Parameters:**
- `--dataset`: Dataset name (food, games, or yelp)
- `--base_tag`: Base tag type to use (native or basetag)
- `--classified_csv`: Path to classified session data CSV
- `--output_dir`: Output directory for generated tags (default: tags/)

**Outputs:**
- `tags/{dataset}/getag_{base_tag}.json`: Final GeTag tags
- `tags/{dataset}/item_group_mapping_getag_{base_tag}.json`: Item-group tag counts
- `tags/{dataset}/group_tag_frequency_getag_{base_tag}.json`: Group tag frequencies

**How it works:**
1. Loads classified session data with LLM-derived group tags
2. Filters sparse tags using session count threshold (sessions/20)
3. Calculates z-scores for each item-tag association
4. Generates composite tags from frequent co-occurring patterns (food/games only)
5. Merges enhanced tags with base tags

## Quick Start

### 1. BM25 Retrieval

```bash
# Run BM25 on food dataset with GeTag tags
python3 downstream/bm25/retrieval.py --dataset food --tag_name getag_native

# Run on games dataset
python3 downstream/bm25/retrieval.py --dataset games --tag_name native
```

### 2. BiRank Retrieval

**Note: BiRank requires GPU for efficient computation.**

```bash
# Run BiRank on food dataset
python3 downstream/birank/retrieval.py --dataset food --tag_name getag_native
```

### 3. UniSRec Sequential Recommendation

**Note: UniSRec requires GPU for training and inference.**

```bash
# Prepare dataset (preprocessing + BERT embeddings + convert to item-based)
./scripts/prepare_unisrec_dataset.sh food getag_native

# Fine-tune UniSRec (runs on GPU)
python3 downstream/UniSRec/finetune.py \
  --dataset food_getag_native_i \
  --checkpoint checkpoints/UniSRec-FHCKM-300.pth \
  --device cuda
```

### 4. LLMRank

```bash
# First-time setup: Configure API key
cd downstream/LLMRank/llmrank
cp openai_api.yaml.example openai_api.yaml
# Edit openai_api.yaml and add your OpenRouter API key

# Run LLMRank experiment
cd ../../../  # Back to GeTag root
./scripts/run_llmrank.sh food getag_native
```

## Datasets

- **food**: Food commerce data (Chinese product names)
- **games**: Amazon Video Games data
- **yelp**: Yelp business data

## Tag Types

- **native**: Original native tags
- **basetag**: Base tags
- **getag_native**: GeTag-generated tags based on native tags
- **getag_basetag**: GeTag-generated tags based on base tags

## Results

All results are saved to the `results/` directory:
- `results/bm25/{dataset}/`
- `results/birank/{dataset}/`
- UniSRec: `output/logs/`
- LLMRank: `downstream/LLMRank/llmrank/log/`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for complete list
