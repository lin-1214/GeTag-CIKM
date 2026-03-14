# LLMRank

LLM-based ranking for sequential recommendation.

## Setup

### 1. Configure OpenRouter API

Copy the example config and add your API key:

```bash
cd downstream/LLMRank/llmrank
cp openai_api.yaml.example openai_api.yaml
```

Then edit `openai_api.yaml` and replace `YOUR_OPENROUTER_API_KEY_HERE` with your actual API key.

**Get your API key:**
1. Visit https://openrouter.ai/
2. Sign up/login
3. Go to Keys → Create Key
4. Copy your API key

### 2. Run Experiments

From the GeTag root directory:

```bash
# Automated workflow
./scripts/run_llmrank.sh food getag_native

# Or manual steps
python3 scripts/prepare_llmrank_dataset.py --dataset food --tag getag_native
cd downstream/LLMRank/llmrank && python3 evaluate.py -m Rank -d food_getag_native
```

## Supported Datasets

- `food` - Food commerce data
- `games` - Amazon games data
- `yelp` - Yelp business data

## Supported Tags

- `native` - Native tags
- `basetag` - Base tags
- `getag_native` - GeTag with native tags
- `getag_basetag` - GeTag with base tags

## Configuration

Model settings are in `llmrank/props/Rank.yaml`:
- `api_name`: Model to use (e.g., `openai/gpt-4-turbo`, `openai/gpt-3.5-turbo`)
- `temperature`: Sampling temperature (default: 0.2)
- `recall_budget`: Number of candidates to rank (default: 20)
- `max_his_len`: Maximum history length (default: 50)

See available models at: https://openrouter.ai/models

## Results

Results are saved to:
- Console output with metrics (NDCG@10, Recall@10, etc.)
- Log files in `llmrank/log/`
