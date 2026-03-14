# PseudoUser

A project for simulating and analyzing user behavior using machine learning models.

## Project Structure

```
PseudoUser/
├── src/               # Source code directory
│   ├── label_phase.py    # Implementation of the labeling phase
│   ├── predict_phase.py  # Implementation of the prediction phase
│   ├── train.py         # Training implementation
│   ├── predict.py       # Prediction module
│   ├── config.py        # Configuration settings
│   ├── utils.py         # Utility functions
│   ├── finetune.py      # Model fine-tuning functionality
│   ├── gen_cllm_prompt.py # Prompt generation for CLLM
│   ├── prompt.py        # Prompt handling utilities
│   ├── requirements.txt # Python dependencies
│   └── run.sh          # Main execution script
├── data/              # Stores preprocessed session data
└── init_conda.sh      # Conda environment setup script
```

## Prerequisites

- Python 3.x
- CUDA-compatible GPU (recommended)
- conda (recommended for environment management)

## Dependencies

The project requires the following Python packages:
- transformers (4.51.0)
- hf_xet
- pandas
- numpy
- torch
- accelerate
- tqdm
- scipy
- torchvision
- torchaudio

## Setup


1. Install the required packages:
```bash
cd src
pip install -r requirements.txt
```

## Usage

The main execution script `run.sh` orchestrates the entire workflow. It performs multiple iterations of label and predict phases.

To run the project:

```bash
cd src
bash run.sh
```

The script will:
1. Clean up existing checkpoints and JSON files
2. Create necessary directories
3. Run multiple iterations of:
   - Label phase
   - Predict phase

### Configuration

You can modify the following parameters in `run.sh`:
- `NUM_ITERATIONS`: Number of times to run the full cycle (default: 8)
- `LABEL_GPU`: GPU device for label phase (default: 0)
- `PREDICT_GPU`: GPU device for predict phase (default: 1)

## Project Components

- `label_phase.py`: Implementation of the labeling phase
- `predict_phase.py`: Implementation of the prediction phase
- `train.py`: Training implementation
- `predict.py`: Prediction module
- `config.py`: Configuration parameters
- `utils.py`: Utility functions
- `finetune.py`: Model fine-tuning functionality
- `gen_cllm_prompt.py`: Prompt generation for CLLM
- `prompt.py`: Prompt handling utilities

## Output

The program generates several JSON files in the `json` directory:
- `ucb_tracking.json`: Tracks UCB (Upper Confidence Bound) scores
- `iteration_meta.json`: Metadata for each iteration
- `best_prompt.json`: Stores information of the best performing prompts in the previous iteration
- `max_ndcg_history.json`: NDCG (Normalized Discounted Cumulative Gain) history
- `all_time_best_prompt.json`: Stores information of the best performing prompts through the whole process

## Directory Structure

- `src/`: Contains all source code
- `data/`: Stores preprocessed session data
- `checkpoints/`: (Created during execution) Stores model checkpoints
- `json/`: (Created during execution) Stores output JSON files
