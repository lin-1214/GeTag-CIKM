import subprocess
import numpy as np
import json
import os
import glob
import torch
import random

from scipy.sparse import load_npz, save_npz
from config import Config

cfg = Config()

LAMBDA_V = 0.1
TOP_K = 2
EXPLORATION_SCALE = 0.1
SUBSET_RATIO = 0.4  # Use 40% of test interactions per behavioral class
CLLM_DATA_PATH = '../cllm_data'
MODEL_NAME = "Qwen/Qwen3-0.6B"
CLLM_PROCESSED_DATA_PATH = 'user_session_data'

# Allow quick-smoke overrides via environment variables
TRAIN_EPOCHS = int(os.getenv('TRAIN_EPOCHS', '2'))
FINTUNE_EPOCHS = int(os.getenv('FINETUNE_EPOCHS', '50'))
EVAL_EPOCHS = int(os.getenv('EVAL_EPOCHS', '1'))
USE_UCB = os.getenv('USE_UCB', 'true').lower() == 'true'  # Set to 'false' for ablation study

# Early Stopping controls for finetune (configurable via env)
EARLY_STOP = os.getenv('EARLY_STOP', 'true').lower() == 'true'
ES_PATIENCE = int(os.getenv('ES_PATIENCE', '7'))
ES_MIN_DELTA = os.getenv('ES_MIN_DELTA', '1e-3')
ES_MONITOR = os.getenv('ES_MONITOR', 'ndcg')  # combined | val_loss | recall | ndcg

if not os.path.exists('json'):
    os.makedirs('json', exist_ok=True)

def main():
    # Fix random seeds for reproducibility using config
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    
    # Set deterministic behavior for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds fixed to {cfg.SEED} for reproducibility")
    
    # Determine how many datasets were generated dynamically
    dataset_dirs = sorted(glob.glob(os.path.join(CLLM_DATA_PATH, f"{CLLM_PROCESSED_DATA_PATH}_*")))
    num_data = len(dataset_dirs)
    if num_data == 0:
        raise FileNotFoundError(f"No datasets found under {CLLM_DATA_PATH} with prefix {CLLM_PROCESSED_DATA_PATH}_*")
    print(f"Detected {num_data} datasets for evaluation: {[os.path.basename(d) for d in dataset_dirs]}")
    strategy_name = "UCB" if USE_UCB else "Greedy"
    print(f"Using {strategy_name} strategy for prompt selection")

    # Training command
    training_cmd = [
        "python3", "./train.py",
        "--num_data", str(num_data),
        "--lambda_V", str(LAMBDA_V),
        "--data_path", CLLM_DATA_PATH,
        "--model_name", MODEL_NAME,
        "--num_epochs", str(TRAIN_EPOCHS)
    ]
    subprocess.run(training_cmd)

    # Finetuning command
    finetuning_cmd = [
        "python3", "./finetune.py",
        "--num_dataset", str(num_data),
        "--lambda_V", str(LAMBDA_V),
        "--model_name", MODEL_NAME,
        "--data_path", CLLM_DATA_PATH,
        "--num_epochs", str(FINTUNE_EPOCHS)
    ]
    # Pass Early Stopping options
    if EARLY_STOP:
        finetuning_cmd.append("--early_stop")
    finetuning_cmd += [
        "--patience", str(ES_PATIENCE),
        "--min_delta", str(ES_MIN_DELTA),
        "--monitor_metric", ES_MONITOR,
    ]
    print(f"Finetune ES settings: early_stop={EARLY_STOP}, patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA}, monitor={ES_MONITOR}")
    subprocess.run(finetuning_cmd)

    # TODO: Generate EVAL_EPOCHS subsets of test dataset for UCB evaluation, each subset is a random sample of the test dataset
    # The test dataset is now in the format of test_matrix.npz
    # THe random indices for each dataset should be the same for each iteration
    # remove all the test_matrix_*.npz files, in the CLLM_DATA_PATH
    # for i in range(NUM_DATA):
    #     for file in os.listdir(os.path.join(CLLM_DATA_PATH, f"{CLLM_PROCESSED_DATA_PATH}_{i}")):
    #         if file.startswith("test_matrix_") and os.path.isfile(os.path.join(CLLM_DATA_PATH, f"{CLLM_PROCESSED_DATA_PATH}_{i}", file)):
    #             os.remove(os.path.join(CLLM_DATA_PATH, f"{CLLM_PROCESSED_DATA_PATH}_{i}", file))

    # Inspect one sample matrix for shape/nnz only
    sample_matrix = load_npz(os.path.join(dataset_dirs[0], "test_matrix.npz"))

    # TODO: Inspect the information of the sample_matrix: shape, content
    print(f"Sample matrix shape: {sample_matrix.shape}, {sample_matrix.nnz} nonzero elements")
    # print(f"Sample matrix content: {sample_matrix.toarray()}")  # too verbose

    print(f"=== Generating {EVAL_EPOCHS} subset evaluations ===")
    print(f"Each subset will sample {SUBSET_RATIO*100}% of test interactions per behavioral class")

    for j in range(EVAL_EPOCHS):

        for i in range(num_data):
            test_matrix = load_npz(os.path.join(CLLM_DATA_PATH, f"{CLLM_PROCESSED_DATA_PATH}_{i}", f"test_matrix.npz"))

            # Create subset by sampling interactions within each behavioral class
            subset_matrix = test_matrix.copy()
            subset_matrix.data[:] = 0  # Zero out all interactions first

            # For each behavioral class (row), sample subset of their test interactions
            for behavioral_class in range(test_matrix.shape[0]):
                # Get all test items for this behavioral class
                class_test_items = test_matrix[behavioral_class].nonzero()[1]

                if len(class_test_items) > 0:
                    # Sample subset of interactions for this behavioral class
                    subset_size = 0
                    if (j == 0):  # all prediction. warm-up
                         subset_size = test_matrix.shape[1]
                    else:
                        subset_size = max(1, int(len(class_test_items) * SUBSET_RATIO))
                    if subset_size < len(class_test_items):
                        # Use deterministic seed for each iteration and dataset
                        iteration_seed = cfg.SEED + j * 1000 + i * 100 + behavioral_class
                        np.random.seed(iteration_seed)
                        sampled_items = np.random.choice(class_test_items, size=subset_size, replace=False)
                    else:
                        sampled_items = class_test_items  # Keep all if subset would be same size

                    # Set the sampled interactions to 1
                    subset_matrix[behavioral_class, sampled_items] = 1

            save_npz(os.path.join(CLLM_DATA_PATH, f"{CLLM_PROCESSED_DATA_PATH}_{i}", f"test_matrix_{j+1}.npz"), subset_matrix)

    print(f"Completed generating all {EVAL_EPOCHS} evaluation subsets")
    
    prompt_scores = [[] for _ in range(num_data)]
    prompt_counts = [0 for _ in range(num_data)]

    for j in range(EVAL_EPOCHS):

        print(f"=== Iteration {j+1} ===")
        print(f"Current prompt scores: {prompt_scores}")
        print(f"Current prompt counts: {prompt_counts}")

        if j == 0:      # warm-up for the first iteration
            for i in range(num_data):
                dataset_dir = f"{CLLM_PROCESSED_DATA_PATH}_{i}"
                predict_cmd = [
                    "python3", "./predict.py",
                    "--dataset", dataset_dir,
                    "--lambda_V", str(LAMBDA_V),
                    "--model_name", MODEL_NAME,
                    "--data_path", CLLM_DATA_PATH,
                    "--iteration", str(j+1)
                ]
                result = subprocess.run(predict_cmd, capture_output=True, text=True)
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "NDCG Returned:" in line:
                        ndcg = float(line.split(': ')[1])
                        prompt_scores[i].append(ndcg)
                        prompt_counts[i] += 1
                
        else:
            if USE_UCB:
                # Choose the top K performance prompts using safe UCB
                ucb_scores = [0 for _ in range(num_data)]
                total_pulls = int(np.sum(prompt_counts))
                if total_pulls == 0:
                    # Warm-up fallback: explore first TOP_K datasets deterministically
                    top_k_indices = np.arange(TOP_K)
                    print(f"[UCB warm-up] No pulls yet, exploring indices {top_k_indices}")
                else:
                    for i in range(num_data):
                        mean_reward = float(np.mean(prompt_scores[i])) if prompt_scores[i] else 0.0
                        count_i = max(1, prompt_counts[i])
                        explore = EXPLORATION_SCALE * np.sqrt(2 * np.log(max(1, total_pulls)) / count_i)
                        ucb_scores[i] = mean_reward + explore
                    # find the indices of the top K performance prompts
                    top_k_indices = np.argsort(ucb_scores)[-TOP_K:][::-1]

                print(f"[UCB Strategy] Explore on the following prompts at iteration {j}: {top_k_indices}")
            else:
                # Greedy strategy: choose top K prompts based on raw NDCG scores
                mean_scores = [np.mean(prompt_scores[i]) if prompt_scores[i] else 0.0 for i in range(num_data)]
                top_k_indices = np.argsort(mean_scores)[-TOP_K:][::-1]
                print(f"[Greedy Strategy] Select top {TOP_K} prompts based on raw NDCG scores at iteration {j}: {top_k_indices}")
                print(f"[Greedy Strategy] Mean NDCG scores: {[f'{mean_scores[i]:.4f}' for i in top_k_indices]}")
            
            for i in top_k_indices:
                dataset_dir = f"{CLLM_PROCESSED_DATA_PATH}_{i}"
                predict_cmd = [
                    "python3", "./predict.py",
                    "--dataset", dataset_dir,
                    "--lambda_V", str(LAMBDA_V),
                    "--model_name", MODEL_NAME,
                    "--data_path", CLLM_DATA_PATH,
                    "--iteration", str(j+1)
                ]
                result = subprocess.run(predict_cmd, capture_output=True, text=True)
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "NDCG Returned:" in line:
                        ndcg = float(line.split(': ')[1])
                        prompt_scores[i].append(ndcg)
                        prompt_counts[i] += 1
            
            


    # TODO: Find the top k indices in the mean of prompt_scores
    mean_scores = [np.mean(prompt_scores[i]) for i in range(num_data)]
    top_k_indices = np.argsort(mean_scores)[-TOP_K:][::-1]

    print(f"Top {TOP_K} performance prompts: {top_k_indices}")

    prompt_metas = [None for _ in range(num_data)]
    prompt_clusters = [None for _ in range(num_data)]
    # Read meta and clusters data
    try:
        with open('json/iteration_meta.json', 'r') as f:
            current_results = json.load(f)
            for i, result in enumerate(current_results):
                if i < num_data:  # Add bounds check
                    prompt_metas[i] = result.get('meta')
                    prompt_clusters[i] = result.get('clusters')
    except FileNotFoundError:
        print("Warning: No iteration_meta.json found")

    best_prompts = {
        "metas": [],
        "clusters": []
    }
    
    for i in top_k_indices:
        best_prompts["metas"].append(prompt_metas[i])
        best_prompts["clusters"].append(prompt_clusters[i])

    try:
        with open('json/best_prompts.json', 'w') as f:
            json.dump(best_prompts, f, indent=4)
    except Exception as e:
        print(f"Error saving best prompts: {str(e)}")

    
    print(f"Top prompts info saved to json/best_prompts.json")

    

    # TODO: predict on the whole test dataset using the best prompts and save the results
    top_ndcg = []
    top_recall = []

    for i in top_k_indices:
        dataset_dir = f"{CLLM_PROCESSED_DATA_PATH}_{i}"
        predict_cmd = [
            "python3", "./predict.py",
            "--dataset", dataset_dir,
            "--lambda_V", str(LAMBDA_V),
            "--model_name", MODEL_NAME,
            "--data_path", CLLM_DATA_PATH,
            "--iteration", "0"
        ]
        result = subprocess.run(predict_cmd, capture_output=True, text=True)
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if "NDCG Returned:" in line:
                ndcg = float(line.split(': ')[1])
                top_ndcg.append(ndcg)
            if "Recall at 10:" in line:
                recall = float(line.split(': ')[1])
                top_recall.append(recall)
   
    # Load existing iteration history or create new
    history_file = 'json/iteration_metrics.json'
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {"iterations": []}

        # Append current iteration results
        import datetime
        strategy_name = "UCB" if USE_UCB else "Greedy"
        current_iteration = {
            "iteration": len(history["iterations"]) + 1,
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy": strategy_name,
            "selected_prompts": top_k_indices.tolist(),
            "results": {
                "ndcg_10": top_ndcg,
                "recall_10": top_recall,
                "avg_ndcg_10": np.mean(top_ndcg) if top_ndcg else 0,
                "avg_recall_10": np.mean(top_recall) if top_recall else 0
            }
        }

        history["iterations"].append(current_iteration)

        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"Iteration {current_iteration['iteration']} metrics appended to {history_file}")
        print(f"Strategy used: {strategy_name}")
        print(f"Average NDCG@10: {current_iteration['results']['avg_ndcg_10']:.4f}")
        print(f"Average Recall@10: {current_iteration['results']['avg_recall_10']:.4f}")

    except Exception as e:
        print(f"Error saving iteration metrics: {str(e)}")


if __name__ == "__main__":
    main()