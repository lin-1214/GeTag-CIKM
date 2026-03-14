"""
Run SASRec baseline on i3fresh dataset (without pre-training)

Usage:
    python src/run_sasrec_baseline.py --dataset i3fresh --device cuda
"""

import argparse
import sys
import os
import torch
from logging import getLogger

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.utils import init_seed, init_logger, get_trainer, set_color

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SASRec baseline on i3fresh dataset')
    parser.add_argument(
        '--model', '-m', type=str, default='SASRec',
        help='Model name (default: SASRec)'
    )
    parser.add_argument(
        '--dataset', '-d', type=str, default='i3fresh',
        help='Dataset name (default: i3fresh)'
    )
    parser.add_argument(
        '--config_files', type=str, default='src/models/unisrec/props/finetune.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--hidden_size', type=int, default=300,
        help='Hidden size (default: 300)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device: cpu or cuda (default: cpu)'
    )
    parser.add_argument(
        '--epochs', type=int, default=300,
        help='Number of training epochs (default: 300)'
    )
    parser.add_argument(
        '--stopping_step', type=int, default=10,
        help='Early stopping patience (default: 10)'
    )

    args, _ = parser.parse_known_args()

    print("="*80)
    print("RUNNING SASREC BASELINE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Device: {args.device}")
    print("="*80)

    # Parse config files
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    # Additional config
    config_dict = {
        'data_path': 'data/unisrec',
        'checkpoint_dir': 'output/checkpoints',
        'hidden_size': args.hidden_size,
        'device': args.device,
        'epochs': args.epochs,
        'stopping_step': args.stopping_step,
    }

    # Initialize configuration
    config = Config(
        model=SASRec,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict
    )

    # Set negative sampling
    config['valid_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 100}
    config['test_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 100}

    # Initialize seed
    init_seed(config['seed'], config['reproducibility'])

    # Initialize logger
    os.makedirs('output/logs', exist_ok=True)
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # Load dataset
    dataset_obj = create_dataset(config)
    logger.info(dataset_obj)

    # Split dataset
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    # Initialize model
    model = SASRec(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # Get trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # Train model
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # Load best checkpoint with weights_only=False for PyTorch 2.6
    best_model_path = trainer.saved_model_file
    logger.info(f'Loading best model from {best_model_path}')
    checkpoint = torch.load(
        best_model_path,
        weights_only=False,
        map_location=config['device']
    )
    model.load_state_dict(checkpoint['state_dict'])

    # Evaluate on test set
    test_result = trainer.evaluate(
        test_data, load_best_model=False, show_progress=config['show_progress']
    )

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    print("\n" + "="*80)
    print("BASELINE TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation: {best_valid_result}")
    print(f"Test result: {test_result}")
    print("="*80)
