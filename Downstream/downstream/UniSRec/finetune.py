"""
Fine-tune UniSRec on downstream dataset

Usage:
    python downstream/UniSRec/finetune.py --dataset food_native --checkpoint checkpoints/UniSRec-FHCKM-300.pth
    python downstream/UniSRec/finetune.py --dataset games_native --checkpoint checkpoints/UniSRec-FHCKM-300.pth
    python downstream/UniSRec/finetune.py --dataset yelp_native --checkpoint checkpoints/UniSRec-FHCKM-300.pth
"""

import argparse
import os
import sys
import pickle
from logging import getLogger
import torch

# Add current directory to path to import unisrec module
sys.path.insert(0, os.path.dirname(__file__))

from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color

# Register recbole.config module as 'config' for checkpoint loading compatibility
import recbole.config
sys.modules['config'] = recbole.config

# Import from the unisrec module
from unisrec import UniSRec
from unisrec.data.dataset import UniSRecDataset


def finetune(dataset, pretrained_file, data_path='data/preprocessed/UniSRec', fix_enc=True, **kwargs):
    """
    Fine-tune pre-trained UniSRec model on downstream dataset

    Args:
        dataset: dataset name (e.g., 'food_native', 'games_native', 'yelp_native')
        pretrained_file: path to pre-trained checkpoint
        data_path: base path for data directory (default: 'data/preprocessed/UniSRec')
        fix_enc: whether to fix encoder parameters during fine-tuning
        **kwargs: additional config parameters
    """
    # Configuration files (relative to this script's directory)
    script_dir = os.path.dirname(__file__)
    props = [
        os.path.join(script_dir, 'unisrec/props/UniSRec.yaml'),
        os.path.join(script_dir, 'unisrec/props/finetune.yaml')
    ]
    print(f"Loading configs: {props}")

    # Override data path to use our data directory
    # RecBole automatically appends dataset name, so just use base path
    # Training checkpoints will be saved to output/checkpoints/
    config_dict = {
        'data_path': data_path,
        'checkpoint_dir': 'output/checkpoints',  # Training output (gitignored)
        **kwargs
    }

    # Initialize configuration
    config = Config(
        model=UniSRec,
        dataset=dataset,
        config_file_list=props,
        config_dict=config_dict
    )

    # Set negative sampling for evaluation
    config['valid_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 100}
    config['test_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 100}

    # Initialize seed for reproducibility
    init_seed(config['seed'], config['reproducibility'])

    # Initialize logger
    os.makedirs('output/logs', exist_ok=True)
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # Load dataset
    dataset_obj = UniSRecDataset(config)
    logger.info(dataset_obj)

    # Split dataset
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    # Initialize model
    model = UniSRec(config, train_data.dataset).to(config['device'])

    # Load pre-trained checkpoint
    if pretrained_file and os.path.exists(pretrained_file):
        checkpoint = torch.load(
            pretrained_file,
            weights_only=False,
            map_location=config['device']
        )
        logger.info(f'Loading from {pretrained_file}')

        # Get source dataset name from checkpoint
        try:
            source_dataset = checkpoint['config']['dataset']
        except (KeyError, TypeError):
            source_dataset = 'pretrained'
        logger.info(f'Transfer [{source_dataset}] -> [{dataset}]')

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if fix_enc:
            logger.info('Fixing encoder parameters.')
            for param in model.position_embedding.parameters():
                param.requires_grad = False
            for param in model.trm_encoder.parameters():
                param.requires_grad = False
    else:
        logger.warning(f'Checkpoint not found: {pretrained_file}. Training from scratch.')

    logger.info(model)

    # Get trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # Train model
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # Manually load best checkpoint (fix for PyTorch 2.6 weights_only issue)
    best_model_path = trainer.saved_model_file
    logger.info(f'Loading best model from {best_model_path}')
    checkpoint = torch.load(
        best_model_path,
        weights_only=False,  # Required for PyTorch 2.6+
        map_location=config['device']
    )
    model.load_state_dict(checkpoint['state_dict'])

    # Evaluate on test set
    test_result = trainer.evaluate(
        test_data, load_best_model=False, show_progress=config['show_progress']
    )

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'model': config['model'],
        'dataset': config['dataset'],
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune UniSRec on downstream dataset')
    parser.add_argument(
        '--dataset', '-d', type=str, required=True,
        help='Dataset name (e.g., food_native, games_native, yelp_native)'
    )
    parser.add_argument(
        '--data_path', type=str, default='data/preprocessed/UniSRec',
        help='Base path for data directory (default: data/preprocessed/UniSRec)'
    )
    parser.add_argument(
        '--checkpoint', '-p', type=str, default='checkpoints/UniSRec-FHCKM-300.pth',
        help='Path to pre-trained checkpoint (default: checkpoints/UniSRec-FHCKM-300.pth)'
    )
    parser.add_argument(
        '--fix_enc', '-f', type=bool, default=True,
        help='Fix encoder parameters during fine-tuning'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device to use (cpu or cuda)'
    )
    parser.add_argument(
        '--epochs', type=int, default=300,
        help='Maximum number of epochs'
    )
    parser.add_argument(
        '--stopping_step', type=int, default=10,
        help='Number of epochs for early stopping'
    )

    args, unparsed = parser.parse_known_args()

    print("="*80)
    print("FINE-TUNING UNISREC")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Fix encoder: {args.fix_enc}")
    print("="*80)

    result = finetune(
        args.dataset,
        pretrained_file=args.checkpoint,
        data_path=args.data_path,
        fix_enc=args.fix_enc,
        device=args.device,
        epochs=args.epochs,
        stopping_step=args.stopping_step
    )

    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"Best validation {result['model']}: {result['best_valid_result']}")
    print(f"Test result: {result['test_result']}")
    print("="*80)
