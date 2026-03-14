'''
MIT License
Copyright (c) 2024 Yaochen Zhu
'''

import re
import os
import sys
import pickle
import argparse
import random
from tqdm import tqdm


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from libs.tokenizer import TokenizerWithUserItemIDTokensBatch

from libs.data import CollaborativeGPTGeneratorBatch
from libs.data import UserItemContentGPTDatasetBatch

from libs.model import GPT4RecommendationBaseModel
from libs.model import CollaborativeGPTwithItemLMHeadBatch
from libs.model import ContentGPTForUserItemWithLMHeadBatch

from config import Config
cfg = Config()

# Configuration for local paths
local_root = "checkpoints"
if not os.path.exists(local_root):
    os.makedirs(local_root, exist_ok=True)

def train_one_data(dataset, lambda_V, data_path, model_name, use_half_precision, share_base_model,device, num_epochs=1, num_pretrained_epochs=1):
    data_root = os.path.join(data_path, dataset)
    meta_path = os.path.join(data_root, "meta.pkl")

    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
        
    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_half_precision and torch.cuda.is_available():
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # device_map="auto",
            device_map={"": f"cuda:{cfg.GPU_INDEX}"},
            torch_dtype=torch.float16
        )
    else:
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # device_map="auto",
            device_map={"": f"cuda:{cfg.GPU_INDEX}"},
            torch_dtype=torch.float32
        )

    # Create extended tokenizer
    extended_tokenizer = TokenizerWithUserItemIDTokensBatch(
        model_name,
        num_users,
        num_items
    )

    # Load data generators
    review_path = os.path.join(data_root, "user_item_texts", "review.pkl")
    review_data_gen = UserItemContentGPTDatasetBatch(extended_tokenizer, review_path)

    train_mat_path = os.path.join(data_root, "train_matrix.npz")
    train_mat = load_npz(train_mat_path)
    collaborative_data_gen = CollaborativeGPTGeneratorBatch(extended_tokenizer, train_mat)

    # Setup config
    config = qwen_model.config
    config.num_users = num_users
    config.num_items = num_items
    config.vocab_size = len(extended_tokenizer.tokenizer)

    # Create models
    content_base_model = GPT4RecommendationBaseModel(config, qwen_model)
    
    # After the first initialization, the config will have the updated (padded) vocab size.
    # We should reuse this updated config for all subsequent models.
    config = content_base_model.config
    
    content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)

    collaborative_base_model = content_base_model if share_base_model else GPT4RecommendationBaseModel(config, qwen_model)
    collaborative_model = CollaborativeGPTwithItemLMHeadBatch(config, collaborative_base_model)

    def setup_model_gradients(model):
        for param in model.parameters():
            param.requires_grad = False
        
        model.base_model.user_embeddings.weight.requires_grad = True
        model.base_model.item_embeddings.weight.requires_grad = True
        model.base_model.qwen_model.get_input_embeddings().weight.requires_grad = True
        
        if hasattr(model, 'lm_head'):
            model.lm_head.weight.requires_grad = True
        
        if hasattr(model, 'item_head'):
            model.item_head.weight.requires_grad = True

    setup_model_gradients(content_model)
    setup_model_gradients(collaborative_model)

    # Training setup
    learning_rate = 1e-3
    batch_size = 8 if use_half_precision else 8
    gradient_accumulation_steps = 4

    # Set up deterministic data loading
    def worker_init_fn(worker_id):
        np.random.seed(cfg.SEED + worker_id)
    
    review_data_loader = DataLoader(review_data_gen, 
                                   batch_size=batch_size, 
                                   collate_fn=review_data_gen.collate_fn,
                                   shuffle=True,
                                   num_workers=0,
                                   worker_init_fn=worker_init_fn,
                                   generator=torch.Generator().manual_seed(cfg.SEED))

    collaborative_data_loader = DataLoader(collaborative_data_gen, 
                                          batch_size=batch_size, 
                                          collate_fn=collaborative_data_gen.collate_fn,
                                          shuffle=True,
                                          num_workers=0,
                                          worker_init_fn=worker_init_fn,
                                          generator=torch.Generator().manual_seed(cfg.SEED))

    content_model.train()
    content_model.to(device)
    collaborative_model.train()
    collaborative_model.to(device)

    review_optimizer = optim.AdamW([p for p in content_model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.01)
    collaborative_optimizer = optim.AdamW([p for p in collaborative_model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.01)

    review_scheduler = optim.lr_scheduler.CosineAnnealingLR(review_optimizer, T_max=num_pretrained_epochs + num_epochs)
    collaborative_scheduler = optim.lr_scheduler.CosineAnnealingLR(collaborative_optimizer, T_max=num_epochs)

    review_best_loss = float('inf')
    collaborative_best_loss = float('inf')

    model_root = os.path.join(local_root, "models", dataset)
    content_model_root = os.path.join(model_root, "content")
    collaborative_model_root = os.path.join(model_root, "collaborative")
    
    os.makedirs(content_model_root, exist_ok=True)
    os.makedirs(collaborative_model_root, exist_ok=True)

    '''
        Define the pretraining loop for the content GPT
    '''
    for epoch in range(num_pretrained_epochs):
        review_total_loss = 0
        
        # Initialize tqdm progress bar
        progress_bar = tqdm(review_data_loader, 
                           desc=f"Content Pretrain Epoch {epoch + 1}", 
                           ncols=100)
        
        for batch_idx, (input_ids_prompt, input_ids_main, attention_mask) in enumerate(progress_bar):
            # Obtain the data
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            try:
                # Forward pass
                outputs = content_model(input_ids_prompt, 
                                       input_ids_main, 
                                       labels_main=input_ids_main,
                                       attention_mask=attention_mask)
                review_loss = outputs[0]
                # print("[Debug] Review Loss", review_loss)
                # Verify loss has gradients
                if not review_loss.requires_grad:
                    print(f"⚠️  Batch {batch_idx}: Loss has no gradients, skipping...")
                    continue
                
                # Scale loss for gradient accumulation
                review_loss = review_loss / gradient_accumulation_steps
                
                # Backward pass
                review_loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(content_model.parameters(), max_norm=1.0)
                    review_optimizer.step()
                    review_optimizer.zero_grad()

                review_total_loss += review_loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({
                    "Loss": f"{review_loss.item() * gradient_accumulation_steps:.4f}",
                    "LR": f"{review_optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        review_average_loss = review_total_loss / len(review_data_loader)
        print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")
        
        # Step scheduler
        review_scheduler.step()

        # Save best model
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            # Save user embeddings
            user_emb_path = os.path.join(content_model_root, f"user_embeddings_{lambda_V}.pt")
            torch.save(content_model.base_model.user_embeddings.state_dict(), user_emb_path)
            print("Content Model Info: ", content_model.base_model.user_embeddings.weight.shape)
            
            # Save item embeddings
            item_emb_path = os.path.join(content_model_root, f"item_embeddings_{lambda_V}.pt")
            torch.save(content_model.base_model.item_embeddings.state_dict(), item_emb_path)
            
            print(f"Saved best content model with loss: {review_best_loss:.4f}")
            
    '''
        Iteratively training the collaborative and content GPT model for recommendations
    '''
    for epoch in range(num_epochs):
        '''
            Optimize the collaborative GPT model
        '''
        collaborative_total_loss = 0
        regularize_total_loss = 0
        
        progress_bar = tqdm(collaborative_data_loader, 
                           desc=f"Epoch {epoch + 1} - Collaborative", 
                           ncols=120)
        
        for batch_idx, (input_ids_prompt, input_ids_main, attention_mask) in enumerate(progress_bar):
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            try:
                # Get content embeddings without gradients
                with torch.no_grad():
                    content_embeds = torch.cat(
                        (content_model.base_model.embed(input_ids_prompt),
                         content_model.base_model.embed(input_ids_main)),
                        axis=1
                    ).to(device)
                    
                # Forward pass of the collaborative GPT
                labels_cleaned = input_ids_main.clone()
                labels_cleaned[labels_cleaned == 151643] = -100  # Replace padding token
                outputs = collaborative_model(input_ids_prompt, 
                                             input_ids_main, 
                                             labels_main=labels_cleaned,
                                             attention_mask=attention_mask,
                                             regularize=True,
                                             lambda_V=lambda_V,
                                             content_embeds=content_embeds)
                collaborative_loss = outputs[0]
                regularize_loss = outputs[1]
                
                # Verify loss has gradients
                if not collaborative_loss.requires_grad:
                    print(f"⚠️  Batch {batch_idx}: Collaborative loss has no gradients, skipping...")
                    continue
                
                # Scale loss for gradient accumulation
                collaborative_loss = collaborative_loss / gradient_accumulation_steps

                # Backward pass
                collaborative_loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(collaborative_model.parameters(), max_norm=1.0)
                    collaborative_optimizer.step()
                    collaborative_optimizer.zero_grad()
                
                collaborative_total_loss += collaborative_loss.item() * gradient_accumulation_steps
                regularize_total_loss += regularize_loss.item()
                
                progress_bar.set_postfix({
                    "Collab": f"{collaborative_loss.item() * gradient_accumulation_steps:.4f}",
                    "Reg": f"{regularize_loss.item():.4f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in collaborative training at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        collaborative_average_loss = collaborative_total_loss / len(collaborative_data_loader)
        regularize_average_loss = regularize_total_loss / len(collaborative_data_loader)
        
        print(f"Epoch {epoch + 1} - Average Collaborative Loss: {collaborative_average_loss:.4f}")
        print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")
        
        # Step scheduler
        collaborative_scheduler.step()
        
        # Save best collaborative model
        if collaborative_average_loss < collaborative_best_loss:
            collaborative_best_loss = collaborative_average_loss

            user_emb_path = os.path.join(collaborative_model_root, f"user_embeddings_{lambda_V}.pt")
            torch.save(collaborative_model.base_model.user_embeddings.state_dict(), user_emb_path)

            item_emb_path = os.path.join(collaborative_model_root, f"item_embeddings_{lambda_V}.pt")
            torch.save(collaborative_model.base_model.item_embeddings.state_dict(), item_emb_path)
            
            print(f"Saved best collaborative model with loss: {collaborative_best_loss:.4f}")

        '''
            Optimize the content GPT model
        '''
        review_total_loss = 0
        regularize_total_loss = 0
        
        progress_bar = tqdm(review_data_loader, 
                           desc=f"Epoch {epoch + 1} - Content", 
                           ncols=120)
        
        for batch_idx, (input_ids_prompt, input_ids_main, attention_mask) in enumerate(progress_bar):
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            try:
                # Get collaborative embeddings without gradients
                with torch.no_grad():
                    collaborative_embeds = collaborative_model.base_model.embed(input_ids_prompt).to(device)
                    
                # Forward pass of the content GPT
                outputs = content_model(input_ids_prompt, 
                                       input_ids_main, 
                                       labels_main=input_ids_main,
                                       attention_mask=attention_mask,
                                       regularize=True,
                                       lambda_V=lambda_V,
                                       collaborative_embeds=collaborative_embeds)
                review_loss = outputs[0]
                regularize_loss = outputs[1]
                
                # Verify loss has gradients
                if not review_loss.requires_grad:
                    print(f"⚠️  Batch {batch_idx}: Content loss has no gradients, skipping...")
                    continue
                
                # Scale loss for gradient accumulation
                review_loss = review_loss / gradient_accumulation_steps

                # Backward pass
                review_loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(content_model.parameters(), max_norm=1.0)
                    review_optimizer.step()
                    review_optimizer.zero_grad()

                review_total_loss += review_loss.item() * gradient_accumulation_steps
                regularize_total_loss += regularize_loss.item()
                
                progress_bar.set_postfix({
                    "Review": f"{review_loss.item() * gradient_accumulation_steps:.4f}",
                    "Reg": f"{regularize_loss.item():.4f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in content training at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        review_average_loss = review_total_loss / len(review_data_loader)
        regularize_average_loss = regularize_total_loss / len(review_data_loader)
        
        print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")
        print(f"Epoch {epoch + 1} - Content Regularize Loss: {regularize_average_loss:.4f}")

        # Step scheduler
        review_scheduler.step()

        # Save best content model
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            user_emb_path = os.path.join(content_model_root, f"user_embeddings_{lambda_V}.pt") 
            torch.save(content_model.base_model.user_embeddings.state_dict(), user_emb_path)
            print("Content Model Info: ", content_model.base_model.user_embeddings.weight.shape)
            
            item_emb_path = os.path.join(content_model_root, f"item_embeddings_{lambda_V}.pt")
            torch.save(content_model.base_model.item_embeddings.state_dict(), item_emb_path)
            
            print(f"Saved best content model with loss: {review_best_loss:.4f}")

    print(f"Training {dataset} completed!")
    print(f"Best content loss: {review_best_loss:.4f}")
    print(f"Best collaborative loss: {collaborative_best_loss:.4f}")

def train (num_data,lambda_V, data_path, use_half_precision, share_base_model, device,  model_name="Qwen/Qwen3-0.6B", num_epochs=1, num_pretrained_epochs=1):
    for i in range(num_data):
        dataset=f"user_session_data_{i}"
        try:
            train_one_data(dataset, lambda_V, data_path, model_name, use_half_precision, share_base_model, device, num_epochs, num_pretrained_epochs)
        except Exception as e:
            print(f"An error occurred during training {dataset}: {e} ")
            sys.exit(1)
    
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
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.GPU_INDEX}")
    else:
        device = torch.device("cpu")

    print(f"Training Using device: {device}")
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_data", type=int, default=1,
        help="specify the number of dataset for experiment")
    parser.add_argument("--lambda_V", type=float, required=True,
        help="specify the regularization parameter")
    parser.add_argument("--data_path", type=str, required=True,
        help="path to your dataset directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
        help="Qwen model name or path")
    parser.add_argument("--use_half_precision", action="store_true",
        help="Use half precision (fp16) for memory efficiency")
    parser.add_argument("--share_base_model", action="store_true",
        help="Share base model between content and collaborative models to save memory")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()
    num_data = args.num_data
    # dataset = args.dataset
    lambda_V = args.lambda_V
    data_path = args.data_path
    model_name = args.model_name
    use_half_precision = args.use_half_precision
    share_base_model = args.share_base_model
    num_epochs = args.num_epochs
    num_pretrained_epochs = args.num_epochs
    train(num_data, lambda_V, data_path, use_half_precision, share_base_model, device,model_name, num_epochs, num_pretrained_epochs)
    print("Training script completed successfully!")

if __name__ == "__main__":
    main()