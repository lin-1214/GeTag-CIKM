from utils import Utils
from prompt import Prompt
from config import Config
import random
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import os
import re

utils = Utils()
prompt = Prompt()
config = Config()

if not os.path.exists('json'):
    os.makedirs('json', exist_ok=True)

prompt_logs = []

def _extract_category_names(cluster_text: str):
    """Extract human-readable category names from clustering output.
    Supports formats like {"Categories": {"Category 1": "Name", ...}} or flat dicts,
    and falls back to regex extraction.
    """
    if not isinstance(cluster_text, str):
        return []
    # Try to locate a JSON object substring
    try:
        start = cluster_text.find('{')
        end = cluster_text.rfind('}')
        candidate = cluster_text[start:end+1] if start != -1 and end != -1 else cluster_text
        data = json.loads(candidate)
        if isinstance(data, dict):
            if 'Categories' in data and isinstance(data['Categories'], dict):
                return [v for v in data['Categories'].values() if isinstance(v, str) and v.strip()]
            # Maybe directly a mapping of Category N -> label
            values = [v for v in data.values() if isinstance(v, str) and v.strip()]
            if values:
                return values
        if isinstance(data, list):
            return [x for x in data if isinstance(x, str) and x.strip()]
    except Exception:
        pass
    # Regex fallback: capture "Category ...": "<label>"
    matches = re.findall(r'\bCategory\s*\d+\b\s*:\s*"([^"]+)"', cluster_text)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    # Another fallback: lines like - Category X: label
    matches = re.findall(r'\bCategory\s*\d+\b\s*[:\-]\s*([\w\-\s/&]+)', cluster_text)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return []

def main():
    utils.set_seed()
    config.check_overall_config()
    
    # Initialize variables for storing previous best prompts
    previous_best_metas = []
    previous_best_clusters = []
    previous_best_tags = []
    previous_best_high_tags = []
    previous_best_low_tags = []

    # Load best prompts info from previous iteration
    try:
        with open('json/best_prompts.json', 'r') as f:
            best_prompts_info = json.load(f)
            previous_best_metas = best_prompts_info.get('metas', [])
            previous_best_clusters = best_prompts_info.get('clusters', [])
            previous_best_tags = best_prompts_info.get('tags', [])
            previous_best_high_tags = best_prompts_info.get('high_tags', [])
            previous_best_low_tags = best_prompts_info.get('low_tags', [])
            
            # Adjust K_AUG: divide total augmentations among TOP_K groups
            # Each group gets K_AUG prompts (1 original + K_AUG-1 augments)
            if previous_best_metas:
                config.K_AUG = config.INITIAL_K_AUG // len(previous_best_metas)
            else:
                config.K_AUG = config.INITIAL_K_AUG
    except FileNotFoundError:
        config.K_AUG = config.INITIAL_K_AUG

    # Domain-specific CSV loading logic
    if "amazon" in config.PREPROCESSED_DATA_PATH.lower():
        import csv
        print("Loading Amazon dataset (comma-separated ASINs, no header)...")
        
        # First pass: find maximum number of columns
        max_cols = 0
        with open(config.PREPROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > max_cols:
                    max_cols = len(row)
        print(f"Amazon dataset: Maximum ASINs per session: {max_cols}")
        
        # Second pass: read all ASINs
        df = pd.read_csv(
            config.PREPROCESSED_DATA_PATH,
            header=None,
            names=list(range(max_cols)),
            engine='python',
            skipinitialspace=True,
            on_bad_lines='skip'
        )
        print(f"Amazon dataset: Loaded {len(df)} sessions")
        
    elif "movie" in config.PREPROCESSED_DATA_PATH.lower():
        import csv
        
        # First pass: find maximum number of columns
        print("Scanning movie dataset to determine column count...")
        max_cols = 0
        with open(config.PREPROCESSED_DATA_PATH, 'r') as f:
            reader = csv.reader(f, quotechar='"')
            for row in reader:
                if len(row) > max_cols:
                    max_cols = len(row)
        print(f"Maximum columns found: {max_cols}")
        
        # Second pass: read with proper column count
        df = pd.read_csv(
            config.PREPROCESSED_DATA_PATH, 
            header=None,
            names=list(range(max_cols)),  # Explicitly set column names
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
            on_bad_lines='skip',
            engine='python'  # Python engine handles variable columns better
        )
        
        # Filter out sessions that are too long to avoid token overflow
        if config.MAX_SESSION_LENGTH is not None:
            session_lengths = df.notna().sum(axis=1)
            original_count = len(df)
            df = df[session_lengths <= config.MAX_SESSION_LENGTH].reset_index(drop=True)
            print(f"Movie dataset: filtered from {original_count} to {len(df)} users (max {config.MAX_SESSION_LENGTH} ratings per user)")
    
    elif "yelp" in config.PREPROCESSED_DATA_PATH.lower():
        import csv
        print("Loading Yelp dataset (comma-separated business_ids, no header)...")
        
        # First pass: find maximum number of columns
        max_cols = 0
        with open(config.PREPROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > max_cols:
                    max_cols = len(row)
        print(f"Yelp dataset: Maximum business_ids per session: {max_cols}")
        
        # Second pass: read all business_ids
        df = pd.read_csv(
            config.PREPROCESSED_DATA_PATH,
            header=None,
            names=list(range(max_cols)),
            engine='python',
            skipinitialspace=True,
            on_bad_lines='skip'
        )
        print(f"Yelp dataset: Loaded {len(df)} sessions")
        
    else:
        # Food/default e-commerce dataset
        try:
            df = pd.read_csv(config.PREPROCESSED_DATA_PATH)
        except pd.errors.ParserError as exc:
            print(f"Standard CSV loader failed: {exc}")
            print("Falling back to ragged CSV loader for food domain...")
            import csv

            max_cols = 0
            with open(config.PREPROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) > max_cols:
                        max_cols = len(row)
            print(f"Detected maximum column count: {max_cols}")

            df = pd.read_csv(
                config.PREPROCESSED_DATA_PATH,
                header=None,
                names=list(range(max_cols)),
                engine="python",
                skipinitialspace=True,
            )

    # pick a slice of sessions based on start index and size from config
    start_index = config.SAMPLE_START_INDEX
    end_index = start_index + config.SAMPLE_SIZE
    sampled_df_for_classification = df.iloc[start_index:end_index].copy()

    """
    Iteration loop start here
    """

    # 1. Generate session clusters
    sampled_df_for_cluster = utils.sample_session(df.copy())
    sessions = [utils.format_session(r, include_tags=True) for _, r in sampled_df_for_cluster.iterrows()]
    body = "\n\n".join(sessions)

    results = []
    current_iterations = []

    # If we have previous best prompts, generate variations for each
    if previous_best_metas:
        # Process each best prompt completely before moving to the next
        # This keeps each group's prompts (original + augments) consecutive
        for idx, (prev_meta, prev_clusters) in enumerate(zip(previous_best_metas, previous_best_clusters)):
            # First, add the original best prompt to the candidates
            current_idx = len(current_iterations)
            iteration_result = {
                "step": current_idx,
                "meta": prev_meta,
                "clusters": prev_clusters,
                "is_original": True,
                "parent_index": idx
            }
            results.append(iteration_result)
            current_iterations.append(iteration_result)

            print(f"Added original best prompt to the candidates (group {idx})")
            prompt_logs.append({
                "type": "original_best",
                "result_step": current_idx,
                "parent_index": idx,
                "meta": prev_meta,
                "clusters": prev_clusters
            })

            # Now generate augmented variations for this specific best prompt
            # Prefer explicit high/low tags; otherwise split 'tags' list in half
            high_ref = previous_best_high_tags[idx] if idx < len(previous_best_high_tags) else None
            low_ref = previous_best_low_tags[idx] if idx < len(previous_best_low_tags) else None
            if (high_ref is None or low_ref is None) and idx < len(previous_best_tags):
                all_tags = previous_best_tags[idx] or []
                split_idx = max(1, len(all_tags)//2) if len(all_tags) > 0 else 0
                high_ref = all_tags[:split_idx]
                low_ref = all_tags[split_idx:]
            
            # Keep original meta/clusters as baseline for all augmentations
            original_meta = prev_meta
            original_clusters = prev_clusters
            
            for step in range(config.K_AUG - 1):
                current_step = len(current_iterations)
                
                # Augment from original meta (not chained)
                augmentation_prompt = prompt.generate_augmentation_prompt(body=original_meta)
                prompt_logs.append({
                    "type": "augmentation",
                    "result_step": current_step,
                    "prompt": augmentation_prompt
                })
                augmented_meta = utils.augment_data(augmentation_prompt)
                
                # Generate cluster using augmented meta
                cluster_prompt, meta = prompt.generate_cluster_prompt(
                    body=body,
                    meta=augmented_meta,
                    previous_clusters=original_clusters,
                    previous_high_tags=high_ref,
                    previous_low_tags=low_ref
                )
                cluster_str = utils.generate_cluster(cluster_prompt)
                
                iteration_result = {
                    "step": current_step,
                    "meta": meta,
                    "clusters": cluster_str,
                    "parent_meta": prev_meta,  # Track which best prompt this came from
                    "parent_index": idx
                }
                results.append(iteration_result)
                current_iterations.append(iteration_result)
                prompt_logs.append({
                    "type": "cluster",
                    "result_step": current_step,
                    "parent_index": idx,
                    "prompt": cluster_prompt,
                    "meta": meta,
                    "high_tags": high_ref,
                    "low_tags": low_ref
                })
            
    else:
        # First iteration - generate initial prompt and variations
        # Step 0: Generate original prompt
        current_step = 0
        cluster_prompt, original_meta = prompt.generate_cluster_prompt(
            body=body,
            meta=None,
            previous_clusters=None
        )
        cluster_str = utils.generate_cluster(cluster_prompt)
        
        iteration_result = {
            "step": current_step,
            "meta": original_meta,
            "clusters": cluster_str,
        }
        results.append(iteration_result)
        current_iterations.append(iteration_result)
        prompt_logs.append({
            "type": "cluster",
            "result_step": current_step,
            "parent_index": None,
            "prompt": cluster_prompt,
            "meta": original_meta,
            "high_tags": None,
            "low_tags": None
        })
        
        # Steps 1 to K_AUG-1: Generate augmented variations from original
        for step in range(1, config.K_AUG):
            current_step = len(current_iterations)
            
            # Augment from original meta
            augmentation_prompt = prompt.generate_augmentation_prompt(body=original_meta)
            prompt_logs.append({
                "type": "augmentation",
                "result_step": current_step,
                "prompt": augmentation_prompt
            })
            augmented_meta = utils.augment_data(augmentation_prompt)
            
            # Generate cluster using augmented meta
            cluster_prompt, meta = prompt.generate_cluster_prompt(
                body=body,
                meta=augmented_meta,
                previous_clusters=cluster_str
            )
            cluster_str = utils.generate_cluster(cluster_prompt)
            
            iteration_result = {
                "step": current_step,
                "meta": meta,
                "clusters": cluster_str,
            }
            results.append(iteration_result)
            current_iterations.append(iteration_result)
            prompt_logs.append({
                "type": "cluster",
                "result_step": current_step,
                "parent_index": None,
                "prompt": cluster_prompt,
                "meta": meta,
                "high_tags": None,
                "low_tags": None
            })

    # Save iterations data
    with open('json/iteration_meta.json', 'w') as f:
        json.dump(current_iterations, f, indent=4)

    print(f"Generated {len(results)} variations")
    

    # 2. Generate classification results
    for step, it in enumerate(tqdm(results, desc="Generating classification results")):

        valid_indices = []
        formatted_sessions = []
        all_predictions = []
        
        # First, format all sessions and collect valid ones
        for idx, row in sampled_df_for_classification.iterrows():
            formatted = utils.format_session(row)
            # Skip empty sessions entirely
            if formatted == "Empty session":
                continue
                
            valid_indices.append(idx)
            formatted_sessions.append(formatted)

        for i in range(0, len(formatted_sessions), config.BATCH_SIZE):
            size = config.BATCH_SIZE
            if (i + size) > len(formatted_sessions):
                size = len(formatted_sessions) - i
            
            batch_sessions = formatted_sessions[i:i+size]
            
            # Join the formatted sessions with clear separators
            batch_text = "\n\n### SESSION " + "\n\n### SESSION ".join([str(j+1) + ":\n" + session for j, session in enumerate(batch_sessions)])

            # Extract clean category names for the classification prompt
            category_names = _extract_category_names(it["clusters"]) or []
            # Fallback: if nothing extracted, try to pass raw text but warn
            if not category_names:
                print("Warning: Failed to extract category names from clusters; passing raw clusters text to the classifier.")
                categories_for_prompt = it["clusters"]
            else:
                categories_for_prompt = json.dumps(category_names, ensure_ascii=False)

            classification_prompt = prompt.generate_classification_prompt(batch_size=size, categories=categories_for_prompt, body=batch_text)
            prompt_logs.append({
                "type": "classification",
                "result_step": step,
                "batch_start_index": i,
                "batch_size": size,
                "categories_reference": categories_for_prompt,
                "prompt": classification_prompt
            })
            
            # Get predictions for the batch
            success = False
            max_retries = config.MAX_RETRIES
            retry_count = 0
            
            while not success and retry_count < max_retries:
                try:
                    result = utils.predict_cluster(classification_prompt)
                    predictions_list = json.loads(result)

                    if config.MULTI_LABEL:
                        # Normalize to list of label lists
                        def normalize_to_labels(item):
                            labels = []
                            if isinstance(item, dict):
                                if "predicted_classes" in item and isinstance(item["predicted_classes"], list):
                                    labels = [str(x).strip() for x in item["predicted_classes"] if isinstance(x, str) and str(x).strip()]
                                elif "predicted_class" in item and isinstance(item["predicted_class"], str):
                                    labels = [item["predicted_class"].strip()]
                            elif isinstance(item, list):
                                labels = [str(x).strip() for x in item if isinstance(x, str) and str(x).strip()]
                            elif isinstance(item, str):
                                s = item.strip()
                                if s.startswith('[') and s.endswith(']'):
                                    try:
                                        parsed = json.loads(s)
                                        if isinstance(parsed, list):
                                            labels = [str(x).strip() for x in parsed if isinstance(x, str) and str(x).strip()]
                                    except Exception:
                                        pass
                                if not labels:
                                    labels = [p.strip() for p in s.split(config.LABEL_SEPARATOR) if p.strip()]
                            # Deduplicate and respect bounds
                            deduped = []
                            seen = set()
                            for lab in labels:
                                if lab not in seen:
                                    seen.add(lab)
                                    deduped.append(lab)
                            if len(deduped) > config.MAX_LABELS_PER_SESSION:
                                deduped = deduped[:config.MAX_LABELS_PER_SESSION]
                            if len(deduped) < config.MIN_LABELS_PER_SESSION:
                                # if empty, fallback to Unknown
                                if not deduped:
                                    deduped = ["Unknown"]
                            return deduped
                        batch_predictions = [normalize_to_labels(pred) for pred in predictions_list]
                    else:
                        # Single-label behavior
                        batch_predictions = [pred if isinstance(pred, str) else pred.get("predicted_class", "Unknown") for pred in predictions_list]

                    if len(batch_predictions) != size:
                        print(f"Prediction count mismatch. Expected {size}, got {len(batch_predictions)}. Retrying...")
                        retry_count += 1
                        continue
                    
                    # If we get here, the batch was successful
                    success = True
                    all_predictions.extend(batch_predictions)
                    # print(f"Successfully processed batch starting at index {i}")
                    
                except json.JSONDecodeError:
                    print(f"Error parsing JSON response: {result}")
                    retry_count += 1
                    print(f"Retrying batch (attempt {retry_count}/{max_retries})...")
            
            if not success:
                print(f"Failed to process batch after {max_retries} attempts. Skipping batch starting at index {i}")
                # Add placeholder predictions to maintain alignment with indices
                if config.MULTI_LABEL:
                    all_predictions.extend([["Failed_Prediction"]] * size)
                else:
                    all_predictions.extend(["Failed_Prediction"] * size)

        # Filter the dataframe to keep only valid rows
        labeled_df = sampled_df_for_classification.loc[valid_indices].copy()
        
        # Add predictions as a new column at the beginning of the dataframe
        if config.MULTI_LABEL:
            labeled_df.insert(0, 'Class', [json.dumps(p, ensure_ascii=False) for p in all_predictions])
            # remove failed predictions & maintain only the most frequent classes (T_CLUSTER)
            # Parse to lists
            labels_lists = labeled_df['Class'].apply(lambda s: json.loads(s) if isinstance(s, str) else (s if isinstance(s, list) else []))
            # Remove rows where prediction failed
            mask_ok = labels_lists.apply(lambda lst: isinstance(lst, list) and len(lst) > 0 and lst != ["Failed_Prediction"])
            labeled_df = labeled_df[mask_ok].copy()
            labels_lists = labels_lists[mask_ok]
            # Count class frequencies across all labels
            from collections import Counter
            counts = Counter()
            for labs in labels_lists:
                counts.update(labs)
            print(f"Number of original classes: {len(counts)}")
            # Filter to top classes if needed
            top_n = config.T_CLUSTERS + config.T_VARIANCE
            if len(counts) > top_n:
                top_classes = set([cls for cls, _ in counts.most_common(top_n)])
                # Filter labels per row
                filtered_labels = labels_lists.apply(lambda labs: [l for l in labs if l in top_classes])
                # Drop rows that become empty after filtering
                nonempty_mask = filtered_labels.apply(lambda labs: len(labs) > 0)
                labeled_df = labeled_df[nonempty_mask].copy()
                labeled_df.loc[:, 'Class'] = filtered_labels[nonempty_mask].apply(lambda x: json.dumps(x, ensure_ascii=False))
                # Recompute counts
                counts = Counter()
                for labs in filtered_labels[nonempty_mask]:
                    counts.update(labs)
            else:
                # ensure 'Class' is json string
                labeled_df.loc[:, 'Class'] = labels_lists.apply(lambda x: json.dumps(x, ensure_ascii=False))
            print(f"Number of classes after filtering: {len(counts)}")
            print(f"SAMPLE_SIZE={len(sampled_df_for_classification)}, valid={len(valid_indices)}, after_failed={labeled_df.shape[0]}, unique_classes={len(counts)}")
        else:
            labeled_df.insert(0, 'Class', all_predictions)
            # remove failed predictions & maintain only the most frequent class (T_CLUSTER)
            labeled_df = labeled_df[labeled_df['Class'] != 'Failed_Prediction']
            class_stats = labeled_df['Class'].value_counts()

            print(f"Number of original classes: {len(class_stats)}")
            
            if len(class_stats) > config.T_CLUSTERS + config.T_VARIANCE:
                top_classes = class_stats.head(config.T_CLUSTERS + config.T_VARIANCE).index.tolist()
                labeled_df = labeled_df[labeled_df['Class'].isin(top_classes)]
                # Recompute stats after filtering for accurate reporting
                class_stats = labeled_df['Class'].value_counts()

            print(f"Number of classes after filtering: {labeled_df['Class'].nunique()}")
            print(f"SAMPLE_SIZE={len(sampled_df_for_classification)}, "
            f"valid={len(valid_indices)}, "
            f"after_failed={labeled_df.shape[0]}, "
            f"unique_classes={labeled_df['Class'].nunique()}")

        labeled_df.to_csv(f"{config.CLASSIFIED_DATA_PATH}_{step}.csv", index=False)
        print(f"Predictions saved to {config.CLASSIFIED_DATA_PATH}_{step}.csv")

    try:
        with open('json/iteration_prompts.json', 'w', encoding='utf-8') as f:
            json.dump(prompt_logs, f, ensure_ascii=False, indent=2)
        print("Prompt logs saved to json/iteration_prompts.json")
    except Exception as e:
        print(f"Warning: Failed to save prompt logs: {e}")



if __name__ == "__main__":
    main()
    
    
    
    


