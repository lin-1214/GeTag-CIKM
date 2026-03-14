# multibeam_label.py

import pandas as pd
import json
import csv
import re
import os
from collections import Counter
from tqdm import tqdm
from config import Config
from utils import Utils
from prompt import Prompt

config = Config()
utils = Utils()
prompt = Prompt()

def main():
    # ========== PHASE 1: 生成 20 个标签（单次）==========
    print("Phase 1: Generating 20 category labels...")
    
    # 1.1 加载数据
    df = load_dataset()  # 复用现有逻辑
    
    # 1.2 Sample sessions for clustering
    sampled_df_for_cluster = utils.sample_session(df.copy())
    sessions = [utils.format_session(r, include_tags=True) 
                for _, r in sampled_df_for_cluster.iterrows()]
    body = "\n\n".join(sessions)
    
    # 1.3 生成 cluster（1次）
    cluster_prompt, meta = prompt.generate_cluster_prompt(body=body, meta=None)
    clusters_str = utils.generate_cluster(cluster_prompt)
    category_names = extract_category_names(clusters_str)
    
    print(f"Generated {len(category_names)} categories: {category_names}")
    
    # ========== PHASE 2: Multi-beam Classification ==========
    print(f"\nPhase 2: Multi-beam classification (k={config.MULTIBEAM_K})...")
    
    # 2.1 准备所有 sessions (连续取一段用于分类)
    # 从 SAMPLE_START_INDEX 开始，连续取 SAMPLE_SIZE 笔
    start_index = config.SAMPLE_START_INDEX
    end_index = start_index + config.SAMPLE_SIZE
    sampled_df_for_classification = df.iloc[start_index:end_index].reset_index(drop=True)
    print(f"Classification range: [{start_index}:{end_index}], total {len(sampled_df_for_classification)} sessions")
    
    # 2.2 准备多个 beam 的结果容器
    all_beam_results = [[] for _ in range(config.MULTIBEAM_K)]  # 每个 beam 的完整结果
    all_final_results = []  # 最终投票后的结果
    all_prompts = []  # 记录所有 prompts
    
    # 2.3 对每个 session 进行 multi-beam 分类
    for idx, row in tqdm(sampled_df_for_classification.iterrows(), 
                         total=len(sampled_df_for_classification)):
        formatted = utils.format_session(row)
        if formatted == "Empty session":
            all_final_results.append([])
            for beam_i in range(config.MULTIBEAM_K):
                all_beam_results[beam_i].append([])
            continue
        
        # Multi-beam: 分类 k 次
        votes = []
        session_prompts = {
            "session_index": idx,
            "formatted_session": formatted,
            "beams": []
        }
        
        for beam_i in range(config.MULTIBEAM_K):
            labels, prompt_used = classify_single_session_with_prompt(
                formatted, 
                category_names, 
                temperature=config.MULTIBEAM_TEMPERATURE
            )
            votes.extend(labels)
            all_beam_results[beam_i].append(labels)  # 记录该 beam 的结果
            
            session_prompts["beams"].append({
                "beam_id": beam_i,
                "prompt": prompt_used,
                "predicted_labels": labels
            })
        
        # 统计频率，筛选最终标签
        final_labels = filter_by_frequency(
            votes, 
            k=config.MULTIBEAM_K,
            threshold=config.MULTIBEAM_MIN_FREQ,
            top_n=config.MULTIBEAM_TOP_N
        )
        all_final_results.append(final_labels)
        session_prompts["final_labels"] = final_labels
        session_prompts["vote_distribution"] = dict(Counter(votes))
        all_prompts.append(session_prompts)
    
    # ========== PHASE 3: 保存结果 ==========
    print("\nPhase 3: Saving results...")
    save_results(
        sampled_df_for_classification, 
        all_beam_results, 
        all_final_results, 
        all_prompts,
        category_names
    )
    print("Done!")

def classify_single_session_with_prompt(formatted_session, category_names, temperature):
    """对单个 session 分类一次，并返回使用的 prompt"""
    # 生成 classification prompt
    classification_prompt = prompt.generate_classification_prompt(
        batch_size=1,
        categories=json.dumps(category_names, ensure_ascii=False),
        body=f"### SESSION 1:\n{formatted_session}"
    )
    
    # 调用 LLM (传递自定义 temperature，不使用固定 seed 以增加多样性)
    response = utils.predict_cluster(classification_prompt, temperature=temperature, use_seed=False)
    
    # 解析返回的标签
    labels = parse_classification_response(response)
    return labels, classification_prompt

def filter_by_frequency(votes, k, threshold, top_n):
    """
    根据频率筛选标签（寧缺勿濫原则）
    
    Args:
        votes: 所有投票结果（可能有重复）
        k: 总共投票次数
        threshold: 最低频率阈值（0-1），例如 0.40 表示至少出现 40% 次数
        top_n: 最多返回几个标签（上限，不是强制数量）
    
    Returns:
        满足阈值的标签列表，按频率降序排列，最多 top_n 个
        如果满足阈值的标签少于 top_n，则只返回实际满足的标签（不凑数）
    
    Examples:
        - 满足阈值的有 3 个，top_n=5 → 返回 3 个 ✅
        - 满足阈值的有 6 个，top_n=5 → 返回前 5 个 ✅
        - 没有满足阈值的 → 返回空列表 ✅
    """
    if not votes:
        return []
    
    label_counts = Counter(votes)
    min_count = int(k * threshold)
    
    # 1. 筛选：只保留频率 >= threshold 的标签
    qualifying = [
        (label, count) for label, count in label_counts.items()
        if count >= min_count
    ]
    
    # 2. 排序：频率降序，同频率按字母序（保证稳定性）
    qualifying.sort(key=lambda x: (-x[1], x[0]))
    
    # 3. 取前 top_n 个（如果不够 top_n 个，就返回所有满足阈值的）
    filtered = qualifying[:top_n]
    
    return [label for label, count in filtered]

def save_results(df, all_beam_results, all_final_results, all_prompts, category_names):
    """
    保存结果：
    1. 每个 beam 的 CSV (包含原始 session 数据)
    2. 最终投票后的 CSV (包含原始 session 数据)
    3. 所有 prompts 到 JSON
    
    输出目录根据 DOMAIN 和 INCLUDE_TAG 自动命名
    """
    # 根据 domain 和 tag 自动生成目录名
    domain = config.DOMAIN
    tag_type = config.INCLUDE_TAG
    output_dir = f'../label_data/multibeam_{domain}_{tag_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存每个 beam 的结果
    for beam_i, beam_results in enumerate(all_beam_results):
        beam_df = df.copy()
        # 将标签列表转换为字符串格式
        if config.MULTI_LABEL:
            beam_df.insert(0, 'Class', [json.dumps(labels, ensure_ascii=False) for labels in beam_results])
        else:
            beam_df.insert(0, 'Class', [labels[0] if labels else '' for labels in beam_results])
        
        beam_path = f'{output_dir}/classified_data_multibeam_beam_{beam_i}.csv'
        beam_df.to_csv(beam_path, index=False)
        print(f"Saved beam {beam_i} to {beam_path}")
    
    # 2. 保存最终投票结果（与 classified_data_0.csv 格式一致）
    final_df = df.copy()
    if config.MULTI_LABEL:
        final_df.insert(0, 'Class', [json.dumps(labels, ensure_ascii=False) for labels in all_final_results])
    else:
        final_df.insert(0, 'Class', [labels[0] if labels else '' for labels in all_final_results])
    
    final_path = f'../label_data/classified_data_multibeam_{domain}_{tag_type}.csv'
    final_df.to_csv(final_path, index=False)
    print(f"Saved final results to {final_path}")
    
    # 3. 保存所有 prompts 到 JSON
    prompts_meta = {
        "categories_used": category_names,
        "config": {
            "MULTIBEAM_K": config.MULTIBEAM_K,
            "MULTIBEAM_TEMPERATURE": config.MULTIBEAM_TEMPERATURE,
            "MULTIBEAM_MIN_FREQ": config.MULTIBEAM_MIN_FREQ,
            "MULTIBEAM_TOP_N": config.MULTIBEAM_TOP_N
        },
        "total_sessions": len(all_prompts),
        "sessions": all_prompts
    }
    
    prompts_path = f'{output_dir}/multibeam_prompts.json'
    with open(prompts_path, 'w', encoding='utf-8') as f:
        json.dump(prompts_meta, f, ensure_ascii=False, indent=2)
    print(f"Saved prompts to {prompts_path}")
    
    # 4. 保存统计信息
    stats = {
        "total_sessions": len(all_final_results),
        "empty_sessions": sum(1 for labels in all_final_results if not labels),
        "label_distribution": dict(Counter([label for labels in all_final_results for label in labels])),
        "avg_labels_per_session": sum(len(labels) for labels in all_final_results) / len(all_final_results) if all_final_results else 0
    }
    
    stats_path = f'{output_dir}/multibeam_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved statistics to {stats_path}")

def load_dataset():
    """从 config.PREPROCESSED_DATA_PATH 加载数据集"""
    # Domain-specific CSV loading logic (复制自 label_phase.py)
    if "amazon" in config.PREPROCESSED_DATA_PATH.lower():
        print("Loading Amazon dataset (comma-separated ASINs, no header)...")
        max_cols = 0
        with open(config.PREPROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > max_cols:
                    max_cols = len(row)
        print(f"Amazon dataset: Maximum ASINs per session: {max_cols}")
        
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
        print("Scanning movie dataset to determine column count...")
        max_cols = 0
        with open(config.PREPROCESSED_DATA_PATH, 'r') as f:
            reader = csv.reader(f, quotechar='"')
            for row in reader:
                if len(row) > max_cols:
                    max_cols = len(row)
        print(f"Maximum columns found: {max_cols}")
        
        df = pd.read_csv(
            config.PREPROCESSED_DATA_PATH, 
            header=None,
            names=list(range(max_cols)),
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
            on_bad_lines='skip',
            engine='python'
        )
        
        if config.MAX_SESSION_LENGTH is not None:
            session_lengths = df.notna().sum(axis=1)
            original_count = len(df)
            df = df[session_lengths <= config.MAX_SESSION_LENGTH].reset_index(drop=True)
            print(f"Movie dataset: filtered from {original_count} to {len(df)} users")
    else:
        # Food/default e-commerce dataset
        try:
            df = pd.read_csv(config.PREPROCESSED_DATA_PATH)
        except pd.errors.ParserError:
            print("Standard CSV loader failed. Using ragged CSV loader...")
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
    
    return df

def extract_category_names(cluster_text: str):
    """Extract category names from cluster JSON output"""
    if not isinstance(cluster_text, str):
        return []
    
    try:
        start = cluster_text.find('{')
        end = cluster_text.rfind('}')
        candidate = cluster_text[start:end+1] if start != -1 and end != -1 else cluster_text
        data = json.loads(candidate)
        
        if isinstance(data, dict):
            if 'Categories' in data and isinstance(data['Categories'], dict):
                return [v for v in data['Categories'].values() if isinstance(v, str) and v.strip()]
            values = [v for v in data.values() if isinstance(v, str) and v.strip()]
            if values:
                return values
        if isinstance(data, list):
            return [str(item).strip() for item in data if isinstance(item, str) and item.strip()]
    except Exception as e:
        print(f"JSON parsing failed: {e}, falling back to regex")
    
    # Fallback: regex extraction
    pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, cluster_text)
    if matches:
        return [v for k, v in matches if v.strip()]
    
    return []

def parse_classification_response(response):
    """解析 LLM 返回的分类结果"""
    try:
        predictions_list = json.loads(response)
        
        if not isinstance(predictions_list, list) or len(predictions_list) == 0:
            return []
        
        # 获取第一个元素（batch_size=1）
        item = predictions_list[0]
        labels = []
        
        if isinstance(item, dict):
            if "predicted_classes" in item and isinstance(item["predicted_classes"], list):
                labels = [str(x).strip() for x in item["predicted_classes"] 
                         if isinstance(x, str) and str(x).strip()]
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
                        labels = [str(x).strip() for x in parsed 
                                 if isinstance(x, str) and str(x).strip()]
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
            if not deduped:
                deduped = ["Unknown"]
        
        return deduped
        
    except Exception as e:
        print(f"Error parsing classification response: {e}")
        return []

if __name__ == "__main__":
    main()