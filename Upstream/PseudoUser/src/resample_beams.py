#!/usr/bin/env python3
"""
从已有的多个 beam CSV 文件中重新采样，生成新的投票结果
不需要重新调用 LLM，只需要重新计算投票
"""

import pandas as pd
import json
import os
import sys
from collections import Counter
from config import Config

config = Config()

def filter_by_frequency(votes, k, threshold, top_n):
    """
    根据频率筛选标签（寧缺勿濫原则）
    复用 multibeam.py 的逻辑
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

def resample_beams(beam_dir, beam_indices, threshold=0.40, top_n=5):
    """
    从指定的 beam CSV 文件中重新采样并投票
    
    Args:
        beam_dir: beam CSV 文件所在目录
        beam_indices: 要使用的 beam 索引列表，例如 [0, 1, 2, 3, 4]
        threshold: 最低频率阈值（0-1）
        top_n: 最多保留几个标签
    
    Returns:
        final_results: 投票后的最终结果列表
        beam_dfs: 读取的 beam DataFrames
    """
    k = len(beam_indices)
    beam_dfs = []
    
    print(f"读取 {k} 个 beam CSV 文件...")
    for idx in beam_indices:
        beam_path = os.path.join(beam_dir, f"classified_data_multibeam_beam_{idx}.csv")
        if not os.path.exists(beam_path):
            raise FileNotFoundError(f"文件不存在: {beam_path}")
        df = pd.read_csv(beam_path)
        beam_dfs.append(df)
        print(f"  ✅ Beam {idx}: {len(df)} sessions")
    
    # 确保所有 beam 的行数一致
    num_sessions = len(beam_dfs[0])
    for i, df in enumerate(beam_dfs[1:], 1):
        if len(df) != num_sessions:
            raise ValueError(f"Beam {beam_indices[i]} 的行数 ({len(df)}) 与 Beam {beam_indices[0]} ({num_sessions}) 不一致！")
    
    print(f"\n开始重新投票 (threshold={threshold}, top_n={top_n})...")
    final_results = []
    
    for session_idx in range(num_sessions):
        votes = []
        
        # 收集该 session 在所有 beam 中的标签
        for beam_df in beam_dfs:
            class_value = beam_df.iloc[session_idx]['Class']
            
            # 解析标签（支持 JSON 格式）
            if pd.isna(class_value) or class_value == '':
                continue
            
            try:
                # 尝试解析 JSON (multi-label)
                if isinstance(class_value, str) and class_value.startswith('['):
                    labels = json.loads(class_value)
                    if isinstance(labels, list):
                        votes.extend(labels)
                else:
                    # Single label
                    votes.append(str(class_value))
            except json.JSONDecodeError:
                # 如果不是 JSON，直接当作单标签
                votes.append(str(class_value))
        
        # 投票筛选
        final_labels = filter_by_frequency(votes, k, threshold, top_n)
        final_results.append(final_labels)
    
    print(f"✅ 完成！共处理 {num_sessions} 个 sessions")
    return final_results, beam_dfs

def save_results(base_df, final_results, output_path, k, beam_indices):
    """保存重新采样的结果"""
    output_df = base_df.copy()
    
    # 根据 MULTI_LABEL 决定格式
    if config.MULTI_LABEL:
        output_df['Class'] = [json.dumps(labels, ensure_ascii=False) for labels in final_results]
    else:
        output_df['Class'] = [labels[0] if labels else '' for labels in final_results]
    
    # 将 Class 列移到第一列
    cols = ['Class'] + [col for col in output_df.columns if col != 'Class']
    output_df = output_df[cols]
    
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ 结果已保存到: {output_path}")
    print(f"   使用的 beams: {beam_indices}")
    print(f"   K = {k}")
    
    # 统计信息
    non_empty = sum(1 for labels in final_results if labels)
    empty = len(final_results) - non_empty
    print(f"\n📊 统计:")
    print(f"   总 sessions: {len(final_results)}")
    print(f"   有标签: {non_empty}")
    print(f"   无标签: {empty}")
    
    # 标签分布
    all_labels = [label for labels in final_results for label in labels]
    if all_labels:
        label_dist = Counter(all_labels)
        print(f"\n   标签分布 (Top 10):")
        for label, count in label_dist.most_common(10):
            print(f"     {label}: {count}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="从已有的 beam CSV 文件中重新采样并投票",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用前 5 个 beams (0-4)
  python3 resample_beams.py --beams 0 1 2 3 4
  
  # 使用指定的 beams
  python3 resample_beams.py --beams 0 2 4 6 8
  
  # 使用所有 10 个 beams，但调整阈值
  python3 resample_beams.py --beams 0 1 2 3 4 5 6 7 8 9 --threshold 0.3
  
  # 指定输入输出目录
  python3 resample_beams.py --beam-dir ../label_data/multibeam --output ../label_data/resampled_5beam.csv --beams 0 1 2 3 4
        """
    )
    
    parser.add_argument(
        '--beam-dir',
        default='../label_data/multibeam',
        help='beam CSV 文件所在目录 (默认: ../label_data/multibeam)'
    )
    parser.add_argument(
        '--beams',
        nargs='+',
        type=int,
        required=True,
        help='要使用的 beam 索引，例如: 0 1 2 3 4'
    )
    parser.add_argument(
        '--output',
        help='输出文件路径 (默认: ../label_data/resampled_Kbeam.csv)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.40,
        help='最低频率阈值 0-1 (默认: 0.40)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='最多保留几个标签 (默认: 5)'
    )
    
    args = parser.parse_args()
    
    # 验证 beam 索引
    beam_indices = sorted(set(args.beams))  # 去重并排序
    k = len(beam_indices)
    
    if k == 0:
        print("错误: 至少需要指定 1 个 beam")
        sys.exit(1)
    
    print("="*80)
    print(f"重新采样 Beam 投票结果")
    print("="*80)
    print(f"Beam 目录: {args.beam_dir}")
    print(f"使用 beams: {beam_indices} (共 {k} 个)")
    print(f"阈值: {args.threshold} (至少出现 {int(k * args.threshold)}/{k} 次)")
    print(f"Top-N: {args.top_n}")
    print()
    
    # 重新采样
    final_results, beam_dfs = resample_beams(
        args.beam_dir,
        beam_indices,
        threshold=args.threshold,
        top_n=args.top_n
    )
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = f'../label_data/resampled_{k}beam.csv'
    
    # 保存结果（使用第一个 beam 的 DataFrame 作为基础）
    save_results(beam_dfs[0], final_results, output_path, k, beam_indices)
    
    print("\n✅ 完成！")

if __name__ == "__main__":
    main()
