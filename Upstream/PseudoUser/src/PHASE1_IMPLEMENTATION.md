# Phase 1 Implementation: Strong/Sparse Tags in Prompts

## 🎯 目標

改進迭代式 prompt 生成，使用**基於性能的標籤分類**而非單純的頻率統計：
- **Strong Tags (高性能標籤)**: P60 以上的標籤，應該被優先使用
- **Sparse Tags (稀疏標籤)**: Q1 以下的標籤，應該被避免

## 📝 修改內容

### 1. `gen_getag.py` - 新增 Strong/Sparse Tags 輸出

#### 修改位置
在 Step 5 之後新增 Step 5.5

#### 新增功能
- 輸出 `{tag_name}_strong_tags.json`: 包含 P60 以上的強標籤
- 輸出 `{tag_name}_sparse_tags.json`: 包含 Q1 以下的稀疏標籤

#### 輸出格式
```json
{
  "tags": [
    ["標籤名稱", 頻率],
    ["健康意識", 145],
    ["品牌忠誠", 132],
    ...
  ],
  "threshold": "P60 (>= 98.5)",
  "count": 5
}
```

#### 好處
- 明確標示哪些標籤是統計上顯著的（strong）
- 明確標示哪些標籤應該避免（sparse）
- 提供頻率排序，便於選擇 Top-K

---

### 2. `run.sh` - 改進 best_prompts.json 生成邏輯

#### 修改位置
選擇 Top-K prompts 後的 Python 腳本部分（約第 177-220 行）

#### 主要改動

**舊邏輯**：
```python
# 從 classified_data CSV 統計標籤頻率
tag_counts = Counter()
for row in csv:
    tags = parse_labels(row['Class'])
    tag_counts.update(tags)

# 按頻率排序並按比例分割
sorted_tags = [t for t, _ in tag_counts.most_common()]
split_idx = int(len(sorted_tags) * 0.5)
high_tags = sorted_tags[:split_idx]  # 前 50%
low_tags = sorted_tags[split_idx:]   # 後 50%
```

**新邏輯**：
```python
# 讀取 gen_getag.py 生成的 strong/sparse tags
strong_tags_path = f"../json/tags/{tag_name}_strong_tags.json"
sparse_tags_path = f"../json/tags/{tag_name}_sparse_tags.json"

# high_tags = Strong Tags (P60 以上)
# low_tags = Sparse Tags (Q1 以下)
high_tags = [tag for tag, freq in strong_data['tags']]
low_tags = [tag for tag, freq in sparse_data['tags']]

# 如果檔案不存在，fallback 到舊邏輯
```

#### 好處
- **語意正確**: high_tags 現在真的是高性能標籤
- **統計支持**: 基於 z-score 和分位數的科學分類
- **向後兼容**: 保留 fallback 機制

---

### 3. `run.sh` - 新增歸檔邏輯

#### 修改位置
每次迭代結束的歸檔部分

#### 新增內容
```bash
# Archive strong/sparse tags files
cp ../json/tags/getags_zscore_iter${i}_*_strong_tags.json "$ARCHIVE_JSON_PATH/"
cp ../json/tags/getags_zscore_iter${i}_*_sparse_tags.json "$ARCHIVE_JSON_PATH/"
```

---

## 🔄 完整流程

### Iteration N 的流程：

```
1. label_phase.py
   └─> 生成 classified_data_*.csv

2. evaluation_phase.py (對每個 CSV)
   ├─> gen_getag.py
   │   ├─> 生成 getags_zscore_iter{N}_{idx}.json
   │   ├─> 生成 getags_zscore_iter{N}_{idx}_strong_tags.json  ✨ 新增
   │   └─> 生成 getags_zscore_iter{N}_{idx}_sparse_tags.json  ✨ 新增
   └─> retrieval_v2.py
       └─> 生成 summary_*.json

3. 選擇 Top-K prompts (基於 ndcg@10/100/test)

4. 生成 best_prompts.json ✨ 改進
   ├─> 讀取 strong_tags.json → high_tags
   ├─> 讀取 sparse_tags.json → low_tags
   └─> 輸出供 Iteration N+1 使用

5. Iteration N+1 的 label_phase.py
   └─> prompt.py 使用 high_tags 和 low_tags
       ├─> "Reference High-Frequency Tags": strong tags
       └─> "Low-Frequency Tags to Avoid": sparse tags
```

---

## 🧪 測試方法

### 快速測試（單次迭代）
```bash
cd /data2/b11902154/PULLRS/PseudoUser/src
./test_phase1.sh
```

這將：
1. 運行一次 label_phase
2. 運行 evaluation_phase
3. 驗證 strong/sparse tags 檔案生成
4. 測試 best_prompts.json 生成邏輯
5. 顯示範例標籤

### 完整測試（多次迭代）
```bash
cd /data2/b11902154/PULLRS/PseudoUser/src
# 修改 run.sh: NUM_ITERATIONS=3 (測試用)
./run.sh
```

檢查點：
- ✅ `json/tags/*_strong_tags.json` 是否生成
- ✅ `json/tags/*_sparse_tags.json` 是否生成
- ✅ `json/best_prompts.json` 的 high_tags 是否來自 strong_tags
- ✅ `json/best_prompts.json` 的 low_tags 是否來自 sparse_tags
- ✅ 第二輪的 prompt 是否包含這些標籤

---

## 📊 預期效果

### 改進前
```
Prompt 中的 high_tags: ["標籤A", "標籤B", "標籤C", ...]
                        ↑ 按頻率排序，前 50%
                        ↑ 可能包含高頻但低性能的標籤

Prompt 中的 low_tags:  ["標籤X", "標籤Y", "標籤Z", ...]
                        ↑ 按頻率排序，後 50%
                        ↑ 可能包含低頻但高性能的標籤
```

### 改進後
```
Prompt 中的 high_tags: ["健康意識", "品牌忠誠", ...]
                        ↑ P60 以上的強標籤
                        ↑ 統計上顯著、檢索性能好

Prompt 中的 low_tags:  ["極端稀有行為", "噪音標籤", ...]
                        ↑ Q1 以下的稀疏標籤
                        ↑ 統計上不可靠、應避免
```

### 對下一輪迭代的影響
- ✅ LLM 更明確知道哪些標籤是成功的
- ✅ LLM 會避免生成稀疏標籤
- ✅ 加速收斂到高質量標籤
- ✅ 減少迭代次數

---

## 🔍 驗證方法

### 1. 檢查 Strong Tags 質量
```bash
# 查看某個 strong_tags 檔案
cat ../json/tags/getags_zscore_iter1_0_strong_tags.json | jq '.tags[:5]'

# 應該看到高頻且語意清晰的標籤
# 例如: [["健康意識", 145], ["品牌忠誠", 132], ...]
```

### 2. 檢查 Sparse Tags
```bash
# 查看某個 sparse_tags 檔案
cat ../json/tags/getags_zscore_iter1_0_sparse_tags.json | jq '.tags[:5]'

# 應該看到低頻且可能不穩定的標籤
# 例如: [["極端稀有", 8], ["噪音標籤", 5], ...]
```

### 3. 檢查 best_prompts.json
```bash
cat json/best_prompts.json | jq '.high_tags, .low_tags'

# 驗證:
# - high_tags 應該包含語意清晰的常見行為標籤
# - low_tags 應該是稀疏標籤
# - 數量合理（不是簡單的 50-50 分割）
```

### 4. 檢查下一輪的 Prompt
在第二輪迭代時，查看 LLM 實際收到的 prompt：
```python
# 在 label_phase.py 中暫時添加打印
print(cluster_prompt)
```

應該看到：
```
4. Reference Category Labels
- For reference, here are previously successful category labels:
...

5. Reference High-Frequency Tags (from previous iteration)
- Prioritize aligning category naming with these tags.
["健康意識", "品牌忠誠", "精打細算", ...]

6. Low-Frequency Tags to Avoid
- Avoid creating categories centered on these low-frequency tags unless strongly justified.
["極端稀有行為", "噪音標籤", ...]
```

---

## 🚀 下一步

Phase 1 完成後，可以考慮：

### Phase 2: Top-K Strong Tags 檢索（提議 1）
- 在 `gen_getag.py` 中添加配置：只保留 Top-K 強標籤
- 對比實驗：全標籤 vs Top-K 強標籤

### Phase 3: 保留最佳組（提議 2）
- 將最佳 prompt 的標籤完整保留到下一輪
- 作為 baseline 確保性能不退化

---

## 📝 Notes

- 所有修改都保持向後兼容（fallback 機制）
- 不影響現有的 retrieval 邏輯
- 只改進 prompt generation 的質量
- 檔案命名遵循現有規範：`{tag_name}_strong_tags.json`

---

## 🐛 常見問題

### Q: 如果 strong_tags.json 不存在會怎樣？
A: 會 fallback 到舊的 CSV 統計方法，不會中斷流程。

### Q: 為什麼是 P60 和 Q1？
A: 這是 `gen_getag.py` 中的預設值：
- `STRONG_PERCENTILE = 60` (P60)
- `SPARSE_PERCENTILE = 25` (Q1)
可以通過修改這些參數來調整。

### Q: high_tags 和 low_tags 的數量會相等嗎？
A: 不會！這正是改進的重點：
- high_tags 數量 = strong_tags 數量（統計顯著的標籤）
- low_tags 數量 = sparse_tags 數量（應該避免的標籤）
- 可能會有 8 個 high_tags 但只有 3 個 low_tags（或反之）

### Q: 如果某個 prompt 沒有 strong tags 怎麼辦？
A: 罕見但可能發生。此時：
1. high_tags 會是空列表
2. Prompt 不會包含 "Reference High-Frequency Tags" 段落
3. 不影響其他部分的執行

---

## ✅ Checklist

- [x] `gen_getag.py` 輸出 strong_tags.json
- [x] `gen_getag.py` 輸出 sparse_tags.json
- [x] `run.sh` 讀取這些檔案
- [x] `run.sh` 歸檔這些檔案
- [x] 保持向後兼容（fallback）
- [x] 創建測試腳本
- [ ] 運行測試驗證
- [ ] 運行完整 iteration 驗證
- [ ] 分析結果並調整參數

---

Last updated: 2025-11-12
