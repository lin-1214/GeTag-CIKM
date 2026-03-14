# Phase 1 Implementation Summary

## ✅ 已完成的修改

### 1. gen_getag.py
**新增功能：輸出 Strong/Sparse Tags**

- 位置：Step 5 之後新增 Step 5.5
- 新增輸出：
  - `{tag_name}_strong_tags.json`: P60 以上的強標籤
  - `{tag_name}_sparse_tags.json`: Q1 以下的稀疏標籤
- 更新 SUMMARY 輸出顯示這些新檔案

### 2. run.sh
**改進 best_prompts.json 生成邏輯**

- 修改位置：選擇 Top-K prompts 後的 Python 腳本
- 主要變更：
  - 從 `strong_tags.json` 讀取 → `high_tags`
  - 從 `sparse_tags.json` 讀取 → `low_tags`
  - 保留 fallback 機制（向後兼容）
- 新增歸檔：
  - 複製 `*_strong_tags.json` 到 archive
  - 複製 `*_sparse_tags.json` 到 archive

### 3. 測試和文檔
- 創建 `test_phase1.sh`: 單次迭代測試腳本
- 創建 `PHASE1_IMPLEMENTATION.md`: 詳細實作說明

## 🎯 核心改進

### 改進前
```
high_tags = 前 50% 頻率的標籤（可能包含低性能標籤）
low_tags = 後 50% 頻率的標籤（可能包含高性能標籤）
```

### 改進後
```
high_tags = Strong Tags (P60+) - 統計顯著、檢索性能好
low_tags = Sparse Tags (Q1-) - 統計不可靠、應避免使用
```

## 📊 預期效果

1. **更精準的 Prompt 引導**
   - LLM 明確知道哪些標籤是成功的（strong tags）
   - LLM 會避免生成不可靠的標籤（sparse tags）

2. **加速收斂**
   - 減少無效標籤的生成
   - 更快找到高質量的標籤組合

3. **統計支持**
   - 基於 z-score 和分位數的科學分類
   - 不再是簡單的頻率排序

## 🧪 如何測試

### 快速測試（推薦先執行）
```bash
cd /data2/b11902154/PULLRS/PseudoUser/src
./test_phase1.sh
```

### 完整測試
```bash
cd /data2/b11902154/PULLRS/PseudoUser/src
# 可選：修改 run.sh 中的 NUM_ITERATIONS=2 (減少測試時間)
./run.sh
```

### 驗證重點
1. ✅ 確認生成 `*_strong_tags.json` 和 `*_sparse_tags.json`
2. ✅ 確認 `json/best_prompts.json` 使用這些檔案
3. ✅ 確認第二輪迭代的 prompt 包含 strong/sparse tags
4. ✅ 確認歸檔正確

## 📝 相關檔案

- `/data2/b11902154/PULLRS/PseudoUser/src/gen_getag.py` (已修改)
- `/data2/b11902154/PULLRS/PseudoUser/src/run.sh` (已修改)
- `/data2/b11902154/PULLRS/PseudoUser/src/test_phase1.sh` (新增)
- `/data2/b11902154/PULLRS/PseudoUser/src/PHASE1_IMPLEMENTATION.md` (新增)

## 🔄 與現有系統的兼容性

✅ **完全向後兼容**
- 如果 strong/sparse tags 檔案不存在，會 fallback 到舊邏輯
- 不影響 retrieval_v2.py 的執行
- 不改變檔案命名規範
- prompt.py 已經支持 high/low tags，無需修改

## 🚀 下一步建議

### 短期（測試階段）
1. 運行 `./test_phase1.sh` 驗證基本功能
2. 檢查生成的 strong/sparse tags 質量
3. 運行 2-3 次完整迭代觀察效果

### 中期（優化階段）
1. 調整 `STRONG_PERCENTILE` 和 `SPARSE_PERCENTILE` 參數
2. 分析迭代收斂速度的改善
3. 比較改進前後的標籤質量

### 長期（擴展階段）
- Phase 2: 實現 Top-K Strong Tags 檢索（提議 1）
- Phase 3: 評估是否需要保留最佳組機制（提議 2）

## ⚠️ 注意事項

1. **首次運行**可能需要清除舊檔案：
   ```bash
   rm -f ../json/tags/getags_zscore_*
   rm -f json/best_prompts.json
   ```

2. **檢查 GPU 記憶體**：
   - 如果 OOM，減少 `BATCH_SIZE` 或 `SAMPLE_SIZE`

3. **監控標籤數量**：
   - 如果 strong_tags 太少（< 3），考慮降低 STRONG_PERCENTILE
   - 如果 sparse_tags 太多，考慮提高 SPARSE_PERCENTILE

## 📞 疑難排解

### 問題：找不到 strong_tags.json
**原因**：evaluation_phase.py 可能沒有成功運行
**解決**：檢查 gen_getag.py 的輸出日誌

### 問題：best_prompts.json 的 high_tags 是空的
**原因**：可能所有標籤都低於 P60 閾值
**解決**：降低 `STRONG_PERCENTILE` 參數（例如從 60 改為 50）

### 問題：下一輪 prompt 沒有包含 strong/sparse tags
**原因**：best_prompts.json 可能沒有正確生成
**解決**：檢查 run.sh 的 Python 腳本輸出

---

Created: 2025-11-12
Status: Ready for Testing
