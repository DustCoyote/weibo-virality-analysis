# Weibo Virality Prediction 分析程式整理

這個資料夾整理「簡報中出現的分析、圖表、模型」是怎麼做出來的。

原始資料不建議丟 GitHub。組員下載後，請把資料放在 repo 根目錄，或放在這個資料夾的上一層：

```text
dataset_weibo.txt
```

## 資料夾內容

```text
presentation_analysis_code/
  README.md
  requirements.txt
  run_all.ps1
  scripts/
    01_analyze_weibo_dataset.py
    02_make_analysis_projects.py
    03_train_baseline_and_tree_models.py
    04_plot_prediction_fit_line.py
    05_plot_prediction_fit_line_zoom.py
    06_train_improved_models.py
  docs/
    analysis_notes.md
    outputs_manifest.csv
```

## 一鍵重跑

如果組員使用 Windows PowerShell：

```powershell
cd "presentation_analysis_code"
.\run_all.ps1
```

如果組員使用 macOS、Linux，或想直接用 Python：

```bash
cd presentation_analysis_code
python run_all.py
```

會重新產生：

- `weibo_summary_full.csv`
- `analysis_projects/` 裡的各類分析圖與 CSV
- `early_features_prediction/` 裡的 baseline 與傳統模型結果
- `early_features_prediction_v2/` 裡的 log scale、非線性模型、Top 10% 評估結果

## 簡報主軸對應

1. 長尾現象：
   - 使用 `02_make_analysis_projects.py`
   - 主要輸出：`analysis_projects/07_long_tail/long_tail_distribution.png`
   - 數字：前 1% 貢獻約 43.83%，前 10% 貢獻約 75.33%

2. 早期訊號：
   - 使用 `03_train_baseline_and_tree_models.py`
   - 觀察窗：前 1 小時
   - 預測窗：24 小時總傳播規模
   - baseline：只用前 1 小時傳播數做線性回歸

3. 散點圖與扇形誤差：
   - 使用 `04_plot_prediction_fit_line.py`
   - 使用 `05_plot_prediction_fit_line_zoom.py`
   - 主要輸出：`early_features_prediction/baseline_prediction_fit_line_zoom_20000.png`

4. 改進方向：
   - 使用 `06_train_improved_models.py`
   - 方法：log target、Ridge、HistGradientBoosting、two-stage top-10-aware regressor
   - 評估：R2、MAE、RMSE、log-RMSE、Top 10% recall、分段誤差

## 重要註記

- 簡報中的 `R² = 0.7471` 是「前 1 小時傳播數預測 24 小時總傳播規模」的 baseline 結果。
- 這不是官方 CasFlow 深度模型結果，而是為期末報告建立的可解釋分析 baseline。
- 模型分析不是全部 119,313 筆都進入訓練；它使用前 1 小時至少有 10 個傳播路徑的 active cascade，共 69,038 筆。
- 原始資料中的 `root_id:0` 代表原始發文節點，因此簡報中更精準的說法是「傳播規模 / 傳播路徑數」，不是純粹扣除原作者後的轉發數。
