$ErrorActionPreference = "Stop"

$PipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Scripts = Join-Path $PipelineRoot "scripts"

if (Test-Path (Join-Path $PipelineRoot "dataset_weibo.txt")) {
    $ProjectRoot = $PipelineRoot
}
else {
    $ProjectRoot = Split-Path -Parent $PipelineRoot
}

if (-not (Test-Path (Join-Path $ProjectRoot "dataset_weibo.txt"))) {
    throw "Cannot find dataset_weibo.txt. Put it next to run_all.ps1 or in the parent project folder before running this pipeline."
}

Push-Location $ProjectRoot
try {
    Write-Host "Step 1/6: build cascade summary CSV"
    python (Join-Path $Scripts "01_analyze_weibo_dataset.py") --input dataset_weibo.txt --output weibo_summary_full.csv --stats weibo_stats_full.txt

    Write-Host "Step 2/6: build analysis_projects charts and CSVs"
    python (Join-Path $Scripts "02_make_analysis_projects.py")

    Write-Host "Step 3/6: train baseline and tree models"
    python (Join-Path $Scripts "03_train_baseline_and_tree_models.py")

    Write-Host "Step 4/6: plot full-scale prediction fit lines"
    python (Join-Path $Scripts "04_plot_prediction_fit_line.py")

    Write-Host "Step 5/6: plot zoomed baseline prediction chart"
    python (Join-Path $Scripts "05_plot_prediction_fit_line_zoom.py")

    Write-Host "Step 6/6: train improved models and robust evaluation"
    python (Join-Path $Scripts "06_train_improved_models.py")

    Write-Host "Done. Outputs are in analysis_projects, early_features_prediction, and early_features_prediction_v2."
}
finally {
    Pop-Location
}
