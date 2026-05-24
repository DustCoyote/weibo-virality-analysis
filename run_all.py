import subprocess
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
SCRIPTS = PIPELINE_ROOT / "scripts"


def find_project_root():
    candidates = [PIPELINE_ROOT, PIPELINE_ROOT.parent]
    for candidate in candidates:
        if (candidate / "dataset_weibo.txt").exists():
            return candidate
    return PIPELINE_ROOT


PROJECT_ROOT = find_project_root()


STEPS = [
    (
        "Step 1/6: build cascade summary CSV",
        [
            SCRIPTS / "01_analyze_weibo_dataset.py",
            "--input",
            "dataset_weibo.txt",
            "--output",
            "weibo_summary_full.csv",
            "--stats",
            "weibo_stats_full.txt",
        ],
    ),
    (
        "Step 2/6: build analysis_projects charts and CSVs",
        [SCRIPTS / "02_make_analysis_projects.py"],
    ),
    (
        "Step 3/6: train baseline and tree models",
        [SCRIPTS / "03_train_baseline_and_tree_models.py"],
    ),
    (
        "Step 4/6: plot full-scale prediction fit lines",
        [SCRIPTS / "04_plot_prediction_fit_line.py"],
    ),
    (
        "Step 5/6: plot zoomed baseline prediction chart",
        [SCRIPTS / "05_plot_prediction_fit_line_zoom.py"],
    ),
    (
        "Step 6/6: train improved models and robust evaluation",
        [SCRIPTS / "06_train_improved_models.py"],
    ),
]


def main():
    if not (PROJECT_ROOT / "dataset_weibo.txt").exists():
        raise FileNotFoundError(
            "Cannot find dataset_weibo.txt. Put it next to run_all.py "
            "or in the parent project folder before running this pipeline."
        )

    for label, args in STEPS:
        print(label, flush=True)
        command = [sys.executable, *[str(arg) for arg in args]]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print(
        "Done. Outputs are in analysis_projects, early_features_prediction, "
        "and early_features_prediction_v2."
    )


if __name__ == "__main__":
    main()
