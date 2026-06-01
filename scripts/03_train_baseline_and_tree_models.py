import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


DATASET = Path("dataset_weibo.txt")
OUTPUT = Path("early_features_prediction")
OBSERVATION_TIME = 3600
PREDICTION_TIME = 86400
MIN_OBSERVED = 10
RANDOM_STATE = 42


FEATURE_COLUMNS = [
    "observed_1h_count",
    "unique_users_1h",
    "max_depth_1h",
    "avg_delay_1h",
    "max_delay_1h",
    "root_out_degree_1h",
    "max_out_degree_1h",
    "leaf_count_1h",
    "branching_factor_1h",
    "first_10min_count",
    "first_30min_count",
    "first_1h_count",
]


def parse_token(token):
    path, _, delay = token.partition(":")
    nodes = [node for node in path.split("/") if node]
    try:
        delay_seconds = int(delay)
    except ValueError:
        delay_seconds = None
    return nodes, delay_seconds


def summarize_cascade(parts):
    cascade_id, root_id, timestamp = parts[:3]
    tokens = " ".join(parts[4:]).split()

    observed_paths = []
    target_24h = 0
    for token in tokens:
        nodes, delay = parse_token(token)
        if delay is None:
            continue
        if delay < PREDICTION_TIME:
            target_24h += 1
        if delay < OBSERVATION_TIME:
            observed_paths.append((nodes, delay))

    if len(observed_paths) < MIN_OBSERVED:
        return None

    unique_users = set()
    depths = []
    delays = []
    source_out = Counter()
    targets = set()
    first_10 = 0
    first_30 = 0

    for nodes, delay in observed_paths:
        if not nodes:
            continue
        unique_users.update(nodes)
        depths.append(len(nodes))
        delays.append(delay)
        if delay < 600:
            first_10 += 1
        if delay < 1800:
            first_30 += 1
        if len(nodes) >= 2:
            source = nodes[-2]
            target = nodes[-1]
            source_out[source] += 1
            targets.add(target)

    observed_count = len(observed_paths)
    root_out = source_out[root_id]
    max_out = max(source_out.values()) if source_out else 0
    source_nodes = set(source_out.keys())
    leaf_count = len(targets - source_nodes)

    return {
        "cascade_id": cascade_id,
        "root_id": root_id,
        "timestamp": int(timestamp),
        "observed_1h_count": observed_count,
        "unique_users_1h": len(unique_users),
        "max_depth_1h": max(depths) if depths else 0,
        "avg_delay_1h": sum(delays) / len(delays) if delays else 0,
        "max_delay_1h": max(delays) if delays else 0,
        "root_out_degree_1h": root_out,
        "max_out_degree_1h": max_out,
        "leaf_count_1h": leaf_count,
        "branching_factor_1h": observed_count / len(source_nodes) if source_nodes else 0,
        "first_10min_count": first_10,
        "first_30min_count": first_30,
        "first_1h_count": observed_count,
        "actual_24h_total": target_24h,
        "actual_increment_after_1h": max(target_24h - observed_count, 0),
    }


def build_features():
    rows = []
    total = 0
    with DATASET.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            row = summarize_cascade(parts)
            if row:
                rows.append(row)
    return pd.DataFrame(rows), total


def mape(y_true, y_pred):
    y_true = pd.Series(y_true).clip(lower=1)
    y_pred = pd.Series(y_pred)
    return ((y_true - y_pred).abs() / y_true).mean()


def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }


def save_prediction_file(path, meta, y_true, predictions):
    output = meta.copy()
    output["actual_24h_total"] = y_true.values
    for name, pred in predictions.items():
        output[f"{name}_predicted_24h_total"] = pred
        output[f"{name}_error"] = pred - y_true.values
        output[f"{name}_absolute_error"] = abs(pred - y_true.values)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def plot_prediction_scatter(path, y_true, y_pred, title):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)
    max_axis = max(max(y_true), max(y_pred))
    plt.plot([0, max_axis], [0, max_axis], color="red", linewidth=1)
    plt.xlabel("Actual 24h total")
    plt.ylabel("Predicted 24h total")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_feature_importance(path, model, feature_names):
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5))
    top = importance.head(12)
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(path.stem.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_model_comparison(path, metrics):
    plot_df = metrics[metrics["model"].isin(["baseline_observed_1h", "random_forest", "xgboost"])].copy()
    plot_df["label"] = plot_df["model"].map(
        {
            "baseline_observed_1h": "Linear baseline",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
        }
    )
    plot_df = plot_df.sort_values("r2", ascending=True)

    plt.figure(figsize=(8, 4.8))
    bars = plt.barh(plot_df["label"], plot_df["r2"], color=["#60a5fa", "#34d399", "#f59e0b"])
    plt.xlim(0, 1)
    plt.xlabel("R2")
    plt.title("Linear Regression vs Random Forest vs XGBoost")
    for bar, value in zip(bars, plot_df["r2"]):
        plt.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def main():
    OUTPUT.mkdir(exist_ok=True)
    features, total_rows = build_features()
    features.to_csv(OUTPUT / "early_features_dataset.csv", index=False, encoding="utf-8-sig")

    train_df, test_df = train_test_split(features, test_size=0.15, random_state=RANDOM_STATE)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["actual_24h_total"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["actual_24h_total"]

    baseline = LinearRegression()
    baseline.fit(X_train[["observed_1h_count"]], y_train)
    baseline_pred = baseline.predict(X_test[["observed_1h_count"]])
    baseline_pred = baseline_pred.clip(min=1)

    random_forest = RandomForestRegressor(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    random_forest.fit(X_train, y_train)
    rf_pred = random_forest.predict(X_test).clip(min=1)

    gradient_boosting = GradientBoostingRegressor(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=4,
        random_state=RANDOM_STATE,
    )
    gradient_boosting.fit(X_train, y_train)
    gb_pred = gradient_boosting.predict(X_test).clip(min=1)

    xgboost = XGBRegressor(
        n_estimators=320,
        max_depth=4,
        learning_rate=0.055,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    xgboost.fit(X_train, y_train)
    xgb_pred = xgboost.predict(X_test).clip(min=1)

    metrics = pd.DataFrame(
        [
            evaluate("baseline_observed_1h", y_test, baseline_pred),
            evaluate("random_forest", y_test, rf_pred),
            evaluate("xgboost", y_test, xgb_pred),
            evaluate("gradient_boosting", y_test, gb_pred),
        ]
    ).sort_values("r2", ascending=False)
    metrics.to_csv(OUTPUT / "model_metrics.csv", index=False, encoding="utf-8-sig")

    predictions = {
        "baseline": baseline_pred,
        "random_forest": rf_pred,
        "xgboost": xgb_pred,
        "gradient_boosting": gb_pred,
    }
    save_prediction_file(
        OUTPUT / "prediction_vs_actual_all_models.csv",
        test_df[["cascade_id", "root_id", "observed_1h_count", "actual_increment_after_1h"]].reset_index(drop=True),
        y_test.reset_index(drop=True),
        predictions,
    )

    plot_prediction_scatter(OUTPUT / "baseline_prediction_vs_actual.png", y_test, baseline_pred, "Baseline Prediction vs Actual")
    plot_prediction_scatter(OUTPUT / "random_forest_prediction_vs_actual.png", y_test, rf_pred, "Random Forest Prediction vs Actual")
    plot_prediction_scatter(OUTPUT / "xgboost_prediction_vs_actual.png", y_test, xgb_pred, "XGBoost Prediction vs Actual")
    plot_prediction_scatter(OUTPUT / "gradient_boosting_prediction_vs_actual.png", y_test, gb_pred, "Gradient Boosting Prediction vs Actual")
    plot_model_comparison(OUTPUT / "model_comparison_linear_rf_xgb.png", metrics)
    plot_feature_importance(OUTPUT / "random_forest_feature_importance.png", random_forest, FEATURE_COLUMNS)
    plot_feature_importance(OUTPUT / "xgboost_feature_importance.png", xgboost, FEATURE_COLUMNS)
    plot_feature_importance(OUTPUT / "gradient_boosting_feature_importance.png", gradient_boosting, FEATURE_COLUMNS)

    best = metrics.iloc[0]
    with (OUTPUT / "report.txt").open("w", encoding="utf-8", newline="") as report:
        report.write("Early Features Prediction Report\n")
        report.write("================================\n\n")
        report.write("Task: use the first 1 hour of Weibo cascade features to predict 24-hour cascade size.\n")
        report.write(f"Original cascades scanned: {total_rows:,}\n")
        report.write(f"Valid cascades with at least {MIN_OBSERVED} reposts in first hour: {len(features):,}\n")
        report.write(f"Train rows: {len(train_df):,}\n")
        report.write(f"Test rows: {len(test_df):,}\n\n")
        report.write("Models compared:\n")
        report.write("- Baseline: linear regression using only observed_1h_count.\n")
        report.write("- Random Forest: hand-crafted early cascade features.\n")
        report.write("- Gradient Boosting: hand-crafted early cascade features.\n\n")
        report.write("Best model by R2:\n")
        report.write(f"- {best['model']}: R2={best['r2']:.4f}, MAE={best['mae']:.2f}, RMSE={best['rmse']:.2f}, MAPE={best['mape'] * 100:.2f}%\n\n")
        report.write("Interpretation:\n")
        report.write("This is a lightweight alternative to CasFlow. It keeps the same prediction idea, but replaces graph embeddings with interpretable early-spread features.\n")
        report.write("It is easier to explain in a report and runs much faster than full deep graph embedding.\n")

    print(f"Wrote early feature prediction project to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
