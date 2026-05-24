import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET = Path("dataset_weibo.txt")
OUTPUT = Path("early_features_prediction_v2")
OBSERVATION_TIME = 3600
PREDICTION_TIME = 86400
MIN_OBSERVED = 10
RANDOM_STATE = 42


FEATURE_COLUMNS = [
    "count_0_5min",
    "count_5_10min",
    "count_10_30min",
    "count_30_60min",
    "observed_1h_count",
    "unique_users_1h",
    "source_count_1h",
    "max_depth_1h",
    "avg_depth_1h",
    "root_out_degree_1h",
    "max_out_degree_1h",
    "max_out_degree_share_1h",
    "leaf_count_1h",
    "leaf_ratio_1h",
    "branching_factor_1h",
    "source_entropy_1h",
    "avg_delay_1h",
    "std_delay_1h",
    "median_delay_1h",
    "max_delay_1h",
    "growth_ratio_30_60_vs_0_30",
    "growth_ratio_10_60_vs_0_10",
    "late_10min_share",
    "log_observed_1h_count",
    "log_unique_users_1h",
    "log_max_out_degree_1h",
]


def parse_token(token):
    path, _, delay = token.partition(":")
    nodes = [node for node in path.split("/") if node]
    try:
        delay_seconds = int(delay)
    except ValueError:
        delay_seconds = None
    return nodes, delay_seconds


def entropy_from_counts(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    return -sum(p * math.log(p) for p in probs)


def summarize_cascade(parts):
    cascade_id, root_id, timestamp = parts[:3]
    tokens = " ".join(parts[4:]).split()

    observed_paths = []
    actual_24h_total = 0
    for token in tokens:
        nodes, delay = parse_token(token)
        if delay is None:
            continue
        if delay < PREDICTION_TIME:
            actual_24h_total += 1
        if delay < OBSERVATION_TIME:
            observed_paths.append((nodes, delay))

    observed_count = len(observed_paths)
    if observed_count < MIN_OBSERVED:
        return None

    unique_users = set()
    depths = []
    delays = []
    source_out = Counter()
    targets = set()

    count_0_5 = 0
    count_5_10 = 0
    count_10_30 = 0
    count_30_60 = 0
    late_10 = 0

    for nodes, delay in observed_paths:
        if not nodes:
            continue
        unique_users.update(nodes)
        depths.append(len(nodes))
        delays.append(delay)

        if delay < 300:
            count_0_5 += 1
        elif delay < 600:
            count_5_10 += 1
        elif delay < 1800:
            count_10_30 += 1
        else:
            count_30_60 += 1
        if delay >= 3000:
            late_10 += 1

        if len(nodes) >= 2:
            source = nodes[-2]
            target = nodes[-1]
            source_out[source] += 1
            targets.add(target)

    source_nodes = set(source_out.keys())
    source_count = len(source_nodes)
    root_out = source_out[root_id]
    max_out = max(source_out.values()) if source_out else 0
    leaf_count = len(targets - source_nodes)
    first_30 = count_0_5 + count_5_10 + count_10_30
    first_10 = count_0_5 + count_5_10

    delay_array = np.array(delays, dtype=float) if delays else np.array([0.0])
    depth_array = np.array(depths, dtype=float) if depths else np.array([0.0])

    return {
        "cascade_id": cascade_id,
        "root_id": root_id,
        "timestamp": int(timestamp),
        "count_0_5min": count_0_5,
        "count_5_10min": count_5_10,
        "count_10_30min": count_10_30,
        "count_30_60min": count_30_60,
        "observed_1h_count": observed_count,
        "unique_users_1h": len(unique_users),
        "source_count_1h": source_count,
        "max_depth_1h": int(depth_array.max()),
        "avg_depth_1h": float(depth_array.mean()),
        "root_out_degree_1h": root_out,
        "max_out_degree_1h": max_out,
        "max_out_degree_share_1h": max_out / observed_count if observed_count else 0,
        "leaf_count_1h": leaf_count,
        "leaf_ratio_1h": leaf_count / len(targets) if targets else 0,
        "branching_factor_1h": observed_count / source_count if source_count else 0,
        "source_entropy_1h": entropy_from_counts(source_out.values()),
        "avg_delay_1h": float(delay_array.mean()),
        "std_delay_1h": float(delay_array.std()),
        "median_delay_1h": float(np.median(delay_array)),
        "max_delay_1h": int(delay_array.max()),
        "growth_ratio_30_60_vs_0_30": count_30_60 / max(first_30, 1),
        "growth_ratio_10_60_vs_0_10": (count_10_30 + count_30_60) / max(first_10, 1),
        "late_10min_share": late_10 / observed_count,
        "log_observed_1h_count": math.log1p(observed_count),
        "log_unique_users_1h": math.log1p(len(unique_users)),
        "log_max_out_degree_1h": math.log1p(max_out),
        "actual_24h_total": actual_24h_total,
        "actual_increment_after_1h": max(actual_24h_total - observed_count, 0),
        "log_actual_24h_total": math.log1p(actual_24h_total),
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
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows), total


def mape(y_true, y_pred):
    y_true = np.clip(np.asarray(y_true, dtype=float), 1, None)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred) / y_true)


def log_rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))))


def top_recall(y_true, y_pred, top_share=0.10):
    threshold = np.quantile(y_true, 1 - top_share)
    pred_threshold = np.quantile(y_pred, 1 - top_share)
    actual_top = y_true >= threshold
    pred_top = y_pred >= pred_threshold
    return recall_score(actual_top, pred_top), threshold, pred_threshold


def evaluate(name, y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1, None)
    recall, actual_threshold, pred_threshold = top_recall(np.asarray(y_true), y_pred)
    return {
        "model": name,
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "log_rmse": log_rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "top10_recall": recall,
        "actual_top10_threshold": actual_threshold,
        "predicted_top10_threshold": pred_threshold,
    }


def bucket_errors(df, prediction_columns):
    q1 = df["actual_24h_total"].quantile(1 / 3)
    q2 = df["actual_24h_total"].quantile(2 / 3)

    def bucket(value):
        if value <= q1:
            return "small"
        if value <= q2:
            return "medium"
        return "large"

    rows = []
    df = df.copy()
    df["size_bucket"] = df["actual_24h_total"].apply(bucket)
    for model, column in prediction_columns.items():
        for bucket_name, group in df.groupby("size_bucket"):
            rows.append(
                {
                    "model": model,
                    "size_bucket": bucket_name,
                    "rows": len(group),
                    "actual_min": group["actual_24h_total"].min(),
                    "actual_max": group["actual_24h_total"].max(),
                    "mae": mean_absolute_error(group["actual_24h_total"], group[column]),
                    "rmse": math.sqrt(mean_squared_error(group["actual_24h_total"], group[column])),
                    "log_rmse": log_rmse(group["actual_24h_total"], group[column]),
                }
            )
    return pd.DataFrame(rows)


def scatter_plot(path, y_true, y_pred, title, log_scale=False, limit=None):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=8, alpha=0.32)
    max_axis = max(np.max(y_true), np.max(y_pred))
    if limit:
        max_axis = limit
        plt.xlim(0, limit)
        plt.ylim(0, limit)
    plt.plot([0, max_axis], [0, max_axis], color="red", linewidth=1.2, label="Perfect prediction")
    plt.xlabel("Actual 24h total")
    plt.ylabel("Predicted 24h total")
    plt.title(title)
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def bar_metrics(path, metrics):
    plot_df = metrics[["model", "r2", "top10_recall"]].copy()
    x = np.arange(len(plot_df))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, plot_df["r2"], width, label="R2")
    plt.bar(x + width / 2, plot_df["top10_recall"], width, label="Top 10% recall")
    plt.xticks(x, plot_df["model"], rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def feature_importance(path, model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    importance = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    top = importance.head(14)
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Two-stage regressor feature importance")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def main():
    OUTPUT.mkdir(exist_ok=True)
    features, total_rows = build_features()
    features.to_csv(OUTPUT / "v2_features_dataset.csv", index=False, encoding="utf-8-sig")

    train_df, test_df = train_test_split(features, test_size=0.15, random_state=RANDOM_STATE)
    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train = train_df["actual_24h_total"].to_numpy()
    y_test = test_df["actual_24h_total"].to_numpy()
    y_train_log = np.log1p(y_train)

    # 1. Log baseline: log target using only early observed count.
    log_baseline = LinearRegression()
    log_baseline.fit(train_df[["log_observed_1h_count"]], y_train_log)
    log_baseline_pred = np.expm1(log_baseline.predict(test_df[["log_observed_1h_count"]]))

    # 2. Ridge regression: standardized full v2 features, log target.
    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
    )
    ridge.fit(X_train, y_train_log)
    ridge_pred = np.expm1(ridge.predict(X_test))

    # 3. HistGradientBoosting: full features, log target.
    hist_gb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=260,
        learning_rate=0.055,
        max_leaf_nodes=31,
        l2_regularization=0.04,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    hist_gb.fit(X_train, y_train_log)
    hist_gb_pred = np.expm1(hist_gb.predict(X_test))

    # 4. Two-stage: classifier-like top 10% signal via HistGB + separate regressor.
    top_threshold = np.quantile(y_train, 0.90)
    train_top = y_train >= top_threshold
    two_stage_low = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.055,
        max_leaf_nodes=31,
        l2_regularization=0.04,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    two_stage_high = RandomForestRegressor(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    top_classifier = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    top_classifier.fit(X_train, train_top.astype(float))
    two_stage_low.fit(X_train[~train_top], np.log1p(y_train[~train_top]))
    two_stage_high.fit(X_train[train_top], np.log1p(y_train[train_top]))

    top_probability = np.clip(top_classifier.predict(X_test), 0, 1)
    low_pred = np.expm1(two_stage_low.predict(X_test))
    high_pred = np.expm1(two_stage_high.predict(X_test))
    two_stage_pred = ((1 - top_probability) * low_pred) + (top_probability * high_pred)

    predictions = {
        "log_baseline": log_baseline_pred,
        "ridge": ridge_pred,
        "hist_gradient_boosting": hist_gb_pred,
        "two_stage": two_stage_pred,
    }

    metrics = pd.DataFrame([evaluate(name, y_test, pred) for name, pred in predictions.items()])
    metrics = metrics.sort_values("r2", ascending=False)
    metrics.to_csv(OUTPUT / "v2_model_metrics.csv", index=False, encoding="utf-8-sig")

    pred_df = test_df[["cascade_id", "root_id", "observed_1h_count", "actual_increment_after_1h"]].reset_index(drop=True)
    pred_df["actual_24h_total"] = y_test
    for name, pred in predictions.items():
        pred = np.clip(pred, 1, None)
        pred_df[f"{name}_predicted_24h_total"] = pred
        pred_df[f"{name}_error"] = pred - y_test
        pred_df[f"{name}_absolute_error"] = np.abs(pred - y_test)
    pred_df.to_csv(OUTPUT / "v2_prediction_vs_actual_all_models.csv", index=False, encoding="utf-8-sig")

    bucket_errors(
        pred_df,
        {
            "log_baseline": "log_baseline_predicted_24h_total",
            "ridge": "ridge_predicted_24h_total",
            "hist_gradient_boosting": "hist_gradient_boosting_predicted_24h_total",
            "two_stage": "two_stage_predicted_24h_total",
        },
    ).to_csv(OUTPUT / "v2_bucket_errors.csv", index=False, encoding="utf-8-sig")

    bar_metrics(OUTPUT / "v2_model_comparison.png", metrics)
    best_model = metrics.iloc[0]["model"]
    best_pred = predictions[best_model]
    scatter_plot(OUTPUT / f"v2_{best_model}_prediction_vs_actual_log.png", y_test, best_pred, f"{best_model}: log-scale actual vs predicted", log_scale=True)
    scatter_plot(OUTPUT / f"v2_{best_model}_prediction_vs_actual_zoom_20000.png", y_test, best_pred, f"{best_model}: zoomed to 20,000", limit=20000)
    scatter_plot(OUTPUT / "v2_hist_gradient_boosting_prediction_vs_actual_log.png", y_test, hist_gb_pred, "HistGradientBoosting: log-scale actual vs predicted", log_scale=True)
    scatter_plot(OUTPUT / "v2_hist_gradient_boosting_prediction_vs_actual_zoom_20000.png", y_test, hist_gb_pred, "HistGradientBoosting: zoomed to 20,000", limit=20000)
    feature_importance(OUTPUT / "v2_two_stage_feature_importance.png", two_stage_high, FEATURE_COLUMNS)

    with (OUTPUT / "v2_report.txt").open("w", encoding="utf-8", newline="") as report:
        best = metrics.iloc[0]
        report.write("Weibo Virality Prediction V2\n")
        report.write("============================\n\n")
        report.write("Data: full Weibo cascade dataset.\n")
        report.write("Observation window: first 1 hour; prediction window: 24 hours.\n")
        report.write(f"Original cascades scanned: {total_rows:,}\n")
        report.write(f"Valid cascades: {len(features):,}\n")
        report.write(f"Train rows: {len(train_df):,}; Test rows: {len(test_df):,}\n\n")
        report.write("Feature upgrades: time-slice counts, growth ratios, structure features, entropy/max-share, and log count features.\n")
        report.write("Models: Log baseline, Ridge regression, HistGradientBoosting, and a two-stage top-10% aware regressor.\n\n")
        report.write(
            f"Best by R2: {best['model']} | R2={best['r2']:.4f}, MAE={best['mae']:.2f}, "
            f"RMSE={best['rmse']:.2f}, log-RMSE={best['log_rmse']:.4f}, Top10 recall={best['top10_recall']:.4f}\n"
        )
        report.write("Interpretation: log targets reduce long-tail distortion; Top10 recall measures whether the model catches viral cases, not only average fit.\n")

    print(f"Wrote V2 prediction project to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
