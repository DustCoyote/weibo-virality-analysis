from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


PROJECT = Path("early_features_prediction")
INPUT = PROJECT / "prediction_vs_actual_all_models.csv"


def plot_fit(model_prefix, title):
    df = pd.read_csv(INPUT)
    x = df[["actual_24h_total"]]
    y = df[f"{model_prefix}_predicted_24h_total"]

    fit = LinearRegression()
    fit.fit(x, y)
    fitted = fit.predict(x)
    fit_r2 = r2_score(y, fitted)

    max_axis = max(df["actual_24h_total"].max(), y.max())
    line_x = pd.DataFrame({"actual_24h_total": [0, max_axis]})
    line_y = fit.predict(line_x)

    plt.figure(figsize=(7, 7))
    plt.scatter(df["actual_24h_total"], y, s=8, alpha=0.32, label="Cascades")
    plt.plot([0, max_axis], [0, max_axis], color="red", linewidth=1, label="Perfect prediction")
    plt.plot(line_x["actual_24h_total"], line_y, color="black", linewidth=2, label="Regression fit")
    plt.xlabel("Actual 24h total")
    plt.ylabel("Predicted 24h total")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_png = PROJECT / f"{model_prefix}_prediction_fit_line.png"
    plt.savefig(output_png, dpi=160)
    plt.close()

    return {
        "model": model_prefix,
        "fit_slope": fit.coef_[0],
        "fit_intercept": fit.intercept_,
        "fit_r2_predicted_on_actual": fit_r2,
        "output_png": str(output_png),
    }


def main():
    rows = [
        plot_fit("baseline", "Baseline: Actual vs Predicted with Fit Line"),
        plot_fit("random_forest", "Random Forest: Actual vs Predicted with Fit Line"),
        plot_fit("xgboost", "XGBoost: Actual vs Predicted with Fit Line"),
        plot_fit("gradient_boosting", "Gradient Boosting: Actual vs Predicted with Fit Line"),
    ]
    pd.DataFrame(rows).to_csv(PROJECT / "prediction_fit_line_summary.csv", index=False, encoding="utf-8-sig")
    print("Wrote prediction fit line charts and summary.")


if __name__ == "__main__":
    main()
