from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


PROJECT = Path("early_features_prediction")
INPUT = PROJECT / "prediction_vs_actual_all_models.csv"
LIMIT = 20000


def main():
    df = pd.read_csv(INPUT)
    y = df["baseline_predicted_24h_total"]
    x = df[["actual_24h_total"]]

    fit = LinearRegression()
    fit.fit(x, y)

    zoom = df[(df["actual_24h_total"] <= LIMIT) & (df["baseline_predicted_24h_total"] <= LIMIT)].copy()
    zoom_r2 = r2_score(zoom["actual_24h_total"], zoom["baseline_predicted_24h_total"])

    line_x = pd.DataFrame({"actual_24h_total": [0, LIMIT]})
    line_y = fit.predict(line_x)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        zoom["actual_24h_total"],
        zoom["baseline_predicted_24h_total"],
        s=10,
        alpha=0.34,
        label="Cascades under 20,000",
    )
    plt.plot([0, LIMIT], [0, LIMIT], color="red", linewidth=1.2, label="Perfect prediction")
    plt.plot(line_x["actual_24h_total"], line_y, color="black", linewidth=2, label="Regression fit")
    plt.xlim(0, LIMIT)
    plt.ylim(0, LIMIT)
    plt.xlabel("Actual 24h total")
    plt.ylabel("Predicted 24h total")
    plt.title("Baseline: Actual vs Predicted, Zoomed to 20,000")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT / "baseline_prediction_fit_line_zoom_20000.png", dpi=170)
    plt.close()

    pd.DataFrame(
        [
            {
                "model": "baseline",
                "axis_limit": LIMIT,
                "points_in_zoom": len(zoom),
                "zoom_r2": zoom_r2,
                "fit_slope_all_test": fit.coef_[0],
                "fit_intercept_all_test": fit.intercept_,
            }
        ]
    ).to_csv(PROJECT / "baseline_prediction_fit_line_zoom_20000_summary.csv", index=False, encoding="utf-8-sig")

    print("Wrote zoomed baseline fit chart.")


if __name__ == "__main__":
    main()
