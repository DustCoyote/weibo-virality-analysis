import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("analysis_projects")
SUMMARY_CSV = Path("weibo_summary_full.csv")
RAW_TXT = Path("dataset_weibo.txt")


PROJECTS = {
    "scale": ROOT / "01_spread_scale",
    "depth": ROOT / "02_spread_depth",
    "speed": ROOT / "03_spread_speed",
    "viral": ROOT / "04_viral_post_features",
    "structure": ROOT / "05_network_structure",
    "key_nodes": ROOT / "06_key_nodes",
    "long_tail": ROOT / "07_long_tail",
}


def ensure_dirs():
    ROOT.mkdir(exist_ok=True)
    for folder in PROJECTS.values():
        folder.mkdir(parents=True, exist_ok=True)


def write_report(path, title, lines):
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(f"{title}\n")
        file.write("=" * len(title) + "\n\n")
        for line in lines:
            file.write(f"{line}\n")


def save_bar(df, x, y, title, output, rotation=0):
    plt.figure(figsize=(10, 5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_scatter(df, x, y, title, output, sample=8000):
    plot_df = df[[x, y]].dropna()
    if len(plot_df) > sample:
        plot_df = plot_df.sample(sample, random_state=7)
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df[x], plot_df[y], s=8, alpha=0.35)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def parse_path_token(token):
    path, _, delay = token.partition(":")
    nodes = [node for node in path.split("/") if node]
    try:
        delay_seconds = int(delay)
    except ValueError:
        delay_seconds = None
    return nodes, delay_seconds


def scan_raw_dataset():
    delay_bins = Counter()
    per_cascade_speed = {}
    per_cascade_structure = {}
    global_source_out = Counter()
    global_target_count = Counter()
    top_source_by_cascade = []

    bin_rules = [
        ("within_1_hour", 3600),
        ("within_6_hours", 21600),
        ("within_24_hours", 86400),
        ("within_3_days", 259200),
        ("within_7_days", 604800),
    ]

    with RAW_TXT.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue

            cascade_id, root_id, timestamp = parts[:3]
            tokens = " ".join(parts[4:]).split()

            source_counts = Counter()
            edge_count = 0
            delay_values = []
            early_1h = 0
            early_6h = 0
            early_24h = 0

            for token in tokens:
                nodes, delay_seconds = parse_path_token(token)
                if len(nodes) < 2:
                    continue

                edge_count += 1
                source_node = nodes[-2]
                target_node = nodes[-1]
                source_counts[source_node] += 1
                global_source_out[source_node] += 1
                global_target_count[target_node] += 1

                if delay_seconds is not None:
                    delay_values.append(delay_seconds)
                    placed = False
                    for label, ceiling in bin_rules:
                        if delay_seconds <= ceiling:
                            delay_bins[label] += 1
                            placed = True
                            break
                    if not placed:
                        delay_bins["after_7_days"] += 1
                    if delay_seconds <= 3600:
                        early_1h += 1
                    if delay_seconds <= 21600:
                        early_6h += 1
                    if delay_seconds <= 86400:
                        early_24h += 1

            if edge_count:
                max_source, max_out = source_counts.most_common(1)[0]
                per_cascade_speed[cascade_id] = {
                    "cascade_id": cascade_id,
                    "root_id": root_id,
                    "timestamp": timestamp,
                    "edge_count": edge_count,
                    "early_1h_edges": early_1h,
                    "early_6h_edges": early_6h,
                    "early_24h_edges": early_24h,
                    "early_1h_ratio": early_1h / edge_count,
                    "early_6h_ratio": early_6h / edge_count,
                    "early_24h_ratio": early_24h / edge_count,
                    "median_delay_seconds": sorted(delay_values)[len(delay_values) // 2]
                    if delay_values
                    else "",
                }
                per_cascade_structure[cascade_id] = {
                    "cascade_id": cascade_id,
                    "root_id": root_id,
                    "edge_count": edge_count,
                    "max_source_node": max_source,
                    "max_source_out_degree": max_out,
                    "max_source_share": max_out / edge_count,
                }
                top_source_by_cascade.append(
                    {
                        "cascade_id": cascade_id,
                        "root_id": root_id,
                        "key_node": max_source,
                        "direct_downstream_count": max_out,
                        "edge_count": edge_count,
                        "share_in_cascade": max_out / edge_count,
                    }
                )

    return {
        "delay_bins": delay_bins,
        "per_cascade_speed": per_cascade_speed,
        "per_cascade_structure": per_cascade_structure,
        "global_source_out": global_source_out,
        "global_target_count": global_target_count,
        "top_source_by_cascade": top_source_by_cascade,
    }


def analyze_scale(summary):
    folder = PROJECTS["scale"]
    stats = pd.DataFrame(
        [
            {
                "cascade_count": len(summary),
                "total_observed_paths": int(summary["observed_path_count"].sum()),
                "mean_observed_paths": summary["observed_path_count"].mean(),
                "median_observed_paths": summary["observed_path_count"].median(),
                "max_observed_paths": int(summary["observed_path_count"].max()),
                "mean_unique_nodes": summary["unique_node_count"].mean(),
            }
        ]
    )
    stats.to_csv(folder / "scale_summary.csv", index=False, encoding="utf-8-sig")

    top = summary.sort_values("observed_path_count", ascending=False).head(30)
    top.to_csv(folder / "top_30_largest_cascades.csv", index=False, encoding="utf-8-sig")

    bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, math.inf]
    labels = ["1-10", "11-50", "51-100", "101-500", "501-1000", "1001-5000", "5001-10000", "10000+"]
    scale_bins = summary.assign(
        scale_bin=pd.cut(summary["observed_path_count"], bins=bins, labels=labels, include_lowest=True)
    )["scale_bin"].value_counts().sort_index().reset_index()
    scale_bins.columns = ["observed_path_count_bin", "cascade_count"]
    scale_bins.to_csv(folder / "scale_distribution_bins.csv", index=False, encoding="utf-8-sig")
    save_bar(scale_bins, "observed_path_count_bin", "cascade_count", "Spread Scale Distribution", folder / "scale_distribution.png", rotation=30)

    write_report(
        folder / "report.txt",
        "傳播規模分析",
        [
            f"共有 {len(summary):,} 筆傳播事件。",
            f"平均每筆傳播有 {summary['observed_path_count'].mean():.2f} 條轉發路徑，中位數為 {summary['observed_path_count'].median():.0f}。",
            f"最大傳播事件有 {int(summary['observed_path_count'].max()):,} 條轉發路徑。",
            "可用於討論：微博傳播規模是否集中在少數大型事件，或多數事件都屬於小規模擴散。",
        ],
    )


def analyze_depth(summary):
    folder = PROJECTS["depth"]
    depth_dist = summary["max_depth"].value_counts().sort_index().reset_index()
    depth_dist.columns = ["max_depth", "cascade_count"]
    depth_dist.to_csv(folder / "depth_distribution.csv", index=False, encoding="utf-8-sig")
    deepest = summary.sort_values(["max_depth", "observed_path_count"], ascending=False).head(30)
    deepest.to_csv(folder / "top_30_deepest_cascades.csv", index=False, encoding="utf-8-sig")
    save_bar(depth_dist, "max_depth", "cascade_count", "Maximum Depth Distribution", folder / "depth_distribution.png")

    write_report(
        folder / "report.txt",
        "傳播深度分析",
        [
            f"最常見的最大深度是 {int(depth_dist.sort_values('cascade_count', ascending=False).iloc[0]['max_depth'])} 層。",
            f"資料中的最大深度為 {int(summary['max_depth'].max())} 層。",
            f"平均最大深度為 {summary['max_depth'].mean():.2f} 層。",
            "可用於討論：微博傳播多半是淺層大量轉發，還是會形成多層級連鎖傳播。",
        ],
    )


def analyze_speed(summary, raw_stats):
    folder = PROJECTS["speed"]
    speed_df = pd.DataFrame(raw_stats["per_cascade_speed"].values())
    speed_df.to_csv(folder / "cascade_speed_metrics.csv", index=False, encoding="utf-8-sig")

    delay_order = ["within_1_hour", "within_6_hours", "within_24_hours", "within_3_days", "within_7_days", "after_7_days"]
    delay_bins = pd.DataFrame(
        [{"delay_bin": label, "edge_count": raw_stats["delay_bins"].get(label, 0)} for label in delay_order]
    )
    delay_bins["edge_share"] = delay_bins["edge_count"] / delay_bins["edge_count"].sum()
    delay_bins.to_csv(folder / "delay_distribution_bins.csv", index=False, encoding="utf-8-sig")
    save_bar(delay_bins, "delay_bin", "edge_count", "Repost Delay Distribution", folder / "delay_distribution.png", rotation=30)

    large = speed_df[speed_df["edge_count"] >= speed_df["edge_count"].quantile(0.9)].copy()
    fastest_large = large.sort_values(["early_1h_ratio", "edge_count"], ascending=False).head(30)
    fastest_large.to_csv(folder / "top_30_fast_large_cascades.csv", index=False, encoding="utf-8-sig")

    write_report(
        folder / "report.txt",
        "傳播速度分析",
        [
            f"1 小時內完成的轉發邊數：{int(delay_bins.loc[delay_bins['delay_bin'] == 'within_1_hour', 'edge_count'].iloc[0]):,}。",
            f"24 小時內完成的轉發邊數：{int(delay_bins.loc[delay_bins['delay_bin'] == 'within_24_hours', 'edge_count'].iloc[0]):,}。",
            "cascade_speed_metrics.csv 可用來比較每筆傳播在 1 小時、6 小時、24 小時內完成的比例。",
            "可用於討論：大量轉發是否集中在早期，以及早期轉發比例高的事件是否更容易形成大型擴散。",
        ],
    )


def analyze_viral(summary, raw_stats):
    folder = PROJECTS["viral"]
    speed_df = pd.DataFrame(raw_stats["per_cascade_speed"].values())
    viral = summary.merge(speed_df[["cascade_id", "early_1h_ratio", "early_6h_ratio", "early_24h_ratio"]], on="cascade_id", how="left")
    viral["log_scale"] = viral["observed_path_count"].apply(lambda value: math.log1p(value))
    viral["scale_score"] = viral["log_scale"] / viral["log_scale"].max()
    viral["depth_score"] = viral["max_depth"] / viral["max_depth"].max()
    viral["speed_score"] = viral["early_24h_ratio"].fillna(0)
    viral["viral_score"] = (viral["scale_score"] * 0.5) + (viral["depth_score"] * 0.2) + (viral["speed_score"] * 0.3)

    top = viral.sort_values("viral_score", ascending=False).head(50)
    top.to_csv(folder / "top_50_viral_feature_cascades.csv", index=False, encoding="utf-8-sig")
    viral[["observed_path_count", "unique_node_count", "max_depth", "avg_delay_seconds", "early_24h_ratio", "viral_score"]].describe().to_csv(
        folder / "viral_feature_descriptive_stats.csv", encoding="utf-8-sig"
    )
    save_scatter(viral, "observed_path_count", "max_depth", "Scale vs Depth", folder / "scale_vs_depth.png")

    write_report(
        folder / "report.txt",
        "爆紅貼文特徵分析",
        [
            "viral_score 是本專案自訂的綜合指標：傳播規模 50%、傳播深度 20%、24 小時內早期轉發比例 30%。",
            "top_50_viral_feature_cascades.csv 列出最符合爆紅特徵的傳播事件。",
            "可用於討論：爆紅事件是否同時具備大規模、較深層傳播，以及較高早期擴散比例。",
        ],
    )


def analyze_structure(summary, raw_stats):
    folder = PROJECTS["structure"]
    structure = pd.DataFrame(raw_stats["per_cascade_structure"].values())
    structure = structure.merge(summary[["cascade_id", "max_depth", "observed_path_count", "unique_node_count"]], on="cascade_id", how="left")

    def classify(row):
        if row["max_depth"] <= 3 and row["max_source_share"] >= 0.5:
            return "star_like"
        if row["max_depth"] >= 6 and row["max_source_share"] < 0.2:
            return "chain_like"
        return "tree_like"

    structure["structure_type"] = structure.apply(classify, axis=1)
    structure.to_csv(folder / "cascade_structure_metrics.csv", index=False, encoding="utf-8-sig")
    counts = structure["structure_type"].value_counts().reset_index()
    counts.columns = ["structure_type", "cascade_count"]
    counts.to_csv(folder / "structure_type_counts.csv", index=False, encoding="utf-8-sig")
    save_bar(counts, "structure_type", "cascade_count", "Network Structure Types", folder / "structure_type_counts.png", rotation=20)

    write_report(
        folder / "report.txt",
        "網路結構分析",
        [
            "本專案用 max_source_share 和 max_depth 粗分三種型態：star_like、tree_like、chain_like。",
            "star_like 代表多數轉發集中在少數來源；chain_like 代表較深且較分散的鏈式傳播；tree_like 介於兩者之間。",
            f"最多的結構類型是 {counts.iloc[0]['structure_type']}，共有 {int(counts.iloc[0]['cascade_count']):,} 筆。",
            "可用於討論：微博擴散是偏向中心化星狀傳播，還是多層樹狀/鏈狀傳播。",
        ],
    )


def analyze_key_nodes(raw_stats):
    folder = PROJECTS["key_nodes"]
    global_source = pd.DataFrame(
        [
            {"node_id": node, "direct_downstream_count": count}
            for node, count in raw_stats["global_source_out"].most_common(100)
        ]
    )
    global_source.to_csv(folder / "top_100_global_direct_influencers.csv", index=False, encoding="utf-8-sig")

    global_targets = pd.DataFrame(
        [
            {"node_id": node, "times_as_reposter": count}
            for node, count in raw_stats["global_target_count"].most_common(100)
        ]
    )
    global_targets.to_csv(folder / "top_100_most_frequent_reposters.csv", index=False, encoding="utf-8-sig")

    per_cascade = pd.DataFrame(raw_stats["top_source_by_cascade"])
    per_cascade.sort_values(["direct_downstream_count", "share_in_cascade"], ascending=False).head(500).to_csv(
        folder / "top_key_node_by_cascade.csv", index=False, encoding="utf-8-sig"
    )
    save_bar(global_source.head(20), "node_id", "direct_downstream_count", "Top Direct Influencers", folder / "top_direct_influencers.png", rotation=45)

    write_report(
        folder / "report.txt",
        "關鍵節點分析",
        [
            "此分析把 source 視為帶動後續轉發的人，direct_downstream_count 表示他直接帶出多少轉發者。",
            "top_100_global_direct_influencers.csv 是全資料中直接帶動最多人的節點。",
            "top_key_node_by_cascade.csv 是各傳播事件內部最能帶動直接轉發的節點。",
            "注意：若不同 cascade 之間節點沒有交集，較嚴謹的說法是「單一傳播事件中的關鍵擴散者」，不要直接稱為全微博影響者。",
        ],
    )


def analyze_long_tail(summary):
    folder = PROJECTS["long_tail"]
    sorted_summary = summary.sort_values("observed_path_count", ascending=False).reset_index(drop=True)
    total_paths = sorted_summary["observed_path_count"].sum()
    rows = []
    for percent in [1, 5, 10, 20, 50]:
        cutoff = max(1, int(len(sorted_summary) * percent / 100))
        contribution = sorted_summary.head(cutoff)["observed_path_count"].sum() / total_paths
        rows.append({"top_percent": percent, "cascade_count": cutoff, "path_share": contribution})
    contribution_df = pd.DataFrame(rows)
    contribution_df.to_csv(folder / "top_percent_contribution.csv", index=False, encoding="utf-8-sig")

    bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, math.inf]
    labels = ["1-10", "11-50", "51-100", "101-500", "501-1000", "1001-5000", "5001-10000", "10000+"]
    tail_bins = summary.assign(
        scale_bin=pd.cut(summary["observed_path_count"], bins=bins, labels=labels, include_lowest=True)
    )["scale_bin"].value_counts().sort_index().reset_index()
    tail_bins.columns = ["observed_path_count_bin", "cascade_count"]
    tail_bins.to_csv(folder / "long_tail_bins.csv", index=False, encoding="utf-8-sig")
    save_bar(tail_bins, "observed_path_count_bin", "cascade_count", "Long Tail Distribution", folder / "long_tail_distribution.png", rotation=30)

    write_report(
        folder / "report.txt",
        "長尾現象分析",
        [
            f"前 1% 傳播事件貢獻了 {contribution_df.loc[contribution_df['top_percent'] == 1, 'path_share'].iloc[0] * 100:.2f}% 的轉發路徑。",
            f"前 10% 傳播事件貢獻了 {contribution_df.loc[contribution_df['top_percent'] == 10, 'path_share'].iloc[0] * 100:.2f}% 的轉發路徑。",
            "若少數事件貢獻大量轉發，即可支持社群媒體傳播具有長尾分布或爆量集中現象。",
        ],
    )


def main():
    ensure_dirs()
    summary = pd.read_csv(SUMMARY_CSV)
    summary["cascade_id"] = summary["cascade_id"].astype(str)
    raw_stats = scan_raw_dataset()

    analyze_scale(summary)
    analyze_depth(summary)
    analyze_speed(summary, raw_stats)
    analyze_viral(summary, raw_stats)
    analyze_structure(summary, raw_stats)
    analyze_key_nodes(raw_stats)
    analyze_long_tail(summary)

    print(f"Analysis projects written to {ROOT.resolve()}")


if __name__ == "__main__":
    main()
