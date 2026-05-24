import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_path_token(token):
    path, _, delay = token.partition(":")
    nodes = [node for node in path.split("/") if node]
    try:
        delay_value = int(delay)
    except ValueError:
        delay_value = None
    return nodes, delay_value


def summarize_line(line):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None

    cascade_id, root_id, timestamp, declared_count = parts[:4]
    tokens = " ".join(parts[4:]).split()

    depths = []
    delays = []
    unique_nodes = set()

    for token in tokens:
        nodes, delay = parse_path_token(token)
        if nodes:
            depths.append(len(nodes))
            unique_nodes.update(nodes)
        if delay is not None:
            delays.append(delay)

    try:
        timestamp_value = int(timestamp)
        utc_time = datetime.fromtimestamp(timestamp_value, tz=timezone.utc).isoformat()
    except ValueError:
        timestamp_value = ""
        utc_time = ""

    observed_count = len(tokens)
    max_depth = max(depths) if depths else 0
    max_delay = max(delays) if delays else 0
    avg_delay = round(sum(delays) / len(delays), 2) if delays else 0

    return {
        "cascade_id": cascade_id,
        "root_id": root_id,
        "timestamp": timestamp_value,
        "utc_time": utc_time,
        "declared_repost_count": declared_count,
        "observed_path_count": observed_count,
        "unique_node_count": len(unique_nodes),
        "max_depth": max_depth,
        "max_delay_seconds": max_delay,
        "avg_delay_seconds": avg_delay,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset_weibo.txt into a compact cascade-level CSV summary."
    )
    parser.add_argument("--input", default="dataset_weibo.txt")
    parser.add_argument("--output", default="weibo_summary.csv")
    parser.add_argument("--stats", default="weibo_stats.txt")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N rows. Use 0 to process the whole file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats)

    fieldnames = [
        "cascade_id",
        "root_id",
        "timestamp",
        "utc_time",
        "declared_repost_count",
        "observed_path_count",
        "unique_node_count",
        "max_depth",
        "max_delay_seconds",
        "avg_delay_seconds",
    ]

    row_count = 0
    path_count_total = 0
    node_count_total = 0
    depth_counter = Counter()
    top_cascades = []

    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as source:
        with output_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()

            for line in source:
                if args.limit and row_count >= args.limit:
                    break

                summary = summarize_line(line)
                if summary is None:
                    continue

                writer.writerow(summary)
                row_count += 1
                path_count_total += summary["observed_path_count"]
                node_count_total += summary["unique_node_count"]
                depth_counter[summary["max_depth"]] += 1
                top_cascades.append(
                    (
                        summary["observed_path_count"],
                        summary["cascade_id"],
                        summary["root_id"],
                        summary["max_depth"],
                    )
                )
                top_cascades = sorted(top_cascades, reverse=True)[:10]

    avg_paths = round(path_count_total / row_count, 2) if row_count else 0
    avg_nodes = round(node_count_total / row_count, 2) if row_count else 0

    with stats_path.open("w", encoding="utf-8", newline="") as stats:
        stats.write("Weibo dataset summary\n")
        stats.write("=====================\n")
        stats.write(f"Input file: {input_path}\n")
        stats.write(f"Output CSV: {output_path}\n")
        stats.write(f"Rows processed: {row_count}\n")
        stats.write(f"Total observed paths: {path_count_total}\n")
        stats.write(f"Average paths per cascade: {avg_paths}\n")
        stats.write(f"Average unique nodes per cascade: {avg_nodes}\n\n")

        stats.write("Max-depth distribution:\n")
        for depth, count in sorted(depth_counter.items()):
            stats.write(f"- depth {depth}: {count}\n")

        stats.write("\nTop 10 cascades by observed paths:\n")
        for observed_paths, cascade_id, root_id, max_depth in top_cascades:
            stats.write(
                f"- cascade {cascade_id}, root {root_id}: "
                f"{observed_paths} paths, max depth {max_depth}\n"
            )

    print(f"Wrote {row_count} rows to {output_path}")
    print(f"Wrote statistics to {stats_path}")


if __name__ == "__main__":
    main()
