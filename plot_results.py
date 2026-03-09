#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_bool(s: str) -> bool:
    value = s.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {s}")


def marker_for_method(method: str) -> str:
    lower = method.lower()
    if "torch" in lower or "cute" in lower:
        return "^"
    if "sage" in lower:
        return "s"
    return "o"


def category_for_method(method: str) -> str:
    lower = method.lower()
    if "torch" in lower or "cute" in lower:
        return "torch_cute"
    if "sage" in lower:
        return "sage"
    return "other"


def load_rows(csv_path: Path, causal_filter: str) -> list[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    parsed_rows: list[dict] = []
    for row in rows:
        if row.get("status") != "OK":
            continue
        if causal_filter != "all":
            row_causal = parse_bool(row["causal"])
            if row_causal != parse_bool(causal_filter):
                continue
        parsed_rows.append(
            {
                "model": row["model"],
                "method": row["method"],
                "category": category_for_method(row["method"]),
                "seq_len": int(row["seq_len"]),
                "tflops": float(row["tflops"]),
            }
        )
    return parsed_rows


def build_topn_series(rows: list[dict], top_n: int):
    # model -> seq_len -> [rows sorted by tflops desc]
    model_seq_rows: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model_seq_rows[row["model"]][row["seq_len"]].append(row)

    model_plot_data: dict[str, dict] = {}
    for model, seq_map in model_seq_rows.items():
        seq_lens = sorted(seq_map.keys())
        top_rows_by_seq: dict[int, list[dict]] = {}
        top_methods_union: set[str] = set()
        for seq in seq_lens:
            category_rows: dict[str, list[dict]] = defaultdict(list)
            for row in seq_map[seq]:
                category_rows[row["category"]].append(row)

            top_rows: list[dict] = []
            for category in ("torch_cute", "sage", "other"):
                top_rows.extend(
                    sorted(category_rows[category], key=lambda r: r["tflops"], reverse=True)[:top_n]
                )
            top_rows_by_seq[seq] = top_rows
            for r in top_rows:
                top_methods_union.add(r["method"])

        # method -> seq_len -> tflops; non-top seqs are NaN so lines only connect top points.
        seq_to_index = {seq: idx for idx, seq in enumerate(seq_lens)}
        method_series: dict[str, list[float]] = {}
        method_peak: dict[str, float] = {}
        for method in top_methods_union:
            y = [math.nan] * len(seq_lens)
            for seq in seq_lens:
                maybe_row = next((r for r in top_rows_by_seq[seq] if r["method"] == method), None)
                if maybe_row is not None:
                    y[seq_to_index[seq]] = maybe_row["tflops"]
            method_series[method] = y
            method_peak[method] = max(v for v in y if not math.isnan(v))

        model_plot_data[model] = {
            "seq_lens": seq_lens,
            "method_series": method_series,
            "method_order": sorted(method_series.keys(), key=lambda m: method_peak[m], reverse=True),
        }

    return model_plot_data


def plot_results(model_plot_data: dict[str, dict], output_path: Path, top_n: int, causal_filter: str) -> None:
    model_names = sorted(model_plot_data.keys())
    if not model_names:
        raise RuntimeError("No rows to plot after filtering.")

    fig, axes = plt.subplots(
        nrows=len(model_names),
        ncols=1,
        figsize=(14, max(4, 4 * len(model_names))),
        sharex=False,
        constrained_layout=True,
    )
    if len(model_names) == 1:
        axes = [axes]

    for ax, model in zip(axes, model_names):
        data = model_plot_data[model]
        seq_lens = data["seq_lens"]
        x = list(range(len(seq_lens)))
        for method in data["method_order"]:
            y = data["method_series"][method]
            ax.plot(x, y, marker=marker_for_method(method), linewidth=1.8, label=method)

        ax.set_title(f"{model} (top {top_n} per category per sequence)")
        ax.set_ylabel("TFLOPs")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in seq_lens])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

    axes[-1].set_xlabel("Sequence Length")
    fig.suptitle(f"Attention Benchmark Results (causal={causal_filter})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"[info] wrote plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark results.csv with one vertical subplot per model and top-N methods per sequence length."
    )
    parser.add_argument("--input", type=str, default="results.csv", help="Path to results CSV.")
    parser.add_argument("--output", type=str, default="results_plot.png", help="Path to output image.")
    parser.add_argument("--top-n", type=int, default=2, help="Number of top methods to keep per sequence length.")
    parser.add_argument(
        "--causal",
        type=str,
        default="false",
        choices=["all", "true", "false"],
        help="Filter by causal flag in CSV.",
    )
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    rows = load_rows(input_path, args.causal)
    model_plot_data = build_topn_series(rows, args.top_n)
    plot_results(model_plot_data, Path(args.output), args.top_n, args.causal)


if __name__ == "__main__":
    main()
