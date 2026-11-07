"""Roofline + bandwidth plots and a markdown summary from results/bench.csv.

    python bench/plot.py --results results [--peak-gbs 273]

Outputs:
  results/roofline.png       classic roofline: achieved GFLOP/s vs arithmetic intensity
  results/bandwidth.png      effective GB/s vs tokens (M), one panel per hidden size
  results/summary.md         per-shape table: time, GB/s, % of copy BW, speedup vs eager/compile
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order (never cycled); a marker per series gives a
# second, non-color channel for identity.
STYLE = {
    "cuda":       dict(color="#2a78d6", marker="o", label="CUDA (hand-written)"),
    "triton":     dict(color="#eb6834", marker="s", label="Triton"),
    "compile":    dict(color="#1baf7a", marker="^", label="torch.compile"),
    "eager":      dict(color="#eda100", marker="D", label="PyTorch eager"),
    "cuda_naive": dict(color="#e87ba4", marker="v", label="CUDA naive (fwd)"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6, "figure.dpi": 150,
})


def roofline(df, meta, peak_gbs, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    peak_flops = meta["fp32_tflops"] * 1e3  # GFLOP/s
    xs = np.logspace(np.log10(0.05), 3, 400)  # dense, so the roof kinks at the ridge
    for bw, ls, name, dy in [(peak_gbs, "-", f"spec DRAM {peak_gbs:.0f} GB/s", 1.6),
                             (meta["copy_gbs"], "--", f"measured copy {meta['copy_gbs']:.0f} GB/s", 0.45)]:
        ax.plot(xs, np.minimum(bw * xs, peak_flops), ls, color=MUTED, lw=1.2)
        ax.text(3, bw * 3 * dy, name, color=MUTED, fontsize=8)
    ax.text(200, peak_flops * 1.1, f"FP32 {meta['fp32_tflops']:.1f} TFLOP/s (measured sgemm)",
            color=MUTED, fontsize=8, ha="right")
    ridge = peak_flops / peak_gbs
    ax.axvline(ridge, color=GRID, lw=1)
    ax.text(ridge * 1.08, 2, f"ridge ≈ {ridge:.0f} FLOP/B", color=MUTED, fontsize=8)

    big = df[df.M == df.M.max()]
    for impl, st in STYLE.items():
        for p, fill in (("fwd", True), ("bwd", False)):
            d = big[(big.impl == impl) & (big["pass"] == p)]
            if d.empty:
                continue
            ax.scatter(d.ai, d.gflops, s=46, marker=st["marker"],
                       facecolor=st["color"] if fill else "white", edgecolor=st["color"],
                       linewidth=1.6, zorder=3, label=f"{st['label']} {p}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.05, 1000); ax.set_ylim(1, peak_flops * 3)
    ax.set_xlabel("Arithmetic intensity (FLOP / byte of ideal fused traffic)")
    ax.set_ylabel("Achieved GFLOP/s")
    ax.set_title(f"Fused add + RMSNorm roofline, {meta['gpu']}, {meta['dtype']}, M={df.M.max()}",
                 loc="left", color=INK, fontsize=11)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def bandwidth(df, meta, peak_gbs, path):
    Ns = sorted(df.N.unique())
    fig, axes = plt.subplots(2, len(Ns), figsize=(3.1 * len(Ns), 6.4), sharey=True, sharex=True)
    for row, p in enumerate(("fwd", "bwd")):
        for col, N in enumerate(Ns):
            ax = axes[row][col]
            ax.axhline(peak_gbs, color=MUTED, lw=1)
            ax.axhline(meta["copy_gbs"], color=MUTED, lw=1, ls="--")
            for impl, st in STYLE.items():
                d = df[(df.impl == impl) & (df["pass"] == p) & (df.N == N)].sort_values("M")
                if d.empty:
                    continue
                ax.plot(d.M, d.gbs, color=st["color"], marker=st["marker"], label=st["label"])
            ax.set_xscale("log", base=2)
            ax.set_title(f"{p}  N={N}", loc="left", fontsize=9.5, color=INK)
            if col == 0:
                ax.set_ylabel("Effective GB/s")
            if row == 1:
                ax.set_xlabel("Tokens M")
    axes[0][0].text(1.2, peak_gbs * 1.02, "spec", color=MUTED, fontsize=7.5)
    axes[0][0].text(1.2, meta["copy_gbs"] * 0.9, "copy", color=MUTED, fontsize=7.5)
    h, l = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(STYLE), fontsize=8.5)
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path); plt.close(fig)


def summary(df, meta, path):
    lines = [f"# Results: {meta['gpu']}, {meta['dtype']}",
             "",
             f"torch {meta['torch']}, triton {meta['triton']}. Measured device-to-device copy: "
             f"**{meta['copy_gbs']:.0f} GB/s**. Effective GB/s = ideal fused bytes / time.",
             ""]
    for p in ("fwd", "bwd"):
        lines += [f"## {p}", "",
                  "| M | N | eager µs | compile µs | triton µs | cuda µs | cuda GB/s (% copy) "
                  "| cuda vs eager | cuda vs compile | cuda vs triton |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        d = df[df["pass"] == p].pivot_table(index=["M", "N"], columns="impl", values="us")
        g = df[(df["pass"] == p) & (df.impl == "cuda")].set_index(["M", "N"])
        for (M, N), r in d.iterrows():
            c = r.get("cuda")
            lines.append(
                f"| {M} | {N} | {r.get('eager', float('nan')):.1f} | {r.get('compile', float('nan')):.1f} "
                f"| {r.get('triton', float('nan')):.1f} | {c:.1f} "
                f"| {g.loc[(M, N), 'gbs']:.0f} ({g.loc[(M, N), 'pct_copy_bw']:.0f}%) "
                f"| {r['eager'] / c:.2f}× | {r['compile'] / c:.2f}× | {r['triton'] / c:.2f}× |")
        lines.append("")
    k = df.dropna(subset=["kernels"])
    if not k.empty:
        lines += ["## Kernel launches per call", "",
                  "| impl | pass | M | N | kernels |", "|---|---|---:|---:|---:|"]
        for _, r in k[k.N == 4096].sort_values(["pass", "impl", "M"]).iterrows():
            lines.append(f"| {r.impl} | {r['pass']} | {r.M} | {r.N} | {int(r.kernels)} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--peak-gbs", type=float, default=273.0,
                    help="spec DRAM bandwidth (GB10 LPDDR5x: 273)")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.results, "bench.csv"))
    meta = json.load(open(os.path.join(a.results, "meta.json")))
    roofline(df, meta, a.peak_gbs, os.path.join(a.results, "roofline.png"))
    bandwidth(df, meta, a.peak_gbs, os.path.join(a.results, "bandwidth.png"))
    summary(df, meta, os.path.join(a.results, "summary.md"))
    print("wrote roofline.png, bandwidth.png, summary.md")


if __name__ == "__main__":
    main()
