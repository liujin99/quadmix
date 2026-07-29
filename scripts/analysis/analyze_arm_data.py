#!/usr/bin/env python3
"""Compare per-arm training-data distributions across downstream A/B arms.

Reads the sharded parquet arms produced by nanochat_mid_compare/prepare_data.py
(<result-dir>/data/{quadmix,random,manual_ratio}_data/shard_*.parquet) and
characterizes each arm to diagnose why min-val_loss selection may pick "easy"
data: low-entropy / repetitive / low-diversity text that drives val_loss down
without building capability.

Outputs (into --output-dir, default = --result-dir):
  fig_arm_length.png      — char-length histograms, arms overlaid (density)
  fig_arm_entropy.png     — char-level Shannon entropy (bits) histograms
  fig_arm_repetition.png  — single-char repetition fraction histograms
  fig_arm_diversity.png   — lexical diversity (type/token ratio) histograms
  fig_arm_domain.png      — domain distribution grouped bars (needs domain col)
  arm_data_comparison.txt — per-arm doc/token counts + metric percentiles + domain mix

Text-based slices run on existing text-only parquets now; the domain slice needs
the "domain" label column (prepare_data preserves it when the source schema has
domain_col). If absent, fig_arm_domain is skipped with a notice.

Usage:
  python scripts/analysis/analyze_arm_data.py \
      --result-dir nanochat_mid_compare/results_stem/<timestamp>
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401  (ensures pyarrow backend present)
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import quadmix  # noqa: F401
except ImportError:
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"),
    )

from quadmix.pipeline.report import _setup_style, _save_fig


# arm label -> display color (cycled for >3 arms)
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ── per-document metrics ──────────────────────────────────────────


def _doc_metrics(text):
    """Return (char_len, entropy_bits, repetition_frac, lexical_diversity).

    Entropy & repetition are computed over Unicode code points via a uint32
    view of the utf-32 encoding (one 32-bit unit per code point, no BOM), so
    both reuse the same array. Lexical diversity uses whitespace tokens.
    """
    if not text:
        return 0, 0.0, 0.0, 0.0
    n = len(text)
    arr = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    m = arr.size
    if m > 1:
        rep = float(np.count_nonzero(arr[1:] == arr[:-1])) / (m - 1)
    else:
        rep = 0.0
    _, counts = np.unique(arr, return_counts=True)
    p = counts / m
    ent = float(-(p * np.log2(p)).sum())
    toks = text.split()
    div = len(set(toks)) / len(toks) if toks else 0.0
    return n, ent, rep, div


# ── shard worker ──────────────────────────────────────────────────


def _scan_shard(task):
    idx, path, has_domain = task
    cols = ["text", "domain"] if has_domain else ["text"]
    table = pq.read_table(path, columns=cols)
    texts = table["text"].to_pylist()
    n = len(texts)
    lens = np.empty(n, dtype=np.int64)
    ents = np.empty(n, dtype=np.float32)
    reps = np.empty(n, dtype=np.float32)
    divs = np.empty(n, dtype=np.float32)
    for i, t in enumerate(texts):
        lens[i], ents[i], reps[i], divs[i] = _doc_metrics(t or "")
    dom = Counter(table["domain"].to_pylist()) if has_domain else None
    return lens, ents, reps, divs, dom


# ── figures ──────────────────────────────────────────────────────


def _fig_hist(per_arm, key, xlabel, fname, out_dir, logx=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pooled = np.concatenate(
        [a[key] for a in per_arm.values() if len(a[key])]
    ) if per_arm else np.array([])
    if pooled.size == 0:
        plt.close(fig)
        print(f"  [skip] no data for {fname}")
        return
    lo, hi = np.percentile(pooled, [1, 99])
    if logx:
        lo = max(1.0, lo)
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)
    else:
        bins = np.linspace(lo, hi, 50)
    for i, (label, arm) in enumerate(per_arm.items()):
        v = arm[key]
        v = v[(v >= lo) & (v <= hi)]
        if v.size == 0:
            continue
        ax.hist(
            v, bins=bins, density=True, histtype="step", lw=1.6,
            label=f"{label} (n={len(arm[key]):,})", color=_COLORS[i % len(_COLORS)],
        )
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(xlabel)
    ax.legend(fontsize=9)
    _save_fig(fig, out_dir, fname)


def _fig_domain(per_arm, out_dir):
    arms_with = {l: a for l, a in per_arm.items() if a["has_domain"] and a["domain"]}
    if not arms_with:
        print("  [skip] fig_arm_domain: no 'domain' column — re-run prepare_data to add it")
        return
    union = Counter()
    for a in arms_with.values():
        union.update(a["domain"])
    domains = [d for d, _ in union.most_common()]
    k = len(arms_with)
    width = 0.8 / k
    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(domains) * k), 4.5))
    for i, (label, arm) in enumerate(arms_with.items()):
        total = sum(arm["domain"].values()) or 1
        fr = [arm["domain"].get(d, 0) / total for d in domains]
        ax.bar(
            x + (i - (k - 1) / 2) * width, fr, width, label=label,
            color=_COLORS[i % len(_COLORS)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in domains], rotation=30, ha="right")
    ax.set_ylabel("fraction of docs")
    ax.set_title("Domain distribution by arm")
    ax.legend(fontsize=9)
    _save_fig(fig, out_dir, "fig_arm_domain.png")


# ── text summary ─────────────────────────────────────────────────


def _write_txt(per_arm, stats, out_dir):
    lines = ["=== Arm Data Comparison ===", ""]
    for label, a in per_arm.items():
        st = stats.get(label, {})
        lines.append(
            f"[{label}]  train_docs={st.get('train_docs', '?')}  "
            f"tokens={st.get('tokens', '?')}  shards={st.get('shards', '?')}"
        )
        for key, name in [
            ("len", "length(chars)"),
            ("ent", "entropy_bits"),
            ("rep", "repetition"),
            ("div", "diversity"),
        ]:
            v = a[key]
            if len(v) == 0:
                continue
            lines.append(
                f"  {name:14s} mean={v.mean():.4f}  median={np.median(v):.4f}  "
                f"p25={np.percentile(v, 25):.4f}  p75={np.percentile(v, 75):.4f}  "
                f"p90={np.percentile(v, 90):.4f}"
            )
        if a["has_domain"] and a["domain"]:
            total = sum(a["domain"].values()) or 1
            top = a["domain"].most_common(10)
            lines.append(
                "  domain top: "
                + ", ".join(f"{d}={c / total:.1%}" for d, c in top)
            )
        lines.append("")
    path = os.path.join(out_dir, "arm_data_comparison.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [Text] Saved: {path}")


# ── main ─────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare per-arm training-data distributions across downstream A/B arms."
    )
    p.add_argument(
        "--result-dir", required=True,
        help="results_stem/<ts> dir containing data/<arm>_data/ + dataset_stats.json",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Where to write figures/txt (default: --result-dir)",
    )
    p.add_argument(
        "--num-workers", type=int, default=None,
        help="multiprocessing workers (default: min(32, cpu_count))",
    )
    return p.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    out_dir = args.output_dir or str(result_dir)
    os.makedirs(out_dir, exist_ok=True)
    num_workers = args.num_workers or min(32, os.cpu_count() or 1)

    data_root = result_dir / "data"
    if not data_root.is_dir():
        sys.exit(f"error: {data_root} not found (need data/<arm>_data/ subdirs)")

    stats_path = result_dir / "dataset_stats.json"
    stats = json.load(open(stats_path)) if stats_path.is_file() else {}

    # discover arms: data/<arm>_data/, train shards = all but last (val)
    arms = {}
    for p in sorted(data_root.glob("*_data")):
        if not p.is_dir():
            continue
        label = p.name.removesuffix("_data")
        shards = sorted(p.glob("shard_*.parquet"))
        train = shards[:-1] if shards else []
        if not train:
            print(f"  [skip] {label}: no train shards")
            continue
        has_domain = "domain" in pq.read_schema(str(train[0])).names
        arms[label] = {"dir": p, "shards": train, "has_domain": has_domain}
        print(f"  arm '{label}': {len(train)} train shards, domain_col={has_domain}")

    if not arms:
        sys.exit(f"error: no arms discovered under {data_root}")

    from multiprocessing import Pool
    from tqdm import tqdm

    per_arm = {}
    for label, info in arms.items():
        tasks = [(i, str(s), info["has_domain"]) for i, s in enumerate(info["shards"])]
        L, E, R, D, dom = [], [], [], [], Counter()
        with Pool(num_workers) as pool:
            for lens, ents, reps, divs, d in tqdm(
                pool.imap_unordered(_scan_shard, tasks, chunksize=1),
                total=len(tasks), desc=f"  {label}", leave=False,
            ):
                L.append(lens); E.append(ents); R.append(reps); D.append(divs)
                if d:
                    dom.update(d)
        per_arm[label] = {
            "len": np.concatenate(L) if L else np.array([], dtype=np.int64),
            "ent": np.concatenate(E) if E else np.array([], dtype=np.float32),
            "rep": np.concatenate(R) if R else np.array([], dtype=np.float32),
            "div": np.concatenate(D) if D else np.array([], dtype=np.float32),
            "domain": dom,
            "has_domain": info["has_domain"],
        }

    print("\n=== Generating figures ===")
    _setup_style()
    _fig_hist(per_arm, "len", "Document length (chars)", "fig_arm_length.png", out_dir, logx=True)
    _fig_hist(per_arm, "ent", "Char entropy (bits)", "fig_arm_entropy.png", out_dir)
    _fig_hist(per_arm, "rep", "Single-char repetition fraction", "fig_arm_repetition.png", out_dir)
    _fig_hist(per_arm, "div", "Lexical diversity (type/token)", "fig_arm_diversity.png", out_dir)
    _fig_domain(per_arm, out_dir)
    _write_txt(per_arm, stats, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
