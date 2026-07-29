#!/usr/bin/env python3
"""Compare per-arm training-data distributions across downstream A/B arms.

Reads the sharded parquet arms produced by nanochat_mid_compare/prepare_data.py
(<result-dir>/data/{quadmix,random,manual_ratio}_data/shard_*.parquet) and
characterizes each arm to diagnose why min-val_loss selection may pick "easy"
data: low-entropy / repetitive / low-diversity / short-fragment text that drives
val_loss down without building capability.

Outputs (into --output-dir, default = --result-dir):
  fig_arm_length.png        — char-length histograms, arms overlaid (density)
  fig_arm_entropy.png       — char-level Shannon entropy (bits) histograms
  fig_arm_repetition.png    — single-char repetition fraction histograms
  fig_arm_diversity.png     — lexical diversity (type/token) histograms
  fig_arm_token_length.png  — TOKEN length histograms + 2k window marker
                              (needs --tokenizer; skipped otherwise)
  fig_arm_boundaries.png    — doc-boundaries per 2k packed row histograms
                              (needs --tokenizer; skipped otherwise)
  fig_arm_domain.png        — domain distribution grouped bars (needs domain col)
  arm_data_comparison.txt   — per-arm doc/token counts + metric percentiles +
                              domain mix + token-length & boundary stats

Text-based slices run on existing text-only parquets now; the token slices need
--tokenizer (nanochat tokenizer.pkl) and the domain slice needs the "domain"
label column (prepare_data preserves it when the source schema has domain_col).

Usage:
  python scripts/analysis/analyze_arm_data.py \
      --result-dir nanochat_mid_compare/results_stem/<timestamp> \
      --tokenizer /home/ma-user/work/nanochat_model_dir/tokenizer
"""

import argparse
import json
import os
import pickle
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


_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# worker globals for tokenization
_ENC = None
_BOS_ID = None
_TOK_THREADS = 1


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


# ── tokenizer worker init ─────────────────────────────────────────


def _init_tok_worker(tokenizer_path, tok_threads):
    global _ENC, _BOS_ID, _TOK_THREADS
    _TOK_THREADS = tok_threads
    if not tokenizer_path:
        return
    pkl = tokenizer_path if os.path.isfile(tokenizer_path) \
        else os.path.join(tokenizer_path, "tokenizer.pkl")
    with open(pkl, "rb") as f:
        _ENC = pickle.load(f)
    _BOS_ID = _ENC.encode_single_token("<|bos|>")


def _tokenize_lens(texts):
    """Return numpy int64 array of token lengths (incl. +1 BOS the dataloader prepends)."""
    if _ENC is None:
        return None
    n = len(texts)
    out = np.empty(n, dtype=np.int64)
    bs = 256
    for s in range(0, n, bs):
        chunk = [t or "" for t in texts[s:s + bs]]
        try:
            encs = _ENC.encode_ordinary_batch(chunk, num_threads=_TOK_THREADS)
        except (AttributeError, TypeError):
            encs = [_ENC.encode_ordinary(t) for t in chunk]
        for j, ids in enumerate(encs):
            out[s + j] = len(ids) + 1  # +1 for BOS prepended by the dataloader
    return out


# ── shard worker ──────────────────────────────────────────────────


def _scan_shard(task):
    idx, path, has_domain, do_tok = task
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
    tok_lens = _tokenize_lens(texts) if do_tok else None
    return lens, ents, reps, divs, dom, tok_lens


# ── packing simulation (best-fit, mirrors nanochat dataloader) ────


def _packing_boundary_dist(tok_lens, seq_len=2048, buffer_size=2000, max_docs=20000):
    """Simulate best-fit packing into 2k-token rows; return boundaries/row array.

    boundaries = docs_in_row - 1 (0 = single doc fills/exceeds the row).
    Mirrors simulate_dataloader.py:117-150 (best-fit decreasing, overflow
    truncates the shortest). Samples up to max_docs (strided for representativity;
    steady-state boundary density is order-invariant under best-fit).
    """
    if tok_lens is None or len(tok_lens) == 0:
        return np.array([], dtype=np.int32)
    a = np.asarray(tok_lens)
    if len(a) > max_docs:
        step = max(1, len(a) // max_docs)
        a = a[::step][:max_docs]
    row_capacity = seq_len + 1
    buf = []
    boundaries = []
    i = 0
    N = len(a)
    while i < N or buf:
        pos = 0
        docs_in_row = 0
        while pos < row_capacity and (i < N or buf):
            while len(buf) < buffer_size and i < N:
                buf.append(int(a[i])); i += 1
            if not buf:
                break
            remaining = row_capacity - pos
            best_idx, best_len = -1, 0
            for k in range(len(buf)):
                dl = buf[k]
                if dl <= remaining and dl > best_len:
                    best_idx = k
                    best_len = dl
            if best_idx >= 0:
                buf.pop(best_idx)
                pos += best_len
            else:
                sk = min(range(len(buf)), key=lambda x: buf[x])
                buf.pop(sk)
                pos += remaining
            docs_in_row += 1
        boundaries.append(docs_in_row - 1)
    return np.array(boundaries, dtype=np.int32)


# ── figures ──────────────────────────────────────────────────────


def _fig_hist(per_arm, key, xlabel, fname, out_dir, logx=False, vline=None):
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
    if vline is not None:
        ax.axvline(vline, color="red", ls="--", lw=1.2, label=f"2k window ({vline})")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(xlabel)
    ax.legend(fontsize=9)
    _save_fig(fig, out_dir, fname)


def _fig_int_hist(per_arm, key, xlabel, fname, out_dir, xmax=None):
    """Bar histogram for small-integer distributions (e.g. boundaries/row)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pooled = np.concatenate(
        [a[key] for a in per_arm.values() if len(a[key])]
    ) if per_arm else np.array([])
    if pooled.size == 0:
        plt.close(fig)
        print(f"  [skip] no data for {fname}")
        return
    hi = int(pooled.max()) if xmax is None else xmax
    bins = np.arange(0, hi + 2) - 0.5
    for i, (label, arm) in enumerate(per_arm.items()):
        v = arm[key]
        if v.size == 0:
            continue
        vc = np.bincount(v.astype(np.int64), minlength=hi + 1).astype(float)
        vc = vc / max(1, vc.sum())
        ax.bar(
            np.arange(hi + 1) + (i - (len(per_arm) - 1) / 2) * 0.8 / len(per_arm),
            vc, 0.8 / len(per_arm), label=label, color=_COLORS[i % len(_COLORS)],
        )
    ax.set_xticks(range(hi + 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of rows")
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


def _write_txt(per_arm, stats, seq_len, out_dir):
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
        tl = a.get("tok_len")
        if tl is not None and len(tl):
            over = float(np.mean(tl > seq_len) * 100)
            lines.append(
                f"  {'tok_len':14s} mean={tl.mean():.1f}  median={np.median(tl):.1f}  "
                f"p25={np.percentile(tl, 25):.1f}  p75={np.percentile(tl, 75):.1f}  "
                f"%>{seq_len}={over:.1f}"
            )
        bd = a.get("boundaries")
        if bd is not None and len(bd):
            lines.append(
                f"  {'bnd/row':14s} mean={bd.mean():.3f}  median={np.median(bd):.0f}  "
                f"%0bnd(rows=1doc)={float(np.mean(bd == 0) * 100):.1f}  "
                f"%<=1bnd={float(np.mean(bd <= 1) * 100):.1f}"
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
    p.add_argument("--output-dir", default=None, help="Where to write outputs (default: --result-dir)")
    p.add_argument("--num-workers", type=int, default=None, help="multiprocessing workers")
    p.add_argument(
        "--tokenizer", default=None,
        help="nanochat tokenizer.pkl or its dir (enables token-length & boundary slices)",
    )
    p.add_argument("--seq-len", type=int, default=2048, help="training context length (tokens)")
    p.add_argument("--tokenizer-threads", type=int, default=1, help="threads per tokenizing worker")
    p.add_argument("--pack-buffer", type=int, default=2000, help="dataloader packing buffer size")
    p.add_argument("--max-pack-docs", type=int, default=20000, help="docs sampled for boundary sim")
    return p.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    out_dir = args.output_dir or str(result_dir)
    os.makedirs(out_dir, exist_ok=True)
    num_workers = args.num_workers or min(32, os.cpu_count() or 1)
    do_tok = bool(args.tokenizer)

    data_root = result_dir / "data"
    if not data_root.is_dir():
        sys.exit(f"error: {data_root} not found (need data/<arm>_data/ subdirs)")

    stats_path = data_root / "dataset_stats.json"
    if not stats_path.is_file():
        stats_path = result_dir / "dataset_stats.json"
    stats = json.load(open(stats_path)) if stats_path.is_file() else {}

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
        print(f"  arm '{label}': {len(train)} train shards, domain_col={has_domain}, tokenize={do_tok}")

    if not arms:
        sys.exit(f"error: no arms discovered under {data_root}")

    from multiprocessing import Pool
    from tqdm import tqdm

    per_arm = {}
    for label, info in arms.items():
        tasks = [(i, str(s), info["has_domain"], do_tok) for i, s in enumerate(info["shards"])]
        L, E, R, D, TOK = [], [], [], [], []
        dom = Counter()
        with Pool(num_workers, initializer=_init_tok_worker,
                  initargs=(args.tokenizer, args.tokenizer_threads)) as pool:
            for lens, ents, reps, divs, d, tl in tqdm(
                pool.imap_unordered(_scan_shard, tasks, chunksize=1),
                total=len(tasks), desc=f"  {label}", leave=False,
            ):
                L.append(lens); E.append(ents); R.append(reps); D.append(divs)
                if tl is not None:
                    TOK.append(tl)
                if d:
                    dom.update(d)
        pa = {
            "len": np.concatenate(L) if L else np.array([], dtype=np.int64),
            "ent": np.concatenate(E) if E else np.array([], dtype=np.float32),
            "rep": np.concatenate(R) if R else np.array([], dtype=np.float32),
            "div": np.concatenate(D) if D else np.array([], dtype=np.float32),
            "domain": dom,
            "has_domain": info["has_domain"],
        }
        if TOK:
            tl_all = np.concatenate(TOK)
            pa["tok_len"] = tl_all
            pa["boundaries"] = _packing_boundary_dist(
                tl_all, args.seq_len, args.pack_buffer, args.max_pack_docs
            )
        else:
            pa["tok_len"] = None
            pa["boundaries"] = None
        per_arm[label] = pa

    print("\n=== Generating figures ===")
    _setup_style()
    _fig_hist(per_arm, "len", "Document length (chars)", "fig_arm_length.png", out_dir, logx=True)
    _fig_hist(per_arm, "ent", "Char entropy (bits)", "fig_arm_entropy.png", out_dir)
    _fig_hist(per_arm, "rep", "Single-char repetition fraction", "fig_arm_repetition.png", out_dir)
    _fig_hist(per_arm, "div", "Lexical diversity (type/token)", "fig_arm_diversity.png", out_dir)
    # token-level slices (need tokenizer)
    if do_tok and any(per_arm[a].get("tok_len") is not None for a in per_arm):
        _fig_hist(per_arm, "tok_len", "Document length (tokens)", "fig_arm_token_length.png",
                  out_dir, logx=True, vline=args.seq_len)
        # boundary histogram: cap x-axis at a sensible max
        bmax = max((int(per_arm[a]["boundaries"].max()) for a in per_arm
                    if per_arm[a].get("boundaries") is not None and len(per_arm[a]["boundaries"])),
                   default=5)
        _fig_int_hist(per_arm, "boundaries", "Doc boundaries per 2k row",
                      "fig_arm_boundaries.png", out_dir, xmax=min(bmax, 8))
    else:
        print("  [skip] fig_arm_token_length & fig_arm_boundaries: pass --tokenizer to enable")
    _fig_domain(per_arm, out_dir)
    _write_txt(per_arm, stats, args.seq_len, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
