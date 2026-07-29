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
  quality_length_corr.txt   — Spearman(quality_signal, char_count) per arm.
                              Signals: raw quality_cols for every arm that
                              preserves them + the merged quality_rank (0=best)
                              for the quadmix_sampled arm only (prepare_data
                              drops quality_rank from the regular quadmix arm;
                              pass --sampled-parquet to recover it).
  fig_quality_length_corr.png — heatmap of the above Spearman matrix
  fig_arm_<quality>.png       — per-arm quality-signal histograms

Text-based slices run on existing text-only parquets now; the token slices need
--tokenizer (nanochat tokenizer.pkl) and the domain slice needs the "domain"
label column (prepare_data preserves it when the source schema has domain_col).

Usage:
  python scripts/analysis/analyze_arm_data.py \
      --result-dir nanochat_mid_compare/results_stem/<timestamp> \
      --tokenizer /home/ma-user/work/nanochat_model_dir/tokenizer \
      --sampled-parquet <search_output_dir>/sampled_dataset.parquet
"""

import argparse
import hashlib
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
    idx, path, has_domain, do_tok, quality_cols = task
    schema_names = set(pq.read_schema(path).names)
    cols = ["text"]
    if has_domain and "domain" in schema_names:
        cols.append("domain")
    extra_cols = []
    for c in ["char_count", "token_count", "quality_rank"] + (quality_cols or []):
        if c and c in schema_names and c not in cols:
            cols.append(c)
            extra_cols.append(c)
    cols = list(dict.fromkeys(cols))
    table = pq.read_table(path, columns=cols)
    texts = table["text"].to_pylist()
    n = len(texts)
    lens = np.empty(n, dtype=np.int64)
    ents = np.empty(n, dtype=np.float32)
    reps = np.empty(n, dtype=np.float32)
    divs = np.empty(n, dtype=np.float32)
    for i, t in enumerate(texts):
        lens[i], ents[i], reps[i], divs[i] = _doc_metrics(t or "")
    dom = None
    dom_labels = None
    if has_domain and "domain" in table.column_names:
        dom_labels = table["domain"].to_pylist()
        dom = Counter(dom_labels)
    tok_lens = _tokenize_lens(texts) if do_tok else None

    text_hashes = [hashlib.sha1((t or "").encode("utf-8")).hexdigest() for t in texts]

    result = {
        "lens": lens, "ents": ents, "reps": reps, "divs": divs,
        "dom": dom, "tok_lens": tok_lens,
        "dom_labels": dom_labels,
        "text_hashes": text_hashes,
    }
    for c in extra_cols:
        if c in table.column_names:
            arr = table[c].to_pylist()
            if c in ("char_count", "token_count"):
                result[c] = np.array(arr, dtype=np.int64)
            else:
                result[c] = np.array(arr, dtype=np.float64)
    return result


def _scan_sampled_arm(parquet_path, quality_cols, domain_names):
    """Scan a QuadMix sampled_dataset.parquet as an extra 'quadmix_sampled' arm.

    sampled_dataset.parquet (batch_sampler.save_sampled_dataset) carries
    char_count / quality_rank (merged, 0=best) / raw quality_cols / <domain_col>
    (int labels). prepare_data drops quality_rank when building arm parquets, so
    this is the only arm that can show rank<->length. Domain column name is the
    schema's domain_col (category_name for STEM); falls back to 'domain'.

    Text-free: reads only the needed columns (no text, no tokenizer, no packing
    sim) -> fast & low-memory. Length is taken from char_count (== len(text) as
    written by save_sampled_dataset). ent/rep/div are left empty so the
    text-only histograms simply skip this arm (they are redundant with the
    quadmix arm, which is the same data pre-packing).
    """
    schema_names = set(pq.read_schema(parquet_path).names)
    domain_col = None
    for c in ("category_name", "domain"):
        if c in schema_names:
            domain_col = c
            break

    want = []
    for c in ["char_count", "quality_rank"] + list(quality_cols or []):
        if c in schema_names and c not in want:
            want.append(c)
    if domain_col and domain_col not in want:
        want.append(domain_col)
    table = pq.read_table(parquet_path, columns=want)

    char_count = np.asarray(table["char_count"].to_numpy(), dtype=np.int64) \
        if "char_count" in table.column_names else np.array([], dtype=np.int64)

    dom_labels = None
    dom = Counter()
    if domain_col and domain_col in table.column_names:
        dom_labels = np.asarray(table[domain_col].to_numpy())
        dom = Counter(dom_labels.tolist())
        if domain_names:
            dom = _normalize_domain(dom, domain_names)

    pa = {
        "len": char_count,
        "ent": np.array([], dtype=np.float32),
        "rep": np.array([], dtype=np.float32),
        "div": np.array([], dtype=np.float32),
        "domain": dom,
        "has_domain": domain_col is not None,
        "dom_labels": dom_labels,
        "text_hashes": [],
        "tok_len": None,
        "boundaries": None,
        "char_count": char_count,
    }
    if "quality_rank" in table.column_names:
        pa["quality_rank"] = np.asarray(table["quality_rank"].to_numpy(), dtype=np.float64)
    for c in (quality_cols or []):
        if c in table.column_names:
            pa[c] = np.asarray(table[c].to_numpy(), dtype=np.float64)
    return pa


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
        vc = np.bincount(np.clip(v.astype(np.int64), 0, hi), minlength=hi + 1).astype(float)
        vc = vc / max(1, vc.sum())
        ax.bar(
            np.arange(hi + 1) + (i - (len(per_arm) - 1) / 2) * 0.8 / len(per_arm),
            vc, 0.8 / len(per_arm), label=label, color=_COLORS[i % len(_COLORS)],
        )
    ax.set_xticks(range(hi + 1))
    if xmax is not None:
        ax.set_xticklabels([str(i) if i < hi else f"{hi}+" for i in range(hi + 1)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of rows")
    ax.set_title(xlabel)
    ax.legend(fontsize=9)
    _save_fig(fig, out_dir, fname)


def _normalize_domain(dom, domain_names):
    """Map integer domain codes (from quadmix sampled parquet) to names.

    The quadmix pipeline's metadata_manager stores domain as int codes via
    pd.CategoricalDtype(categories=domain_names), so code i = domain_names[i].
    Manual/random arms already carry string names from source parquets.
    """
    if not domain_names or not dom:
        return dom
    out = Counter()
    for k, v in dom.items():
        if isinstance(k, bool):
            out[k] += v
        elif isinstance(k, (int, np.integer)):
            idx = int(k)
            if 0 <= idx < len(domain_names):
                out[domain_names[idx]] += v
            else:
                out[f"D{idx}"] += v
        else:
            out[k] += v
    return out


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


# ── length by domain ──────────────────────────────────────────────


def _fig_length_by_domain(per_arm, domain_names, out_dir):
    arms_with = {l: a for l, a in per_arm.items()
                 if a.get("dom_labels") is not None and len(a["dom_labels"])}
    if not arms_with:
        print("  [skip] fig_length_by_domain: no per-doc domain labels")
        return
    all_doms = set()
    for a in arms_with.values():
        all_doms.update(a["dom_labels"])
    domains = sorted(all_doms, key=str)
    n_arms = len(arms_with)
    fig, axes = plt.subplots(1, n_arms, figsize=(5 * n_arms, 5), squeeze=False)
    for i, (label, arm) in enumerate(arms_with.items()):
        ax = axes[0][i]
        data = []
        labels = []
        for d in domains:
            mask = arm["dom_labels"] == d
            vals = arm["len"][mask]
            if len(vals):
                data.append(np.log10(np.maximum(vals, 1)))
                labels.append(str(d))
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"{label}")
        ax.set_ylabel("log10(char length)")
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle("Document length by domain", fontsize=13)
    _save_fig(fig, out_dir, "fig_length_by_domain.png")


# ── quality-length correlation ───────────────────────────────────


def _spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx_m = rx - rx.mean()
    ry_m = ry - ry.mean()
    den = np.sqrt(np.sum(rx_m ** 2) * np.sum(ry_m ** 2))
    return float(np.sum(rx_m * ry_m) / den) if den > 0 else 0.0


def _quality_length_correlation(per_arm, quality_cols, out_dir):
    length_key = "char_count"
    if not any(length_key in per_arm[a] for a in per_arm):
        length_key = "len"
    signal_keys = list(quality_cols or [])
    if "quality_rank" not in signal_keys:
        for a in per_arm:
            if "quality_rank" in per_arm[a]:
                signal_keys.append("quality_rank")
                break
    if not signal_keys:
        print("  [skip] quality_length_correlation: no quality signal columns")
        return
    arms = [a for a in per_arm if length_key in per_arm[a] and len(per_arm[a][length_key])]
    if not arms:
        print("  [skip] quality_length_correlation: no length data")
        return

    lines = ["=== Quality–Length Spearman Correlation ===", ""]
    header = f"  {'arm':>16s}"
    for sk in signal_keys:
        header += f"  {sk:>18s}"
    lines.append(header)
    lines.append("  " + "-" * (16 + 20 * len(signal_keys)))

    matrix = []
    for arm_label in arms:
        a = per_arm[arm_label]
        length = a[length_key]
        row = []
        row_str = f"  {arm_label:>16s}"
        for sk in signal_keys:
            if sk in a and len(a[sk]) == len(length):
                rho = _spearman(a[sk], length)
                row.append(rho)
                row_str += f"  {rho:>18.4f}"
            else:
                row.append(None)
                row_str += f"  {'N/A':>18s}"
        matrix.append(row)
        lines.append(row_str)
    lines.append("")

    path = os.path.join(out_dir, "quality_length_corr.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [Text] Saved: {path}")

    has_data = any(any(v is not None for v in row) for row in matrix)
    if has_data:
        n_arms = len(arms)
        n_sigs = len(signal_keys)
        fig, ax = plt.subplots(figsize=(max(6, 2 * n_sigs), max(3, 0.6 * n_arms)))
        data = np.full((n_arms, n_sigs), np.nan)
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                if v is not None:
                    data[i, j] = v
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(n_sigs))
        ax.set_xticklabels(signal_keys, rotation=30, ha='right')
        ax.set_yticks(range(n_arms))
        ax.set_yticklabels(arms)
        ax.set_title("Spearman(signal, length)")
        fig.colorbar(im, ax=ax, label="Spearman ρ")
        for i in range(n_arms):
            for j in range(n_sigs):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center', fontsize=8)
        _save_fig(fig, out_dir, "fig_quality_length_corr.png")


# ── quality signal histograms ────────────────────────────────────


def _fig_quality_hist(per_arm, quality_cols, out_dir):
    signal_keys = list(quality_cols or [])
    for a in per_arm:
        if "quality_rank" in per_arm[a] and "quality_rank" not in signal_keys:
            signal_keys.append("quality_rank")
            break
    for sk in signal_keys:
        arms_with = {l: a for l, a in per_arm.items() if sk in a and len(a[sk])}
        if not arms_with:
            continue
        pooled = np.concatenate([a[sk] for a in arms_with.values()])
        if pooled.size == 0:
            continue
        lo, hi = np.percentile(pooled, [1, 99])
        bins = np.linspace(lo, hi, 50)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (label, arm) in enumerate(arms_with.items()):
            v = arm[sk]
            v = v[(v >= lo) & (v <= hi)]
            if v.size == 0:
                continue
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6,
                    label=f"{label} (n={len(arm[sk]):,})",
                    color=_COLORS[i % len(_COLORS)])
        ax.set_xlabel(sk)
        ax.set_ylabel("density")
        ax.set_title(sk)
        ax.legend(fontsize=9)
        _save_fig(fig, out_dir, f"fig_arm_{sk}.png")


# ── duplicate detection (report only, no dedup) ───────────────────


def _detect_duplicates(per_arm, domain_names, out_dir):
    lines = ["=== Duplicate Detection (report only) ===", ""]
    for label, a in per_arm.items():
        hashes = a.get("text_hashes")
        if not hashes:
            lines.append(f"[{label}] no text hashes available")
            lines.append("")
            continue
        n_total = len(hashes)
        counts = Counter(hashes)
        n_unique = len(counts)
        n_dup_docs = n_total - n_unique
        dup_mult = Counter(counts.values())
        lines.append(f"[{label}]  total={n_total:,}  unique={n_unique:,}  "
                     f"dup_docs={n_dup_docs:,} ({n_dup_docs / max(1, n_total) * 100:.1f}%)")
        for mult, cnt in sorted(dup_mult.items()):
            if mult > 1:
                lines.append(f"  {mult}x: {cnt:,} docs "
                             f"({cnt * mult:,} rows, {cnt * (mult - 1):,} extra)")
        lengths = a.get("len", np.array([]))
        if len(lengths) == n_total:
            hash_to_len = {}
            for i, h in enumerate(hashes):
                hash_to_len.setdefault(h, []).append(int(lengths[i]))
            dup_chars = sum(lens[0] * (len(lens) - 1)
                            for lens in hash_to_len.values() if len(lens) > 1)
            total_chars = int(lengths.sum())
            if total_chars > 0:
                lines.append(f"  Duplicate char fraction: {dup_chars / total_chars * 100:.1f}%")
        dom_labels = a.get("dom_labels")
        if dom_labels is not None and len(dom_labels) == n_total and len(lengths) == n_total:
            lines.append("  Per-domain duplicate docs:")
            domains = sorted(set(str(d) for d in dom_labels))
            for d in domains:
                d_mask = np.array([str(dl) == d for dl in dom_labels])
                d_hashes = [hashes[i] for i in range(n_total) if d_mask[i]]
                d_counts = Counter(d_hashes)
                d_dup = sum(1 for v in d_counts.values() if v > 1)
                d_total = len(d_hashes)
                if d_total > 0:
                    lines.append(f"    {d:>12s}: {d_dup:,}/{d_total:,} "
                                 f"({d_dup / d_total * 100:.1f}% unique-doc dup)")
        lines.append("")
    path = os.path.join(out_dir, "duplicate_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [Text] Saved: {path}")


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
        dom_labels = a.get("dom_labels")
        if dom_labels is not None and len(dom_labels) and len(a["len"]):
            all_doms = sorted(set(str(d) for d in dom_labels))
            lines.append("  per-domain length (chars):")
            for d in all_doms:
                mask = np.array([str(dl) == d for dl in dom_labels])
                dv = a["len"][mask]
                if len(dv):
                    lines.append(
                        f"    {d:>12s}  n={len(dv):>8,}  mean={dv.mean():>8.0f}  "
                        f"median={np.median(dv):>8.0f}  p25={np.percentile(dv, 25):>8.0f}  "
                        f"p75={np.percentile(dv, 75):>8.0f}"
                    )
        for qk in ("char_count", "token_count"):
            qv = a.get(qk)
            if qv is not None and len(qv):
                lines.append(
                    f"  {qk:14s} mean={qv.mean():.1f}  median={np.median(qv):.1f}  "
                    f"p25={np.percentile(qv, 25):.1f}  p75={np.percentile(qv, 75):.1f}"
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
        "--sampled-parquet", default=None,
        help="path to a QuadMix sampled_dataset.parquet (search output); scanned as an "
             "extra 'quadmix_sampled' arm. It is the only arm carrying the merged "
             "quality_rank (0=best) that prepare_data drops, so it enables the "
             "quality_rank<->length Spearman in the quality-length table.",
    )
    p.add_argument("--output-dir", default=None, help="Where to write outputs (default: --result-dir)")
    p.add_argument("--num-workers", type=int, default=None, help="multiprocessing workers")
    p.add_argument(
        "--tokenizer", default=None,
        help="nanochat tokenizer.pkl or its dir (enables token-length & boundary slices; "
             "default: $NANOCHAT_MODEL_DIR/tokenizer)",
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
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        model_dir = os.environ.get(
            "NANOCHAT_MODEL_DIR", "/home/ma-user/work/nanochat_model_dir")
        default_tok = os.path.join(model_dir, "tokenizer")
        default_pkl = os.path.join(default_tok, "tokenizer.pkl")
        if os.path.isfile(default_pkl):
            tokenizer_path = default_tok
            print(f"  [info] tokenizer auto-detected: {default_pkl}")
    do_tok = bool(tokenizer_path)

    data_root = result_dir / "data"
    if not data_root.is_dir():
        sys.exit(f"error: {data_root} not found (need data/<arm>_data/ subdirs)")

    stats_path = data_root / "dataset_stats.json"
    if not stats_path.is_file():
        stats_path = result_dir / "dataset_stats.json"
    stats = json.load(open(stats_path)) if stats_path.is_file() else {}
    domain_names = (stats.get("config") or {}).get("domain_names")
    quality_cols = (stats.get("config") or {}).get("quality_cols", [])

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
        tasks = [(i, str(s), info["has_domain"], do_tok, quality_cols)
                 for i, s in enumerate(info["shards"])]
        L, E, R, D, TOK = [], [], [], [], []
        dom = Counter()
        dom_labels_all = []
        text_hashes_all = []
        extra_arrays = {}
        with Pool(num_workers, initializer=_init_tok_worker,
                  initargs=(tokenizer_path, args.tokenizer_threads)) as pool:
            for result in tqdm(
                pool.imap_unordered(_scan_shard, tasks, chunksize=1),
                total=len(tasks), desc=f"  {label}", leave=False,
            ):
                L.append(result["lens"]); E.append(result["ents"])
                R.append(result["reps"]); D.append(result["divs"])
                if result["tok_lens"] is not None:
                    TOK.append(result["tok_lens"])
                if result["dom"]:
                    dom.update(result["dom"])
                if result["dom_labels"] is not None:
                    dom_labels_all.extend(result["dom_labels"])
                if result["text_hashes"]:
                    text_hashes_all.extend(result["text_hashes"])
                for key, val in result.items():
                    if key in ("lens", "ents", "reps", "divs", "dom", "tok_lens",
                               "dom_labels", "text_hashes"):
                        continue
                    if isinstance(val, np.ndarray):
                        extra_arrays.setdefault(key, []).append(val)
        if domain_names:
            dom = _normalize_domain(dom, domain_names)
        pa = {
            "len": np.concatenate(L) if L else np.array([], dtype=np.int64),
            "ent": np.concatenate(E) if E else np.array([], dtype=np.float32),
            "rep": np.concatenate(R) if R else np.array([], dtype=np.float32),
            "div": np.concatenate(D) if D else np.array([], dtype=np.float32),
            "domain": dom,
            "has_domain": info["has_domain"],
            "dom_labels": np.array(dom_labels_all) if dom_labels_all else None,
            "text_hashes": text_hashes_all,
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
        for key, arrs in extra_arrays.items():
            if arrs:
                pa[key] = np.concatenate(arrs)
        per_arm[label] = pa

    if args.sampled_parquet:
        sp = args.sampled_parquet
        if os.path.isfile(sp):
            print(f"\n  arm 'quadmix_sampled': scanning {sp}")
            per_arm["quadmix_sampled"] = _scan_sampled_arm(sp, quality_cols, domain_names)
            print(f"    n_docs={len(per_arm['quadmix_sampled']['len']):,}  "
                  f"has quality_rank={'quality_rank' in per_arm['quadmix_sampled']}")
        else:
            print(f"  [warn] --sampled-parquet not found: {sp}")

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
    _fig_length_by_domain(per_arm, domain_names, out_dir)
    _quality_length_correlation(per_arm, quality_cols, out_dir)
    _fig_quality_hist(per_arm, quality_cols, out_dir)
    _detect_duplicates(per_arm, domain_names, out_dir)
    _write_txt(per_arm, stats, args.seq_len, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
