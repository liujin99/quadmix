#!/usr/bin/env python3
"""
Analyze Essential-Web v1.0 data distribution:
  1. FDC L1 / L2 label distribution
  2. FDC code prefix distribution (for L2 mapping design)
  3. Quality signal distribution per domain
  4. Matplotlib visualization output

Usage:
  python scripts/analysis/analyze_essential_web_distribution.py \
      --input-dir /home/ma-user/work/QuaDMix/data/essential-web \
      --output-dir results/essential_web_analysis \
      --limit 100
"""

import argparse, json, os, sys, glob, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from quadmix.constants import DOMAIN_MAP, DOMAIN_NAMES, FASTTEXT_FIELDS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

L1_SHORT = {
    "Industrial arts, Technology, and Engineering": "Industrial",
    "Social sciences": "Social Sci",
    "Science and Natural history": "Science",
    "Religion": "Religion",
    "Philology; or, Language and languages": "Philology",
    "Literature": "Literature",
    "History and Geography": "History",
    "General works, books and libraries, information sciences": "General",
    "Philosophy and psychology": "Philosophy",
    "Arts": "Arts",
}

QUALITY_SHORT = ["DCLM", "Edu", "Eng", "MathG", "MathO"]
QUALITY_COLORS = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"]
COLOR_L1 = "#5B9BD5"
COLOR_L2 = "#ED7D31"
COLOR_GRID = "#D0D0D0"

L1_COLORS = [
    "#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5",
    "#70AD47", "#264478", "#9B59B6", "#E74C3C", "#1ABC9C",
]

HIST_BINS = np.linspace(0, 1, 51)


def _setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def _save_fig(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  [Figure] Saved: {path}")
    return path


def extract_fdc_info(eai_taxonomy):
    if isinstance(eai_taxonomy, str):
        try:
            eai_taxonomy = json.loads(eai_taxonomy)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(eai_taxonomy, dict):
        return None
    fdc = eai_taxonomy.get("free_decimal_correspondence", {})
    if not isinstance(fdc, dict):
        return None
    primary = fdc.get("primary", {})
    if not isinstance(primary, dict):
        return None
    code = primary.get("code", "")
    labels = primary.get("labels", {})
    if not isinstance(labels, dict):
        labels = {}
    level_1 = labels.get("level_1", "") or ""
    level_2 = labels.get("level_2", "") or ""
    prefix_2 = ""
    if isinstance(code, str) and len(code) >= 2:
        prefix_2 = code[:2]
    return {
        "code": code if isinstance(code, str) else "",
        "prefix_2": prefix_2,
        "level_1": level_1,
        "level_2": level_2,
    }


def extract_quality(quality_signals):
    if not isinstance(quality_signals, dict):
        return [0.0] * len(FASTTEXT_FIELDS)
    fasttext = quality_signals.get("fasttext", {})
    if not isinstance(fasttext, dict):
        return [0.0] * len(FASTTEXT_FIELDS)
    return [(fasttext.get(f, 0.0) or 0.0) for f in FASTTEXT_FIELDS]


def process_shard(shard_path):
    try:
        df = pd.read_parquet(shard_path, columns=["eai_taxonomy", "quality_signals"])
    except Exception as e:
        print(f"  ERROR reading {shard_path}: {e}")
        return None

    n = len(df)
    fdc_infos = df["eai_taxonomy"].apply(extract_fdc_info)
    valid_mask = fdc_infos.notna()
    valid_infos = fdc_infos[valid_mask].tolist()

    l1_counter = Counter()
    l2_counter = Counter()
    prefix_counter = Counter()
    l1_l2_pairs = Counter()
    l2_empty_by_l1 = Counter()
    l2_has_by_l1 = Counter()

    for info in valid_infos:
        l1 = info["level_1"]
        l2 = info["level_2"]
        pfx = info["prefix_2"]
        if l1:
            l1_counter[l1] += 1
            if l2:
                l2_counter[l2] += 1
                l1_l2_pairs[(l1, l2)] += 1
                l2_has_by_l1[l1] += 1
            else:
                l2_empty_by_l1[l1] += 1
        if pfx:
            prefix_counter[pfx] += 1

    quality = df["quality_signals"].apply(extract_quality)
    quality_matrix = np.stack(quality.to_numpy())

    quality_hists = []
    for qi in range(len(FASTTEXT_FIELDS)):
        hist, _ = np.histogram(quality_matrix[:, qi], bins=HIST_BINS)
        quality_hists.append(hist.tolist())

    quality_by_l1_hists = {}
    for l1_name in set(info["level_1"] for info in valid_infos if info and info["level_1"]):
        mask = np.array([
            (fdc_infos.iloc[i] is not None and fdc_infos.iloc[i]["level_1"] == l1_name)
            for i in range(n)
        ])
        if mask.sum() == 0:
            continue
        sub = quality_matrix[mask]
        hists = []
        for qi in range(len(FASTTEXT_FIELDS)):
            h, _ = np.histogram(sub[:, qi], bins=HIST_BINS)
            hists.append(h.tolist())
        quality_by_l1_hists[l1_name] = hists

    return {
        "n_docs": n,
        "n_valid": len(valid_infos),
        "l1_counter": l1_counter,
        "l2_counter": l2_counter,
        "prefix_counter": prefix_counter,
        "l1_l2_pairs": l1_l2_pairs,
        "l2_empty_by_l1": l2_empty_by_l1,
        "l2_has_by_l1": l2_has_by_l1,
        "quality_hists": quality_hists,
        "quality_by_l1_hists": quality_by_l1_hists,
    }


def merge_results(results):
    total_docs = 0
    total_valid = 0
    l1_counter = Counter()
    l2_counter = Counter()
    prefix_counter = Counter()
    l1_l2_pairs = Counter()
    l2_empty_by_l1 = Counter()
    l2_has_by_l1 = Counter()
    n_bins = len(HIST_BINS) - 1
    quality_hists = [np.zeros(n_bins) for _ in range(len(FASTTEXT_FIELDS))]
    quality_by_l1_hists = defaultdict(lambda: [np.zeros(n_bins) for _ in range(len(FASTTEXT_FIELDS))])

    for r in results:
        if r is None:
            continue
        total_docs += r["n_docs"]
        total_valid += r["n_valid"]
        l1_counter.update(r["l1_counter"])
        l2_counter.update(r["l2_counter"])
        prefix_counter.update(r["prefix_counter"])
        l1_l2_pairs.update(r["l1_l2_pairs"])
        l2_empty_by_l1.update(r["l2_empty_by_l1"])
        l2_has_by_l1.update(r["l2_has_by_l1"])
        for qi in range(len(FASTTEXT_FIELDS)):
            quality_hists[qi] += np.array(r["quality_hists"][qi])
        for l1_name, hists in r["quality_by_l1_hists"].items():
            for qi in range(len(FASTTEXT_FIELDS)):
                quality_by_l1_hists[l1_name][qi] += np.array(hists[qi])

    total_l2_empty = sum(l2_empty_by_l1.values())
    total_l2_has = sum(l2_has_by_l1.values())

    return {
        "total_docs": total_docs,
        "total_valid": total_valid,
        "total_l2_empty": total_l2_empty,
        "total_l2_has": total_l2_has,
        "l1_counter": l1_counter,
        "l2_counter": l2_counter,
        "prefix_counter": prefix_counter,
        "l1_l2_pairs": l1_l2_pairs,
        "l2_empty_by_l1": l2_empty_by_l1,
        "l2_has_by_l1": l2_has_by_l1,
        "quality_hists": quality_hists,
        "quality_by_l1_hists": dict(quality_by_l1_hists),
    }


# ── Text output ──

def print_separator(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_distribution(counter, title, total, top_n=None):
    print_separator(title)
    items = counter.most_common(top_n)
    print(f"  {'Label':<60} {'Count':>10} {'%':>7}")
    print(f"  {'-' * 60} {'-' * 10} {'-' * 7}")
    for label, count in items:
        pct = count / max(total, 1) * 100
        display = label if len(label) <= 58 else label[:55] + "..."
        print(f"  {display:<60} {count:>10,} {pct:>6.2f}%")
    print(f"\n  Total unique labels: {len(counter)}")
    print(f"  Total count: {sum(counter.values()):,}")


def print_l1_l2_tree(l1_counter, l1_l2_pairs):
    print_separator("FDC L1 -> L2 Tree")
    for l1, l1_count in l1_counter.most_common():
        print(f"\n  [{l1}] ({l1_count:,} docs)")
        pairs = [(k, v) for k, v in l1_l2_pairs.items() if k[0] == l1]
        pairs.sort(key=lambda x: -x[1])
        if not pairs:
            print(f"    (no L2 labels)")
            continue
        for (_, l2_name), count in pairs:
            pct = count / l1_count * 100
            display = l2_name if len(l2_name) <= 50 else l2_name[:47] + "..."
            print(f"    |- {display:<50} {count:>8,} ({pct:>5.1f}%)")
        l2_total = sum(v for _, v in pairs)
        no_l2 = l1_count - l2_total
        if no_l2 > 0:
            pct = no_l2 / l1_count * 100
            print(f"    |- {'(no L2 label)':<50} {no_l2:>8,} ({pct:>5.1f}%)")


# ── Figures ──

def make_fig1_l1_distribution(l1_counter, total_valid, output_dir):
    _setup_style()
    items = l1_counter.most_common()
    labels = [L1_SHORT.get(l1, l1[:20]) for l1, _ in items]
    counts = [c for _, c in items]
    pcts = [c / max(total_valid, 1) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, pcts, width=0.7, color=[L1_COLORS[i % len(L1_COLORS)] for i in range(len(labels))],
                  edgecolor="white", linewidth=0.5)
    ax.set_xlabel("FDC L1 Domain")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"FDC L1 Distribution ({total_valid:,} docs with valid FDC)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{pct:.1f}%\n({cnt/1e6:.1f}M)", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig1_l1_distribution.png")


def make_fig2_l2_distribution(l2_counter, total_valid, output_dir, top_n=30):
    _setup_style()
    items = l2_counter.most_common(top_n)
    if not items:
        return None
    labels = [l for l, _ in reversed(items)]
    counts = [c for _, c in reversed(items)]
    pcts = [c / max(total_valid, 1) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.3)))
    y = np.arange(len(labels))
    bars = ax.barh(y, pcts, height=0.7, color=COLOR_L2, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Percentage (%)")
    ax.set_title(f"FDC L2 Distribution (Top {top_n}, {len(l2_counter)} unique labels)")
    ax.set_yticks(y)
    ax.set_yticklabels([l[:45] + "..." if len(l) > 45 else l for l in labels], fontsize=8)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.2f}% ({cnt:,})", ha="left", va="center", fontsize=7)

    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig2_l2_distribution_top30.png")


def make_fig3_prefix_by_l1(prefix_counter, l1_counter, l1_l2_pairs, output_dir):
    _setup_style()

    l1_order = [l1 for l1, _ in l1_counter.most_common()]
    all_prefixes = sorted(prefix_counter.keys())

    prefix_by_l1 = defaultdict(Counter)
    for (l1, l2), count in l1_l2_pairs.items():
        for pfx in all_prefixes:
            if l2 and pfx == str(int(hash(l1 + l2) % 100)).zfill(2):
                pass

    dewey_l1_map = {
        "0": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"],
        "1": ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
        "2": ["20", "21", "22", "23", "24", "25", "26", "27", "28", "29"],
        "3": ["30", "31", "32", "33", "34", "35", "36", "37", "38", "39"],
        "4": ["40", "41", "42", "43", "44", "45", "46", "47", "48", "49"],
        "5": ["50", "51", "52", "53", "54", "55", "56", "57", "58", "59"],
        "6": ["60", "61", "62", "63", "64", "65", "66", "67", "68", "69"],
        "7": ["70", "71", "72", "73", "74", "75", "76", "77", "78", "79"],
        "8": ["80", "81", "82", "83", "84", "85", "86", "87", "88", "89"],
        "9": ["90", "91", "92", "93", "94", "95", "96", "97", "98", "99"],
    }

    l1_to_dewey_prefix = {}
    for l1_name in l1_order:
        for l1_digit, prefixes in dewey_l1_map.items():
            for pfx in prefixes:
                if pfx in prefix_counter:
                    if l1_name not in l1_to_dewey_prefix:
                        l1_to_dewey_prefix[l1_name] = []
                    l1_to_dewey_prefix[l1_name].append(pfx)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, l1_name in enumerate(l1_order[:10]):
        ax = axes[idx]
        short = L1_SHORT.get(l1_name, l1_name[:15])

        prefixes_for_l1 = sorted(set(
            pfx for pfx in all_prefixes
            if pfx[0] == str(idx)
        ))

        counts = [prefix_counter.get(pfx, 0) for pfx in prefixes_for_l1]
        total_for_l1 = sum(counts) if counts else 1

        if counts and max(counts) > 0:
            pcts = [c / total_for_l1 * 100 for c in counts]
            ax.bar(range(len(prefixes_for_l1)), pcts,
                   color=L1_COLORS[idx], edgecolor="white", linewidth=0.3)
            ax.set_xticks(range(len(prefixes_for_l1)))
            ax.set_xticklabels(prefixes_for_l1, rotation=45, ha="right", fontsize=6)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")

        ax.set_title(f"{short} ({idx}xx)", fontsize=9)
        ax.set_ylabel("%" if idx % 5 == 0 else "", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7)

    plt.suptitle("FDC Code Prefix Distribution by L1 Domain (2-digit prefix)", fontsize=13, y=1.01)
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig3_prefix_by_l1.png")


def make_fig4_l2_coverage(l1_counter, l2_has_by_l1, l2_empty_by_l1, output_dir):
    _setup_style()
    items = l1_counter.most_common()
    labels = [L1_SHORT.get(l1, l1[:15]) for l1, _ in items]
    has_l2 = [l2_has_by_l1.get(l1, 0) / max(cnt, 1) * 100 for l1, cnt in items]
    no_l2 = [l2_empty_by_l1.get(l1, 0) / max(cnt, 1) * 100 for l1, cnt in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.6

    ax.bar(x, has_l2, width, label="Has L2 label", color="#70AD47", edgecolor="white", linewidth=0.5)
    ax.bar(x, no_l2, width, bottom=has_l2, label="No L2 label", color="#E74C3C", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("FDC L1 Domain")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("L2 Label Coverage per L1 Domain")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    for i, (h, n) in enumerate(zip(has_l2, no_l2)):
        if h > 5:
            ax.text(i, h / 2, f"{h:.0f}%", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        if n > 5:
            ax.text(i, h + n / 2, f"{n:.0f}%", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig4_l2_coverage.png")


def make_fig5_quality_histograms(quality_hists, output_dir):
    _setup_style()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    bin_centers = (HIST_BINS[:-1] + HIST_BINS[1:]) / 2

    for qi, ax in enumerate(axes):
        counts = quality_hists[qi]
        total = sum(counts)
        if total > 0:
            pcts = [c / total * 100 for c in counts]
        else:
            pcts = [0] * len(counts)

        ax.bar(bin_centers, pcts, width=HIST_BINS[1] - HIST_BINS[0],
               color=QUALITY_COLORS[qi], edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.set_title(QUALITY_SHORT[qi], fontsize=11)
        ax.set_xlabel("Score" if qi == 2 else "", fontsize=9)
        ax.set_ylabel("% of docs" if qi == 0 else "", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)

        nonzero = [c for c in counts if c > 0]
        if nonzero:
            mean_val = sum(b * c for b, c in zip(bin_centers, counts)) / max(total, 1)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(mean_val, ax.get_ylim()[1] * 0.9, f"μ={mean_val:.3f}",
                    fontsize=7, color="red", ha="center")

    plt.suptitle("Quality Signal Score Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig5_quality_histograms.png")


def make_fig6_quality_by_domain(quality_by_l1_hists, l1_counter, output_dir):
    _setup_style()
    l1_order = [l1 for l1, _ in l1_counter.most_common()]
    n_l1 = len(l1_order)
    n_q = len(FASTTEXT_FIELDS)

    mean_matrix = np.zeros((n_l1, n_q))
    for i, l1 in enumerate(l1_order):
        if l1 not in quality_by_l1_hists:
            continue
        hists = quality_by_l1_hists[l1]
        bin_centers = (HIST_BINS[:-1] + HIST_BINS[1:]) / 2
        for qi in range(n_q):
            total = sum(hists[qi])
            if total > 0:
                mean_matrix[i, qi] = sum(b * c for b, c in zip(bin_centers, hists[qi])) / total

    fig, ax = plt.subplots(figsize=(10, max(4, n_l1 * 0.4)))
    im = ax.imshow(mean_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(n_q))
    ax.set_xticklabels(QUALITY_SHORT, fontsize=9)
    ax.set_yticks(range(n_l1))
    ax.set_yticklabels([L1_SHORT.get(l1, l1[:20]) for l1 in l1_order], fontsize=9)
    ax.set_title("Mean Quality Score by L1 Domain")

    for i in range(n_l1):
        for j in range(n_q):
            val = mean_matrix[i, j]
            color = "white" if val > mean_matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Score", fontsize=9)
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig6_quality_by_domain.png")


def make_fig7_l1_l2_heatmap(l1_counter, l1_l2_pairs, l2_counter, output_dir, top_l2=20):
    _setup_style()
    l1_order = [l1 for l1, _ in l1_counter.most_common()]
    l2_top = [l2 for l2, _ in l2_counter.most_common(top_l2)]

    matrix = np.zeros((len(l1_order), len(l2_top)))
    for i, l1 in enumerate(l1_order):
        for j, l2 in enumerate(l2_top):
            matrix[i, j] = l1_l2_pairs.get((l1, l2), 0)

    fig, ax = plt.subplots(figsize=(max(10, len(l2_top) * 0.6), max(4, len(l1_order) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(l2_top)))
    ax.set_xticklabels([l2[:25] + "..." if len(l2) > 25 else l2 for l2 in l2_top],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(l1_order)))
    ax.set_yticklabels([L1_SHORT.get(l1, l1[:20]) for l1 in l1_order], fontsize=9)
    ax.set_title(f"L1 x L2 Co-occurrence Heatmap (Top {top_l2} L2 labels)")

    max_val = matrix.max() if matrix.max() > 0 else 1
    for i in range(len(l1_order)):
        for j in range(len(l2_top)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > max_val * 0.5 else "black"
                if val >= 1000:
                    text = f"{val/1000:.0f}K"
                else:
                    text = str(int(val))
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Doc Count", fontsize=9)
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig7_l1_l2_heatmap.png")


def make_fig8_prefix_treemap(prefix_counter, output_dir):
    _setup_style()

    l1_groups = {}
    for pfx, count in prefix_counter.items():
        l1_digit = pfx[0]
        if l1_digit not in l1_groups:
            l1_groups[l1_digit] = {}
        l1_groups[l1_digit][pfx] = count

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    l1_names = [
        "General (0xx)", "Philosophy (1xx)", "Religion (2xx)",
        "Social Sci (3xx)", "Philology (4xx)", "Science (5xx)",
        "Industrial (6xx)", "Arts (7xx)", "Literature (8xx)", "History (9xx)"
    ]

    for idx in range(10):
        ax = axes[idx]
        digit = str(idx)
        if digit in l1_groups:
            prefixes = sorted(l1_groups[digit].keys())
            counts = [l1_groups[digit][p] for p in prefixes]
            total = sum(counts)
            pcts = [c / max(total, 1) * 100 for c in counts]

            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(prefixes)))
            ax.bar(range(len(prefixes)), pcts, color=colors, edgecolor="white", linewidth=0.3)
            ax.set_xticks(range(len(prefixes)))
            ax.set_xticklabels(prefixes, rotation=45, ha="right", fontsize=6)

            for i, (pfx, pct) in enumerate(zip(prefixes, pcts)):
                if pct > 5:
                    ax.text(i, pct + 0.5, f"{pct:.0f}%", ha="center", fontsize=5)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")

        ax.set_title(l1_names[idx], fontsize=9)
        ax.set_ylabel("%" if idx % 5 == 0 else "", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7)

    plt.suptitle("FDC 2-Digit Prefix Distribution (within each L1 domain)", fontsize=13, y=1.01)
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig8_prefix_treemap.png")


def main():
    p = argparse.ArgumentParser(description="Analyze Essential-Web FDC & quality distribution")
    p.add_argument("--input-dir",
                   default="/home/ma-user/work/QuaDMix/data/essential-web",
                   help="Directory containing raw parquet shards")
    p.add_argument("--output-dir",
                   default=None,
                   help="Output directory for figures and report (default: results/essential_web_analysis)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of shards to process")
    p.add_argument("--workers", type=int, default=32,
                   help="Number of parallel workers")
    args = p.parse_args()

    if args.output_dir is None:
        from quadmix.constants import PROJECT_DIR
        args.output_dir = "/home/ma-user/work/QuaDMix/data/data_analysis"

    os.makedirs(args.output_dir, exist_ok=True)

    shard_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not shard_paths:
        print(f"Error: no parquet files found in {args.input_dir}")
        return 1

    if args.limit:
        shard_paths = shard_paths[:args.limit]

    print(f"Scanning {len(shard_paths)} shards from {args.input_dir}")
    t0 = time.time()

    results = []
    workers = min(args.workers, len(shard_paths))
    if len(shard_paths) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_shard, sp): sp for sp in shard_paths}
            done = 0
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                done += 1
                if done % 50 == 0 or done == len(shard_paths):
                    print(f"  [{done}/{len(shard_paths)}] shards processed...")
    else:
        results.append(process_shard(shard_paths[0]))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    stats = merge_results(results)

    print(f"\n{'#' * 70}")
    print(f"#  Essential-Web v1.0 Distribution Report")
    print(f"#  Shards scanned: {len(shard_paths)}")
    print(f"#  Total docs: {stats['total_docs']:,}")
    print(f"#  Valid FDC: {stats['total_valid']:,} ({stats['total_valid']/max(stats['total_docs'],1)*100:.1f}%)")
    print(f"#  L2 coverage: {stats['total_l2_has']:,} ({stats['total_l2_has']/max(stats['total_valid'],1)*100:.1f}%)")
    print(f"#  L2 missing: {stats['total_l2_empty']:,} ({stats['total_l2_empty']/max(stats['total_valid'],1)*100:.1f}%)")
    print(f"{'#' * 70}")

    print_distribution(stats["l1_counter"], "FDC L1 Distribution", stats["total_valid"])
    print_distribution(stats["l2_counter"], "FDC L2 Distribution (Top 50)", stats["total_valid"], top_n=50)
    print_l1_l2_tree(stats["l1_counter"], stats["l1_l2_pairs"])

    print(f"\n  Generating figures to {args.output_dir} ...")

    make_fig1_l1_distribution(stats["l1_counter"], stats["total_valid"], args.output_dir)
    make_fig2_l2_distribution(stats["l2_counter"], stats["total_valid"], args.output_dir, top_n=30)
    make_fig3_prefix_by_l1(stats["prefix_counter"], stats["l1_counter"], stats["l1_l2_pairs"], args.output_dir)
    make_fig4_l2_coverage(stats["l1_counter"], stats["l2_has_by_l1"], stats["l2_empty_by_l1"], args.output_dir)
    make_fig5_quality_histograms(stats["quality_hists"], args.output_dir)
    make_fig6_quality_by_domain(stats["quality_by_l1_hists"], stats["l1_counter"], args.output_dir)
    make_fig7_l1_l2_heatmap(stats["l1_counter"], stats["l1_l2_pairs"], stats["l2_counter"], args.output_dir)
    make_fig8_prefix_treemap(stats["prefix_counter"], args.output_dir)

    report = {
        "total_docs": stats["total_docs"],
        "total_valid": stats["total_valid"],
        "total_l2_empty": stats["total_l2_empty"],
        "total_l2_has": stats["total_l2_has"],
        "l1_distribution": dict(stats["l1_counter"].most_common()),
        "l2_distribution": dict(stats["l2_counter"].most_common(100)),
        "prefix_distribution": dict(sorted(stats["prefix_counter"].items())),
        "l1_l2_pairs": {f"{k[0]}||{k[1]}": v for k, v in stats["l1_l2_pairs"].most_common()},
    }
    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [Report] Saved: {report_path}")

    print(f"\n{'=' * 70}")
    print(f"  All outputs saved to: {args.output_dir}")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
