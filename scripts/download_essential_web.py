#!/usr/bin/env python3
"""
Download Essential-Web v1.0 parquet shards from Hugging Face.

Usage:
    python scripts/download_essential_web.py [--num-files N] \
        [--output-dir DIR] [--crawl CRAWL_NAME]

Downloads N shards (default 50) from the specified CommonCrawl dump
into the output directory. Uses the Hugging Face Hub API to list
files and downloads them via streaming HTTP requests.

Examples:
    # Download 50 files (default) to data/essential-web/
    python scripts/download_essential_web.py

    # Download just 2 files (for testing)
    python scripts/download_essential_web.py --num-files 2

    # Custom output
    python scripts/download_essential_web.py \
        --output-dir ./data/essential-web \
        --num-files 50
"""

import argparse
import os
import sys
import json
import urllib.request
import urllib.error
import time

# ---------------------------------------------------------------------------
REPO_ID = "EssentialAI/essential-web-v1.0"
DEFAULT_CRAWL = "CC-MAIN-2024-38"
DEFAULT_NUM_FILES = 50
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "essential-web")

HF_API_TREE = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}?download=true"


def list_crawl_files(repo: str, crawl: str) -> list[dict]:
    """List all parquet files in a given crawl directory via the HF tree API."""
    path = f"data/crawl={crawl}"
    url = HF_API_TREE.format(repo=repo, path=path)
    parts = []

    # The API returns up to 1000 entries.  We paginate using cursor from
    # response headers if available; otherwise iterate through all entries
    # via the Link header / next URL.
    cursor = None
    while True:
        api_url = url + (f"?cursor={cursor}" if cursor else "")
        req = urllib.request.Request(api_url)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"[Error] Failed to list files: {e}")
            sys.exit(1)

        for item in data:
            if item["type"] == "file" and item["path"].endswith(".parquet"):
                parts.append(item)

        # Check for Link header (pagination)
        link = resp.headers.get("Link", "") if cursor is not None else ""
        if link and 'rel="next"' in link:
            cursor = resp.headers.get("X-Cursor") or (data and data[-1].get("path"))
        else:
            break

    return parts


def format_size(mb: float) -> str:
    if mb < 1024:
        return f"{mb:.0f} MB"
    return f"{mb / 1024:.1f} GB"


def download_file(url: str, dest: str, index: int, total: int, size_mb: float,
                  show_progress: bool = True) -> bool:
    """Download a single file with progress indication."""
    try:
        if show_progress:
            print(f"  [{index+1}/{total}] {os.path.basename(dest)} ({size_mb:.0f} MB) ... ",
                  end="", flush=True)

        urllib.request.urlretrieve(url, dest)

        # Verify size
        actual = os.path.getsize(dest) / (1024 * 1024)
        if show_progress:
            print(f"done ({actual:.0f} MB)")
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        if show_progress:
            print(f"FAILED: {e}")
        # Remove partial file if any
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Essential-Web v1.0 parquet shards from Hugging Face."
    )
    parser.add_argument("--num-files", type=int, default=DEFAULT_NUM_FILES,
                        help=f"Number of parquet files to download (default: {DEFAULT_NUM_FILES})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--crawl", type=str, default=DEFAULT_CRAWL,
                        help=f"CommonCrawl dump name (default: {DEFAULT_CRAWL})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: List files on HF ──
    print(f"[1/3] Listing parquet files in {args.crawl}...")
    files = list_crawl_files(REPO_ID, args.crawl)
    if not files:
        print(f"[Error] No parquet files found in crawl '{args.crawl}'")
        print(f"  Check: {HF_API_TREE.format(repo=REPO_ID, path=f'data/crawl={args.crawl}')}")
        sys.exit(1)

    total_remote = len(files)
    total_size_mb = sum(f["size"] for f in files) / (1024 * 1024)
    print(f"  Found {total_remote} files ({format_size(total_size_mb)}) in crawl '{args.crawl}'")

    # ── Step 2: Check what's already downloaded ──
    print(f"[2/3] Checking existing files in {output_dir}...")
    existing = set()
    if os.path.isdir(output_dir):
        for fname in os.listdir(output_dir):
            if fname.endswith(".parquet"):
                existing.add(fname)
    need = [f for f in files if os.path.basename(f["path"]) not in existing]
    have = total_remote - len(need)

    if have >= args.num_files:
        print(f"  Already have {have} files (>= {args.num_files} requested). Nothing to do.")
        sys.exit(0)

    need = need[:args.num_files - have]
    size_need_mb = sum(f["size"] for f in need) / (1024 * 1024)
    print(f"  Already have: {have} | Need to download: {len(need)} ({format_size(size_need_mb)})")
    print(f"  Output: {output_dir}")

    # ── Step 3: Download ──
    print(f"[3/3] Downloading {len(need)} files...")
    start = time.time()
    ok = 0
    fail = 0
    for i, f in enumerate(need):
        url = HF_RESOLVE.format(repo=REPO_ID, path=f["path"])
        dest = os.path.join(output_dir, os.path.basename(f["path"]))
        size_mb = f["size"] / (1024 * 1024)
        if download_file(url, dest, i, len(need), size_mb, show_progress=not args.quiet):
            ok += 1
        else:
            fail += 1
            if fail >= 3:
                print("  [Abort] Too many download failures")
                break

    elapsed = time.time() - start
    speed = size_need_mb / max(1, elapsed) if ok > 0 else 0
    print(f"\n  Downloaded: {ok}/{len(need)} files ({format_size(size_need_mb)} in {elapsed:.0f}s, {speed:.1f} MB/s)")

    # ── Summary ──
    final_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".parquet"))
    final_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in final_files) / (1024 * 1024)
    print(f"\n  Total files in {output_dir}: {len(final_files)} ({format_size(final_size)})")
    if final_files:
        print(f"  Range: {final_files[0]} ... {final_files[-1]}")


if __name__ == "__main__":
    main()
