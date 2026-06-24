"""
Extract a specific document from a parquet shard for inspection.

Usage:
    python3 extract_document.py \
        --data-dir /path/to/quality_data_fineweb_edu \
        --parquet-idx 43 \
        --row-group 5 \
        --doc-idx 155 \
        --output extracted_doc.txt
"""

import argparse
import os
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--parquet-idx", type=int, required=True)
    parser.add_argument("--row-group", type=int, required=True)
    parser.add_argument("--doc-idx", type=int, required=True)
    parser.add_argument("--output", default="extracted_doc.txt")
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".parquet"))
    filepath = os.path.join(args.data_dir, files[args.parquet_idx])
    print(f"Reading: {filepath}")
    print(f"  Parquet index: {args.parquet_idx}, Row group: {args.row_group}, Doc index: {args.doc_idx}")

    pf = pq.ParquetFile(filepath)
    rg = pf.read_row_group(args.row_group)
    texts = rg.column("text").to_pylist()
    text = texts[args.doc_idx]

    print(f"  Length: {len(text):,} chars")
    print(f"  First 500 chars:")
    print(f"  {text[:500]}")
    print(f"\n  Last 500 chars:")
    print(f"  {text[-500:]}")

    with open(args.output, "w") as f:
        f.write(text)
    print(f"\n  Full document saved to: {args.output}")

    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    print(f"\n  Printable chars: {printable:,}/{len(text):,} ({printable/len(text)*100:.1f}%)")
    print(f"  Non-printable: {len(text)-printable:,} ({(len(text)-printable)/len(text)*100:.1f}%)")

    ascii_printable = sum(1 for c in text if 32 <= ord(c) <= 126 or c in "\n\r\t")
    print(f"  ASCII printable: {ascii_printable:,} ({ascii_printable/len(text)*100:.1f}%)")


if __name__ == "__main__":
    main()
