# prepare_data.py Performance Fixes

## Background
- Random path memory exhaustion: 192 spawn workers × ~394MB/shard = ~76GB peak
- Old QuadMix code residual memory + Random path = system OOM
- CPU utilization low: 192 processes competing for disk I/O
- Parquet writes sequential, no parallelism
- Quality path uses wasteful `.to_pandas()` roundtrip

## Fixes

### Fix 1: Parquet write parallelization (`write_dataset`, lines 1047-1062)
Replace sequential `for i in range(n_shards)` with `ThreadPool` (already imported at line 51).

```python
# Current (line 1049):
for i in range(n_shards):
    start = i * args.shard_size
    end = min(start + args.shard_size, len(docs))
    shard_docs = docs[start:end]
    out_path = data_dir / f"shard_{i:05d}.parquet"
    write_shard(shard_docs, str(out_path), args.num_npu)

# New:
shard_specs = []
for i in range(n_shards):
    start = i * args.shard_size
    end = min(start + args.shard_size, len(docs))
    shard_docs = docs[start:end]
    out_path = data_dir / f"shard_{i:05d}.parquet"
    shard_specs.append((shard_docs, str(out_path), args.num_npu))
nw = min(n_shards, 8)
with ThreadPool(nw) as pool:
    list(pool.map(lambda spec: write_shard(*spec), shard_specs))
```

### Fix 2: Quality path `.to_pandas()` removal (line 488)
```python
# Old:
scores = pq.read_table(str(prep_files[shard_id]), columns=[quality_col]).to_pandas()[quality_col].to_numpy()
# New:
scores = pq.read_table(str(prep_files[shard_id]), columns=[quality_col])[quality_col].to_numpy()
```

### Fix 3: Row-group-level reading in `_read_docs_from_shard_tagged` (lines 246-283)
Replace `pq.read_table(shard_path, columns=cols)` (reads ENTIRE shard ~394MB) with `ParquetFile.read_row_group()` (reads one row group ~40MB, freed before reading next).

```python
def _read_docs_from_shard_tagged(args):
    shard_id, shard_path, doc_indices, text_col, domain_col, quality_cols, char_count_col = args
    shard_names = set(pq.read_schema(shard_path).names)
    want = [c for c in [text_col, domain_col, char_count_col] + (quality_cols or [])
            if c and c in shard_names]
    cols = list(dict.fromkeys(want))

    pf = pq.ParquetFile(shard_path)
    n_rgs = pf.num_row_groups

    # Build row-group offset boundaries
    rg_offsets = [0]
    for i in range(n_rgs):
        rg_offsets.append(rg_offsets[-1] + pf.metadata.row_group(i).num_rows)

    # Sort doc_indices for efficient row-group assignment
    sort_order = sorted(range(len(doc_indices)), key=lambda i: doc_indices[i])
    sorted_di = [doc_indices[i] for i in sort_order]

    # Assign sorted doc_indices to row groups (linear scan since sorted_di is sorted)
    rg_to_local = {}
    di_pos = 0
    for rg_idx in range(n_rgs):
        rg_start = rg_offsets[rg_idx]
        rg_end = rg_offsets[rg_idx + 1]
        local = []
        while di_pos < len(sorted_di) and rg_start <= sorted_di[di_pos] < rg_end:
            local.append(sorted_di[di_pos] - rg_start)
            di_pos += 1
        if local:
            rg_to_local[rg_idx] = local

    # Read only needed row groups, one at a time, freeing each
    sorted_texts = []
    sorted_domains = []
    sorted_quality = {qc: [] for qc in (quality_cols or []) if qc in shard_names}
    sorted_cc = []
    for rg_idx, local_indices in rg_to_local.items():
        rg_table = pf.read_row_group(rg_idx, columns=cols)
        rg_taken = rg_table.take(pa.array(local_indices))
        n_local = len(local_indices)
        if text_col in rg_taken.column_names:
            sorted_texts.extend(rg_taken[text_col].to_pylist())
        else:
            sorted_texts.extend([""] * n_local)
        if domain_col and domain_col in rg_taken.column_names:
            sorted_domains.extend(rg_taken[domain_col].to_pylist())
        else:
            sorted_domains.extend([None] * n_local)
        for qc in sorted_quality:
            if qc in rg_taken.column_names:
                sorted_quality[qc].extend(rg_taken[qc].to_pylist())
            else:
                sorted_quality[qc].extend([None] * n_local)
        if char_count_col and char_count_col in rg_taken.column_names:
            sorted_cc.extend(rg_taken[char_count_col].to_pylist())
        else:
            sorted_cc.extend([len(t) for t in sorted_texts[-n_local:]])
        del rg_table, rg_taken

    # Build docs in sorted order, then unsort to original doc_indices order
    sorted_docs = []
    for i in range(len(sorted_di)):
        doc = {
            "text": sorted_texts[i],
            "char_count": sorted_cc[i] if sorted_cc else len(sorted_texts[i]),
            "token_count": len(sorted_texts[i]) // 4,
            "domain": sorted_domains[i],
        }
        for qc, arr in sorted_quality.items():
            doc[qc] = arr[i]
        sorted_docs.append(doc)

    # Restore original order
    docs = [None] * len(doc_indices)
    for i, si in enumerate(sort_order):
        docs[si] = sorted_docs[i]

    return shard_id, docs
```

### Fix 4: Separate read pool with fewer workers (lines 62-96)
Add `_read_pool` global and `_get_read_pool()`. Close `_io_pool` after scanning.

```python
_io_pool = None
_read_pool = None
_token_pool = None

def _get_io_pool(num_workers=None):
    global _io_pool
    if _io_pool is None:
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 256) or 1
        _io_pool = _SPAWN_CTX.Pool(num_workers)
    return _io_pool

def _close_io_pool():
    global _io_pool
    if _io_pool is not None:
        _io_pool.close()
        _io_pool.join()
        _io_pool = None

def _get_read_pool(num_workers=None):
    global _read_pool
    if _read_pool is None:
        if num_workers is None:
            num_workers = min(mp.cpu_count() // 4, 48) or 1
        _read_pool = _SPAWN_CTX.Pool(num_workers)
    return _read_pool

def _cleanup_pools():
    global _io_pool, _read_pool, _token_pool
    for pool, setter in [(_io_pool, '_io_pool'), (_read_pool, '_read_pool'), (_token_pool, '_token_pool')]:
        if pool is not None:
            pool.close()
            pool.join()
    _io_pool = None
    _read_pool = None
    _token_pool = None
```

In `read_docs_from_shards` (line 299), change:
```python
# Old:
pool = _get_io_pool(num_workers)
# New:
pool = _get_read_pool(min(num_workers, 48) if num_workers else None)
```

After `scan_shards` returns (line 897), add:
```python
_close_io_pool()  # Free 192 scan workers, release ~38GB
```

### Fix 5: Memory cleanup between baselines (after write_dataset calls, ~line 1072)
After each `write_dataset` call, free the docs list:
```python
if not skip_quadmix:
    write_dataset(quadmix_train, quadmix_dir, "QuadMix", quadmix_val)
    del quadmix_train, quadmix_val; gc.collect()
if not skip_random:
    write_dataset(random_train, random_dir, "Random", random_val)
    del random_train, random_val; gc.collect()
if do_manual_ratio:
    write_dataset(manual_ratio_train, manual_ratio_dir, manual_ratio_label, manual_ratio_val)
    del manual_ratio_train, manual_ratio_val; gc.collect()
for m, qt in quality_trains.items():
    write_dataset(qt, quality_dirs[m], f"Quality ({m})", quality_vals[m])
    del qt, quality_vals[m]
# Don't del quality_trains dict itself until after the loop
gc.collect()
```

## Memory Impact Summary
| Component | Before | After |
|-----------|--------|-------|
| Read workers (192 × 394MB) | 76GB | 48 × 40MB = 1.9GB (Fix 3+4) |
| Scan pool residual | 38GB | 0 (closed after scan, Fix 4) |
| QuadMix residual | ~133GB | ~3GB (two-pass already pushed) |
| Quality path pandas | ~1GB/shard | 0 (Fix 2) |
| Between baselines | accumulates | freed (Fix 5) |

## Verification
1. `python3 -m py_compile prepare_data.py` — syntax check
2. Re-run smoke test `/tmp/opencode/test_two_pass.py` — verify row-group reading works
3. `git diff --stat` — verify net line count is reasonable
4. Commit + push
5. Server: `git pull` + re-run `run_stem_experiment.sh`
