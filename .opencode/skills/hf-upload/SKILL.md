---
name: hf-upload
description: Upload files to HuggingFace Hub datasets or models. Use when user wants to upload data, models, or other files to HuggingFace repositories.
---

# HuggingFace Upload Workflow

This skill provides a reliable workflow for uploading files to HuggingFace Hub, handling common issues like proxy errors and repo type confusion.

## Key Principles

1. **Use `hf` CLI, not `huggingface-cli`** (deprecated)
2. **Always specify `--repo-type=dataset` for dataset repos** (default is model)
3. **Upload files one at a time** (CLI doesn't support batch uploads)
4. **Use mirror as fallback** when direct upload fails due to proxy/network issues
5. **Verify upload success** by checking file sizes on the remote repo

## Basic Upload Command

```bash
cd /path/to/files
hf upload <repo_id> <local_file> <remote_path> --repo-type=dataset
```

**Important:** 
- `<local_file>` and `<remote_path>` are relative paths
- For dataset repos, `--repo-type=dataset` is **required**
- For model repos, omit `--repo-type` (default)

## Example: Upload Dataset Files

```bash
cd /home/liujin99/quadmix/data

# Upload tokenized data
hf upload liujin99/quadmix-core-22tasks core_22tasks_tokenized.pt core_22tasks_tokenized.pt --repo-type=dataset

# Upload parquet
hf upload liujin99/quadmix-core-22tasks core_22tasks.parquet core_22tasks.parquet --repo-type=dataset

# Upload README from different directory
cd /home/liujin99/quadmix
hf upload liujin99/quadmix-core-22tasks scripts/validation_set/README.md README.md --repo-type=dataset
```

## Handling Proxy/Network Errors

If direct upload fails with `Connection reset by peer` or similar network errors, use the HuggingFace mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com hf upload <repo_id> <local_file> <remote_path> --repo-type=dataset
```

**Note:** The mirror uploads to the same official HuggingFace repo, just routes through a different API endpoint.

## Verify Upload Success

After uploading, verify file sizes match expectations:

```python
from huggingface_hub import HfApi

api = HfApi()
info = api.repo_info('<repo_id>', repo_type='dataset', files_metadata=True)

for s in info.siblings:
    size_str = f'{s.size / 1024**2:.1f} MB' if s.size else 'unknown'
    print(f'{s.rfilename}: {size_str}')
```

Check recent commits:

```python
commits = api.list_repo_commits('<repo_id>', repo_type='dataset')
for c in commits[:3]:
    print(f'{c.commit_id[:12]}  {c.created_at}  {c.title}')
```

## Common Pitfalls

1. **Wrong repo type**: Uploading to model repo instead of dataset repo
   - Solution: Always use `--repo-type=dataset` for dataset repos
   
2. **Proxy connection reset**: Network issues with direct upload
   - Solution: Use `HF_ENDPOINT=https://hf-mirror.com`
   
3. **File not updated**: Local file changed but remote still shows old version
   - Solution: Check file sizes via API to confirm upload succeeded
   
4. **Authentication errors**: Token expired or invalid
   - Solution: Run `hf auth login` to re-authenticate

## Complete Workflow Example

```bash
# 1. Navigate to data directory
cd /home/liujin99/quadmix/data

# 2. Upload main data file (with mirror fallback if needed)
hf upload liujin99/quadmix-core-22tasks core_22tasks_tokenized.pt core_22tasks_tokenized.pt --repo-type=dataset

# If fails, try mirror:
# HF_ENDPOINT=https://hf-mirror.com hf upload liujin99/quadmix-core-22tasks core_22tasks_tokenized.pt core_22tasks_tokenized.pt --repo-type=dataset

# 3. Upload additional files
hf upload liujin99/quadmix-core-22tasks core_22tasks.parquet core_22tasks.parquet --repo-type=dataset

cd /home/liujin99/quadmix
hf upload liujin99/quadmix-core-22tasks scripts/validation_set/README.md README.md --repo-type=dataset

# 4. Verify upload
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
info = api.repo_info('liujin99/quadmix-core-22tasks', repo_type='dataset', files_metadata=True)
for s in info.siblings:
    size_str = f'{s.size / 1024**2:.1f} MB' if s.size else 'unknown'
    print(f'{s.rfilename}: {size_str}')
"
```

## Authentication Check

Before uploading, verify you're logged in:

```bash
hf auth whoami
```

If not logged in:

```bash
hf auth login
```
