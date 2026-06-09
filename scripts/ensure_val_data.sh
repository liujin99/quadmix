#!/usr/bin/env bash
# Shared validation set download helper with version checking.
# Source this file in demo scripts: source "$QUADMIX_DIR/scripts/ensure_val_data.sh"

# Get remote file size from HuggingFace via HEAD request
# Args: $1=repo_id, $2=filename
# Returns: size in bytes, or 0 if failed
_hf_remote_size() {
    local repo_id="$1"
    local filename="$2"
    local hf_endpoint="${HF_ENDPOINT:-https://huggingface.co}"
    local url="$hf_endpoint/datasets/$repo_id/resolve/main/$filename"

    local size=0
    if command -v curl &>/dev/null; then
        size=$(curl -sIL --connect-timeout 5 --max-time 10 "$url" 2>/dev/null \
            | grep -i "^content-length:" | tail -1 | awk '{print $2}' | tr -d '\r')
    elif command -v wget &>/dev/null; then
        size=$(wget --spider --timeout=10 -S "$url" 2>&1 \
            | grep -i "Content-Length:" | tail -1 | awk '{print $2}' | tr -d '\r')
    fi

    echo "${size:-0}"
}

# Get local file size in bytes
# Args: $1=file_path
# Returns: size in bytes, or 0 if file doesn't exist
_local_size() {
    local file_path="$1"
    if [ -f "$file_path" ]; then
        stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# Ensure validation set is up-to-date.
# Args:
#   $1 = repo_id (e.g., "liujin99/quadmix-core-22tasks")
#   $2 = filename (e.g., "core_22tasks_tokenized.pt")
#   $3 = local_path (e.g., "/path/to/data/core_22tasks_tokenized.pt")
#
# Behavior:
#   - If local file missing: download
#   - If local file size != remote size: re-download
#   - If can't connect to HF: warn and continue with local file (if exists)
ensure_val_data() {
    local repo_id="$1"
    local filename="$2"
    local local_path="$3"
    local hf_endpoint="${HF_ENDPOINT:-https://huggingface.co}"

    mkdir -p "$(dirname "$local_path")"

    local local_size
    local_size=$(_local_size "$local_path")

    local remote_size
    remote_size=$(_hf_remote_size "$repo_id" "$filename")

    if [ "$remote_size" -eq 0 ] 2>/dev/null; then
        if [ "$local_size" -gt 0 ]; then
            echo ""
            echo "  [Warning] Cannot connect to HuggingFace to check $filename"
            echo "            Using local file ($(du -h "$local_path" | cut -f1))"
            echo "            To force re-download, delete: $local_path"
            echo ""
            return 0
        else
            echo ""
            echo "  [Error] Cannot connect to HuggingFace and no local file"
            echo "          Please download manually from:"
            echo "            https://huggingface.co/datasets/$repo_id"
            echo "          And place at: $local_path"
            echo ""
            return 1
        fi
    fi

    if [ "$local_size" -eq "$remote_size" ]; then
        echo "  ✓ Validation set up-to-date: $filename ($(du -h "$local_path" | cut -f1))"
        return 0
    fi

    if [ "$local_size" -gt 0 ]; then
        echo ""
        echo "  [Update] $filename version mismatch"
        echo "           Local:  $(du -h "$local_path" | cut -f1)"
        echo "           Remote: $(( remote_size / 1024 / 1024 )) MB"
        echo "           Re-downloading..."
        rm -f "$local_path"
    else
        echo ""
        echo "  [Download] $filename not found locally"
        echo "             Downloading from $repo_id..."
    fi

    if [ "$hf_endpoint" != "https://huggingface.co" ]; then
        echo "             Using HF mirror: $hf_endpoint"
    fi

    local download_url="$hf_endpoint/datasets/$repo_id/resolve/main/$filename?download=true"

    if command -v wget &>/dev/null; then
        wget -q --show-progress "$download_url" -O "$local_path"
    else
        curl -L -o "$local_path" "$download_url"
    fi

    if [ -f "$local_path" ]; then
        echo "  ✓ Downloaded: $local_path ($(du -h "$local_path" | cut -f1))"
        echo ""
        return 0
    else
        echo "  [Error] Download failed"
        return 1
    fi
}
