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
        size=$(curl -ksIL --connect-timeout 5 --max-time 10 "$url" 2>/dev/null \
            | grep -i "^content-length:" | tail -1 | awk '{print $2}' | tr -d '\r')
    elif command -v wget &>/dev/null; then
        size=$(wget --no-check-certificate --spider --timeout=10 -S "$url" 2>&1 \
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

    local dl_exit=0
    if command -v wget &>/dev/null; then
        wget --no-check-certificate --progress=bar:force "$download_url" -O "$local_path" 2>&1
        dl_exit=$?
    else
        curl -L -k --fail --progress-bar -o "$local_path" "$download_url" 2>&1
        dl_exit=$?
    fi

    if [ "$dl_exit" -ne 0 ]; then
        rm -f "$local_path"
        echo ""
        echo "  [Error] Download failed (exit code: $dl_exit)"
        echo "          URL: $download_url"
        echo "          You can manually download from:"
        echo "            https://huggingface.co/datasets/$repo_id"
        echo ""
        return 1
    fi

    local downloaded_size
    downloaded_size=$(_local_size "$local_path")
    if [ "$downloaded_size" -eq 0 ]; then
        rm -f "$local_path"
        echo ""
        echo "  [Error] Downloaded file is empty (0 bytes)"
        echo "          URL: $download_url"
        echo ""
        return 1
    fi

    if [ "$remote_size" -gt 0 ] && [ "$downloaded_size" -ne "$remote_size" ]; then
        rm -f "$local_path"
        echo ""
        echo "  [Error] Downloaded file size mismatch"
        echo "          Expected: $(( remote_size / 1024 / 1024 )) MB"
        echo "          Got:      $(( downloaded_size / 1024 / 1024 )) MB"
        echo ""
        return 1
    fi

    echo "  ✓ Downloaded: $local_path ($(du -h "$local_path" | cut -f1))"
    echo ""
    return 0
}

# Ensure validation set based on --val-set argument.
# Args:
#   $1 = val_set (e.g., "openhermes", "core", "core_bmk_v6")
#   $2 = data_dir (e.g., "/path/to/data")
#
# Behavior:
#   - Downloads the appropriate validation set based on val_set name
#   - Supports: openhermes, core, core_bmk_v3/v4/v4.2/v4.3/v5/v6
ensure_val_set() {
    local val_set="$1"
    local data_dir="$2"
    
    case "$val_set" in
        openhermes)
            ensure_val_data "liujin99/quadmix-openhermes-10k" \
                "openhermes_10k_assistant_tokenized.pt" \
                "$data_dir/openhermes_10k_assistant_tokenized.pt"
            ;;
        core)
            ensure_val_data "liujin99/quadmix-core-22tasks" \
                "core_22tasks_tokenized.pt" \
                "$data_dir/core_22tasks_tokenized.pt"
            ;;
        core_bmk_v3)
            ensure_val_data "liujin99/quadmix-core-bmk-v3" \
                "core_bmk_21tasks_v3_tokenized.pt" \
                "$data_dir/core_bmk_21tasks_v3_tokenized.pt"
            ;;
        core_bmk_v4)
            ensure_val_data "liujin99/quadmix-core-bmk-v4" \
                "core_bmk_21tasks_v4_tokenized.pt" \
                "$data_dir/core_bmk_21tasks_v4_tokenized.pt"
            ;;
        core_bmk_v4.2)
            ensure_val_data "liujin99/quadmix-core-bmk-v4.2" \
                "core_bmk_21tasks_v4.2_tokenized.pt" \
                "$data_dir/core_bmk_21tasks_v4.2_tokenized.pt"
            ;;
        core_bmk_v4.3)
            ensure_val_data "liujin99/quadmix-core-bmk-v4.3" \
                "core_bmk_21tasks_v4.3_tokenized.pt" \
                "$data_dir/core_bmk_21tasks_v4.3_tokenized.pt"
            ;;
        core_bmk_v5)
            ensure_val_data "liujin99/quadmix-core-bmk-v5" \
                "core_bmk_21tasks_v5_tokenized.pt" \
                "$data_dir/core_bmk_21tasks_v5_tokenized.pt"
            ;;
        core_bmk_v6)
            ensure_val_data "liujin99/quadmix-core-bmk-v6" \
                "core_bmk_21tasks_v6_tokenized.pt" \
                "$data_dir/core_bmk_21tasks_v6_tokenized.pt"
            ;;
        *)
            echo "  [Error] Unknown val_set: $val_set"
            echo "          Supported: openhermes, core, core_bmk_v3/v4/v4.2/v4.3/v5/v6"
            return 1
            ;;
    esac
}
