"""
Shared constants for the QuaDMix project.

Centralizes domain names, quality criteria, HuggingFace paths, and
directory defaults that were previously duplicated across scripts.
"""

import os

DOMAIN_NAMES = [
    "Industrial arts, Technology, and Engineering",
    "Social sciences",
    "Science and Natural history",
    "Religion",
    "Philology; or, Language and languages",
    "Literature",
    "History and Geography",
    "General works, books and libraries...",
    "Philosophy and psychology",
    "Arts",
]

DOMAIN_MAP = {
    "Industrial arts, Technology, and Engineering": 0,
    "Social sciences": 1,
    "Science and Natural history": 2,
    "Religion": 3,
    "Philology; or, Language and languages": 4,
    "Literature": 5,
    "History and Geography": 6,
    "General works, books and libraries, information sciences": 7,
    "Philosophy and psychology": 8,
    "Arts": 9,
}

QUALITY_NAMES = ["dclm", "fineweb_edu", "english", "math_general", "math_openweb"]

QUALITY_COLUMNS = [
    "qs_dclm", "qs_fineweb_edu_approx", "qs_english",
    "qs_eai_general_math", "qs_eai_open_web_math",
]

FASTTEXT_FIELDS = [
    "dclm", "fineweb_edu_approx", "english",
    "eai_general_math", "eai_open_web_math",
]

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HF_RESOLVE = f"{HF_ENDPOINT}/datasets/{{repo}}/resolve/main/{{file}}"

HF_OPENHERMES_DATASET = "liujin99/quadmix-openhermes-10k"
HF_OPENHERMES_FILENAME = "openhermes_10k_assistant_tokenized.pt"

HF_CORE_DATASET = "liujin99/quadmix-core-22tasks"
HF_CORE_FILENAME = "core_22tasks_tokenized.pt"

HF_CORE_BMK_V2_DATASET = "liujin99/quadmix-core-bmk-v2"
HF_CORE_BMK_V2_FILENAME = "core_bmk_10tasks_v2_tokenized.pt"

HF_CORE_BMK_V3_DATASET = "liujin99/quadmix-core-bmk-v3"
HF_CORE_BMK_V3_FILENAME = "core_bmk_10tasks_v3_tokenized.pt"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_TEMP_DIR = os.environ.get(
    "QUADMIX_TEMP_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "QuaDMix", "temp"),
)
DEFAULT_TOKEN_CACHE_DIR = os.path.join(DEFAULT_TEMP_DIR, "token_cache")
DEFAULT_PREPROCESSED_DIR = os.path.join(DEFAULT_TEMP_DIR, "preprocessed")
DEFAULT_VAL_DIR = os.path.join(PROJECT_DIR, "data")

DEFAULT_EVAL_BUNDLE = os.environ.get(
    "EVAL_BUNDLE_DIR",
    "/home/ma-user/work/nanochat-master-multi/eval_bundle",
)
