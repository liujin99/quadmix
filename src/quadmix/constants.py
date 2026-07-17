"""
Shared constants for the QuaDMix project.

Centralizes domain names, quality criteria, HuggingFace paths, and
directory defaults that were previously duplicated across scripts.
"""

import os

DOMAIN_NAMES = [
    "Mathematics",
    "Chemistry",
    "Biology",
    "Physics",
]

FDC_PREFIX_TO_DOMAIN = {
    "00": 0,
    "01": 1, "02": 1, "03": 1, "04": 1, "05": 1,
    "06": 1, "07": 1, "08": 1, "09": 1,
    "10": 2, "11": 2, "12": 2, "13": 2, "14": 2,
    "15": 2, "16": 2, "17": 2, "18": 2, "19": 2,
    "20": 3, "21": 3, "22": 3, "23": 3, "24": 3,
    "25": 3, "26": 3, "27": 3, "28": 3, "29": 3,
    "32": 4, "34": 4,
    "33": 5,
    "37": 6,
    "30": 7, "31": 7, "35": 7, "36": 7, "38": 7, "39": 7, "92": 7,
    "40": 8, "41": 8, "42": 8,
    "51": 9,
    "53": 10, "54": 10,
    "50": 11, "52": 11, "55": 11, "56": 11, "57": 11, "58": 11, "59": 11,
    "61": 12,
    "65": 13,
    "60": 14, "62": 14, "66": 14, "67": 14, "68": 14, "69": 14,
    "63": 15,
    "70": 16, "71": 16, "72": 16, "73": 16, "74": 16,
    "75": 16, "76": 16, "77": 16, "78": 16,
    "79": 17,
    "80": 18, "81": 18, "82": 18, "83": 18, "84": 18,
    "85": 18, "86": 18, "87": 18, "88": 18, "89": 18,
    "90": 19, "93": 19, "94": 19, "95": 19, "96": 19,
    "97": 19, "98": 19, "99": 19,
    "91": 20,
    "64": 21,
}

NUM_DOMAINS = 4

DOMAIN_SHORT_NAMES = [
    "Math", "Chem", "Bio", "Physics",
]

QUALITY_NAMES = [
    "stem_relevance",
    "knowledge_value",
    "notation_fidelity",
    "rigor_coherence",
    "noise_level",
]

QUALITY_COLUMNS = [
    "stem_relevance",
    "knowledge_value",
    "notation_fidelity",
    "rigor_coherence",
    "noise_level",
]

FASTTEXT_FIELDS = [
    "stem_relevance",
    "knowledge_value",
    "notation_fidelity",
    "rigor_coherence",
    "noise_level",
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

HF_CORE_BMK_V4_DATASET = "liujin99/quadmix-core-bmk-v4"
HF_CORE_BMK_V4_FILENAME = "core_bmk_10tasks_v4_tokenized.pt"

HF_CORE_BMK_V42_DATASET = "liujin99/quadmix-core-bmk-v4.2"
HF_CORE_BMK_V42_FILENAME = "core_bmk_21tasks_v4.2_tokenized.pt"

HF_CORE_BMK_V43_DATASET = "liujin99/quadmix-core-bmk-v4.3"
HF_CORE_BMK_V43_FILENAME = "core_bmk_21tasks_v4.3_tokenized.pt"

HF_CORE_BMK_V5_DATASET = "liujin99/quadmix-core-bmk-v5"
HF_CORE_BMK_V5_FILENAME = "core_bmk_21tasks_v5_tokenized.pt"

HF_CORE_BMK_V6_DATASET = "liujin99/quadmix-core-bmk-v6"
HF_CORE_BMK_V6_FILENAME = "core_bmk_21tasks_v6_tokenized.pt"

HF_CAP_V1_DATASET = "liujin99/quadmix-cap-v1"
HF_CAP_V1_FILENAME = "cap_v1_tokenized.pt"

HF_STEM_V1_DATASET = "liujin99/quadmix-stem-v1"
HF_STEM_V1_FILENAME = "stem_v1_tokenized.pt"

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
