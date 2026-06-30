"""
Shared constants for the QuaDMix project.

Centralizes domain names, quality criteria, HuggingFace paths, and
directory defaults that were previously duplicated across scripts.
"""

import os

DOMAIN_NAMES = [
    "Computers_and_Electronics",
    "News_and_General_Works",
    "Philosophy_and_Psychology",
    "Religion",
    "Law_and_Government",
    "Economics_and_Finance",
    "Education",
    "People_and_Society",
    "English_Language",
    "Other_Languages",
    "Mathematics",
    "Physics_and_Chemistry",
    "Earth_and_Life_Sciences",
    "Medicine_and_Health",
    "Business_and_Management",
    "Engineering",
    "Agriculture",
    "Arts_and_Entertainment",
    "Sports_and_Recreation",
    "Books_and_Literature",
    "History",
    "Geography_and_Travel",
    "Home_Economics",
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
    "43": 9, "44": 9, "45": 9, "46": 9, "47": 9, "48": 9, "49": 9,
    "51": 10,
    "53": 11, "54": 11,
    "50": 12, "52": 12, "55": 12, "56": 12, "57": 12, "58": 12, "59": 12,
    "61": 13,
    "65": 14,
    "60": 15, "62": 15, "66": 15, "67": 15, "68": 15, "69": 15,
    "63": 16,
    "70": 17, "71": 17, "72": 17, "73": 17, "74": 17,
    "75": 17, "76": 17, "77": 17, "78": 17,
    "79": 18,
    "80": 19, "81": 19, "82": 19, "83": 19, "84": 19,
    "85": 19, "86": 19, "87": 19, "88": 19, "89": 19,
    "90": 20, "93": 20, "94": 20, "95": 20, "96": 20,
    "97": 20, "98": 20, "99": 20,
    "91": 21,
    "64": 22,
}

NUM_DOMAINS = 23

DOMAIN_SHORT_NAMES = [
    "Computers", "News", "Philosophy", "Religion", "Law",
    "Economics", "Education", "People", "English", "OtherLang",
    "Math", "Physics", "EarthLife", "Medicine", "Business",
    "Engineering", "Agriculture", "Arts", "Sports", "Books",
    "History", "Geography", "HomeEcon",
]

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
