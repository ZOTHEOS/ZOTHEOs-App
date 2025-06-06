# FILE: modules/config_settings_public.py

import os
import sys
import logging
from huggingface_hub import hf_hub_download # ✅ IMPORT THE DOWNLOADER

logger = logging.getLogger("ZOTHEOS_Config")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- ✅ WEB DEPLOYMENT & LOCAL PATH CONFIG ---

# Standard way to detect if we are running in a Hugging Face Space
IS_WEB_MODE = "HF_SPACE_ID" in os.environ 

_is_frozen = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

if _is_frozen:
    APP_DIR = os.path.dirname(sys.executable)
else:
    try:
        APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        APP_DIR = os.getcwd()
        logger.warning(f"__file__ not defined, APP_DIR set to CWD: {APP_DIR}")

# This directory structure is primarily for local/desktop use.
# On Hugging Face, these paths will be created in the virtual environment.
BASE_MODELS_DIR = os.path.join(APP_DIR, "models")
BASE_DATA_SUBDIR = "zotheos_public_data"
BASE_DATA_PATH = os.path.join(APP_DIR, BASE_DATA_SUBDIR)
CORE_DIRS_TO_VERIFY = {
    "data_base": BASE_DATA_PATH, "memory": os.path.join(BASE_DATA_PATH, "zotheos_memory"),
    "cache": os.path.join(BASE_DATA_PATH, "cache"), "cache_transformers": os.path.join(BASE_DATA_PATH, "cache", "transformers"),
    "cache_huggingface": os.path.join(BASE_DATA_PATH, "cache", "huggingface"), "cache_hf_hub": os.path.join(BASE_DATA_PATH, "cache", "huggingface", "hub"),
    "logs": os.path.join(BASE_DATA_PATH, "logs"), "temp": os.path.join(BASE_DATA_PATH, "temp_files"),
    "models_root": BASE_MODELS_DIR
}

for dir_key, dir_path in CORE_DIRS_TO_VERIFY.items():
    try:
        if dir_key == "models_root" and (_is_frozen or IS_WEB_MODE):
            continue # Skip creating models folder for frozen app or web app
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")

# --- ✅ CONDITIONAL MODEL PATHS & GPU CONFIG ---

if IS_WEB_MODE:
    logger.info("✅✅✅ RUNNING IN WEB MODE (Hugging Face Space) ✅✅✅")
    logger.info("Model paths will be resolved by hf_hub_download.")
    
    # Download models from the Hub instead of looking for local files
    MODEL_PATHS = {
        "mistral": hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        "gemma": hf_hub_download(repo_id="google/gemma-2b-it-gguf", filename="gemma-2b-it.Q4_K_M.gguf"),
        "qwen": hf_hub_download(repo_id="Qwen/Qwen1.5-1.8B-Chat-GGUF", filename="qwen1.5-1.8b-chat.Q4_K_M.gguf")
    }
    # Free Hugging Face Spaces are CPU-only, so GPU layers MUST be 0
    N_GPU_LAYERS_FALLBACK = 0
    logger.info("N_GPU_LAYERS_FALLBACK forced to 0 for CPU-only web environment.")

else:
    logger.info("✅✅✅ RUNNING IN LOCAL MODE (Desktop/PC) ✅✅✅")
    logger.info(f"Models will be loaded from local directory: {BASE_MODELS_DIR}")
    
    # Use local file paths
    # Note: I've updated your paths to match the ones from the hf_hub_download for consistency.
    # Please ensure your local folder structure matches these paths.
    MODEL_PATHS = {
        "mistral": os.path.join(BASE_MODELS_DIR, "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        "gemma": os.path.join(BASE_MODELS_DIR, "gemma-2b-it.Q4_K_M.gguf"),
        "qwen": os.path.join(BASE_MODELS_DIR, "qwen1.5-1.8b-chat.Q4_K_M.gguf"),
    }
    # On local machine, use your GPU
    N_GPU_LAYERS_FALLBACK = -1 # -1 means offload all to GPU
    logger.info("N_GPU_LAYERS_FALLBACK set to -1 for local GPU acceleration.")


# --- Shared Configurations ---

MAX_RAM_MODELS_GB = 23.8
MAX_CONCURRENT_MODELS = 3
N_CTX_FALLBACK = 2048
N_THREADS_FALLBACK = 8
VERBOSE_LLAMA_CPP = True

MODEL_SPECIFIC_PARAMS = {
    "mistral": { "chat_format": "mistral-instruct", "n_ctx": N_CTX_FALLBACK },
    "gemma": { "chat_format": "gemma", "n_ctx": N_CTX_FALLBACK },
    "qwen": { "chat_format": "chatml", "n_ctx": N_CTX_FALLBACK },
    "_default": {
        "f16_kv": True, "use_mmap": True, "use_mlock": False,
        "verbose": VERBOSE_LLAMA_CPP,
        "n_gpu_layers": N_GPU_LAYERS_FALLBACK, # This now uses the correct value for web or local
        "n_threads": N_THREADS_FALLBACK,
        "n_ctx": N_CTX_FALLBACK
    }
}

INFERENCE_PRESETS = {
    "balanced": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "mirostat_mode": 0, "max_tokens": 1024},
    "precise": {"temperature": 0.2, "top_p": 0.7, "top_k": 20, "repeat_penalty": 1.05, "mirostat_mode": 0, "max_tokens": 1536},
    "creative": {"temperature": 0.9, "top_p": 0.95, "top_k": 60, "repeat_penalty": 1.15, "mirostat_mode": 2, "mirostat_tau": 4.0, "mirostat_eta": 0.1, "max_tokens": 1024},
    "passthrough": {}
}
DEFAULT_INFERENCE_PRESET = "balanced"

DEFAULT_SYSTEM_PROMPT = "You are ZOTHEOS, an ethical AI developed to help humanity. Provide clear, concise, and helpful responses. Be respectful and avoid harmful content."
SYSTEM_PERSONAS = {
    "default": DEFAULT_SYSTEM_PROMPT, "helpful_assistant": "You are a helpful AI assistant. Your goal is to provide accurate and informative answers.",
    "philosopher": "You are an AI philosopher. Engage with complex questions thoughtfully and explore different perspectives.",
    "coder": "You are an expert AI programmer. Provide code examples and explain them clearly. Assume a senior developer audience.",
    "concise_summarizer": "You are an AI tasked with providing very concise summaries. Get straight to the point. Use bullet points where appropriate.",
}

MODEL_ROLES = { "mistral": "analyst", "gemma": "humanist", "qwen": "skeptic" }
MODEL_ROLE_SYSTEM_PROMPTS = {
    "analyst": "You are an impartial analyst. Focus on facts, clarity, and cause-effect logic. Provide structured, evidence-based reasoning.",
    "humanist": "You are a human-centered assistant. Focus on emotion, empathy, ethical considerations, and the potential human impact or experience related to the query.",
    "skeptic": "You are a critical evaluator and a respectful skeptic. Your role is to challenge assumptions, highlight potential risks, point out biases, and explore alternative or less obvious interpretations. Question the premises if necessary.",
    "general": DEFAULT_SYSTEM_PROMPT
}

MODEL_WEIGHTS = { "mistral": 1.0, "gemma": 0.9, "qwen": 1.1 }
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'

ENV_VARS_TO_SET = {
    "TRANSFORMERS_CACHE": CORE_DIRS_TO_VERIFY["cache_transformers"], "HF_HOME": CORE_DIRS_TO_VERIFY["cache_huggingface"],
    "HF_HUB_CACHE": CORE_DIRS_TO_VERIFY["cache_hf_hub"], "TOKENIZERS_PARALLELISM": "false",
}

ZOTHEOS_VERSION = "Public Beta 1.4 (Web Enabled)"

logger.info(f"Config settings loaded. Version: {ZOTHEOS_VERSION}")
logger.info(f"APP_DIR: {APP_DIR} (Frozen: {_is_frozen}) | Web Mode: {IS_WEB_MODE}")