import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"
for _d in (DATA_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(exist_ok=True)

RAW_DEV_FILE = DATA_DIR / "musique_ans_v1.0_dev.jsonl"
SAMPLE_FILE = DATA_DIR / "musique_100.jsonl"

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3.8-27b",
        "key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "key_env": "GEMINI_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "key_env": "OPENROUTER_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b-instruct",
        "key_env": None,
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
_preset = PROVIDERS[LLM_PROVIDER]
LLM_MODEL = os.getenv("LLM_MODEL") or _preset["model"]
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or _preset["base_url"]
LLM_KEY_ENV = _preset["key_env"]
LLM_API_KEY = (
    (os.getenv("LLM_API_KEY") or (os.getenv(LLM_KEY_ENV) if LLM_KEY_ENV else None) or "").strip().strip("\"'")
    or (None if LLM_KEY_ENV else "not-needed")
)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
# Client-side input-tokens-per-minute pacing (0 = off). Groq free tier allows 7000 ITPM.
LLM_INPUT_TPM = int(os.getenv("LLM_INPUT_TPM", "6000" if LLM_PROVIDER == "groq" else "0"))
LLM_OUTPUT_TPM = int(os.getenv("LLM_OUTPUT_TPM", "950" if LLM_PROVIDER == "groq" else "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256" if LLM_PROVIDER == "groq" else "400"))

# "paragraphs": BM25 over the 20 paragraphs shipped with each MuSiQue question.
# "wikipedia": live MediaWiki search API.
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "paragraphs").lower()
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "3"))

MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

HTTP_HEADERS = {
    "User-Agent": "KGIRS-MuSiQue-Agent/1.0 (student research project; contact: none)"
}
