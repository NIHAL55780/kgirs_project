"""OpenAI-compatible chat client (Groq, Gemini, OpenAI, OpenRouter, Ollama) with caching and retries."""
import re
import threading
import time
from dataclasses import dataclass

import openai

import config
from cache import Cache


class DailyLimitReached(RuntimeError):
    """The provider's per-day quota is exhausted; resume the run later."""


class FatalLLMError(RuntimeError):
    """Misconfiguration (bad/missing key, unknown model): retrying other questions is pointless."""


_WAIT = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.I)
_DAILY = re.compile(r"per day|\(TPD\)|\(RPD\)|daily", re.I)


def _retry_after(exc) -> float | None:
    m = _WAIT.search(str(exc))
    if m and any(m.groups()):
        h, mnt, s = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mnt * 60 + s
    try:
        return float(exc.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


_LIMIT_KIND = re.compile(r"on ([\w\s]+?\(\w+\): Limit \d+)")


class TokenPacer:
    """Client-side sliding-window limiter on input tokens per minute, so free-tier
    limits are respected up front instead of through 429 retry storms."""

    def __init__(self, tokens_per_minute: int):
        self.limit = tokens_per_minute
        self.window = []
        self.lock = threading.Lock()

    def acquire(self, tokens: int):
        """Block until `tokens` fit in the last-minute budget; returns a handle for settle()."""
        if self.limit <= 0:
            return None
        tokens = min(tokens, self.limit)
        while True:
            with self.lock:
                now = time.time()
                self.window = [e for e in self.window if now - e[0] < 60]
                used = sum(e[1] for e in self.window)
                if used + tokens <= self.limit:
                    entry = [now, tokens]
                    self.window.append(entry)
                    return entry
                freed, wait = 0, 1.0
                for t, n in self.window:
                    freed += n
                    if used - freed + tokens <= self.limit:
                        wait = 60 - (now - t) + 0.5
                        break
            time.sleep(max(wait, 0.5))

    def settle(self, handle, actual: int):
        """Replace a reservation with the tokens actually used."""
        if handle is not None:
            with self.lock:
                handle[1] = actual


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


class LLM:
    def __init__(self, model=None, base_url=None, api_key=None, temperature=None, max_tokens=None):
        self.model = model or config.LLM_MODEL
        api_key = api_key or config.LLM_API_KEY
        if not api_key:
            raise FatalLLMError(
                f"No API key found. Put {config.LLM_KEY_ENV}=<your key> in {config.ROOT / '.env'} and save the file."
            )
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.client = openai.OpenAI(
            base_url=base_url or config.LLM_BASE_URL,
            api_key=api_key,
            max_retries=0,
            timeout=120,
        )
        self.cache = Cache("llm")
        self.pacer = TokenPacer(config.LLM_INPUT_TPM)
        self.out_pacer = TokenPacer(config.LLM_OUTPUT_TPM)
        self._usage_lock = threading.Lock()
        self.session_tokens = 0
        self.session_calls = 0

    def chat(self, messages, stop=None, max_retries=12) -> LLMResponse:
        key = Cache.key(self.model, messages, stop, self.temperature, self.max_tokens)
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResponse(**{**hit, "cached": True})

        kwargs = dict(
            model=self.model, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        if stop:
            kwargs["stop"] = stop

        est_tokens = sum(len(m["content"]) for m in messages) // 3 + 20
        for attempt in range(max_retries):
            self.pacer.acquire(est_tokens)
            out_handle = self.out_pacer.acquire(self.max_tokens)
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self.out_pacer.settle(out_handle, getattr(resp.usage, "completion_tokens", 0) or self.max_tokens)
                break
            except openai.AuthenticationError as exc:
                raise FatalLLMError(
                    f"API key rejected by {config.LLM_PROVIDER} ({exc.status_code}). "
                    f"Check {config.LLM_KEY_ENV} in .env."
                ) from exc
            except (openai.PermissionDeniedError, openai.NotFoundError) as exc:
                raise FatalLLMError(
                    f"{config.LLM_PROVIDER} refused model '{self.model}': {str(exc)[:200]}. "
                    "Set LLM_MODEL in .env to a model your account can use."
                ) from exc
            except openai.RateLimitError as exc:
                wait = _retry_after(exc)
                if _DAILY.search(str(exc)) and (wait is None or wait > 300):
                    raise DailyLimitReached(str(exc)[:300]) from exc
                if attempt == max_retries - 1:
                    raise
                wait = min(120, (wait or 2 ** attempt * 2) + 1)
                kind = _LIMIT_KIND.search(str(exc))
                print(f"  [llm] rate limited ({kind.group(1) if kind else 'unknown limit'}), waiting {wait:.0f}s")
                time.sleep(wait)
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                if attempt == max_retries - 1:
                    raise
                wait = min(60, 2 ** attempt * 2)
                print(f"  [llm] {exc.__class__.__name__}, retrying in {wait}s")
                time.sleep(wait)
            except openai.BadRequestError as exc:
                if "stop" in kwargs and "stop" in str(exc).lower():
                    kwargs.pop("stop")
                    continue
                raise

        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        if stop:
            for s in stop:
                if s in text:
                    text = text[: text.index(s)]
        usage = resp.usage
        out = LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        with self._usage_lock:
            self.session_calls += 1
            self.session_tokens += out.prompt_tokens + out.completion_tokens
        self.cache.set(key, {"text": out.text, "prompt_tokens": out.prompt_tokens,
                             "completion_tokens": out.completion_tokens})
        return out
