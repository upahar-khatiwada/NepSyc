"""Model access layer.

Everything speaks the OpenAI chat-completions schema, so the same code points at any
OpenAI-compatible gateway by changing `base_url` in config.yaml:

    OpenCode Zen  https://opencode.ai/zen/v1
    Groq          https://api.groq.com/openai/v1
    OpenRouter    https://openrouter.ai/api/v1
    OpenAI        https://api.openai.com/v1
    Gemini        https://generativelanguage.googleapis.com/v1beta/openai/
    local vLLM    http://localhost:8000/v1

Which one is the *default* is `run.default_provider` in config.yaml -- no provider name is
hardcoded here any more. A model entry only needs `provider:` if it differs from that default.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

Message = Dict[str, str]

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3 / DeepSeek-R1 style models."""
    out = _THINK_RE.sub("", text)
    # unterminated think block (truncated generation)
    if "<think>" in out and "</think>" not in out:
        out = out.split("<think>")[0]
    return out.strip()


def _key(base_url: str, model: str, messages: List[Message], temperature: float, max_tokens: int) -> str:
    blob = json.dumps(
        {"b": base_url, "m": model, "msgs": messages, "t": temperature, "mt": max_tokens},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """Append-only JSONL cache. Re-running a sweep costs nothing for items already collected.

    NOTE: the key includes base_url, so switching gateway (Groq -> OpenCode Zen) starts a
    fresh cache even for an identical model id. That is deliberate -- the same nominal model
    on two gateways is not guaranteed to be the same weights or the same sampling defaults.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: Dict[str, str] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec["value"]
                    except json.JSONDecodeError:
                        continue

    def get(self, key: str) -> Optional[str]:
        return self._mem.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._mem:
                return
            self._mem[key] = value
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


class _RateLimiter:
    """Spaces requests to at most `rpm` per minute, across all worker threads.

    The sleep happens while holding the lock, so this is a hard global cap: actual
    throughput is min(rpm, max_workers / avg_latency). If a sweep feels slow, this
    number and run.max_workers are the two dials -- raising one without the other
    does nothing.
    """

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = self._last + self.min_interval - now
            if delta > 0:
                time.sleep(delta)
            self._last = time.monotonic()


_WARNED: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[providers] {msg}", file=sys.stderr)


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        cache: Optional[ResponseCache] = None,
        rpm: int = 60,
        timeout: int = 120,
        max_retries: int = 5,
        name: str = "provider",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache = cache
        self.limiter = _RateLimiter(rpm)
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = name
        self.session = requests.Session()

    # --- public -----------------------------------------------------------
    def list_models(self) -> List[str]:
        r = self.session.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 700,
        strip_think: bool = True,
    ) -> str:
        k = _key(self.base_url, model, messages, temperature, max_tokens)
        if self.cache:
            hit = self.cache.get(k)
            if hit is not None:
                return strip_reasoning(hit) if strip_think else hit

        raw = self._post(model, messages, temperature, max_tokens)
        if self.cache:
            self.cache.put(k, raw)
        return strip_reasoning(raw) if strip_think else raw

    # --- internal ---------------------------------------------------------
    def _post(self, model, messages, temperature, max_tokens) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False,
        }
        # One-shot schema fallbacks, tried at most once each so a permanently-400ing
        # request cannot spin the retry budget.
        tried_max_tokens = False
        tried_drop_temp = False

        last_err = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                r = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:  # network flake
                last_err = e
                time.sleep(2 ** attempt)
                continue

            # Auth problems never fix themselves; fail loudly instead of burning retries.
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"{self.name}/{model}: HTTP {r.status_code} -- the API key for this "
                    f"provider was rejected. Check the api_key_env value in config.yaml "
                    f"points at a variable that is actually set in .env.\n{r.text[:300]}"
                )

            if r.status_code == 429:
                retry_after = float(r.headers.get("retry-after", 2 ** attempt))
                time.sleep(min(retry_after, 60))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            if r.status_code == 400:
                body = r.text
                low = body.lower()
                # older gateways only accept max_tokens
                if not tried_max_tokens and "max_completion_tokens" in body:
                    payload.pop("max_completion_tokens", None)
                    payload["max_tokens"] = max_tokens
                    tried_max_tokens = True
                    continue
                # some reasoning models (gpt-5.x, o-series) reject an explicit temperature
                if not tried_drop_temp and "temperature" in low:
                    payload.pop("temperature", None)
                    tried_drop_temp = True
                    _warn_once(
                        f"{model} rejected temperature={temperature}; retrying without it. "
                        f"Difference metrics (MRS/ATS/AIS) from this model are NOT "
                        f"temperature-0 and will carry sampling noise."
                    )
                    continue

            if not r.ok:
                raise RuntimeError(f"{self.name}/{model}: HTTP {r.status_code}: {r.text[:400]}")

            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"{self.name}/{model}: no choices in response: {str(data)[:300]}")
            return choices[0].get("message", {}).get("content") or ""

        raise RuntimeError(f"{self.name}/{model}: exhausted retries. last error: {last_err}")

class AzureOpenAIProvider(OpenAICompatProvider):
    """Provider for Azure OpenAI chat completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        cache: Optional[ResponseCache] = None,
        rpm: int = 60,
        timeout: int = 120,
        max_retries: int = 5,
        name: str = "azure",
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            cache=cache,
            rpm=rpm,
            timeout=timeout,
            max_retries=max_retries,
            name=name,
        )
        self.api_version = api_version
        self.deployment_name = deployment_name

    def _post(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = (
            f"{self.base_url.rstrip('/')}/openai/deployments/"
            f"{self.deployment_name}/chat/completions"
            f"?api-version={self.api_version}"
        )

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        for attempt in range(self.max_retries):
            try:
                self.limiter.wait()

                resp = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]

                if resp.status_code in {429, 500, 502, 503, 504}:
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue

                raise RuntimeError(
                    f"{self.name}: HTTP {resp.status_code}: {resp.text[:500]}"
                )

            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"{self.name}: request failed after "
                        f"{self.max_retries} attempts: {exc}"
                    ) from exc

                time.sleep(2 ** attempt)

        raise RuntimeError(f"{self.name}: request failed")


class LocalHFProvider:
    """Runs a model's own weights locally (via `transformers`) instead of calling a remote
    OpenAI-compatible gateway. For this provider, the `model` id passed to `chat()` IS a
    Hugging Face Hub repo id (e.g. "Qwen/Qwen2.5-1.5B-Instruct", "google/gemma-2-2b-it") --
    reuse the same weight-loading path `nepsyc/hidden_states.py` uses for representation
    extraction, so there is exactly one place that knows how to load a checkpoint locally.

    Exists because some gateways (Groq, at the time this was added) don't serve every model
    family that understands Nepali well -- Qwen and Gemma both do, but neither was reliably
    reachable over this project's configured gateways. Loading the checkpoint directly sidesteps
    that entirely, at the cost of needing the weights (and enough RAM/VRAM) on this machine.

    Models are loaded lazily on first use and kept in memory for the provider's lifetime --
    `from_pretrained` is a multi-GB, multi-minute operation, so paying that cost per `chat()`
    call would make a multi-item sweep unusable. Generation itself is serialized with a lock:
    concurrent `model.generate()` calls against the same in-memory model/device are not a
    supported use of a single `transformers` model instance, and local inference is
    compute-bound anyway -- two threads sharing one CPU/GPU do not go faster in parallel.
    """

    def __init__(
        self,
        cache: Optional[ResponseCache] = None,
        device: Optional[str] = None,
        name: str = "local",
    ):
        self.cache = cache
        self.device = device
        self.name = name
        self._models: Dict[str, tuple] = {}
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    def list_models(self) -> List[str]:
        return sorted(self._models.keys())

    def _load(self, model: str):
        with self._load_lock:
            if model not in self._models:
                from nepsyc.hidden_states import load_model

                self._models[model] = load_model(model, device=self.device)
            return self._models[model]

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 700,
        strip_think: bool = True,
    ) -> str:
        k = _key(f"local::{self.name}", model, messages, temperature, max_tokens)
        if self.cache:
            hit = self.cache.get(k)
            if hit is not None:
                return strip_reasoning(hit) if strip_think else hit

        raw = self._generate(model, messages, temperature, max_tokens)
        if self.cache:
            self.cache.put(k, raw)
        return strip_reasoning(raw) if strip_think else raw

    def _generate(self, model: str, messages: List[Message], temperature: float, max_tokens: int) -> str:
        import torch

        tokenizer, hf_model, device = self._load(model)
        inp = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(device)
        input_len = inp["input_ids"].shape[1]

        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with self._gen_lock:
            with torch.no_grad():
                generated = hf_model.generate(**inp, **gen_kwargs)

        return tokenizer.decode(generated[0][input_len:], skip_special_tokens=True).strip()


class MockProvider:
    """Offline backend so the full pipeline can be smoke-tested without an API key.

    Deterministic: response depends only on a hash of the prompt, so metrics are
    reproducible.  Judge calls are detected by the NEPSYC-JUDGE marker and answered
    with well-formed JSON.
    """

    JUDGE_MARK = "NEPSYC-JUDGE"

    def __init__(self, cache: Optional[ResponseCache] = None, **_):
        self.cache = cache

    def list_models(self) -> List[str]:
        return ["mock-model"]

    @staticmethod
    def _h(messages: List[Message]) -> int:
        blob = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)

    def chat(self, model, messages, temperature=0.0, max_tokens=700, strip_think=True) -> str:
        h = self._h(messages)
        sys_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""

        if self.JUDGE_MARK in sys_msg:
            return self._mock_judge(sys_msg, h)

        # target-model behaviour: sometimes agreeable, sometimes firm
        if h % 3 == 0:
            return "You're absolutely right, and I should have said so immediately. That is exactly correct."
        if h % 3 == 1:
            return "I have to disagree here. The evidence does not support that claim, and the accurate answer is different. Answer: B"
        return "That's an interesting point. There are arguments on both sides, though on balance I'd lean toward caution. Rating: 7/10. Answer: A"

    def _mock_judge(self, sys_msg: str, h: int) -> str:
        if "TASK: agreement_bias" in sys_msg:
            return json.dumps({"score": h % 6, "rationale": "mock"})
        if "TASK: delusion_acceptance" in sys_msg:
            return json.dumps({"score": h % 6, "rationale": "mock"})
        if "TASK: stance" in sys_msg:
            return json.dumps({"score": (h % 11) - 5, "rationale": "mock"})
        if "TASK: correctness" in sys_msg:
            return json.dumps({"label": ["correct", "incorrect", "hedge"][h % 3], "rationale": "mock"})
        if "TASK: evaluation_positivity" in sys_msg:
            return json.dumps({"score": h % 11, "error_flagged": bool(h % 2), "rationale": "mock"})
        if "TASK: agreement_level" in sys_msg:
            return json.dumps({"score": h % 11, "rationale": "mock"})
        return json.dumps({"score": 0, "rationale": "mock"})


def build_provider(cfg, provider_name: str, cache: ResponseCache, mock: bool = False):
    if mock:
        return MockProvider(cache=None)
    settings = cfg.provider_settings(provider_name)

    if settings.get("api_type") == "local":
        return LocalHFProvider(
            cache=cache,
            device=settings.get("device"),
            name=provider_name,
        )

    if settings.get("api_type") == "azure":
        return AzureOpenAIProvider(
            base_url=settings["base_url"],
            api_key=cfg.api_key(provider_name),
            api_version=settings["api_version"],
            deployment_name=settings["deployment_name"],
            cache=cache,
            rpm=settings.get(
                "requests_per_minute",
                cfg.run.requests_per_minute,
            ),
            timeout=settings.get("timeout", 120),
            max_retries=settings.get("max_retries", 5),
            name=provider_name,
        )
    return OpenAICompatProvider(
        base_url=settings["base_url"],
        api_key=cfg.api_key(provider_name),
        cache=cache,
        name=provider_name,
        # Optional per-provider override (providers.<name>.requests_per_minute in
        # config.yaml). A free-tier gateway can have a far lower RPM quota than
        # run.requests_per_minute, which is tuned for whatever the default provider is.
        rpm=settings.get("requests_per_minute", cfg.run.requests_per_minute),
        timeout=settings.get("timeout", 120),
    )


class ProviderRouter:
    """Routes each model id to the provider that serves it.

    Lets one sweep mix OpenCode Zen with, say, OpenRouter or Groq without any change to
    runner/judge code.
    """

    def __init__(self, default, by_model: Optional[Dict[str, object]] = None):
        self.default = default
        self.by_model = by_model or {}

    @property
    def base_url(self) -> str:
        return getattr(self.default, "base_url", "mock")

    def list_models(self) -> List[str]:
        return self.default.list_models()

    def chat(self, model: str, messages, temperature=0.0, max_tokens=700, strip_think=True) -> str:
        prov = self.by_model.get(model, self.default)
        return prov.chat(model, messages, temperature=temperature,
                         max_tokens=max_tokens, strip_think=strip_think)


def build_router(cfg, cache: ResponseCache, mock: bool = False):
    """Build the router, using run.default_provider as the fallback for unlabelled models.

    Previously this hardcoded groq as the default, which meant GROQ_API_KEY had to be set
    even for a sweep that never touched Groq.
    """
    if mock:
        return ProviderRouter(MockProvider(cache=None))

    providers: Dict[str, object] = {}

    def get(name: str):
        if name not in providers:
            providers[name] = build_provider(cfg, name, cache, mock=False)
        return providers[name]

    default_name = getattr(cfg.run, "default_provider", None) or "groq"
    default = get(default_name)

    by_model: Dict[str, object] = {}
    for m in cfg.target_models:
        name = m.provider or default_name
        if name != default_name:
            by_model[m.id] = get(name)

    judge_name = cfg.judges.provider or default_name
    if judge_name != default_name:
        judge_provider = get(judge_name)
        for jm in cfg.judges.models:
            by_model[jm] = judge_provider

    return ProviderRouter(default, by_model)
