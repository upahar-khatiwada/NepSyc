"""Model access layer.

Everything speaks the OpenAI chat-completions schema, which is what Groq exposes at
https://api.groq.com/openai/v1 .  That means the exact same code points at OpenRouter,
Together, Fireworks or a local vLLM server by changing `base_url` in config.yaml --
useful because Groq no longer hosts Gemma or DeepSeek.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
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
    """Append-only JSONL cache. Re-running a sweep costs nothing for items already collected."""

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


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        cache: Optional[ResponseCache] = None,
        rpm: int = 60,
        timeout: int = 120,
        max_retries: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache = cache
        self.limiter = _RateLimiter(rpm)
        self.timeout = timeout
        self.max_retries = max_retries
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

            if r.status_code == 429:
                retry_after = float(r.headers.get("retry-after", 2 ** attempt))
                time.sleep(min(retry_after, 60))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 400 and "max_completion_tokens" in r.text:
                # older gateways only accept max_tokens
                payload.pop("max_completion_tokens", None)
                payload["max_tokens"] = max_tokens
                continue
            if not r.ok:
                raise RuntimeError(f"{model}: HTTP {r.status_code}: {r.text[:400]}")

            data = r.json()
            return data["choices"][0]["message"]["content"] or ""

        raise RuntimeError(f"{model}: exhausted retries. last error: {last_err}")


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
        sys = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""

        if self.JUDGE_MARK in sys:
            return self._mock_judge(sys, h)

        # target-model behaviour: sometimes agreeable, sometimes firm
        if h % 3 == 0:
            return "You're absolutely right, and I should have said so immediately. That is exactly correct."
        if h % 3 == 1:
            return "I have to disagree here. The evidence does not support that claim, and the accurate answer is different. Answer: B"
        return "That's an interesting point. There are arguments on both sides, though on balance I'd lean toward caution. Rating: 7/10. Answer: A"

    def _mock_judge(self, sys: str, h: int) -> str:
        if "TASK: agreement_bias" in sys:
            return json.dumps({"score": h % 6, "rationale": "mock"})
        if "TASK: delusion_acceptance" in sys:
            return json.dumps({"score": h % 6, "rationale": "mock"})
        if "TASK: stance" in sys:
            return json.dumps({"score": (h % 11) - 5, "rationale": "mock"})
        if "TASK: correctness" in sys:
            return json.dumps({"label": ["correct", "incorrect", "hedge"][h % 3], "rationale": "mock"})
        if "TASK: evaluation_positivity" in sys:
            return json.dumps({"score": h % 11, "error_flagged": bool(h % 2), "rationale": "mock"})
        if "TASK: agreement_level" in sys:
            return json.dumps({"score": h % 11, "rationale": "mock"})
        return json.dumps({"score": 0, "rationale": "mock"})


def build_provider(cfg, provider_name: str, cache: ResponseCache, mock: bool = False):
    if mock:
        return MockProvider(cache=None)
    settings = cfg.provider_settings(provider_name)
    return OpenAICompatProvider(
        base_url=settings["base_url"],
        api_key=cfg.api_key(provider_name),
        cache=cache,
        # Optional per-provider override (providers.<name>.requests_per_minute in
        # config.yaml) -- a hosted gateway like Gemini can have a much lower RPM quota
        # than run.requests_per_minute, which is really "how fast can I hit Groq".
        rpm=settings.get("requests_per_minute", cfg.run.requests_per_minute),
    )


class ProviderRouter:
    """Routes each model id to the provider that serves it.

    Lets one sweep mix Groq (llama, qwen, gpt-oss) with, say, OpenRouter (gemma,
    deepseek) without any change to runner/judge code.
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
    if mock:
        return ProviderRouter(MockProvider(cache=None))
    providers: Dict[str, object] = {}

    def get(name: str):
        if name not in providers:
            providers[name] = build_provider(cfg, name, cache, mock=False)
        return providers[name]

    default = get("groq")
    by_model = {}
    for m in cfg.target_models:
        if m.provider and m.provider != "groq":
            by_model[m.id] = get(m.provider)
    if cfg.judges.provider and cfg.judges.provider != "groq":
        judge_provider = get(cfg.judges.provider)
        for jm in cfg.judges.models:
            by_model[jm] = judge_provider
    return ProviderRouter(default, by_model)
