"""Configuration loading for NepSyc."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class ModelSpec:
    id: str
    label: str
    # None = use run.default_provider. Only set this when a model lives somewhere other
    # than the default gateway.
    provider: Optional[str] = None
    # Reserved/unused: provider_settings() keys off `provider` (a name under `providers:` in
    # config.yaml) only, and never reads these two fields. Don't rely on setting them per-model.
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    # Some hosted reasoning models emit <think> blocks; we strip them before judging.
    strip_think: bool = True
    # Hugging Face Hub repo id (e.g. "Qwen/Qwen2.5-1.5B-Instruct"), for local hidden-state
    # extraction only (scripts/analyze_hidden_states.py, app/dashboard.py's "Local hidden-state
    # analysis" section). Distinct from `id` above: `id` is whatever the API gateway expects,
    # which for Groq and similar is not a loadable HF repo. None (the default) means this model
    # is not eligible for that feature -- true for any API-only model such as OpenAI's, which
    # has no local weights to load at all.
    hf_repo_id: Optional[str] = None


@dataclass
class GenerationCfg:
    temperature: float = 0.0
    max_tokens: int = 700
    system_prompt: Optional[str] = None


@dataclass
class JudgeCfg:
    models: List[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 500
    aggregate: str = "median"
    # Tasks that are cheap/objective enough for a single judge instead of the full panel.
    single_judge_tasks: List[str] = field(default_factory=lambda: ["correctness"])
    # None = route judge calls the same as target models (i.e. run.default_provider). Set to
    # a key under `providers:` in config.yaml to send every judge call to that gateway
    # instead, regardless of what target models are being evaluated.
    provider: Optional[str] = None


@dataclass
class RunCfg:
    language: str = "en"
    source: str | None = None
    # None = derive from `language` as data/nepsyc_{language}.csv
    dataset: Optional[str] = None
    output_dir: str = "results"
    cache_path: str = ".cache/responses.jsonl"
    # Which key under `providers:` serves any model that doesn't name a provider of its own.
    default_provider: str = "groq"
    max_workers: int = 4
    requests_per_minute: int = 60
    behaviours: List[str] = field(default_factory=list)  # empty = all
    limit_per_behaviour: int = 0  # 0 = no limit
    limit_total: int = 0  # 0 = no limit; hard cap on item count after all other filtering
    # False (default) = keep the first N items per behaviour, as CSV row order has it.
    # True = keep the last N instead, so a small smoke test can exercise a different
    # slice of each seed/authored file than the one every prior run already covered.
    limit_from_end: bool = False
    # ids or labels from target_models; empty = all (same convention as `behaviours` above)
    target_model_ids: List[str] = field(default_factory=list)


@dataclass
class Config:
    run: RunCfg
    generation: GenerationCfg
    judges: JudgeCfg
    target_models: List[ModelSpec]
    providers: Dict[str, Dict[str, Any]]

    @property
    def root(self) -> Path:
        return ROOT

    def provider_settings(self, name: str) -> Dict[str, Any]:
        if name in self.providers:
            p = dict(self.providers[name] or {})
            
            if not p.get("base_url"):
                raise RuntimeError(
                    f"Provider '{name}' in config.yaml has no `base_url`. Add one, e.g.\n"
                    f"  {name}:\n    base_url: https://...\n    api_key_env: {name.upper()}_API_KEY"
                )

            if p.get("api_type") == "azure" and not p.get("api_version"):
                raise RuntimeError(
                    f"Provider '{name}' is configured as Azure but has no `api_version`. "
                    f"Add e.g. `api_version: 2024-08-01-preview`."
                )

            p.setdefault("api_key_env", f"{name.upper()}_API_KEY")
            return p
        if name == "groq":
            return {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"}
        raise RuntimeError(
            f"Provider '{name}' has no block under `providers:` in config.yaml. Add one, e.g.\n"
            f"  {name}:\n    base_url: https://...\n    api_key_env: {name.upper()}_API_KEY"
        )

    def api_key(self, provider: str) -> str:
        env = self.provider_settings(provider)["api_key_env"]
        key = os.environ.get(env, "")
        if not key:
            raise RuntimeError(
                f"Environment variable {env} is not set (needed for provider '{provider}'). "
                f"Copy .env.example to .env and fill it in, or export it in your shell."
            )
        return key

    def provider_for(self, model: ModelSpec) -> str:
        """Which provider name actually serves this model entry."""
        return model.provider or self.run.default_provider


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    models = [ModelSpec(**m) for m in raw.get("target_models", [])]
    return Config(
        run=RunCfg(**raw.get("run", {})),
        generation=GenerationCfg(**raw.get("generation", {})),
        judges=JudgeCfg(**raw.get("judges", {})),
        target_models=models,
        providers=raw.get("providers", {}),
    )
