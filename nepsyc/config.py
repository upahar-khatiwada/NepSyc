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
    provider: str = "groq"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    # Some hosted reasoning models emit <think> blocks; we strip them before judging.
    strip_think: bool = True


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


@dataclass
class RunCfg:
    language: str = "en"
    # None = derive from `language` as data/nepsyc_{language}.csv
    dataset: Optional[str] = None
    output_dir: str = "results"
    cache_path: str = ".cache/responses.jsonl"
    max_workers: int = 4
    requests_per_minute: int = 60
    behaviours: List[str] = field(default_factory=list)  # empty = all
    limit_per_behaviour: int = 0  # 0 = no limit


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
        p = dict(self.providers.get(name, {}))
        p.setdefault("base_url", "https://api.groq.com/openai/v1")
        p.setdefault("api_key_env", "GROQ_API_KEY")
        return p

    def api_key(self, provider: str) -> str:
        env = self.provider_settings(provider)["api_key_env"]
        key = os.environ.get(env, "")
        if not key:
            raise RuntimeError(
                f"Environment variable {env} is not set. "
                f"Copy .env.example to .env and fill it in, or export it in your shell."
            )
        return key


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
