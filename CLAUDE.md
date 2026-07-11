# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NepSyc is a benchmark + evaluation harness for six sycophancy behaviours in LLMs, scored
across open-weight models served over the Groq API (or any OpenAI-compatible gateway). It
ships three parallel language splits: English (`en`), Nepali (`ne`, Devanagari), and
Romanized Nepali (`ne_rom`, Latin-script Nepali as typed by Nepali speakers texting).

There is no test suite, linter, or CI config in this repo. Correctness is checked by running
the pipeline itself (`--mock` mode) and by reading `results/nepsyc_summary_latest.txt`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                     # add GROQ_API_KEY (NOTE: .env.example is currently
                                          # missing from the repo -- see below)

python run.py build                      # seeds -> data/nepsyc_{en,ne,ne_rom}.csv (154 items each)
python run.py build --languages ne       # just one split
python run.py build --out foo.csv --languages ne   # --out only valid with a single language

python run.py check-models               # hits /v1/models live, prints OK/MISSING per configured id

python run.py evaluate --mock            # full pipeline offline, no API key, ~5s
python run.py evaluate                   # the real sweep, run.language from config.yaml
python run.py evaluate --language ne     # override run.language for one invocation
python run.py evaluate --behaviours agreement_bias mirroring --limit 5
python run.py evaluate --human data/human_annotations.csv   # adds Krippendorff's alpha

python scripts/convert_public_datasets.py --n-truthfulqa 200 --n-csqa 200
                                          # scales up the English seed pool from the real
                                          # TruthfulQA/CommonsenseQA (requires `pip install datasets`)
```

`--mock` uses a deterministic hash-based fake provider (`MockProvider` in `providers.py`) so
the whole pipeline — build, run, judge, score, report — can be exercised with no network
call and no API key. Use it after any change to `build_dataset.py`, `metrics.py`, `judge.py`,
or `report.py` to confirm nothing crashes before doing a real (costed) sweep.

**Known issue:** `.env.example` is referenced by the README/setup flow but is not currently
present in the repo. If you need it, recreate it with a single line: `GROQ_API_KEY=`.

## Architecture

### The central design decision: an item is a set of named conditions, not a prompt

Four of the six behaviours can only be measured as a *difference between two independent
conversations* (never two turns in one chat):

| behaviour | metric | conditions | signal |
|---|---|---|---|
| agreement bias | AGS 0..5 | `main` | judge score |
| delusion acceptance | DAS 0..5 | `main` | judge score |
| revision under pressure | RPS 0..5 | `pressure` (3 turns) | turn-wise correctness |
| mirroring | MRS −5..+5 | `stance_pro`, `stance_con` | stance(pro) − stance(con) |
| attribution bias | ATS −5..+5 | `self_authored`, `anonymous` | positivity(self) − positivity(anon) |
| authority influence | AIS −5..+5 | `self_opinion`, `authority_cue` | agree(auth) − agree(self) |

This is why `build_dataset.py` produces `item = {..., conditions: {name: {turns: [...]}}}`
rather than a flat prompt list, and why `runner.run_item()` always executes each named
condition as its own fresh message history. Scores are reported separately per behaviour,
never summed — a model can be robust under repeated disagreement (RPS ≈ 0) and still inflate
praise for the user's own writing (ATS ≫ 0); collapsing them would hide that structure.

### Pipeline data flow

```
data/seeds/*.csv, data/authored/*.csv   (hand-edited, per-language)
         |  build_dataset.build(language=...)
         v
data/nepsyc_<language>.csv              (generated, one row per turn — do not hand-edit)
         |  build_dataset.load()
         v
runner.collect()                        -> results/raw_responses.csv
         |  metrics.score_item() via judge.JudgePanel
         v
results/item_scores.csv, summary.json, results/nepsyc_summary_*.txt (report.py)
```

### Module map (`nepsyc/`)

- `config.py` — typed config loaded from `config.yaml`. `RunCfg.dataset` is `None` by default
  and derived at runtime as `data/nepsyc_<language>.csv`.
- `tables.py` — the CSV I/O layer all datasets go through. Two conventions carry structure CSV
  lacks: pipe-separated list columns (`correct_variants`), and dict columns split into named
  columns (`choice_a`..`choice_e`). A blank cell reads as `None`, never `""` — treating a blank
  `false_claim` as `""` would build a prompt asserting nothing instead of asserting nothing at all.
- `build_dataset.py` — expands seed/authored CSVs into items. All language-specific
  conversational wrapper text (the "quick check before I put this in my assignment..." kind of
  phrasing) lives in `TEMPLATES[language]`; seed/authored files are located by suffix
  (`factual_{language}.csv`, `delusion_{language}.csv`, ...). Adding a language means one
  `TEMPLATES` entry plus matching CSVs in `data/seeds/` and `data/authored/` — nothing else
  changes. One deliberate exception: the forced-answer anchors `'Answer: <letter>'` and
  `'Rating: X/10'` stay in English in every language, because `metrics.py`'s regexes match
  those literal English words for deterministic MCQ flip detection and the rating-delta
  cross-check — translating them would silently break both.
- `providers.py` — OpenAI-compatible chat client (`OpenAICompatProvider`), an append-only
  JSONL response cache keyed by `(base_url, model, messages, temperature, max_tokens)`
  (`ResponseCache`), a token-bucket-style rate limiter, a `MockProvider` for offline runs, and
  `ProviderRouter`, which lets one sweep mix providers (e.g. Groq for Llama/Qwen/GPT-OSS,
  OpenRouter for Gemma/DeepSeek) by routing each model id to whichever provider block it's
  tagged with in `config.yaml`. `strip_think` strips `<think>...</think>` blocks (Qwen3,
  DeepSeek-R1) before the judge ever sees a reply.
- `runner.py` — executes each condition's turns as one growing message history per
  conversation; writes `results/raw_responses.csv`.
- `judge.py` — `JudgePanel` runs N judge models against a fixed rubric per task and takes the
  median (Syco-bench / MT-Bench convention). Rubrics are English regardless of dataset
  language — a judge call passes (possibly Nepali) item content + reply into an English system
  prompt, on the assumption that judge models are multilingual enough to read the content and
  follow English grading instructions. `correctness` uses a single judge (objective enough),
  cutting ~40% of judge calls (`judges.single_judge_tasks` in config.yaml).
- `metrics.py` — per-item scoring per behaviour, plus aggregation: bootstrap CIs, Spearman,
  Krippendorff's alpha (for human-annotation agreement). `parse_answer_letter` /
  `parse_rating` are the deterministic (non-judge) cross-checks mentioned above.
- `report.py` — renders the `.txt` summary (`results/nepsyc_summary_<timestamp>.txt` and
  `..._latest.txt`), including the co-occurrence/correlation section and judge-reliability
  section (Section 5 is the only evidence judge scores are trustworthy).
- `cli.py` — `build` / `check-models` / `evaluate` subcommands; `run.py` is a one-line
  entrypoint into `nepsyc.cli.main()`.

### Seed vs. authored datasets

Both live under `data/` and both feed `build()`, but the label describes provenance, not
quality:

- **`data/seeds/`** — items adapted from, or written in the style of, an existing public
  benchmark (TruthfulQA for `factual_*.csv`, CommonsenseQA for `mcq_*.csv`) that ships a
  ready-made ground truth. Feeds agreement bias and revision-under-pressure only. Has a
  scale-up path: `scripts/convert_public_datasets.py` (English only — TruthfulQA/CommonsenseQA
  are English-only datasets).
- **`data/authored/`** — items with no public-benchmark equivalent (delusion, mirroring,
  attribution, authority), written from scratch because they need multi-turn structure, paired
  conditions, or an unfalsifiable belief that no existing dataset provides. No scale-up
  script; growing this set means writing more items by hand in the same style.

Every seed/authored file has three language copies (`_en`, `_ne`, `_ne_rom`) with identical
`seed_id`s and row counts — only the natural-language content differs.

### Validation philosophy

`tables.require()` and the seed loaders in `build_dataset.py` raise loudly at build time (a
`distractor_key` with no matching `choice_` column, a missing `false_claim`, an
`answer_key == distractor_key`) rather than silently producing a malformed prompt. This is
intentional: these CSVs are meant to be hand-edited, and hand-editing wants loud failures over
best-effort recovery. Preserve that pattern when touching the seed/authored loaders.
