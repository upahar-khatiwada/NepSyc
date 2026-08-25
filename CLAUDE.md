# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NepSyc is a benchmark + evaluation harness for six sycophancy behaviours in LLMs, scored
across open-weight models served over the Groq API (or any OpenAI-compatible gateway). It
ships three parallel language splits: English (`en`), Nepali (`ne`, Devanagari), and
Romanized Nepali (`ne_rom`, Latin-script Nepali as typed by Nepali speakers texting).

There is no linter or CI config in this repo, and the sycophancy pipeline itself has no test
suite — correctness there is checked by running the pipeline (`--mock` mode) and reading
`results/nepsyc_summary_latest.txt`. The one exception is the language competence probe
(`nepsyc/competence.py`, see below), which does have `tests/test_competence.py` (stdlib
`unittest`, no pytest dependency) covering its scoring/threshold logic and a mock end-to-end run.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                     # add the API key(s) for the provider(s) you use

python run.py build                      # seeds -> data/nepsyc_{en,ne,ne_rom}.csv (154 items each)
python run.py build --languages ne       # just one split
python run.py build --out foo.csv --languages ne   # --out only valid with a single language

python run.py check-models               # hits /v1/models live, prints OK/MISSING per configured id

python run.py evaluate --mock            # full pipeline offline, no API key, ~5s
python run.py evaluate                   # the real sweep, run.language from config.yaml
python run.py evaluate --language ne     # override run.language for one invocation
python run.py evaluate --behaviours agreement_bias mirroring --limit 5
python run.py evaluate --domain government_civics --mock   # scope to one or more
                                          # domains (item's `domain` field: general_knowledge /
                                          # everyday_reasoning / education / government_civics
                                          # today); default: all domains present in the dataset
python run.py evaluate --limit-total 1 --mock   # alias for --limit: N items PER behaviour,
                                          # not N total -- --limit-total 1 runs 1 item x
                                          # 6 behaviours (6 items), not 1 item total
python run.py evaluate --target-models Llama-3.1-8B --mock   # subset of target_models, by
                                          # id or label; empty (default) = all configured
python run.py evaluate --limit-total 2 --from-end --mock   # last 2 items per behaviour
                                          # instead of the first 2 -- covers a different
                                          # slice of each seed/authored file than the default
python run.py evaluate --human data/human_annotations.csv   # adds Krippendorff's alpha

python scripts/convert_public_datasets.py --n-truthfulqa 200 --n-csqa 200
                                          # scales up the English seed pool from the real
                                          # TruthfulQA/CommonsenseQA (requires `pip install datasets`)

python run.py competence --mock          # Nepali language competence probe (BLEU/chrF++),
                                          # fully separate from evaluate -- see below
python run.py competence --target-models Llama-3.1-8B
python -m unittest tests.test_competence -v   # the one test suite in this repo

streamlit run app/dashboard.py          # dashboard: pick models, run, see charts (needs
                                          # streamlit/plotly/pandas -- in requirements.txt)

# Representation-level analysis (sycophantic vs. neutral hidden states) -- a third standalone
# axis alongside evaluate/competence, see "Representation-level analysis" below.
python scripts/build_neutral_pairs.py            # data/nepsyc_* -> data/representation/neutral_*.csv
python scripts/validate_neutral_pairs.py         # asserts pairing integrity, prints coverage
python scripts/extract_representations.py --dry-run   # lists model x pair x variant counts, extracts nothing
python scripts/extract_representations.py --model Qwen2.5-1.5B --limit 2 --attn-layers none
                                          # only target_models with hf_repo_id set are eligible;
                                          # writes results/representations/ (gitignored)
python scripts/analyze_representation_drift.py   # pure reader of results/representations/index.csv;
                                          # writes data/representation/metrics/{layer_cosine,layer_agg,...}
python scripts/analyze_representation_research.py     # optional PCA/direction/CKA/RuP-drift/fertility panels
python scripts/make_representation_report_figures.py  # static PNGs for docs/REPRESENTATION_LEARNING_REPORT.md
```

`--mock` uses a deterministic hash-based fake provider (`MockProvider` in `providers.py`) so
the whole pipeline — build, run, judge, score, report — can be exercised with no network
call and no API key. Use it after any change to `build_dataset.py`, `metrics.py`, `judge.py`,
or `report.py` to confirm nothing crashes before doing a real (costed) sweep.

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

All of the above, end to end, is `pipeline.run_evaluation()`; `cli.cmd_evaluate` (CLI) and
`app/dashboard.py` (Streamlit) are both just callers of it.

### Module map (`nepsyc/`)

- `config.py` — typed config loaded from `config.yaml`. `RunCfg.dataset` is `None` by default
  and derived at runtime as `data/nepsyc_<language>.csv`. `RunCfg.target_model_ids` is the
  same empty-means-all convention as `behaviours`, filtering `target_models` by id or label.
  `RunCfg.domains` (empty = all) is the same convention again, filtering items by their
  `domain` field (`--domain` on the CLI, a "Domains" multiselect in the dashboard).
  `RunCfg.limit_from_end` (default `False`) flips `limit_per_behaviour` / `limit_total` from
  keeping each behaviour's first N items to keeping its last N, so a small sweep can be pointed
  at a slice of the dataset a prior small sweep hasn't already covered.
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
  cross-check — translating them would silently break both. Every seed/authored row also
  carries an *optional* `domain` column (`base(..., domain=s.get("domain"))`): free text like
  `topic`, but unlike `topic` it's actually filterable (`--domain` / `cfg.run.domains` /
  the dashboard's "Domains" multiselect) and shows up in the report header. A row with no
  `domain` column, or a blank cell, falls back to the module-level `DOMAIN` constant
  (`"general_knowledge"`). Every bundled seed/authored row sets `domain` explicitly, split into
  four categories: `general_knowledge` (science/history/geography misconceptions and expert-claim
  items with no Nepal-government or education-policy angle), `everyday_reasoning` (the
  commonsense MCQ items in `mcq_*.csv`), `education` (school/university policy, pedagogy, and
  academic-context items), and `government_civics` (Nepal government, constitution, law, and
  policy items — identifiable by their `nepal_*`/`*_policy` topic values). Adding a new domain,
  or moving items between existing ones, is purely a data change (edit the `domain` column on the
  rows that belong to it in `data/seeds/*.csv` / `data/authored/*.csv`, keeping the three language
  copies of a file in sync by `seed_id` since domain is a content category, not language-specific
  text), never a code change. `--domain <name>` (CLI) / `cfg.run.domains` / the dashboard's
  "Domains" multiselect then scope a sweep to exactly that category.
- `providers.py` — OpenAI-compatible chat client (`OpenAICompatProvider`), an append-only
  JSONL response cache keyed by `(base_url, model, messages, temperature, max_tokens)`
  (`ResponseCache`), a token-bucket-style rate limiter, a `MockProvider` for offline runs, and
  `ProviderRouter`, which lets one sweep mix providers (e.g. Groq for Llama/Qwen/GPT-OSS,
  OpenRouter for Gemma/DeepSeek) by routing each model id to whichever provider block it's
  tagged with in `config.yaml`. `strip_think` strips `<think>...</think>` blocks (Qwen3,
  DeepSeek-R1) before the judge ever sees a reply. OpenAI and Gemini are first-class providers
  here too — both expose OpenAI-compatible endpoints, so the same `OpenAICompatProvider`
  handles them with no separate client; set `provider: openai` / `provider: gemini` on a
  `target_models` entry (`config.yaml`) or `judges.provider` to route there, given
  `OPENAI_API_KEY` / `GEMINI_API_KEY` in `.env`.
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
  section (Section 5 is the only evidence judge scores are trustworthy). The header line reads
  the domain(s) actually present in `items` (`sorted({i["domain"] for i in items})`) rather
  than a hardcoded "Education & General Knowledge" string, so it reflects whatever `--domain`
  scoped the sweep to.
- `pipeline.py` — `run_evaluation(cfg, *, mock=False, human_file=None, progress=None) -> dict`
  is the evaluate pipeline (dataset build/load, provider routing, collect, judge, score,
  aggregate, write item_scores.csv / summary.json / judge_detail.csv / the .txt report) as an
  importable function, so a caller other than the CLI (e.g. a Streamlit dashboard) can run a
  sweep in-process and get `{items, scores, summary, report_text, paths}` back without
  shelling out or re-parsing files. Writes the exact same output files at the same paths as
  `cli.cmd_evaluate` always has -- this is a pure extraction, not a schema change. `progress`,
  if given, is called at coarse milestones (`dataset ready`, `responses collected`, `scoring
  done`, `report written`) with a `(fraction, message)` pair; `collect()`'s own tqdm bar is
  unchanged and still prints for CLI use regardless. Domain filtering (`if cfg.run.domains:
  items = [i for i in items if i["domain"] in cfg.run.domains]`) runs right after the
  behaviour filter and before either limit, so `--limit`/`--limit-total` count items within
  the already-domain-scoped set. `_limit_per_behaviour(items, n, from_end)`
  backs both `limit_per_behaviour` and `limit_total`: it keeps each behaviour's first N items,
  or its last N when `cfg.run.limit_from_end` is set, selecting by `id()` so the output keeps
  `items`' original relative order rather than regrouping by behaviour. `list_configured_models(cfg)`
  reads `target_models` / `judges` / `providers` out of config with no network call, for populating
  a model-selection UI before a run.
- `cli.py` — `build` / `check-models` / `evaluate` / `competence` subcommands; `run.py` is a
  one-line entrypoint into `nepsyc.cli.main()`. `cmd_evaluate` only parses args into `cfg`
  (including `--target-models`, which sets `cfg.run.target_model_ids`) and calls
  `pipeline.run_evaluation` — the pipeline itself lives in `pipeline.py`. `cmd_competence`
  is the same shape, calling `competence.run_competence_sweep`.
- `competence.py` — the Nepali language competence probe: a standalone axis, never combined
  with AGS/DAS/RPS/MRS/ATS/AIS. See "Language competence probe" below.

### Language competence probe (`nepsyc/competence.py`)

A model that scores badly on the six sycophancy behaviours in the `ne`/`ne_rom` splits might
simply not understand Nepali well — that confound is a stated limitation of the benchmark
proposal. This module measures it directly, as a **standalone axis**: it is never combined
with AGS/DAS/RPS/MRS/ATS/AIS, has its own probe set, own CSVs, own CLI subcommand
(`python run.py competence`), and own dashboard section — no changes anywhere to
`build_dataset.py`, `runner.py`, `pipeline.py`, `data/nepsyc_*.csv`, `raw_responses.csv`, or
`item_scores.csv`.

```
data/seeds/competence_probes.csv   (24 hand-authored rows: direction, script, source_text,
    |                                reference_texts, notes -- one file, not per-language,
    |                                since `script`/`direction` columns already cover both
    |                                Nepali scripts and all three probe types)
    |  load_probe_set()
    v
run_competence_sweep(cfg)          -- reuses providers.build_router(), the exact client
    |                                  evaluate uses (same ResponseCache, same MockProvider
    |                                  for --mock), never a second client
    v
competence_scores.csv              (long: one row per model/item/metric)
competence_summary.csv             (one row per model per {en_to_ne, ne_to_en, comprehension,
                                     overall}; `verdict` populated only on the overall row)
```

`direction` is `en_to_ne` | `ne_to_en` | `comprehension`; `script` is `devanagari` |
`romanized`. Scoring is sacreBLEU BLEU + chrF++ (`score_reply()`) against
`reference_texts` (pipe-separated, multiple references supported natively by sacrebleu).
chrF++ (`sacrebleu.CHRF(word_order=2)`) is the primary signal — character n-grams reward a
near-miss on Nepali verb morphology that word-level BLEU zeroes out. BLEU's tokenizer is
chosen per row by `_bleu_tokenize_for(direction, script)`: `intl` when the *reference* text is
Devanagari (sacreBLEU's default `13a` under-segments it), `13a` otherwise — `ne_to_en` is
always `13a` since its reference is English regardless of the source's `script`. Each scored
row carries its own sacreBLEU signature string, computed after scoring (calling
`get_signature()` before any `sentence_score()` call raises, since sacrebleu needs to see the
reference count first).

`verdict_for(bleu, chrfpp, comp_cfg)` maps thresholds from `config.yaml`'s `competence:` block
(`understands_chrfpp_min`, `understands_bleu_min`, `partial_chrfpp_min` — not hardcoded) to
`Understands` / `Partial` / `Poor`: `Understands` needs *both* metrics to clear their minimums,
`Partial` needs only chrF++ to clear its own lower minimum. Computed only on the `aggregate()`
overall row (pooled across all three directions) — the per-direction rows carry no verdict,
matching the dashboard's "headline badge, breakdown on expand" design. `write_scores()`/
`load_scores()` are a plain per-row parse (no group-and-merge like `build_dataset.load()`
needs): every field on a `competence_scores.csv` row is independently meaningful, unlike
`grading_json`, which is heavy enough to only carry on an item's first row.

`run_competence_sweep(cfg, *, mock=False)` filters target models via
`pipeline._filter_target_models(cfg)` — same `cfg.run.target_model_ids` convention as
`evaluate`, imported directly from `pipeline.py` rather than reimplemented.

### Representation-level analysis (sycophantic vs. neutral hidden states)

A third standalone axis alongside `evaluate`/`competence`: compares a model's internal hidden
states under an existing sycophancy-framed condition against a matched **neutral** (unframed)
counterpart, layer by layer, for open-weight models only. Additive throughout — no change to
`build_dataset.py`, `data/nepsyc_*.csv`, `runner.py`, or `pipeline.py`. Full writeup with actual
computed numbers: `docs/REPRESENTATION_LEARNING_REPORT.md`; design rationale and what was found
to already exist vs. missing before this was built: `docs/REPRESENTATION_ANALYSIS_PLAN.md`.

```
data/nepsyc_{en,ne,ne_rom}.csv (untouched)
    |  nepsyc/neutral_pairs.py -- one FIXED mechanical rule per behaviour (drop the claim/
    |  stance/authority cue, never a hand-rewrite): scripts/build_neutral_pairs.py
    v
data/representation/neutral_{en,ne,ne_rom}.csv, pairs_manifest.csv   (705 pairs, all 3
    |                                                                  languages, validated by
    |                                                                  scripts/validate_neutral_pairs.py:
    |                                                                  no orphans either direction,
    |                                                                  behaviour/domain/language match)
    |  scripts/extract_representations.py -- only target_models with `hf_repo_id` set in
    |  config.yaml are eligible; nepsyc/neutral_templates.py (a SEPARATE neutral-template set,
    |  extraction-path-only) supplies the neutral condition's turns
    v
results/representations/{<model>/.../tensors.npz, index.csv, runs/<run_id>.json}   (gitignored,
    |                                                                                like results/)
    |  scripts/analyze_representation_drift.py -- pure reader, no model load
    v
data/representation/metrics/{layer_cosine.csv/.parquet, layer_agg.csv, layer_ranking.csv,
                              model_layer_matrix_*.csv, summary.json}   (committed)
    |  scripts/analyze_representation_research.py -- optional, SEPARATE script (never edits
    |  analyze_representation_drift.py's five files above); five extra panels
    v
data/representation/metrics/research_{pca_layers,directions,direction_stability,cka,
                                        rup_drift,fertility}.csv, research_directions.npz
```

`nepsyc/representation.py` is the extraction/metric core (kept separate from
`nepsyc/hidden_states.py`, whose last-token-only, no-attention output shape is a documented
dependency of `scripts/analyze_hidden_states.py` and the Prompt Inspector's own "Local
hidden-state analysis" sub-block — changing its return shape there would break both). Two
pooling conventions, computed at every layer for every turn: `last_token` (final-token hidden
state from a prompt-only forward pass — the pre-answer state) and `mean_pooled` (mean over the
generated reply's token positions, from a teacher-forced prompt+reply pass — how the model
represents its own answer). `cosine_similarity(a, b)` and `linear_cka(X, Y)` (Gram-matrix/HSIC
form, cheap when hidden_dim exceeds sample count) both return NaN on a degenerate input (zero
vector; n<2) rather than raising — every downstream table treats NaN as "flagged", not a crash.
`confidence_logit_metrics()` reads the softmax/logit gap between the correct and incorrect answer
tokens right after the prompt (mcq: answer-key vs. distractor-key letter; open: first token of
`correct_answer` vs. `false_claim`/`incorrect_answer`) — `None` for `paired`-mode items
(mirroring/attribution_bias/authority_influence), which have no single correct answer.

Per-behaviour neutral-turn rule (`nepsyc/neutral_pairs.py`'s docstring is the canonical source):
drop the confident false claim (`agreement_bias`), keep `pressure`'s own turn 0 verbatim
(`revision_under_pressure`), drop the `belief` prefix but keep the `ask` (`delusion_acceptance`),
drop the "I strongly believe" opener (`mirroring`), a new neutral template with no
self-authorship claim (`attribution_bias` — see below), a new neutral template with no "I think"
ownership framing and no authority citation (`authority_influence`). **`attribution_bias` is the
one exception with no separate neutral item at all**: its existing `anonymous` condition already
carries no self-authorship claim, so it doubles as the neutral proxy
(`is_neutral_proxy=True` in `results/representations/index.csv`,
`NO_NEUTRAL_NEEDED` in `nepsyc/neutral_templates.py`) — every downstream reader
(`analyze_representation_drift.py`, `analyze_representation_research.py`) resolves "which variant
is neutral for this pair" the same way: the literal `"neutral"` variant if one was extracted,
else whichever variant is flagged `is_neutral_proxy`.

**As of the last extraction run** (`results/representations/`, gitignored — the numbers below are
current only until someone re-runs or extends it): 1 model (`Qwen2.5-1.5B-Instruct`, added to
`config.yaml`'s `target_models` specifically for this, because the originally-planned
`GPT-OSS-20B` needs ~80GB RAM at this CPU-only build's required float32 and doesn't fit this
machine's 16GB), English only, 11 item-pairs across all 6 behaviours (`attribution_bias` has
exactly 1). The 705-pair pool in `data/representation/` (all 3 languages) is ready for a larger
run whenever more compute or another `hf_repo_id`-bearing model is available; extending coverage
needs zero code changes, just re-running `extract_representations.py` with a wider
`--language`/`--model` selection. Real, computed findings (largest-divergence layers, CKA,
direction stability, fertility correlation) are in `docs/REPRESENTATION_LEARNING_REPORT.md` —
notably, no model currently has *both* representation data and sycophancy-judge scores, so a
representation-drift-vs-judge-score relationship is not yet measurable at all (a structural gap,
not a weak result).

Dashboard page: the Representational Learning page (`app/pages/3_Representational_Learning.py`,
see below) follows the Language Competence precedent (own place in the nav, pure reader of
precomputed files, never triggers extraction itself) with a "Research panels (optional)"
sub-section reading the five `research_*.csv` files independently — each panel's own caption
states its `n` so a low-sample result is never presented as if it were settled.

The one place extraction *can* be triggered from inside the app is the main dashboard's sidebar
(`app/dashboard.py`, not this page): "Representational analysis" → "Auto-extract after the
sweep" (default on, disabled in Mock mode). After a live `run_evaluation` sweep, it filters the
sweep's selected target models down to whichever have `hf_repo_id` set (the groq/local
open-weight models eligible for this feature at all) and calls
`scripts.extract_representations.run_extraction()` for them, passing `items_by_language` —
the literal `items` list(s) `run_evaluation()` just scored for each language, accumulated
across every `(language, domain)` combo the sweep ran — so extraction covers exactly the
prompts the sweep benchmarked rather than an independently reselected slice; `run_extraction`'s
own `behaviours`/`limit` reselection (still what the CLI uses by default) is bypassed entirely
whenever `items_by_language` is given. Then it calls
`scripts.analyze_representation_drift.run_drift_analysis()` to refresh
`data/representation/metrics/`, then `st.cache_data.clear()` so the Representational Learning
page picks up the new model without a manual terminal step or app restart. Both scripts were
split CLI-driver/callable-core the same way `pipeline.run_evaluation` was split from
`cli.cmd_evaluate`: `run_extraction()` in `scripts/extract_representations.py` is `main()`'s
former body, and `run_drift_analysis()` in `scripts/analyze_representation_drift.py` is
`main()`'s former body — `main()` in each is now a thin CLI wrapper, so both remain runnable
standalone exactly as documented above. `run_extraction()` catches one model's load/extraction
failure (OOM, a checkpoint too large for this machine — e.g. GPT-OSS-20B/120B per their
`config.yaml` comments — a network error fetching the weights) per model rather than letting it
abort the rest, since the dashboard may hand it several models unattended; failures land in the
returned `model_errors` list and are shown as `st.warning`s rather than crashing the app.
Because `run_drift_analysis()` regenerates `data/representation/metrics/*.csv` from whatever
`results/representations/index.csv` exists locally (gitignored) rather than merging with the
previously-committed version, running this against a fresh clone whose `index.csv` doesn't yet
include a previously-extracted model (e.g. the committed metrics' `Qwen2.5-1.5B` numbers, from a
prior extraction run on a different machine) will locally overwrite those committed CSVs to
contain only the model(s) just re-extracted, until that other model is re-extracted too —
`data/representation/` (unlike `results/`) is git-tracked on purpose (the neutral-pair pool and
these aggregate metrics are meant to be shared via clone; only the raw per-model tensors under
`results/representations/` are gitignored for size), so this shows up as a normal working-tree
diff on those files after any live sweep with auto-extraction on, not a bug.

### Dashboard (`app/dashboard.py`) — entry point, sidebar, Status/Prompt inspector

Streamlit UI over the same pipeline the CLI uses — additive only, no pipeline/config/output
schema changes. The app is a multi-page Streamlit app now: `app/dashboard.py` is still the
entry point (`streamlit run app/dashboard.py`) and owns the sidebar (so it's the only page
that can trigger a run), but two sections that used to live on it were pulled out to their own
pages as the page grew — per-behaviour scoring to `app/pages/2_Scoring.py` and representational
learning to `app/pages/3_Representational_Learning.py` (both described below). What's left on
the main page after a run: the "Status" gauges, a "Reading" section (plain-language claims the
run does/doesn't support), the "Prompt inspector" (walk one behaviour → model → item down to
the actual conversation, including a "Local hidden-state analysis" sub-block), a "Files"
section with download buttons, and two short stub sections pointing at the Scoring and
Representational Learning pages for anyone who remembers the old single-page layout.

Sidebar builds a `Config` from `load_config()` plus widget values (target models via
`cfg.run.target_model_ids`, judge models/provider via `cfg.judges.*`, behaviours,
`limit_per_behaviour`, a "Take last N instead of first N" toggle wired to `limit_from_end` (CLI
equivalent: `--from-end`), and the mock toggle) and calls `pipeline.run_evaluation(cfg, mock=...,
progress=...)`; a `RuntimeError` from a provider (missing API key, exhausted retries) is caught
and shown as `st.error` instead of a traceback, with a pointer back to mock mode. Right after
that (still inside the "Run benchmark" click handler), if the sweep was live and the sidebar's
"Representational analysis" → "Auto-extract after the sweep" toggle is on, `_run_auto_
representational()` runs `scripts.extract_representations.run_extraction()` +
`scripts.analyze_representation_drift.run_drift_analysis()` for whichever selected target
models have `hf_repo_id` set — see "Representation-level analysis" above for the full behaviour
and its caveats (per-model failure handling, local metrics files getting overwritten rather
than merged).

"Languages" and "Domains" are both multiselects, not the single dropdowns a single sweep's
`cfg.run.language` / `cfg.run.domains` might suggest, and they compose: picking N languages ×
M domains runs N×M separate `run_evaluation` calls (one per `(language, domain)` combo), each
against its own freshly-`load_config()`-ed `Config` (never a shared/mutated one — `run_evaluation`
writes the resolved dataset path back onto `cfg.run.dataset` as a side effect, so reusing one
`Config` object across combos would make every combo after the first silently reuse the first
one's dataset). `_discover_domains()` (`st.cache_data`) unions whichever `domain` values are
actually present across whatever `data/nepsyc_{en,ne,ne_rom}.csv` files already exist on disk —
never hardcoded — falling back to `build_dataset.DOMAIN` if no dataset has been built yet, so
the multiselect always has at least one option. Output directory follows the same
one-sweep-shouldn't-clobber-another logic as before, generalized to two dimensions: the plain
`results/` dir (unchanged) when exactly one combo is selected, so "Load last results" still
lines up with a normal single-combo run or a CLI run; otherwise `results/<parts>` where `parts`
includes only whichever dimension(s) actually vary (a language-only multi-run still lands in
exactly `results/<lang>`, as before domains existed).

Results land in `st.session_state["dash_results"]`, a `{key: {summary, paths, report_text,
meta}}` dict keyed by `language` when only one domain is selected (unchanged from before
domains existed) or by `f"{language}__{domain}"` when more than one domain is in play;
`"Load last results"` populates it by scanning for every `summary.json` under `results/`
(`_discover_result_summaries()`, not just the plain `results/summary.json`), so it also finds
whatever a prior multi-language/multi-domain sweep wrote to `results/<lang>`, `results/<domain>`,
or `results/<lang>/<domain>` — the button is disabled only when none exist anywhere. Each is
keyed by its directory's path relative to `results/` (`"(root)"` for the plain
`results/summary.json`), with language/domain best-effort inferred from that path for display.
A "Viewing" radio appears whenever it holds more than one entry, using `dash_common.result_label
(meta, key)` to render each option as `"{language label} ({lang}) · {domain}"` (or just the
bare key when no language could be inferred, e.g. `"(root)"`) — scores are still never pooled across languages *or*
domains, matching the CLI's report caveat that AGS in `en` vs. `ne`, or in one domain vs.
another, has to be diffed by hand, not averaged.

Judge model options follow the judge provider, not a fixed list: `_judge_model_options(cfg,
info, provider)` returns `judges.models` from config.yaml as-is only when `provider` matches
`cfg.judges.provider` (that list was hand-curated to avoid a judge grading its own target's
replies, so pulling in target model ids here for the matching-provider case would quietly
reintroduce that overlap); for any other provider it offers target model ids already routed
there via `target_models[].provider`, plus `gemini-2.5-flash` as a starting point for `gemini`
specifically (mirroring the `--gemini-judge` CLI shorthand) since no target may use it yet. The
"Judge models" multiselect's widget `key` includes the provider (`judge_models_<provider>`), so
switching provider resets the selection to that provider's full option set instead of carrying
over model ids that belonged to whichever provider was picked before.

The Prompt inspector and downloads read straight from the files `run_evaluation` already writes
(`item_scores.csv`, `raw_responses.csv`, `judge_detail.csv`, `summary.json`, the `.txt` report)
for whichever combo is active — there is no separate in-memory result path, so "Run
benchmark" and "Load last results" render through the same code. The Prompt inspector filters
behaviour first, then model, then a single scored item, because the driving
use case is "show me the prompt behind this model's AIS score" rather than browsing a flat
table; its "Scored item" dropdown label is otherwise just ids and scores (nothing language-
specific to localize), so each option also gets a `_preview_snippet` of that item's actual first
prompt (`preview_of`, a `{(model, item_id): turn}` lookup built once via `raw_df.groupby(["model",
"item_id"]).first()` on `raw_responses.csv` -- group-first preserves the file's own row order,
i.e. that item's first condition and turn) capped at 50 chars, so the picker reads in whatever
language the active run's dataset is in rather than only ever showing English. Once an item is
picked, it renders the exact conversation from `raw_responses.csv`'s `turn`/`reply` columns as
chat bubbles grouped by condition (`main`, `stance_pro`/`stance_con`, `self_opinion`/
`authority_cue`, ...) -- each turn's text run through `dash_common.render_markdown()` (a
`markdown-it-py` instance with the `gfm_plugin` for tables, `html=False` so any literal
`<script>`/HTML a prompt or an adversarial reply contains gets escaped rather than passed
through into the `unsafe_allow_html=True` block downstream) so a reply's own bold/headings/
tables/rules actually render instead of showing up as literal `**`/`##`/`---` characters --
a colored score hero plus behaviour-specific badges parsed out of
`detail_json` (`DETAIL_FIELDS` in `app/dash_common.py`), and the judge panel's votes from
`judge_detail.csv` as rationale cards with the raw grading prompt available in an expander — no
new files or schema, purely a different read of the same CSVs. Needs `streamlit` / `plotly` /
`pandas`, listed in `requirements.txt` alongside the CLI's dependencies.

A separate "Language Competence" section (own sidebar button "Run language competence check",
own `st.session_state["dash_competence"]` key, independent of `dash_results`/`dash_lang_view`)
runs/reads the probe from "Language competence probe" above: a color-coded verdict badge per
model (`_competence_badge_html`, reusing the `.ns-gauge` CSS class from the Status gauges) as
the headline, with a per-model expander showing the BLEU/chrF++ breakdown by direction. It
renders unconditionally, above the `dash_results` branch, since it has no dependency on whether
a sycophancy sweep has been run or which language is being viewed.

### Scoring page (`app/pages/2_Scoring.py`)

Pulled off the main page once it grew too long: the headline scores table (models × the six
metrics), coverage, collection/judging health, and one Plotly bar chart per behaviour with 95%
bootstrap CIs — the 0..5 "higher = more sycophantic" metrics (AGS/DAS/RPS) and the signed
−5..+5 difference metrics (MRS/ATS/AIS) are charted differently, per `report.DIRECTION`, never
treated the same way. Model→color is assigned once by declaration order (`color_map()`) so a
model keeps its color across every chart, per the dataviz skill's "color follows the entity,
never its rank." A pure reader of `st.session_state["dash_results"]` — it never runs a sweep
itself; if `dash_results` isn't set yet it points back at the main page and stops, same pattern
as the Human Annotation page. Same "Viewing" radio (`dash_common.result_label`) as the main
page when more than one `(language, domain)` combo is loaded.

### Representational Learning page (`app/pages/3_Representational_Learning.py`)

Also pulled off the main page: sycophantic-vs-neutral hidden-state analysis (see
"Representation-level analysis" above), plus the five optional research panels. Unlike the
Scoring page, this one does **not** read `st.session_state["dash_results"]` at all — it's a
third standalone axis, so it goes straight to `data/representation/metrics/` and
`results/representations/` on disk, independent of whether any sycophancy sweep has been run.
Filters (model, behaviour, domain, language, prompt pair) drive which precomputed rows get
plotted; nothing on this page loads model weights or triggers extraction — that only ever
happens via `scripts/extract_representations.py`, run separately, offline.

### Shared dashboard helpers (`app/dash_common.py`)

Constants and pure-render helpers used by `app/dashboard.py` and every page under
`app/pages/` live here, not in `dashboard.py` itself: the `CSS` block (`ns-*` classes), the
categorical color palette and `color_map()`, `SIGNED_METRICS`/`LANGUAGE_LABELS`,
`COND_LABELS`/`DETAIL_FIELDS`, the html builders (`hero_html`, `badges_html`,
`conversation_html`, `judge_cards_html`, plus the annotation page's `prompts_only_html`/
`replies_only_html` split), and — moved here in the page-split that produced the Scoring and
Representational Learning pages, since both those pages need them too — the coverage/health
helpers (`coverage_summary`, `collection_health`, `judge_health`, `entry`, `status_color`,
`meter_html`) that used to be private to `dashboard.py`, plus `result_label(meta, key)` (the
`"{language label} ({lang}) · {domain}"` formatter for a `dash_results` key, used by every page
that renders the "Viewing" radio). This module has no top-level widget calls or
`st.set_page_config`, so importing it is side-effect free — the reason it's a plain module and
not something `dashboard.py` exposes for the pages to import, since importing a Streamlit
*page* script (as opposed to a helpers module) re-executes everything in it, sidebar included.
Every page that uses the `ns-*` markup must call `st.markdown(dash_common.CSS,
unsafe_allow_html=True)` itself — CSS injected by `dashboard.py` does not carry over to other
pages, each gets a fresh DOM on navigation.

### Human annotation page (`app/pages/1_Human_Annotation.py`)

A second page, reachable from the sidebar nav Streamlit generates automatically for anything
under `app/pages/` (classic pages-directory convention — no `st.navigation`/`st.Page` call
needed, and `dashboard.py` stays the entry point run via `streamlit run app/dashboard.py`).
Built for comparing models and annotating by hand, which the main page's "Prompt inspector"
doesn't do well: that one filters down to a single (behaviour, model, item) triple, so
comparing how three models answered the *same* prompt means re-picking the model and losing
your place. This page instead picks a (behaviour, item) pair — independent of model — shows
the shared prompt once (`prompts_only_html`, pulled from whichever selected model has it in
`raw_responses.csv`, since the turns are identical across models for one item), then renders
one column per model (native `st.container(border=True)`, not a hand-rolled HTML div — a div
opened in one `st.markdown` call and closed in another does not actually nest around
Streamlit-native elements written in between, they render as sibling DOM nodes) holding that
model's own replies (`replies_only_html`), its score for the behaviour (`hero_html`, compact
form), behaviour-specific badges, and its judge-panel votes/rationale from `judge_detail.csv`
(`judge_cards_html`) with the grading prompt in an expander. Above the columns, a compact
`ns-strip` of gauges gives the score for every selected model at a glance before drilling into
any one of them. More than four selected models switches from columns to `st.tabs` so each
model's block stays full width instead of being squeezed. Reads `st.session_state["dash_results"]`
and `st.session_state["dash_lang_view"]`, written by `dashboard.py`'s "Run benchmark"/"Load last
results" — session state is shared across pages in one browser session, so this page never runs
a sweep itself; if `dash_results` isn't set yet it just points back at the main page and stops.

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

Both kinds also accept the same optional `domain` column (see `build_dataset.py` above) —
domain is orthogonal to seed-vs-authored: a new domain can mix rows from either kind of file,
same as the six behaviours already do.

### Validation philosophy

`tables.require()` and the seed loaders in `build_dataset.py` raise loudly at build time (a
`distractor_key` with no matching `choice_` column, a missing `false_claim`, an
`answer_key == distractor_key`) rather than silently producing a malformed prompt. This is
intentional: these CSVs are meant to be hand-edited, and hand-editing wants loud failures over
best-effort recovery. Preserve that pattern when touching the seed/authored loaders.
