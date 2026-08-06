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

All of the above, end to end, is `pipeline.run_evaluation()`; `cli.cmd_evaluate` (CLI) and
`app/dashboard.py` (Streamlit) are both just callers of it.

### Module map (`nepsyc/`)

- `config.py` — typed config loaded from `config.yaml`. `RunCfg.dataset` is `None` by default
  and derived at runtime as `data/nepsyc_<language>.csv`. `RunCfg.target_model_ids` is the
  same empty-means-all convention as `behaviours`, filtering `target_models` by id or label.
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
  cross-check — translating them would silently break both.
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
  section (Section 5 is the only evidence judge scores are trustworthy).
- `pipeline.py` — `run_evaluation(cfg, *, mock=False, human_file=None, progress=None) -> dict`
  is the evaluate pipeline (dataset build/load, provider routing, collect, judge, score,
  aggregate, write item_scores.csv / summary.json / judge_detail.csv / the .txt report) as an
  importable function, so a caller other than the CLI (e.g. a Streamlit dashboard) can run a
  sweep in-process and get `{items, scores, summary, report_text, paths}` back without
  shelling out or re-parsing files. Writes the exact same output files at the same paths as
  `cli.cmd_evaluate` always has -- this is a pure extraction, not a schema change. `progress`,
  if given, is called at coarse milestones (`dataset ready`, `responses collected`, `scoring
  done`, `report written`) with a `(fraction, message)` pair; `collect()`'s own tqdm bar is
  unchanged and still prints for CLI use regardless. `_limit_per_behaviour(items, n, from_end)`
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

### Dashboard (`app/dashboard.py`)

Streamlit UI over the same pipeline the CLI uses — additive only, no pipeline/config/output
schema changes. Sidebar builds a `Config` from `load_config()` plus widget values (target
models via `cfg.run.target_model_ids`, judge models/provider via `cfg.judges.*`, behaviours,
`limit_per_behaviour`, a "Take last N instead of first N" toggle wired to `limit_from_end` (CLI
equivalent: `--from-end`), and the mock toggle) and calls `pipeline.run_evaluation(cfg, mock=...,
progress=...)`; a `RuntimeError` from a provider (missing API key, exhausted retries) is caught
and shown as `st.error` instead of a traceback, with a pointer back to mock mode. Charts are
Plotly bar-with-CI, one per behaviour, respecting `report.DIRECTION` (0..5 vs. signed −5..+5,
never treating the two the same way); model→color is assigned once by declaration order so a
model keeps its color across every chart, per the dataviz skill's "color follows the entity,
never its rank."

"Languages" is a multiselect, not the single dropdown a single sweep's `cfg.run.language` might
suggest: picking more than one runs a separate `run_evaluation` call per language, each against
its own freshly-`load_config()`-ed `Config` (never a shared/mutated one — `run_evaluation`
writes the resolved dataset path back onto `cfg.run.dataset` as a side effect, so reusing one
`Config` object across languages would make every language after the first silently reuse the
first one's dataset). Output directory follows the same one-sweep-shouldn't-clobber-another
logic: the plain `results/` dir (unchanged) when exactly one language is selected, so "Load last
results" still lines up with a normal single-language run or a CLI run; `results/<language>/`
per language when more than one is selected, so they don't overwrite each other's CSVs.
Results land in `st.session_state["dash_results"]`, a `{language: {summary, paths, report_text,
meta}}` dict (`"Load last results"` populates it with the single sentinel key
`"(loaded from disk)"`); a "Viewing" radio appears whenever it holds more than one entry, letting
you switch which language's results the rest of the page renders — scores are still never
pooled across languages, matching the CLI's report caveat that AGS in `en` vs. `ne` has to be
diffed by hand, not averaged.

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

The item explorer and downloads read straight from the files `run_evaluation` already writes
(`item_scores.csv`, `raw_responses.csv`, `judge_detail.csv`, `summary.json`, the `.txt` report)
for whichever language is active — there is no separate in-memory result path, so "Run
benchmark" and "Load last results" render through the same code. The item explorer ("Prompt
inspector") filters behaviour first, then model, then a single scored item, because the driving
use case is "show me the prompt behind this model's AIS score" rather than browsing a flat
table; its "Scored item" dropdown label is otherwise just ids and scores (nothing language-
specific to localize), so each option also gets a `_preview_snippet` of that item's actual first
prompt (`preview_of`, a `{(model, item_id): turn}` lookup built once via `raw_df.groupby(["model",
"item_id"]).first()` on `raw_responses.csv` -- group-first preserves the file's own row order,
i.e. that item's first condition and turn) capped at 50 chars, so the picker reads in whatever
language the active run's dataset is in rather than only ever showing English. Once an item is
picked, it renders the exact conversation from `raw_responses.csv`'s `turn`/`reply` columns as
chat bubbles grouped by condition (`main`, `stance_pro`/`stance_con`, `self_opinion`/
`authority_cue`, ...), a colored score hero plus behaviour-specific badges parsed out of
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

### Shared dashboard helpers (`app/dash_common.py`)

Constants and pure-render helpers used by both `app/dashboard.py` and every page under
`app/pages/` live here, not in `dashboard.py` itself: the `CSS` block (`ns-*` classes), the
categorical color palette and `color_map()`, `SIGNED_METRICS`/`LANGUAGE_LABELS`,
`COND_LABELS`/`DETAIL_FIELDS`, and the html builders (`hero_html`, `badges_html`,
`conversation_html`, `judge_cards_html`, plus the annotation page's `prompts_only_html`/
`replies_only_html` split). This module has no top-level widget calls or `st.set_page_config`,
so importing it is side-effect free — the reason it's a plain module and not something
`dashboard.py` exposes for the pages to import, since importing a Streamlit *page* script
(as opposed to a helpers module) re-executes everything in it, sidebar included. Every page
that uses the `ns-*` markup must call `st.markdown(dash_common.CSS, unsafe_allow_html=True)`
itself — CSS injected by `dashboard.py` does not carry over to other pages, each gets a fresh
DOM on navigation.

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

### Validation philosophy

`tables.require()` and the seed loaders in `build_dataset.py` raise loudly at build time (a
`distractor_key` with no matching `choice_` column, a missing `false_claim`, an
`answer_key == distractor_key`) rather than silently producing a malformed prompt. This is
intentional: these CSVs are meant to be hand-edited, and hand-editing wants loud failures over
best-effort recovery. Preserve that pattern when touching the seed/authored loaders.
