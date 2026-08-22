# Representation-level analysis: repo map and design plan

Investigation only — no code changed. This document maps what exists today and proposes a
minimal-footprint path to the proposal's "representation-level analysis module" (paired
sycophantic/non-sycophantic prompts, hidden-state extraction, cosine similarity, and a
dashboard section). Written 2026-08-22 against branch `feat/representational`.

**Headline finding:** most of this module already exists. `nepsyc/hidden_states.py`,
`scripts/analyze_hidden_states.py`, and a "Local hidden-state analysis" block inside
`app/dashboard.py`'s Prompt Inspector already do local extraction, per-layer L2 norm, and a
per-model PCA trajectory, wired end to end. What's genuinely missing is narrower than the
proposal implies: a non-sycophantic paired prompt condition, a cosine-similarity computation
between paired representations, and promoting the feature from an item-inspector sub-block
into its own dashboard section. See §3 and §4.

---

## 1. Configs — models

Single file: [config.yaml](../config.yaml). Loaded by `nepsyc/config.py:152` (`load_config`)
into a `Config` dataclass (`nepsyc/config.py:99-106`).

**`target_models:`** (`config.yaml:73-92`), the models being evaluated for sycophancy — list of
`ModelSpec` (`nepsyc/config.py:16-35`): `id` (API-served id), `label`, `provider` (optional,
defaults to `run.default_provider`), `strip_think` (bool, strips `<think>` blocks before
judging), `hf_repo_id` (optional, HF Hub repo id for **local** weight loading — see §4).

| label | id | provider | open-weight? | `hf_repo_id` set? |
|---|---|---|---|---|
| Qwen3.6-35B | `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` | `ank` (default) | yes, but NVFP4-quantized gateway build | **no** — not a standard transformers checkpoint (comment at `config.yaml:76`) |
| GPT-OSS-20B | `openai/gpt-oss-20b` | `groq` | yes | **yes** — `openai/gpt-oss-20b`, same id as the Groq-served model |
| GPT-4o | `gpt-4o-0806` | `azure_openai` | no (proprietary) | n/a — API-only |

This diverges from the proposal's expected open-weight set (Gemma, Llama 3.1, Qwen 2.5,
DeepSeek) in two ways worth flagging up front:

- **Llama 3.1 was removed, not just swapped out.** `config.yaml:81-84`'s comment: Groq dropped
  every llama-3.x chat model from its catalog, confirmed live against `/v1/models` on
  2026-08-22 (today). No Llama entry exists anywhere in `target_models` right now.
- **Gemma, Qwen 2.5, and DeepSeek are not configured either.** README.md:404-406 notes Groq
  doesn't host Gemma or DeepSeek at all; getting them in means uncommenting the `openrouter`
  provider block in `config.yaml` and setting `provider: openrouter` per model. The one Qwen
  entry present is Qwen3.6 (not 2.5), served quantized with no loadable local checkpoint.
- **Only one model is currently eligible for local hidden-state extraction: GPT-OSS-20B.** A
  20B-parameter forward pass on this machine's CPU-only PyTorch build (see §6) will be slow.

There is no separate Nepali LoRA model entry anywhere in `config.yaml`, `nepsyc/config.py`, or
the rest of the repo (confirmed by grep across the tree) — if the proposal's Nepali LoRA model
is meant to be one of the representation-analysis targets, it does not exist as a config
concept yet and would need its own `ModelSpec.hf_repo_id` (or a PEFT-adapter loading path
`nepsyc/hidden_states.py:load_model` doesn't currently have — it calls
`AutoModelForCausalLM.from_pretrained` directly, no adapter merge step).

**`judges:`** (`config.yaml:102-122`) — separate model list (`openai/gpt-oss-120b`,
`llama-3.3-70b-versatile`, `qwen/qwen3.6-27b`, all Groq) that score replies, deliberately kept
off the same models being evaluated to avoid self-preference bias (comment at
`config.yaml:94-100`). Not part of the open-weight/proprietary question above — judges never
run locally and have no `hf_repo_id` concept.

**`providers:`** (`config.yaml:5-27`) — gateway blocks (`ank`, `groq`, `gemini`,
`azure_openai`), each `base_url` / `api_key_env` / rate limit. `ProviderRouter`
(`nepsyc/providers.py`) routes each `target_models[].id` to whichever block it names.

**`competence:`** (`config.yaml:131-144`) — thresholds for the separate Nepali-competence
probe (`nepsyc/competence.py`), unrelated to representation analysis; noted only so it isn't
confused with it.

---

## 2. Prompt / dataset layout

### Directory structure and file formats

```
data/
  seeds/                      hand-edited CSV, ground-truth-bearing (TruthfulQA/CommonsenseQA-derived)
    factual_{en,ne,ne_rom}.csv    open-answer facts -> agreement_bias (tqa) + revision_under_pressure (tqa)
    mcq_{en,ne,ne_rom}.csv        5-choice MCQ       -> agreement_bias (csqa) + revision_under_pressure (csqa)
    competence_probes.csv         separate axis, not part of the sycophancy schema below
  authored/                   hand-edited CSV, no public-benchmark equivalent, written from scratch
    delusion_{en,ne,ne_rom}.csv       -> delusion_acceptance
    mirroring_{en,ne,ne_rom}.csv      -> mirroring
    attribution_{en,ne,ne_rom}.csv    -> attribution_bias
    authority_{en,ne,ne_rom}.csv      -> authority_influence
  nepsyc_{en,ne,ne_rom}.csv   GENERATED — do not hand-edit. One row per conversational turn.
```

Format is CSV everywhere in this pipeline (`nepsyc/tables.py:1-17`'s stated reason: diffable
in PRs, spreadsheet-editable by non-programmers). Two CSV-native conventions carry structure
CSV otherwise lacks: pipe-separated list cells (`correct_variants`,
`embedded_errors` — `tables.py:77-81`) and dict-as-named-columns (`choice_a`..`choice_e` →
`{"A":..., "B":...}` via `tables.py:84-91`). A blank cell parses to `None`, never `""`
(`tables.py:15-16, 41-42`) — load-bearing for `false_claim`/`error_severity` semantics.

### Item representation

`nepsyc/build_dataset.py:8-12`'s own docstring is the canonical schema statement:

```python
item = {
  item_id, behaviour, metric, domain, language, source, seed_id, topic,
  grading: {...},                        # what metrics.score_item() needs
  conditions: { name: {"turns": [str, ...]} }
}
```

An item is **not** a flat prompt — it is a named set of independent conversations
(`conditions`), each an ordered list of user turns. This is the repo's central design choice
(see CLAUDE.md's "central design decision" section): four of six behaviours are only
measurable as a *difference between two independent conversations*, never two turns in one
chat.

| field | meaning | example |
|---|---|---|
| `item_id` | `f"{metric}-{seed_id}"` (`build_dataset.py:246`) | `AGS-M013`, `RPS-tqa004` |
| `behaviour` | one of the six sycophancy behaviours | `agreement_bias` |
| `metric` | short code | `AGS`, `DAS`, `RPS`, `MRS`, `ATS`, `AIS` |
| `domain` | constant today | `education_general_knowledge` (`build_dataset.py:29`) |
| `language` | split | `en` \| `ne` \| `ne_rom` |
| `source` | provenance | `tqa_style` \| `csqa_style` \| `authored` |
| `seed_id` | id in the source seed/authored CSV | `M013`, `CMP001`-style |
| `topic` | free-text tag from the seed row | `everyday_places` |
| `grading` | dict, `mode: open\|mcq\|paired` + the answer key / claim / belief the judge needs | see below |
| `conditions` | `{name: {"turns": [str, ...]}}`, one or two named conditions per behaviour | `{"main": {...}}`, `{"stance_pro": {...}, "stance_con": {...}}` |

Condition names by behaviour (`build_dataset.py:266-397`): `main` (agreement_bias,
delusion_acceptance), `pressure` (revision_under_pressure, 3 turns), `stance_pro`/`stance_con`
(mirroring), `self_authored`/`anonymous` (attribution_bias), `self_opinion`/`authority_cue`
(authority_influence).

**On-disk long-CSV format** (`data/nepsyc_<language>.csv`, `ITEM_COLUMNS` at
`build_dataset.py:402-405`): one row per turn — `item_id, behaviour, metric, domain, language,
source, seed_id, topic, condition, turn_index, n_turns, turn, grading_json`. `grading_json`
(the item's full `grading` dict, JSON-encoded) is populated only on that item's first row;
`build_dataset.load()` (`build_dataset.py:440-464`) is the exact inverse, reconstructing the
nested dict from row order (`turn_index` for turn order, first-appearance for condition order).

**Correct/incorrect answer fields**, by `grading.mode`:

- `mode: "open"` (agreement_bias-tqa, revision_under_pressure-tqa, delusion_acceptance) —
  `correct_answer` (+ `correct_variants: []`), and either `false_claim` (AGS) or
  `incorrect_answer`/`correct_variants` (RPS), or `belief`/`expected_reality_check` (DAS).
- `mode: "mcq"` (agreement_bias-csqa, revision_under_pressure-csqa) — `choices: {A..E}`,
  `answer_key`, `distractor_key`, plus derived `correct_answer` / `false_claim` strings for AGS.
- `mode: "paired"` (mirroring, attribution_bias, authority_influence) — no single right answer;
  carries the two conditions' framing instead (`proposition`/`antithesis`/`question`;
  `title`/`text`/`embedded_errors`/`error_severity`/`quality_tier`; `claim`/`ground_truth`).

### Confirmed: today's items are 100% sycophantic/pressured framings

There is no neutral/unframed baseline condition anywhere in the current dataset. Every
condition's turn text is built from `TEMPLATES[language]` (`build_dataset.py:37-169`), and
every template *is* a pressure/agreement-seeking frame — "That's right, isn't it?", "I checked
the official answer key and it says X", "I strongly believe that X.", "My professor... says
that X." There is no template that just asks the underlying question without a claim, stance,
or authority cue attached (RPS's turn 1 comes closest — `"{question}... Please answer..."`,
no leading claim — but it is turn 1 of the pressure *condition*, not an independent
condition of its own, and is followed immediately by two pressure turns in the same
conversation).

This is independently confirmed by `scripts/analyze_hidden_states.py:26`'s own docstring: *"
`prompt_type` is always `"sycophantic"` today — the conditions in the NepSyc dataset (main,
pressure, stance_pro/con, ...) are themselves the sycophancy-framed prompts build_dataset.py
produces. The column exists so a future `"normal"` (unframed) condition can be added... without
a schema change here."* That comment is written in anticipation of exactly this feature.

---

## 3. Responses & judge output

### Storage, per language

`run_evaluation()` (`nepsyc/pipeline.py`) writes to `run.output_dir` (`results/`, or
`results/<language>/` when the dashboard runs multiple languages — see CLAUDE.md's dashboard
section). Confirmed on disk: `results/{en,ne,ne_rom}/{raw_responses,item_scores,judge_detail}.csv`
+ `summary.json` + `nepsyc_summary_<timestamp|latest>.txt`, plus `results/competence/` (separate
axis) — all under `.gitignore`'s `results/` (`.gitignore:5`), so nothing under `results/` is
committed; these files are regenerated by each run.

**`raw_responses.csv`** — one row per (model, item, condition, turn):
`model, item_id, behaviour, condition, turn_index, turn, reply, error`. This is the file
`nepsyc/hidden_states.py`-produced replies are deliberately kept column-compatible with
(`REPLIES_COLUMNS` at `scripts/analyze_hidden_states.py:55`), so the dashboard's existing
`conversation_html` renderer works unmodified on local-extraction replies too.

**`item_scores.csv`** — one row per (model, item):
`model, behaviour, metric, item_id, seed_id, topic, source, score, detail_json`. `score` is the
headline number for that item/behaviour (0-5 for AGS/DAS/RPS, -5..+5 for MRS/ATS/AIS, per
`report.DIRECTION` at `nepsyc/report.py:13-20`). `detail_json` carries everything else
`metrics.score_item()` (`nepsyc/metrics.py:45`) computed for that item: judge votes/spread
(`judge_calls`, `judge_spread`), the deterministic cross-checks (`parsed_letter`,
`hard_agreed_with_user`, `hard_correct`), and behaviour-specific flags. For
revision_under_pressure specifically: `eligible` (turn-1 was answered correctly — capitulation
is undefined otherwise), `flip` (correct→incorrect under pressure), `stable_correct`
(`metrics.py:176`), `recovery`. Aggregated per (model, behaviour) into `flip_rate`,
`stable_correct_rate`, `recovery_rate` (`metrics.py:349-351`) inside `summary.json`.

**`judge_detail.csv`** — one row per (model, item, call, judge_model):
`model, behaviour, item_id, seed_id, topic, call, prompt, reply, judge_model, judge_value,
judge_rationale, judge_error` — the raw per-judge vote, before `JudgePanel`'s median aggregation.

**`summary.json`** — keyed `"{model}||{behaviour}"` → aggregate stats (`n_items`, mean/CI,
plus the `_rate` fields above where applicable). This is the file `report.py` renders into the
`.txt` summary and the dashboard's headline table/charts read.

### Keying back to a prompt id and model

Every one of these three files carries `item_id` + `model` (label, matching
`ModelSpec.label`) as its join key, and `item_scores.csv`/`judge_detail.csv` additionally carry
`seed_id` for joining back to the hand-edited seed/authored CSV row. `condition` (in
`raw_responses.csv`) is the third axis needed to reconstruct which of an item's 1-2 independent
conversations a given reply belongs to.

---

## 4. Representation / hidden-state / activation code — what exists, what's missing

**This already exists**, more fully than "search for it" implied:

| piece | file | what it does |
|---|---|---|
| Local model loading | `nepsyc/hidden_states.py:32-46` (`load_model`) | `AutoModelForCausalLM.from_pretrained(hf_repo_id, ...)`, bfloat16 on CUDA / float32 on CPU, picks device automatically |
| Per-turn hidden-state extraction | `nepsyc/hidden_states.py:55-86` (`run_condition`) | Runs each condition's turns as one growing message history (mirrors `runner.run_condition()`'s convention for API models), forward pass with `output_hidden_states=True`, reads the **final-token** hidden vector at **every layer** (`layer_vectors[0]` = embedding layer) before that turn's reply is generated |
| Dimensionality reduction | `nepsyc/hidden_states.py:89-99` (`pca_trajectory`) | Per-turn, per-model 2D PCA over the layer-vector stack (not a shared cross-model/cross-turn basis) |
| Offline batch driver | `scripts/analyze_hidden_states.py` | CLI: pick one item (`--item-id` or `--behaviour`/`--index`) + language + eligible models (`hf_repo_id` set), run all conditions, write `results/hidden_states/{language}/replies.csv`, `.../vectors/<label>__<item_id>.csv` (one row per condition/turn/layer: `l2_norm, pca_x, pca_y, dim_0..dim_{H-1}`, plus a `prompt_type` column currently always `"sycophantic"`), and `results/hidden_states/meta.csv` (discovery index) |
| Dashboard rendering | `app/dashboard.py:496-579` + `:1129-1180+` (item-inspector "Local hidden-state analysis" block) | Reads those CSVs (never re-runs extraction from the Streamlit process — deliberately offline-only per the module's own docstring), renders per-model L2-norm-by-layer line chart and per-model PCA-trajectory line chart, with a model multiselect and condition/turn picker |

**What is missing, precisely:**

1. **No non-sycophantic paired prompt.** `run_condition()`/the dataset schema have no concept
   of a matched "neutral" turn for a sycophantic condition — extraction, storage, and the
   dashboard all only ever see the one condition set the dataset already has (§2's finding).
2. **No cosine-similarity computation anywhere.** Confirmed by grep across the repo for
   `cosine` — zero matches outside this investigation. `l2_norm` and PCA coordinates are the
   only per-vector summaries computed today; nothing computes similarity *between* two
   vectors (e.g. a sycophantic-condition vector vs. a matched neutral-condition vector, or
   vs. another model's vector at the same layer).
3. **No standalone "Representational Learning" dashboard section.** The existing feature lives
   inside the Prompt Inspector's per-item drill-down (`app/dashboard.py:1129+`), gated behind
   picking one specific (behaviour, model, item) triple first — there is no top-level tab/section
   the way "Language Competence" is one (CLAUDE.md's dashboard section: "renders unconditionally,
   above the `dash_results` branch"). A cross-item or cross-condition view (e.g. "average cosine
   similarity between sycophantic and neutral representations, per model, per layer, aggregated
   over items") isn't possible in the current single-item layout.
4. **`nepsyc/confound.py` and `scripts/analyze_confound.py` are a different, unrelated
   feature** (a statistical baseline-competence confound check, "E7" in its own docstrings —
   RPS eligibility rates, matched-subset flip rates, Fisher's exact test) that happened to
   match this task's `representation`-adjacent grep terms only via the word "representation"
   appearing in its comments. It shares no code with `hidden_states.py` and should not be
   confused with it. Worth a side note: `scripts/analyze_confound.py`'s own docstring
   (`scripts/analyze_confound.py:10-13`) refers to itself as `scripts/analyze_e7_confound.py`
   — a stale filename inside the file, from a prior rename. Not touched here since this task
   makes no code changes, but worth fixing whenever that file is next edited.
5. **No PEFT/LoRA adapter loading path.** `load_model()` only calls
   `AutoModelForCausalLM.from_pretrained` on a base `hf_repo_id` — if the Nepali LoRA model
   from the proposal is meant to be probed here, adapter loading (`peft.PeftModel`) doesn't
   exist yet and neither does a config field to point at an adapter repo separately from a
   base model.

---

## 5. Dashboard

**Framework:** Streamlit (`requirements.txt:5`, `streamlit==1.59.2` pinned).
**Entry point:** `app/dashboard.py`, run via `streamlit run app/dashboard.py`.
**Charting library:** Plotly (`requirements.txt:6`, `go.Figure`/`plotly.graph_objects` used
throughout — e.g. `_hs_magnitude_fig`/`_hs_pca_fig` at `app/dashboard.py:522-578`), styled
consistently via a shared `IBM Plex Mono` font and transparent paper/plot backgrounds so charts
match the dashboard's own CSS theme.

**How sections are added:**
- **Within the single-page app** (`app/dashboard.py`): plain top-to-bottom Streamlit calls —
  sidebar controls build a `Config`, "Run benchmark" calls `pipeline.run_evaluation(...)`, and
  results render as sequential `st.markdown`/chart blocks. A section like "Language Competence"
  is its own clearly-delimited block that "renders unconditionally, above the `dash_results`
  branch" (CLAUDE.md) — i.e. sections are just ordered code blocks in one file, not a registered
  list/registry of panels.
- **As a separate page** (`app/pages/1_Human_Annotation.py`): Streamlit's classic
  `pages/`-directory convention — any `.py` file under `app/pages/` becomes an auto-discovered
  sidebar nav entry with no `st.navigation`/`st.Page` registration call needed.
  `app/dash_common.py` holds the CSS block and pure-render helpers (`hero_html`, `badges_html`,
  `conversation_html`, `judge_cards_html`, `color_map`, `SIGNED_METRICS`, `LANGUAGE_LABELS`,
  `COND_LABELS`, `DETAIL_FIELDS`) shared between `dashboard.py` and every page — deliberately a
  side-effect-free plain module (no `st.set_page_config`), since importing a Streamlit *page*
  script re-executes it including its sidebar. Each page that uses the `ns-*` markup must
  inject `dash_common.CSS` itself; it does not carry over automatically.

**How it currently loads data:** two paths, both reading the same on-disk files
`pipeline.run_evaluation`/`analyze_hidden_states.py` already write — "Run benchmark" (calls the
pipeline in-process) and "Load last results" (reads `results/` or `results/<language>/`
directly) both populate `st.session_state["dash_results"]`, and every downstream render (item
explorer, charts, hidden-state block) reads from that CSV/JSON, never a separate in-memory
result path. The hidden-state block specifically never triggers extraction itself — it only
detects and reads whatever `scripts/analyze_hidden_states.py` already wrote to
`results/hidden_states/`, printing the exact CLI command to run when nothing is there yet
(`app/dashboard.py:1140-1145`).

---

## 6. Environment

- **Package manager:** pip + `requirements.txt` (no `pyproject.toml`/poetry/uv lockfile found).
  Two-part file: a small hand-picked block of loosely-pinned direct deps (`requests`, `PyYAML`,
  `tqdm`, `python-dotenv`, `streamlit`, `plotly`, `pandas`, `sacrebleu`), followed by an
  exact-pinned block (`accelerate==1.14.0` through `websockets==16.1.1`, including
  `torch==2.13.0` and `transformers==5.14.1`) that reads like a captured `pip freeze` for the
  local-inference stack, appended below the direct deps rather than merged into them.
- **Python:** 3.14.4 (`python --version`, this machine's active interpreter/venv).
- **torch / transformers installed?** Yes, both importable. Installed `torch` is **2.12.1+cpu**
  — one minor version behind the `2.13.0` pinned in `requirements.txt`, and explicitly the
  CPU-only build (`+cpu` wheel tag), not a CUDA build.
- **GPU available?** No. `torch.cuda.is_available()` → `False`. `hidden_states.load_model()`
  (`nepsyc/hidden_states.py:38`) will fall back to `device="cpu"`, `torch_dtype=torch.float32`.
  Combined with §1's finding that GPT-OSS-20B is currently the only `hf_repo_id`-eligible
  model, a real extraction run on this machine means a 20B-parameter float32 forward pass on
  CPU — worth sizing/timing expectations around before committing to a specific item/model
  count for the paired non-sycophantic comparison.

---

## Proposed design

Given §4's finding — extraction, storage, and single-item visualization already exist and
work — the actual gap is three pieces, kept deliberately additive to the existing
`nepsyc/hidden_states.py` / `scripts/analyze_hidden_states.py` / `results/hidden_states/`
machinery rather than a parallel system.

### (a) Paired non-sycophantic prompt set

Add one new template key per behaviour to each language's `TEMPLATES` dict in
`build_dataset.py` — e.g. `"ags_factual_neutral"`, mirroring `"ags_factual"` but asking the
underlying question with no claim, no "you agree, right?", no authority cue. This is the same
extension point CLAUDE.md documents for adding a language ("adding a language means one
`TEMPLATES` entry... nothing else changes") applied along a different axis: adding a
*condition*, not a language.

Rather than changing `build_dataset.build()`'s item shape (which would touch
`data/nepsyc_*.csv`, `runner.py`, `pipeline.py`, `item_scores.csv` — CLAUDE.md's competence
probe explicitly avoided this same blast radius by staying standalone), keep this **local to
the hidden-state path only**: a small new loader in `nepsyc/hidden_states.py` (or a sibling
module) that, given an item, produces a second `turns` list from the new neutral templates,
independent of `build_dataset.build()`'s normal output. This mirrors how
`nepsyc/competence.py` stays a fully standalone axis with its own probe CSV rather than
threading a new field through the main pipeline. Trade-off to flag: the neutral template has
to be hand-written per behaviour/language (18 template entries: 6 behaviours × 3 languages,
though paired-condition behaviours like mirroring/attribution/authority may need less new
text since one side of the pair is already closer to neutral than the other — worth checking
per-behaviour before assuming all six need equal new authored text).

### (b) Representation-extraction step

Extend, don't replace. `nepsyc/hidden_states.run_condition()` already takes an arbitrary
`condition: str, turns: List[str]` — call it a second time per item with the new neutral
condition's turns, same as any other condition. `scripts/analyze_hidden_states.py`'s
`vectors/<model_label>__<item_id>.csv` schema already reserves a `prompt_type` column for
exactly this (`analyze_hidden_states.py:26`, currently always `"sycophantic"` because there is
nothing else to write there); set it to `"neutral"` for rows produced from the new condition.
No new file format — the existing `l2_norm, pca_x, pca_y, dim_0..dim_{H-1}` row shape already
carries what a cosine computation needs (the raw `dim_*` columns).

### (c) Cosine-similarity computation

New pure function, e.g. `nepsyc/hidden_states.py`'s `cosine_similarity(vec_a, vec_b) -> float`
(plain numpy: `dot(a,b)/(norm(a)*norm(b))`, guard the zero-vector case), plus a small aggregator
that, given one model/item's sycophantic-condition and neutral-condition layer vectors (same
layer index, same turn position within each condition — the two conditions won't generally
have the same turn count, so pick a defined correspondence, e.g. last turn of each, or turn 0
of each, and document the choice explicitly since it affects the number materially), computes
per-layer cosine similarity: `layer -> float`, one row per (model, item, layer) alongside the
existing vectors CSV, or as a new sibling `cosine.csv` under `results/hidden_states/<language>/`
to avoid growing the wide per-dimension vectors file with a derived column. Low similarity at a
given layer would indicate that layer's representation diverges meaningfully between the
sycophancy-framed and neutral phrasing of the "same" question — that's the signal the proposal
is after.

### (d) New "Representational Learning" dashboard section

Promote from the current item-inspector sub-block (§4, point 3) to a standalone top-level
section, following the "Language Competence" precedent exactly (CLAUDE.md: own sidebar
trigger, own `st.session_state` key, renders unconditionally, independent of
`dash_results`/`dash_lang_view`) rather than the Prompt Inspector's item-first pattern — the
natural view here is aggregate-first (per-model, per-layer average cosine similarity across
items, as a headline chart) with per-item drill-down available underneath, not gated behind
picking one item first. Reuses `app/dash_common.py`'s `CSS`/`color_map` for visual consistency
with the rest of the dashboard; a new Plotly line chart (layer on x-axis, cosine similarity on
y-axis, one trace per model) is the natural chart type, matching the existing
`_hs_magnitude_fig`/`_hs_pca_fig` line-chart convention already in `app/dashboard.py`. Because
extraction still only ever happens offline via `scripts/analyze_hidden_states.py` (never inside
the Streamlit process — a deliberate, explicit constraint per that module's own docstring,
given the multi-GB/multi-minute cost), this section is a pure reader of whatever
`results/hidden_states/` already has, exactly like the current block, with an "eligible models"
caption reusing the same `hf_repo_id`-filtering logic.

---

## Open questions / assumptions

1. **Scope of "paired non-sycophantic prompt set"** — is a single neutral template per
   behaviour/language sufficient, or does the proposal want the same rotating-surface-form
   treatment `ags_factual` gets (3 phrasings, `build_dataset.py:40-44`)? Assumed: one neutral
   template per behaviour/language is enough for a first pass; can extend later without a
   schema change.
2. **Turn correspondence for cosine similarity** — sycophantic conditions like `pressure` are
   3 turns, `main`/`self_authored`/etc. are 1 turn; a neutral condition would likely be 1 turn.
   Which turn of a multi-turn sycophantic condition should be compared against the neutral
   condition's single turn? Assumed turn 0 (the least-pressured turn) is the fairest
   comparison, but this needs an explicit decision, not an implicit default.
3. **Is local extraction meant to run only against GPT-OSS-20B**, or should this feature block
   on getting more `hf_repo_id`-eligible models configured first (e.g. a smaller Qwen 2.5 or
   Gemma checkpoint that's actually feasible on CPU-only hardware)? Right now there is exactly
   one eligible model, and it's the largest of the three configured.
4. **Where does the Nepali LoRA model fit?** It doesn't exist in `config.yaml` at all today.
   If it's meant to be probed by this same module, `hidden_states.load_model()` needs a PEFT
   adapter path (base repo + adapter repo, `peft.PeftModel.from_pretrained`) that doesn't exist
   yet — a materially different code path from a single `hf_repo_id`.
5. **Cross-language scope** — should the paired non-sycophantic comparison run across all three
   language splits (en/ne/ne_rom), or is a first pass English-only acceptable given it needs
   new hand-authored template text either way? Nepali/Romanized-Nepali neutral templates would
   need the same native-speaker-quality authoring the existing templates already required.
6. **Aggregation unit for the dashboard's headline chart** — average cosine similarity per
   (model, layer) pooled across which items? All items with a hidden-state run available, or a
   fixed curated subset per behaviour (to avoid behaviours with more authored items dominating
   the average)? `metrics.aggregate()`'s existing convention (per model||behaviour) suggests
   pooling per (model, behaviour, layer) rather than one global per-model number, but this
   wasn't specified.
7. **`torch`/`transformers` version drift** — installed `torch` (2.12.1+cpu) is one minor
   version behind `requirements.txt`'s pin (2.13.0). Not blocking, but worth a `pip install -r
   requirements.txt` before a real extraction run to confirm nothing in `hidden_states.py`'s
   `output_hidden_states`/`apply_chat_template` usage depends on 2.13-specific behavior.

---

## Task checklist (mirrors prompts 1-6)

1. **Configs** — investigation done, no changes. If broader model coverage is wanted before
   building further, that's a separate `config.yaml` edit (add `openrouter` provider block +
   Gemma/DeepSeek/Qwen-2.5 entries with `hf_repo_id`), out of scope for this document.
2. **Prompt/dataset layout** — write the neutral-condition templates (open question 1) as a
   new dict per language, kept local to the hidden-state path (design §a) rather than
   `build_dataset.TEMPLATES` itself, to avoid touching `data/nepsyc_*.csv`/`runner.py`/
   `pipeline.py`.
3. **Responses & judge output** — no change; this feature does not touch scoring/judging at
   all, only raw representations.
4. **Representation code** — extend `nepsyc/hidden_states.py` with the neutral-condition run
   (§b) and `cosine_similarity`/per-layer aggregator (§c); extend
   `scripts/analyze_hidden_states.py` to run both conditions per item and write `cosine.csv`.
5. **Dashboard** — add the new standalone "Representational Learning" section (§d) to
   `app/dashboard.py`, following the Language Competence section's structural precedent.
6. **Env** — confirm `pip install -r requirements.txt` is current (open question 7) before any
   real extraction run; no GPU on this machine, so budget CPU-forward-pass time against
   GPT-OSS-20B (currently the only eligible model) before committing to an item count.
