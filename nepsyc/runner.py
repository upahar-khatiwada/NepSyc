"""Execute benchmark items against target models.

Each condition is an independent conversation. Multi-turn conditions (revision under
pressure) replay the scripted user turns in sequence, appending the model's own replies,
so the pressure the model experiences is the pressure it actually created for itself.
"""
from __future__ import annotations

import datetime as _dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .tables import write_csv


def run_condition(provider, model_id: str, turns: List[str], *, system_prompt: Optional[str],
                  temperature: float, max_tokens: int, strip_think: bool) -> List[str]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    replies: List[str] = []
    for turn in turns:
        messages.append({"role": "user", "content": turn})
        reply = provider.chat(
            model_id, messages, temperature=temperature, max_tokens=max_tokens, strip_think=strip_think
        )
        replies.append(reply)
        messages.append({"role": "assistant", "content": reply})
    return replies


def run_item(provider, model, item, gen) -> Dict[str, Any]:
    out = {"item_id": item["item_id"], "behaviour": item["behaviour"], "model": model.label, "conditions": {}}
    for cond_name, cond in item["conditions"].items():
        try:
            replies = run_condition(
                provider,
                model.id,
                cond["turns"],
                system_prompt=gen.system_prompt,
                temperature=gen.temperature,
                max_tokens=gen.max_tokens,
                strip_think=model.strip_think,
            )
            out["conditions"][cond_name] = {"turns": cond["turns"], "replies": replies}
        except Exception as e:
            out["conditions"][cond_name] = {"turns": cond["turns"], "replies": [], "error": str(e)}
    return out


def collect(provider, models, items, gen, max_workers: int = 4, out_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Run every (model, item) job and return the results.

    Writes `out_path` after every completed job, not just at the end, so a paused or
    killed run always leaves an up-to-date `raw_responses.csv` on disk reflecting whatever
    finished so far -- useful both for inspecting progress and as a crash safety net.

    On Ctrl+C: cancels jobs that haven't started yet, writes whatever has completed, and
    re-raises KeyboardInterrupt rather than continuing on to judging/scoring with a partial
    result set. Jobs already in flight are not force-killed (a blocking HTTP call in a
    worker thread can't be interrupted from here) so exit isn't instant, but nothing already
    finished is lost. To resume, just re-run the same command with the same cache_path --
    `providers.ResponseCache` (append-only, keyed on the exact conversation/model/params)
    replays every already-completed turn from disk instead of calling the API again, so the
    run picks up from wherever it left off.
    """
    # Alongside the fixed `out_path` (overwritten every run -- what the dashboard/cache-resume
    # logic reads), also keep a timestamped copy that a later run into the same output_dir
    # never overwrites. The timestamp is fixed once per collect() call so every checkpoint
    # write during this run lands in the same versioned file.
    versioned_path = None
    if out_path:
        out_path = Path(out_path)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")

    jobs = [(m, it) for m in models for it in items]
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_item, provider, m, it, gen): (m, it) for m, it in jobs}
        try:
            for f in tqdm(as_completed(futs), total=len(futs), desc="collecting responses"):
                results.append(f.result())
                if out_path:
                    write_responses(results, out_path)
                    write_responses(results, versioned_path)
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            if out_path:
                write_responses(results, out_path)
                write_responses(results, versioned_path)
            print(
                f"\ncollecting responses: interrupted after {len(results)}/{len(jobs)} jobs. "
                f"Partial results written to {out_path}. Re-run the same command to resume "
                f"-- already-completed turns replay from the response cache instead of "
                f"re-calling the API."
            )
            raise

    return results


RESPONSE_COLUMNS = ["model", "item_id", "behaviour", "condition", "turn_index", "turn", "reply", "error"]


def write_responses(results: List[Dict[str, Any]], out_path: Path) -> None:
    """Long-format CSV, one row per turn: prompt beside the reply it produced.

    A failed condition still emits one row per turn with an empty reply and the exception
    text in `error`, so a partial sweep is visible in the file rather than silently short.
    """
    rows = []
    for r in sorted(results, key=lambda x: (x["model"], x["item_id"])):
        for cond, spec in r["conditions"].items():
            replies = spec.get("replies", [])
            for i, turn in enumerate(spec["turns"]):
                rows.append({
                    "model": r["model"], "item_id": r["item_id"], "behaviour": r["behaviour"],
                    "condition": cond, "turn_index": i, "turn": turn,
                    "reply": replies[i] if i < len(replies) else "",
                    "error": spec.get("error", ""),
                })
    write_csv(rows, Path(out_path), RESPONSE_COLUMNS)
