"""Full representation extraction for open-weight models: per-layer hidden states (two
documented poolings), attention weights, and answer-position logits/confidence metrics --
across sycophantic and neutral prompt variants (nepsyc/neutral_templates.py).

This is a separate, richer extractor from nepsyc/hidden_states.py, not an edit to it:
hidden_states.py's output shape (last-token vectors only, no attention, no logits) is a
documented dependency of scripts/analyze_hidden_states.py and the dashboard's "Local
hidden-state analysis" block, so changing its return shape there would break both. This
module follows the same "standalone axis" precedent nepsyc/competence.py already set --
driven only by scripts/extract_representations.py, imported nowhere else.

Pooling convention (fixed; logged verbatim into every run's run_meta.json so a stored
tensor's meaning never has to be guessed from code later):

  last_token
      The final-token hidden state from a forward pass over the PROMPT ONLY (before
      generation) -- the model's state the instant before it starts answering. Same point
      nepsyc/hidden_states.py samples.
  mean_pooled
      Mean, over the generated REPLY's token positions, of the hidden state at each layer --
      read from a second, teacher-forced forward pass over prompt+reply together (no
      sampling). This captures how the model represents the content of its own answer,
      which last_token (a pre-answer state) cannot.

Both are computed at every layer, for every turn, for every condition (including the
synthetic "neutral" condition). Attention weights (head-averaged, from the same teacher-
forced pass) and the prompt-end next-token logits (from the first pass) are captured only
for a configurable set of layers / always for the single next-token position respectively,
per the proposal's "store compactly" instruction.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .hidden_states import load_model  # noqa: F401  (re-exported: extract_representations.py imports it from here)


@dataclass
class TurnRepresentation:
    condition: str
    turn_index: int
    prompt_text: str
    reply: str
    last_token: List[np.ndarray] = field(default_factory=list)     # [layer] -> (hidden_dim,)
    mean_pooled: List[np.ndarray] = field(default_factory=list)    # [layer] -> (hidden_dim,)
    attentions: Dict[int, np.ndarray] = field(default_factory=dict)  # attn_layer -> (seq, seq), head-averaged
    next_token_logits: Optional[np.ndarray] = None                  # (vocab_size,)
    n_layers: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    device_used: str = ""
    truncated: bool = False


def _resolve_attn_layers(spec: str, num_hidden_layers: int) -> List[int]:
    """"last" -> [num_hidden_layers-1]; "all" -> every layer; "none" -> []; or an explicit
    comma-separated list of 0-based layer indices into the model's transformer blocks
    (NOT the hidden_states index, which is offset by +1 for the embedding layer -- see
    module docstring / run_meta.json's `attn_layer_offset_note`)."""
    spec = (spec or "last").strip().lower()
    if spec == "none":
        return []
    if spec == "last":
        return [num_hidden_layers - 1]
    if spec == "all":
        return list(range(num_hidden_layers))
    return sorted({int(x) for x in spec.split(",") if x.strip() != ""})


def _truncate(inp, max_input_tokens: int) -> bool:
    """Left-truncate (keep the most recent tokens) in place if the prompt exceeds
    max_input_tokens. Returns whether truncation happened, for logging/metadata."""
    seq_len = inp["input_ids"].shape[1]
    if seq_len <= max_input_tokens:
        return False
    for key in ("input_ids", "attention_mask"):
        if key in inp:
            inp[key] = inp[key][:, -max_input_tokens:]
    return True


def _run_with_oom_fallback(fn, model, device: str):
    """Run fn() (a zero-arg closure doing one forward/generate call); on a CUDA OOM, move the
    model to CPU and retry once. This is the "CPU fallback for a small --limit smoke test"
    the extraction proposal asks for -- a best-effort retry, not a batching/sharding system."""
    import torch

    try:
        return fn(), device
    except RuntimeError as e:
        if device == "cuda" and "out of memory" in str(e).lower():
            print("  CUDA OOM -- falling back to CPU for this turn (will be slow)", file=sys.stderr)
            torch.cuda.empty_cache()
            model.to("cpu")
            return fn(), "cpu"
        raise


def run_turn_full(
    tokenizer, model, device: str, messages: List[Dict[str, str]],
    max_new_tokens: int = 120, attn_layers: Optional[List[int]] = None,
    max_input_tokens: int = 4096,
) -> "tuple[TurnRepresentation, str]":
    """One turn: prompt-only forward pass (last_token + next_token_logits), generation, then
    a teacher-forced prompt+reply forward pass (mean_pooled + attentions). Returns the
    populated TurnRepresentation and the (possibly OOM-fallback-updated) device string, so
    callers can carry a CPU fallback forward to subsequent turns instead of re-hitting the
    same OOM every time."""
    import torch

    attn_layers = attn_layers if attn_layers is not None else []

    prompt_inp = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    truncated = _truncate(prompt_inp, max_input_tokens)
    prompt_inp = prompt_inp.to(device)
    prompt_len = prompt_inp["input_ids"].shape[1]

    def _prompt_forward():
        with torch.no_grad():
            return model(**prompt_inp, output_hidden_states=True, return_dict=True)

    prompt_out, device = _run_with_oom_fallback(_prompt_forward, model, device)
    if device == "cpu":
        prompt_inp = {k: v.to("cpu") for k, v in prompt_inp.items()}
    last_token = [h[0, -1].float().cpu().numpy() for h in prompt_out.hidden_states]
    next_token_logits = prompt_out.logits[0, -1].float().cpu().numpy()
    n_layers = len(last_token)
    hidden_size = last_token[0].shape[0]
    vocab_size = next_token_logits.shape[0]
    del prompt_out

    def _generate():
        with torch.no_grad():
            return model.generate(**prompt_inp, max_new_tokens=max_new_tokens, do_sample=False)

    generated, device = _run_with_oom_fallback(_generate, model, device)
    reply = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()

    full_inp = {"input_ids": generated}
    if "attention_mask" in prompt_inp:
        pad = torch.ones(
            (1, generated.shape[1] - prompt_len), dtype=prompt_inp["attention_mask"].dtype,
            device=prompt_inp["attention_mask"].device,
        )
        full_inp["attention_mask"] = torch.cat([prompt_inp["attention_mask"], pad], dim=1)
    full_inp = {k: v.to(device) for k, v in full_inp.items()}

    def _full_forward():
        with torch.no_grad():
            return model(
                **full_inp, output_hidden_states=True,
                output_attentions=bool(attn_layers), return_dict=True,
            )

    full_out, device = _run_with_oom_fallback(_full_forward, model, device)

    reply_len = generated.shape[1] - prompt_len
    if reply_len > 0:
        mean_pooled = [h[0, -reply_len:].float().mean(dim=0).cpu().numpy() for h in full_out.hidden_states]
    else:
        # Degenerate case: the model generated nothing. Fall back to the prompt's own
        # last-token state so callers get a same-shape vector rather than a hole in the array.
        mean_pooled = [v.copy() for v in last_token]

    attentions: Dict[int, np.ndarray] = {}
    if attn_layers and full_out.attentions is not None:
        for layer in attn_layers:
            if 0 <= layer < len(full_out.attentions):
                attentions[layer] = full_out.attentions[layer][0].float().mean(dim=0).cpu().numpy()

    del full_out, generated

    tr = TurnRepresentation(
        condition="", turn_index=-1, prompt_text="", reply=reply,
        last_token=last_token, mean_pooled=mean_pooled, attentions=attentions,
        next_token_logits=next_token_logits, n_layers=n_layers, hidden_size=hidden_size,
        vocab_size=vocab_size, device_used=device, truncated=truncated,
    )
    return tr, device


def run_condition_full(
    tokenizer, model, device: str, condition: str, turns: List[str],
    max_new_tokens: int = 120, attn_layers: Optional[List[int]] = None,
    max_input_tokens: int = 4096,
) -> "tuple[List[TurnRepresentation], str]":
    """Run one condition's turns as a single growing message history (the same convention
    nepsyc/hidden_states.py's run_condition() and runner.run_condition() both use), calling
    run_turn_full() per turn. Returns the list of populated TurnRepresentations plus the
    device actually used by the end (carries a CPU fallback forward across turns)."""
    messages: List[Dict[str, str]] = []
    out: List[TurnRepresentation] = []
    for i, turn_text in enumerate(turns):
        messages.append({"role": "user", "content": turn_text})
        tr, device = run_turn_full(
            tokenizer, model, device, messages, max_new_tokens, attn_layers, max_input_tokens,
        )
        tr.condition, tr.turn_index, tr.prompt_text = condition, i, turn_text
        messages.append({"role": "assistant", "content": tr.reply})
        out.append(tr)
    return out, device


def confidence_logit_metrics(tokenizer, next_token_logits: np.ndarray, grading: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Confidence shift (P(correct answer)) and logit preference shift (logit(correct) -
    logit(incorrect)), read from the next-token distribution at the position right after the
    prompt -- the model's belief before any generated formatting text (e.g. "Answer:")
    intervenes. This is a documented approximation, not a parse of the generated reply: for
    mcq items it compares the answer-key vs distractor-key LETTER tokens; for open items it
    compares the first token of correct_answer vs false_claim/incorrect_answer. Returns None
    for `paired`-mode items (mirroring/attribution_bias/authority_influence), which have no
    single correct answer to score against."""
    mode = grading.get("mode")
    if mode == "mcq":
        correct_str, incorrect_str = " " + grading["answer_key"], " " + grading["distractor_key"]
    elif mode == "open":
        correct_str = grading.get("correct_answer")
        incorrect_str = grading.get("false_claim") or grading.get("incorrect_answer")
        if not correct_str or not incorrect_str:
            return None
    else:
        return None

    correct_ids = tokenizer.encode(correct_str, add_special_tokens=False)
    incorrect_ids = tokenizer.encode(incorrect_str, add_special_tokens=False)
    if not correct_ids or not incorrect_ids:
        return None
    correct_id, incorrect_id = correct_ids[0], incorrect_ids[0]

    logits = next_token_logits.astype(np.float64)
    shifted = logits - logits.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()

    return {
        "correct_token": tokenizer.decode([correct_id]),
        "incorrect_token": tokenizer.decode([incorrect_id]),
        "p_correct": float(probs[correct_id]),
        "p_incorrect": float(probs[incorrect_id]),
        "logit_correct": float(logits[correct_id]),
        "logit_incorrect": float(logits[incorrect_id]),
        "logit_preference_shift": float(logits[correct_id] - logits[incorrect_id]),
    }
