"""Local hidden-state extraction for open-weight models (Hugging Face transformers).

`providers.py`'s `OpenAICompatProvider` only ever sees a model's text reply over an API --
that's all a gateway can expose. Reading a model's internal per-layer activations needs the
real weights loaded locally, which is why `ModelSpec.hf_repo_id` (config.py) is a separate,
optional field from the API-serving `id`: a model can be evaluated over Groq/OpenAI/etc. and
still have no local weights available for this module, or vice versa.

Downloading a checkpoint and running forward passes is multi-GB and multi-minute, so this
module is only ever driven by scripts/analyze_hidden_states.py as an offline batch step, never
imported into the Streamlit process -- see that script's docstring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TurnHiddenStates:
    """One conversational turn's prompt-side hidden states, one vector per layer
    (index 0 = embedding layer, matching transformers' own `output_hidden_states` order)."""
    condition: str
    turn_index: int
    prompt_text: str
    reply: str
    layer_vectors: List[np.ndarray] = field(default_factory=list)


def load_model(hf_repo_id: str, device: Optional[str] = None):
    """Load a causal LM + tokenizer from the Hub. bfloat16 on CUDA (matches the original
    prototype), float32 on CPU since not all CPU kernels support bfloat16 matmuls."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    return tokenizer, model, device


def _prepare(tokenizer, messages: List[Dict[str, str]], device: str):
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)


def run_condition(tokenizer, model, device: str, condition: str, turns: List[str],
                  max_new_tokens: int = 120) -> List[TurnHiddenStates]:
    """Run one condition's turns as a single growing message history -- the same "each
    condition is its own fresh conversation, replies feed the next turn" convention
    runner.run_condition() uses for the API-served models, so the pressure/history a local
    model experiences here matches what the real eval sweep would have produced.

    Hidden states are read from the forward pass over the prompt as sent (before that turn's
    reply is generated), one vector per layer at the final token position -- the same point
    the original prototype script sampled.
    """
    import torch

    messages: List[Dict[str, str]] = []
    out: List[TurnHiddenStates] = []
    for i, turn_text in enumerate(turns):
        messages.append({"role": "user", "content": turn_text})
        inp = _prepare(tokenizer, messages, device)
        input_len = inp["input_ids"].shape[1]

        with torch.no_grad():
            hidden_states = model(**inp, output_hidden_states=True, return_dict=True).hidden_states
        layer_vectors = [h[0, -1].float().cpu().numpy() for h in hidden_states]

        with torch.no_grad():
            generated = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        reply = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True).strip()

        messages.append({"role": "assistant", "content": reply})
        out.append(TurnHiddenStates(condition=condition, turn_index=i, prompt_text=turn_text,
                                     reply=reply, layer_vectors=layer_vectors))
    return out


def pca_trajectory(layer_vectors: List[np.ndarray]) -> np.ndarray:
    """2D PCA projection of one turn's per-layer vectors, one point per layer -- a per-turn
    fit (not a shared cross-model/cross-turn basis), same as the original prototype's PCA
    section. Returns an (n_layers, 2) array; degenerate 0/1-layer input returns zeros rather
    than raising, since PCA needs at least 2 samples to fit 2 components."""
    from sklearn.decomposition import PCA

    stacked = np.stack(layer_vectors)
    if stacked.shape[0] < 2:
        return np.zeros((stacked.shape[0], 2))
    return PCA(n_components=2).fit_transform(stacked)
