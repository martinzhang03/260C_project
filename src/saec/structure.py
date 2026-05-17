"""Layer-type tagging for structure-aware quantization / prioritization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch.nn as nn

from .hf_layers import is_weight_matrix_module


class LayerKind(Enum):
    ATTENTION_PROJ = "attention"
    FFN = "ffn"
    LM_HEAD = "lm_head"
    EMBEDDING = "embedding"
    OTHER = "other"


@dataclass(frozen=True)
class LayerTag:
    name: str
    kind: LayerKind


def classify_linear_name(name: str) -> LayerKind:
    lower = name.lower()
    if "lm_head" in lower:
        return LayerKind.LM_HEAD
    if "wte" in lower or "wpe" in lower or "embed_tokens" in lower or "tok_embeddings" in lower:
        return LayerKind.EMBEDDING
    if (
        ".attn." in lower
        or ".self_attn." in lower
        or ".attention." in lower
    ):
        return LayerKind.ATTENTION_PROJ
    if ".mlp." in lower or ".feed_forward." in lower or ".ffn." in lower:
        return LayerKind.FFN
    if ".attn" in lower and any(
        p in lower for p in ("c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj", "dense")
    ):
        return LayerKind.ATTENTION_PROJ
    return LayerKind.OTHER


def tag_dense_weight_modules(model: nn.Module, prefix: str = "") -> dict[str, LayerTag]:
    out: dict[str, LayerTag] = {}
    for child_name, child in model.named_children():
        fq = f"{prefix}.{child_name}" if prefix else child_name
        out.update(tag_dense_weight_modules(child, fq))
    if is_weight_matrix_module(model):
        fq = prefix
        out[fq] = LayerTag(name=fq, kind=classify_linear_name(fq))
    return out


def kind_priority(kind: LayerKind) -> int:
    """Higher = more likely to keep in higher precision."""
    table = {
        LayerKind.LM_HEAD: 3,
        LayerKind.ATTENTION_PROJ: 2,
        LayerKind.FFN: 1,
        LayerKind.EMBEDDING: 0,
        LayerKind.OTHER: 1,
    }
    return table.get(kind, 0)


def structural_fp16_candidates(tags: dict[str, LayerTag], prefer_attention: bool = True) -> list[str]:
    names = sorted(tags.keys())

    def sort_key(name: str) -> tuple[float, float, str]:
        t = tags[name]
        prio = float(kind_priority(t.kind))
        if not prefer_attention and t.kind == LayerKind.ATTENTION_PROJ:
            prio -= 1.5
        depth = float(name.count("."))
        return (-prio, -depth, name)

    return sorted(names, key=sort_key)
