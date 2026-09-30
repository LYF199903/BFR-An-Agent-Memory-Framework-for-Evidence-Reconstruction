"""Text encoder used by FCA-MS coverage scoring.

Tries ``sentence-transformers`` when installed. Falls back to a hashed
bag-of-words vector so the framework runs without downloading a model.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

_ENCODER = None
_ENCODER_NAME: Optional[str] = None
_USE_ST: Optional[bool] = None

DIM = 384


def _hashed_bow(text: str, dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    toks = [t.lower() for t in (text or "").split() if t]
    if not toks:
        return vec
    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] += 0.5
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def _get_st_encoder(model_name: str):
    global _ENCODER, _ENCODER_NAME, _USE_ST
    if _USE_ST is False:
        return None
    if _ENCODER is not None and _ENCODER_NAME == model_name:
        return _ENCODER
    try:
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer(model_name)
        _ENCODER_NAME = model_name
        _USE_ST = True
        return _ENCODER
    except Exception:
        _USE_ST = False
        _ENCODER = None
        return None


def encode_query(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    enc = _get_st_encoder(model_name)
    if enc is not None:
        return np.asarray(enc.encode([text or ""])[0], dtype=np.float32)
    return _hashed_bow(text or "")
