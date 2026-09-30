# BFR: Evidence Sufficiency for Long-Term Memory QA

**Budgeted Flat Reconstruction (BFR)** builds an evidence set in two stages. FCA-MS selects complementary memories for a question's information requirements. BFR then retrieves additional records from a flat memory store under a shared call and retention budget. This repository contains the retrieval implementation and a synthetic example. It does not contain the manuscript, benchmark datasets, run caches, or model weights.

## Quick start

Requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python examples/toy_bfr.py
```

The example uses five synthetic memories. With a one-record Stage I seed, the camera-order record is initially absent; BFR-MV retrieves it under a fixed budget. It demonstrates **evidence acquisition**, not arithmetic answer generation or benchmark reproduction.

## Use the pipeline

```python
from bfr import run_bfr

memories = [
    {"memory_id": "order", "content": "Ordered a camera on February 5."},
    {"memory_id": "arrival", "content": "The camera arrived on February 10."},
]
out = run_bfr(
    "How many days after ordering the camera did it arrive?",
    memories,
    method="BFR-MV",
    max_set_size=1,
    budget={"max_reasoning_steps": 2, "max_retrieval_calls": 3, "max_new_memories": 2},
)
print(out["stage1_seed_ids"], out["final_evidence_ids"])
```

`method` accepts `FCA-MS`, `OnePass-MV`, `BFR-Text`, or `BFR-MV`. You can pass explicit `requirements` as a list of dictionaries with `requirement_id`, `description`, and `type`; otherwise a small rule-based sketch is used. Each memory needs `memory_id` and `content`. Optional fields include `speaker`, `timestamp` or `date`, `session_id`, `context`, and `keywords`.

The default FCA-MS encoder is a hashed bag-of-words representation. If `sentence-transformers` and `torch` are installed, the code attempts to use `all-MiniLM-L6-v2`; that optional model may need a download. BM25 uses `rank-bm25` when installed and a local fallback otherwise.

## Repository layout

| Path | Purpose |
| --- | --- |
| `bfr/` | Public `run_bfr` entry point |
| `experiments/fca_real_locomo/` | FCA-MS selection, attributes, and prompt templates |
| `analysis/` | Evidence state and budgeted retrieval loop |
| `models/` | Cue generation and candidate routing |
| `retrieval/` | Shared flat and multi-view retrieval |
| `evaluation/` | Offline metrics and diagnostic answer proxy |
| `tools/iclr/` | Method constructors and evaluation adapters |
| `examples/toy_bfr.py` | Synthetic example |

## License

MIT. See `LICENSE`.
