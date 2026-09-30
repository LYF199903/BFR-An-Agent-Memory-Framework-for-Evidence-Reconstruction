#!/usr/bin/env python3
"""Synthetic two-hop example (no dataset files).

Relevance-only retrieval often stops at the camera-shopping memory.
Answering the duration question needs both the order date and the arrival date.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bfr import run_bfr  # noqa: E402

QUESTION = "How many days after ordering the camera did it arrive?"

MEMORIES = [
    {
        "memory_id": "m_order",
        "session_id": "s1",
        "date": "5 February, 2024",
        "timestamp": "8:00 pm on 5 February, 2024",
        "speaker": "Alex",
        "content": "Alex says they ordered a used camera body online on February 5.",
    },
    {
        "memory_id": "m_arrive",
        "session_id": "s2",
        "date": "10 February, 2024",
        "timestamp": "9:00 am on 10 February, 2024",
        "speaker": "Alex",
        "content": "Alex says the camera package finally arrived this morning, February 10.",
    },
    {
        "memory_id": "m_hike",
        "session_id": "s1",
        "date": "5 February, 2024",
        "timestamp": "8:20 pm on 5 February, 2024",
        "speaker": "Alex",
        "content": "Alex is packing a backpack for a weekend hike and lists snacks, a jacket, and extra socks.",
    },
    {
        "memory_id": "m_tripod",
        "session_id": "s3",
        "date": "12 March, 2024",
        "timestamp": "3:00 pm on 12 March, 2024",
        "speaker": "Alex",
        "content": "Alex talks about selling an old 2018 tripod because it is too heavy.",
    },
    {
        "memory_id": "m_lens",
        "session_id": "s0",
        "date": "20 January, 2024",
        "timestamp": "6:00 pm on 20 January, 2024",
        "speaker": "Alex",
        "content": "Alex browsed camera lenses and said a 50mm prime would be fun someday.",
    },
]


def main() -> None:
    reqs = [
        {
            "requirement_id": "R1",
            "description": "date the camera was ordered",
            "type": "temporal_bridge",
        },
        {
            "requirement_id": "R2",
            "description": "date the camera arrived",
            "type": "temporal_bridge",
        },
    ]
    out = run_bfr(
        QUESTION,
        MEMORIES,
        requirements=reqs,
        method="BFR-MV",
        question_id="toy_camera",
        max_set_size=1,
        budget={"max_reasoning_steps": 2, "max_retrieval_calls": 3, "max_new_memories": 2},
    )
    print(json.dumps(
        {
            "method": out["method"],
            "stage1_seed_ids": out["stage1_seed_ids"],
            "final_evidence_ids": out["final_evidence_ids"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
