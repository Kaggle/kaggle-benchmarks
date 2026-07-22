# Copyright 2026 Kaggle Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fairness: model rotation + de-duplication (design doc §10).

Rotate the *author* model (avoid one model's blind spots / home-field advantage)
and the *user-simulator* model, and drop near-duplicate scenarios.

The similarity here is a dependency-free ``difflib`` ratio. The intended
production version embeds scenarios (sentence-transformers) and clusters by
cosine similarity — reuse
``documentation/guides/oulipo.py:calculate_semantic_diversity_score``.
"""

from __future__ import annotations

import difflib
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaggle_benchmarks.agentic.scenario import Scenario, Suite


def pick(items: list, i: int):
    """Round-robin selection — rotate author / user-sim models across tasks."""
    return items[i % len(items)]


def dedup(
    scenarios: list["Scenario"], threshold: float = 0.9
) -> tuple[list["Scenario"], list[tuple[str, str]]]:
    """Drop near-duplicate scenarios; returns (kept, dropped_pairs)."""
    kept: list[Scenario] = []
    dropped: list[tuple[str, str]] = []
    for cand in scenarios:
        clash = next(
            (
                k
                for k in kept
                if difflib.SequenceMatcher(
                    None, cand.signature(), k.signature()
                ).ratio()
                >= threshold
            ),
            None,
        )
        if clash is None:
            kept.append(cand)
        else:
            dropped.append((cand.id, clash.id))
    return kept, dropped


def diversity_report(suite: "Suite") -> dict:
    tags = Counter(t for s in suite for t in s.tags)
    authors = Counter(s.provenance.get("author_model", "?") for s in suite)
    return {
        "n_scenarios": len(suite),
        "tag_coverage": dict(tags),
        "author_distribution": dict(authors),
    }
