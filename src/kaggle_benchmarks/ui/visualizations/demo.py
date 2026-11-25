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

"""Deterministic sample benchmark used to demo / test the chart library.

Having a realistic, self-contained :class:`LeaderboardData` lets anyone preview
the whole dashboard without live benchmark runs -- ``dashboard(demo_data())``.
"""

from __future__ import annotations

import random

from kaggle_benchmarks.ui.visualizations.data import (
    STANDARD_METRICS,
    LeaderboardData,
    Metric,
)


def demo_data(seed: int = 7) -> LeaderboardData:
    """Return a fixed, plausible multi-metric benchmark for demos and tests."""
    rng = random.Random(seed)

    models = [
        "gemini-3-pro",
        "gpt-5.5",
        "claude-opus-4.8",
        "llama-4-405b",
        "mistral-large-3",
        "qwen-3-72b",
    ]

    # Hand-tuned so there is a clear Pareto frontier (some models dominate on
    # the accuracy/cost trade-off, others are dominated).
    base = {
        "gemini-3-pro": (0.912, 3.10, 4200),
        "gpt-5.5": (0.905, 5.40, 5600),
        "claude-opus-4.8": (0.921, 6.80, 6100),
        "llama-4-405b": (0.842, 0.90, 3300),
        "mistral-large-3": (0.808, 0.55, 2600),
        "qwen-3-72b": (0.774, 0.28, 2100),
    }

    metrics = {
        "score": STANDARD_METRICS["score"],
        "cost_usd": STANDARD_METRICS["cost_usd"],
        "latency_ms": STANDARD_METRICS["latency_ms"],
        "output_tokens": Metric(
            "output_tokens", "Output tokens", higher_is_better=False, fmt="number"
        ),
    }

    scores = {}
    for model, (score, cost, latency) in base.items():
        scores[model] = {
            "score": score,
            "cost_usd": cost,
            "latency_ms": latency,
            "output_tokens": int(1200 + score * 2500 + rng.uniform(-150, 150)),
        }

    tasks = ["coding", "math", "reasoning", "retrieval", "tool_use", "long_context"]
    task_scores = {}
    for model in models:
        overall = base[model][0]
        task_scores[model] = {
            t: max(0.0, min(1.0, overall + rng.uniform(-0.25, 0.2))) for t in tasks
        }

    # Pairwise win rates consistent-ish with the score ordering.
    pairwise = {}
    for a in models:
        pairwise[a] = {}
        for b in models:
            if a == b:
                continue
            gap = base[a][0] - base[b][0]
            pairwise[a][b] = max(0.02, min(0.98, 0.5 + gap * 2.2))

    elo = {}
    for model in models:
        elo[model] = (1000 + base[model][0] * 350, rng.uniform(18, 42))

    pass_at_k = {}
    for model in models[:4]:
        p1 = base[model][0] * 0.7
        pass_at_k[model] = {k: round(1 - (1 - p1) ** k, 4) for k in (1, 2, 4, 8, 16)}

    return LeaderboardData(
        name="Frontier Reasoning Bench",
        models=models,
        metrics=metrics,
        scores=scores,
        task_scores=task_scores,
        pairwise=pairwise,
        elo=elo,
        pass_at_k=pass_at_k,
    )
