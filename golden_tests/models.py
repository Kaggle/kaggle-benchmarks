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

"""Model pools for the golden tests, as ready-to-parametrize LLM instances.

Every model is named exactly once, in its provider's table below, alongside what
it can be used for. The pools are filters over those rows, so a live test just
parametrizes over the pool it needs::

    @pytest.mark.parametrize("llm", VISION_MODELS)
    def test_image_base64(llm):
        assert image_base64.run(llm).passed

When no model provider is configured the pools are **empty**, so those tests skip
cleanly (pytest skips an empty parameter set) and the suite stays green without
an API key. Offline coverage comes from each task's scripted test, which builds
a fake model with :func:`fake`::

    def test_image_base64_scripted():
        assert image_base64.run(fake(["Red."])).passed
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import pytest

import kaggle_benchmarks as kbench
from kaggle_benchmarks.testing import ScriptedLLM


@dataclasses.dataclass(frozen=True)
class Model:
    """A model and what the suite may use it for.

    The first flags are provider capabilities. The rest mark a model as curated
    for one suite — either because the scenario needs a dependable model, or
    because this one is known to misbehave there.
    """

    name: str
    #: Provider natively honors response_format / a JSON schema. This is what the
    #: report records as `structured_output`; models without it still get schema
    #: instructions in the prompt, so no pool filters on it today.
    structured: bool = True
    vision: bool = False
    audio: bool = False
    video: bool = False
    #: Returns reasoning traces on the response.
    traces: bool = False
    #: Dependable for multi-participant ChatRoom orchestration.
    chatroom: bool = False
    #: Dependable for streaming tool-call loops.
    streaming_tools: bool = False
    #: Dependable in tool loops, and with tools plus structured output together.
    tool_loops: bool = True
    tools_with_schema: bool = True
    #: Part of the default pool that model-agnostic tests run against.
    default: bool = True
    #: The stand-in for tests where the model isn't the subject (judge, dataset).
    reference: bool = False
    #: Supports the provider parameters test_provider_params exercises.
    probe: bool = False


GOOGLE = [
    Model(
        "google/gemini-3.6-flash",
        vision=True,
        audio=True,
        video=True,
        traces=True,
        chatroom=True,
        reference=True,
        streaming_tools=True,
    ),
    Model("google/gemini-2.5-pro", vision=True, audio=True, video=True, traces=True),
    Model(
        "google/gemini-3-flash-preview",
        vision=True,
        audio=True,
        video=True,
        traces=True,
        chatroom=True,
        probe=True,
    ),
    Model("google/gemini-3.1-flash-lite-preview", vision=True, audio=True),
    Model("google/gemma-4-31b", structured=False, vision=True),
]

ANTHROPIC = [
    Model(
        "anthropic/claude-sonnet-4-6@default",
        vision=True,
        chatroom=True,
        streaming_tools=True,
    ),
    Model("anthropic/claude-opus-4-7@default", vision=True),
]

OPENAI = [
    Model("openai/gpt-5.5-2026-04-23", vision=True),
]

QWEN = [
    Model("qwen/qwen3-235b-a22b-instruct-2507", structured=False),
    Model("qwen/qwen3-next-80b-a3b-instruct", structured=False),
]

DEEPSEEK = [
    # deepseek-r1 is unreliable in tool loops; both are unreliable once tools and
    # structured output are combined.
    Model(
        "deepseek-ai/deepseek-r1-0528",
        structured=False,
        tool_loops=False,
        tools_with_schema=False,
    ),
    Model("deepseek-ai/deepseek-v3.1", structured=False, tools_with_schema=False),
]

#: Every model the suite knows about, including any outside the default pool.
ALL = GOOGLE + ANTHROPIC + OPENAI + QWEN + DEEPSEEK

#: The models that model-agnostic tests run against.
DEFAULT = [m for m in ALL if m.default]


def _params(models: Iterable[Model], apis=("openai", "genai")) -> list:
    """Loads each model once per api, as pytest params (empty when unconfigured)."""
    if not kbench.kaggle.is_configured():
        return []
    # Materialize first: `models` is usually a generator, which a per-api loop
    # would exhaust on the first pass.
    selected = sorted(models, key=lambda model: model.name)
    return [
        pytest.param(kbench.kaggle.load_model(m.name, api=api), id=f"{api}-{m.name}")
        for api in apis
        for m in selected
    ]


def fake(responses: list[Any], *, cycle: bool = False, name: str = "scripted"):
    """A fake model replaying ``responses``, for a task's scripted test."""
    return ScriptedLLM(responses, name=name, cycle=cycle)


def judge_model():
    """The live judge model, or ``None`` when no provider is configured."""
    if not kbench.kaggle.is_configured():
        return None
    return kbench.kaggle.load_model(next(m.name for m in ALL if m.reference))


ALL_MODELS = _params(DEFAULT)
VISION_MODELS = _params(m for m in DEFAULT if m.vision)
AUDIO_MODELS = _params(m for m in DEFAULT if m.audio)
VIDEO_MODELS = _params(m for m in DEFAULT if m.video)
REASONING_TRACE_MODELS = _params(m for m in DEFAULT if m.traces)
CHATROOM_MODELS = _params(m for m in DEFAULT if m.chatroom)
TOOL_MODELS = _params(m for m in DEFAULT if m.tool_loops)
TOOL_SCHEMA_MODELS = _params(m for m in DEFAULT if m.tools_with_schema)
REFERENCE_MODELS = _params(m for m in DEFAULT if m.reference)

# Curated outside the default pool.
STREAMING_TOOL_MODELS = _params(m for m in ALL if m.streaming_tools)

# Provider-parameter probes: one model, pinned to a single api.
OPENAI_PROBE_MODELS = _params((m for m in ALL if m.probe), apis=("openai",))
GENAI_PROBE_MODELS = _params((m for m in ALL if m.probe), apis=("genai",))
