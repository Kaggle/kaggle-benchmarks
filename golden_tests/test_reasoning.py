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

"""Reasoning-parameter tasks, and their golden tests (live-only).

Each task is followed by its test. These verify real provider behavior — that
``reasoning=`` is accepted across providers, and that reasoning traces are
captured on the message for models that expose them. A fake model would verify
nothing here, so every test is parametrized over a live pool only and nothing is
scripted: without a configured provider the pools are empty and pytest skips
these tests cleanly.
"""

import pytest
from models import ALL_MODELS, REASONING_TRACE_MODELS

import kaggle_benchmarks as kbench


# Known failure: gemma-4-31b does not support reasoning (kept unexcluded to match
# the original cookbook behavior; every other model must pass).
@kbench.task(name="reasoning_param")
def reasoning_param(llm):
    """Tests that the unified reasoning parameter works across providers."""
    response = llm.prompt(
        "What is 2 + 2? Reply with just the number.",
        reasoning="low",
    )

    kbench.assertions.assert_contains_regex(
        r"4",
        response,
        expectation="Model should answer 4.",
    )


@pytest.mark.parametrize("llm", ALL_MODELS)
def test_reasoning_param(llm):
    assert reasoning_param.run(llm).passed


@kbench.task(name="reasoning_captures_traces")
def reasoning_captures_traces(llm):
    """Tests that reasoning captures reasoning traces on the message."""
    llm.prompt(
        "How many r's are in the word 'strawberry'? Think step by step.",
        reasoning="high",
    )

    chat = kbench.chats.get_current_chat()
    last_message = chat.messages[-1]
    assert last_message.reasoning_traces is not None, (
        "Reasoning traces should be accessible via message.reasoning_traces"
    )
    assert len(last_message.reasoning_traces) > 0, (
        "Reasoning traces should not be empty"
    )


@pytest.mark.parametrize("llm", REASONING_TRACE_MODELS)
def test_reasoning_captures_traces(llm):
    assert reasoning_captures_traces.run(llm).passed
