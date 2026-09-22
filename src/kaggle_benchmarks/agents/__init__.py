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

"""Agents: a second way to define a benchmark, alongside ``actors.LLMChat``.

``LLMActor`` is the small interface a backend implements; ``LLMAgent`` adds the
instructions, tools and subagents that let agents compose. ``GenAIAgent`` and
``ADKAgent`` are the two backends.
"""

from kaggle_benchmarks.agents.adk import ADKAgent
from kaggle_benchmarks.agents.base import LLMActor, LLMAgent
from kaggle_benchmarks.agents.genai import GenAIAgent

__all__ = ["ADKAgent", "GenAIAgent", "LLMActor", "LLMAgent"]
