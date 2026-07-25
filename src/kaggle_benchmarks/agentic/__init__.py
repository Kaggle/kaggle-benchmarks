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

"""Agentic end-to-end evaluation — **experimental**.

Prototype of the vision in ``design/agentic-evaluation.md``, built on the
library's own types (``Chat`` / ``Message`` / ``LLMMessage`` / ``Usage`` /
``ToolInvocation`` / ``AssertionResult``). Import explicitly:

    from kaggle_benchmarks.agentic import Trajectory, simulate, Examiner

This subpackage is intentionally **not** imported by ``kaggle_benchmarks`` at
top level (it's opt-in and unstable). The LLM/agent parts have mocked stand-ins
(``agentic.demo``) so flows run without a configured model.
"""

from kaggle_benchmarks.agentic.agent import (
    Action,
    Agent,
    Call,
    ConstantAgent,
    LLMAgent,
    PlannedAgent,
    Reason,
    Response,
    Say,
    act_in_current_chat,
    as_agent,
)
from kaggle_benchmarks.agentic.analyzers import (
    Analyzer,
    ErrorClass,
    answer_mentions,
    called_tool,
    error_class_of,
    judge,
    max_steps,
    no_tool_errors,
    reasoning_mentions,
    run_analyzers,
)
from kaggle_benchmarks.agentic.examiner import Examiner, Report
from kaggle_benchmarks.agentic.fairness import dedup, diversity_report, pick
from kaggle_benchmarks.agentic.scenario import Persona, Scenario, Suite
from kaggle_benchmarks.agentic.simulation import (
    EmulatedTool,
    LLMEmulatedTool,
    ToolSpec,
    UserSimulator,
    WorldModel,
    build_toolset,
    emulate,
    simulate,
)
from kaggle_benchmarks.agentic.trajectory import Trajectory

__all__ = [
    # trajectory
    "Trajectory",
    # agent
    "Agent",
    "Response",
    "PlannedAgent",
    "ConstantAgent",
    "LLMAgent",
    "as_agent",
    "act_in_current_chat",
    "Action",
    "Reason",
    "Call",
    "Say",
    # analyzers
    "Analyzer",
    "ErrorClass",
    "run_analyzers",
    "error_class_of",
    "called_tool",
    "no_tool_errors",
    "max_steps",
    "reasoning_mentions",
    "answer_mentions",
    "judge",
    # scenario
    "Persona",
    "Scenario",
    "Suite",
    # simulation / tools
    "EmulatedTool",
    "emulate",
    "ToolSpec",
    "LLMEmulatedTool",
    "WorldModel",
    "build_toolset",
    "UserSimulator",
    "simulate",
    # examiner / fairness
    "Examiner",
    "Report",
    "pick",
    "dedup",
    "diversity_report",
]
