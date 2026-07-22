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

# %% [markdown]
# # LLMAgent + how it maps to the library
#
# The other demos mock the policy with `PlannedAgent`. The **real** path is
# `LLMAgent`, which wraps an `LLMChat` and runs the native tool loop. Here we
# still mock the *model* — a `ScriptedLLM` built against the real `LLMChat` base
# class (the same trick the library's own tests use) — so this runs offline.

# %%
import itertools

from kaggle_benchmarks import actors
from kaggle_benchmarks.agentic import LLMAgent
from kaggle_benchmarks.llm_messages import LLMMessage

# %% [markdown]
# ## Mock the LLM against the real base class


# %%
class ScriptedLLM(actors.LLMChat):
    """Deterministic LLMChat: returns pre-canned responses (cf. tests/mocks.py)."""

    def __init__(self, contents, **kw):
        super().__init__(**kw)
        self._it = itertools.cycle(
            [LLMMessage(sender=None, content=c) for c in contents]
        )

    def invoke(self, messages, tools=None, **kwargs):
        msg = next(self._it)
        msg.sender = self
        return msg


# %% [markdown]
# ## Drive it through LLMAgent (no tools here) to get a real Trajectory

# %%
agent = LLMAgent(
    ScriptedLLM(["Oct 11 is your best bet — sunny and cheaper."]), name="scripted"
)
response = agent.act([actors.user.send("Best October weekend for Barcelona?")])
print("answer:", response.answer)
response.trajectory

# %% [markdown]
# ## How the `agentic` prototype maps onto the library (design §7)
#
# | `kaggle_benchmarks.agentic` | built on / next step |
# | --- | --- |
# | `Trajectory` | `Message`/`LLMMessage`/`ToolInvocation` steps; `from_chat` adopts a `Chat`; → proto `Conversation` |
# | `LLMAgent` | wraps `LLMChat` + `tools.native.native_tool_agent` (real); `PlannedAgent` is the mocked stand-in |
# | `UserSimulator` | → a `ChatRoom` `Participant` from the persona (rooms.py; hidden info via `post(visible_to=...)`) |
# | `EmulatedTool` | a callable over `scenario.environment`; cache → hishel/joblib |
# | analyzers | return `AssertionResult`; judge → `LLMChat.prompt` (rotate a panel — §10) |
# | `Suite` storage | JSON now; → proto (`BenchmarkTaskVersion` + scenario msg) via `clients` / `kaggle/` |
# | running agents × scenarios | → `orchestration.run_tasks(..., n_jobs=)` |
#
# The one real blocker for the fully-real version: tools inside `ChatRoom`
# (`Participant.reply(tools=...)` currently raises `NotImplementedError`).

# %%
