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

"""ADK adapter — run a Google **ADK** agent as an eval Agent (design doc §9).

``ADKAgent`` wraps a ``google.adk`` agent (or builds one from a Gemini model +
Python tools) behind the eval ``act(conversation) -> Response`` contract, mapping
ADK's event stream into a library ``Trajectory`` (tool calls, results, reasoning,
final answer).

**Optional dependency.** This module needs ``google-adk`` — install it separately
(``pip install google-adk``); it is intentionally *not* a core dependency, so the
rest of ``kaggle_benchmarks.agentic`` imports without it. Import explicitly:

    from kaggle_benchmarks.agentic.adk import ADKAgent

**Auth.** Uses the Gemini API via ``GOOGLE_API_KEY`` when
``GOOGLE_GENAI_USE_VERTEXAI`` is unset/``FALSE`` (default here); ADK/`google-genai`
pick the key up from the environment.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from kaggle_benchmarks import actors
from kaggle_benchmarks.agentic.agent import Response
from kaggle_benchmarks.agentic.trajectory import Trajectory
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.messages import Message
from kaggle_benchmarks.tools.base import ToolInvocation, ToolInvocationResult

_USER_ID = "kbench-eval-user"
_SESSION_ID = "kbench-eval-session"


def _require_adk():
    try:
        import google.adk  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "ADKAgent requires the optional dependency 'google-adk' "
            "(pip install google-adk)."
        ) from exc


def _run_sync(coro):
    """Run an async coroutine from sync code, including inside a notebook loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a running loop (e.g. Jupyter): allow a nested run.
    import nest_asyncio

    nest_asyncio.apply(loop)
    return loop.run_until_complete(coro)


def _last_user_text(conversation: list[Message]) -> str:
    for msg in reversed(conversation):
        sender = getattr(msg, "sender", None)
        if sender is not None and getattr(sender, "role", "") == "user":
            return str(msg.content)
    return str(conversation[-1].content) if conversation else ""


def _map_event(event: Any, traj: Trajectory, speaker: actors.Actor) -> None:
    """Translate one ADK ``Event`` into trajectory steps."""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    for part in parts or []:
        call = getattr(part, "function_call", None)
        result = getattr(part, "function_response", None)
        text = getattr(part, "text", None)
        is_thought = bool(getattr(part, "thought", False))
        if call is not None:
            traj.add(ToolInvocation(name=call.name, arguments=dict(call.args or {})))
        elif result is not None:
            traj.add(
                ToolInvocationResult(
                    name=result.name,
                    arguments={},
                    output=getattr(result, "response", None),
                )
            )
        elif text and is_thought:
            traj.add(LLMMessage(content="", sender=speaker, reasoning_traces=text))
        elif text:
            traj.add(LLMMessage(content=text, sender=speaker))


def _final_text(event: Any) -> str | None:
    content = getattr(event, "content", None)
    for part in getattr(content, "parts", None) or []:
        if getattr(part, "text", None):
            return part.text
    actions = getattr(event, "actions", None)
    if actions is not None and getattr(actions, "escalate", None):
        return f"[escalated] {getattr(event, 'error_message', '') or ''}"
    return None


class ADKAgent:
    """A Google ADK agent as an eval Agent (conforms to the ``Agent`` protocol).

    Pass an existing ``google.adk`` agent, or let it build one from a Gemini
    ``model`` + Python ``tools`` (+ ``instruction``). ADK is async; ``act`` bridges
    to sync.
    """

    def __init__(
        self,
        agent: Any = None,
        *,
        model: str = "gemini-2.5-flash",
        tools: list[Callable] | None = None,
        instruction: str | None = None,
        name: str | None = None,
        app_name: str = "kbench-eval",
    ):
        _require_adk()
        from google.adk.agents import Agent as _ADKAgent

        if agent is None:
            agent = _ADKAgent(
                name=name or "adk_agent",
                model=model,
                instruction=instruction or "",
                tools=list(tools or []),
            )
        self._agent = agent
        self.model = model
        self.name = name or getattr(agent, "name", "adk-agent")
        self.app_name = app_name

    def act(self, conversation: list[Message]) -> Response:
        return _run_sync(self._act_async(conversation))

    async def _act_async(self, conversation: list[Message]) -> Response:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=self.app_name, user_id=_USER_ID, session_id=_SESSION_ID
        )
        runner = Runner(
            agent=self._agent, app_name=self.app_name, session_service=session_service
        )

        query = _last_user_text(conversation)
        new_message = types.Content(role="user", parts=[types.Part(text=query)])

        speaker = actors.Actor(name=self.name, role="assistant", avatar="🤖")
        traj = Trajectory()
        traj.add(Message(content=query, sender=actors.user))

        answer: str | None = None
        async for event in runner.run_async(
            user_id=_USER_ID, session_id=_SESSION_ID, new_message=new_message
        ):
            _map_event(event, traj, speaker)
            if event.is_final_response():
                answer = _final_text(event)

        traj.answer = answer
        return Response(
            answer=answer,
            trajectory=traj,
            metadata={"framework": "adk", "model": self.model},
        )
