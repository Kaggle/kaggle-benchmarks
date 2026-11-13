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

"""An agent backed by Google ADK.

Where :class:`~kaggle_benchmarks.agents.GenAIAgent` implements a single
provider call and lets ``LLMActor`` drive the loop, ADK runs a loop of its own.
So this specializes at the other end of the seam — it overrides :meth:`ADKAgent.run`
and translates the events ADK emits back into the agent's session, so a run
looks the same in a transcript whichever backend produced it.

``google-adk`` is imported lazily, so importing this module never requires it.
It is deliberately not declared as an extra: the project pins
``[tool.uv] exclude-newer``, and under that date the index offers only a stub
``google-adk`` release, so declaring it would make the lockfile unresolvable.
Install it alongside instead — ``pip install google-adk``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from kaggle_benchmarks.agents.base import LLMAgent

_USER_ID = "kbench-user"
_SESSION_ID = "kbench-session"


def _require_adk() -> None:
    try:
        import google.adk  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "ADKAgent needs 'google-adk', which is not installed. "
            "Install it with: pip install google-adk"
        ) from exc


def _run_sync(coro):
    """Runs a coroutine from sync code, including inside a notebook loop.

    ``kaggle_benchmarks`` applies ``nest_asyncio`` globally, which makes the
    event loop re-entrant. A side effect is that ``sniffio``/``anyio`` can no
    longer auto-detect the running async library — deep in httpcore,
    ``asyncio.current_task()`` reads ``None`` — which surfaces through ADK as
    ``AsyncLibraryNotFoundError`` as soon as it makes a request. Two things
    avoid it: run the coroutine on a dedicated thread with a fresh loop, and
    pin sniffio's context variable so detection short-circuits. ``asyncio.run``
    copies the worker thread's context, so httpcore sees ``"asyncio"``.
    """
    import sniffio

    result: list[Any] = []
    error: list[BaseException] = []

    def _worker() -> None:
        token = sniffio.current_async_library_cvar.set("asyncio")
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            error.append(exc)
        finally:
            sniffio.current_async_library_cvar.reset(token)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def record_event(event: Any, speaker) -> str | None:
    """Writes one ADK event into the current session; returns any answer text.

    Kept a plain function over duck-typed parts so the translation can be
    tested without ADK installed.
    """
    from kaggle_benchmarks import actors
    from kaggle_benchmarks.llm_messages import LLMMessage
    from kaggle_benchmarks.tools.base import ToolInvocation, ToolInvocationResult

    answer = None
    content = getattr(event, "content", None)
    for part in getattr(content, "parts", None) or []:
        call = getattr(part, "function_call", None)
        result = getattr(part, "function_response", None)
        text = getattr(part, "text", None)

        if call is not None:
            speaker.send(
                LLMMessage(
                    content="",
                    sender=speaker,
                    tool_calls=[
                        ToolInvocation(name=call.name, arguments=dict(call.args or {}))
                    ],
                )
            )
        elif result is not None:
            actors.Tool(name=result.name).send(
                ToolInvocationResult(
                    name=result.name,
                    arguments={},
                    output=getattr(result, "response", None),
                )
            )
        elif text and getattr(part, "thought", False):
            speaker.send(LLMMessage(content="", sender=speaker, reasoning_traces=text))
        elif text:
            speaker.send(LLMMessage(content=text, sender=speaker))
            answer = text
    return answer


class ADKAgent(LLMAgent):
    """An :class:`~kaggle_benchmarks.agents.LLMAgent` that runs on Google ADK.

    Tools and subagents work as they do for any agent: both are handed to ADK
    as callables, so delegation behaves the same here as on the GenAI backend,
    and a subagent still answers inside its own nested session.

    Args:
        model: The Gemini model ADK should drive.
        agent: A ready-made ``google.adk`` agent. Built from ``model``, the
            instructions and the callables when omitted.
        app_name: ADK application name, used for its session bookkeeping.
        **kwargs: Forwarded to ``LLMAgent`` (``tools``, ``subagents``,
            ``instructions``, ``description``, …).
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        agent: Any = None,
        app_name: str = "kbench",
        **kwargs,
    ):
        kwargs.setdefault("name", "adk_agent")
        super().__init__(**kwargs)
        self.model = model
        self.app_name = app_name
        self._agent = agent

    def _build_agent(self) -> Any:
        """The ADK agent, built on first use from this agent's configuration."""
        if self._agent is None:
            _require_adk()
            from google.adk.agents import Agent as _ADKAgent

            self._agent = _ADKAgent(
                name=self.name,
                model=self.model,
                instruction=self.instructions or "",
                tools=self.callables,
            )
        return self._agent

    def run(self, message: str, schema: type = str, **kwargs):
        """Answers ``message`` by running ADK's loop inside this agent's session.

        ADK drives the tool calling itself, so unlike the default this does not
        go through ``native_tool_agent``; the events it emits are recorded into
        the session as they arrive.
        """
        if schema is not str:
            raise NotImplementedError(
                "ADKAgent returns text; structured output is not wired up yet."
            )
        _require_adk()
        from kaggle_benchmarks import actors

        with self.session():
            actors.user.send(message)
            answer = _run_sync(self._run_async(message))
            return self.send(answer or "")

    async def _run_async(self, message: str) -> str | None:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=self.app_name, user_id=_USER_ID, session_id=_SESSION_ID
        )
        runner = Runner(
            agent=self._build_agent(),
            app_name=self.app_name,
            session_service=session_service,
        )

        answer = None
        async for event in runner.run_async(
            user_id=_USER_ID,
            session_id=_SESSION_ID,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            text = record_event(event, self)
            if text is not None:
                answer = text
        return answer
