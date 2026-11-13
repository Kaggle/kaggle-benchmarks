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

"""A second way to write a benchmark, alongside ``LLMChat``.

``LLMChat`` carries the shape of the backend it grew up with: flags for what a
model cannot do, Model-Proxy workarounds for structured output and tool calls,
streaming. Benchmarks written against it inherit all of that. Changing it is not
an option — too much already depends on its behaviour — so this is a parallel
surface, and the two coexist.

What is left once the workarounds are gone is a small interface:

* :meth:`LLMActor.invoke` — one call to a provider, and the only thing a
  backend has to implement.
* :meth:`LLMActor.respond` — one turn: ask for the schema, invoke, parse.
* :meth:`LLMActor.prompt` — the whole exchange, including the tool loop.

:class:`LLMAgent` adds what makes an actor composable: instructions, tools it
owns rather than is handed, and subagents to delegate to. An ``LLMAgent`` with
none of those is just an ``LLMActor`` — which is the shape ``LLMChat`` could
eventually be retired into.
"""

from __future__ import annotations

import contextlib
import re
from typing import Callable, TypeVar

from kaggle_benchmarks import prompting
from kaggle_benchmarks.core import Actor, Status
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.messages import Message

T = TypeVar("T")


class ToolInvocationLimitExhausted(Exception):
    """Raised when the model keeps asking for tools past the allowed rounds."""


class LLMActor(Actor):
    """An actor that answers by calling a language model.

    Backends implement :meth:`invoke`; the rest is provider-agnostic. Unlike
    ``LLMChat`` there are no capability flags: a backend is expected to handle
    structured output and tool calls natively, and to fail if it cannot, rather
    than have this class paper over it.
    """

    def __init__(self, *, name: str | None = None, **kwargs):
        kwargs.setdefault("role", "assistant")
        kwargs.setdefault("avatar", "🤖")
        super().__init__(name=name or type(self).__name__, **kwargs)

    def invoke(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        schema: type = str,
        tools: list[Callable] | None = None,
        **kwargs,
    ) -> LLMMessage:
        """Makes one call to the provider and returns what it said.

        Args:
            messages: The conversation so far, oldest first.
            system: System instructions for this call.
            schema: The type the answer should take. ``str`` asks for text.
            tools: Callables the model may ask for. An implementation reports
                requested calls on the returned message's ``tool_calls``.

        Returns:
            The model's reply, ``content`` still raw text; :meth:`respond`
            parses it.
        """
        raise NotImplementedError

    def respond(
        self,
        *,
        system: str | None = None,
        schema: type[T] = str,
        tools: list[Callable] | None = None,
        input_messages: list[Message] | None = None,
        **kwargs,
    ) -> Message:
        """Takes one turn in the current chat, records it, and parses the reply."""
        from kaggle_benchmarks import actors, chats

        chat = chats.get_current_chat()
        view = input_messages if input_messages is not None else chat.messages

        handler = prompting.process_schema(schema)
        instructions = next(handler)
        if isinstance(instructions, (list, tuple)):
            _, schema = instructions

        response = self.invoke(
            [m for m in view if m.is_visible_to_llm],
            system=system,
            schema=schema,
            tools=tools,
            **kwargs,
        )
        response.sender = self
        response._meta.update(chat=chat, schema=schema, raw_content=response.content)
        chat.append(response)

        if not response.content:
            # Nothing to parse: the model asked for a tool instead of answering.
            return response

        try:
            handler.send(response.content)
        except prompting.ResponseParsingError as error:
            actors.system.send(str(error))
            response.status = Status.FAILED
            raise
        except StopIteration as stop:
            response.content = stop.value
            return response

        raise prompting.SchemaError(
            f"Generator for {schema!r} yielded twice; expected one value."
        )

    def prompt(
        self,
        message: str,
        schema: type[T] = str,
        tools: list[Callable] | None = None,
        max_tool_rounds: int = 10,
        **kwargs,
    ) -> T:
        """Asks ``message`` and returns the answer, running any tool calls.

        The exchange runs in a forked chat, so tool traffic and schema
        instructions stay out of the caller's history; only the answer goes back.
        """
        from kaggle_benchmarks import chats

        with chats.fork(self.name):
            answer = self.exchange(
                message,
                schema=schema,
                tools=tools,
                max_tool_rounds=max_tool_rounds,
                **kwargs,
            )

        self.send(answer)
        return answer

    def exchange(
        self,
        message: str,
        schema: type[T] = str,
        tools: list[Callable] | None = None,
        max_tool_rounds: int = 10,
        **kwargs,
    ) -> T:
        """Runs the ask/answer loop in the *current* chat, tool calls included.

        :meth:`prompt` wraps this in a fork to keep tool traffic out of the
        caller's history. A caller that already owns the chat it wants the
        exchange recorded in — an agent in its session — calls this directly.
        """
        from kaggle_benchmarks import actors
        from kaggle_benchmarks.tools.native import invoke_tool

        actors.user.send(message)

        for _ in range(max_tool_rounds):
            response = self.respond(schema=schema, tools=tools, **kwargs)
            if not (tools and response.tool_calls):
                return response.content
            for call in response.tool_calls:
                actors.Tool(name=call.name).send(invoke_tool(call, tools))

        raise ToolInvocationLimitExhausted(
            f"Model still requesting tools after {max_tool_rounds} rounds."
        )


class LLMAgent(LLMActor):
    """An :class:`LLMActor` carrying its own instructions, tools and subagents.

    An actor is handed its tools per call; an agent *has* them, which is what
    lets agents compose. A subagent is exposed to its parent as one more
    callable, so from the model's side delegating is just another tool call.

    A subagent is not quite a tool, though. A tool is a function — same inputs,
    same outputs, no memory — while a subagent is an actor with a conversation,
    and it should still remember the first question when the parent asks a
    second. So each agent answers inside its own session, opened on first call
    and reused for the rest of the run. ``Session`` is an ``Event``, so that
    session nests in the parent's history: a delegation reads as a branch of the
    transcript, and serializing the parent carries the subagent's turns with it.

    Args:
        name: How the agent is addressed, and the tool name a parent sees.
        instructions: System prompt, sent once when the session opens.
        description: What this agent is for. As a subagent this becomes the tool
            description the parent's model picks from, so it should read as one.
        tools: Callables the model may call.
        subagents: Agents this one may delegate to.
        max_tool_rounds: Cap on tool-calling rounds per :meth:`run`.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        instructions: str | None = None,
        description: str = "",
        tools: list[Callable] | None = None,
        subagents: list["LLMAgent"] | None = None,
        max_tool_rounds: int = 10,
        **kwargs,
    ):
        kwargs.setdefault("avatar", "🕵️")
        super().__init__(name=name, **kwargs)
        self.instructions = instructions
        self.description = description
        self.tools = list(tools or [])
        self.subagents = list(subagents or [])
        self.max_tool_rounds = max_tool_rounds
        self._session = None

    @property
    def callables(self) -> list[Callable]:
        """Everything the model may call: plain tools, then subagents."""
        return [*self.tools, *(agent.as_tool() for agent in self.subagents)]

    def as_tool(self) -> Callable:
        """Exposes this agent to a parent as a callable it can invoke."""

        def call_subagent(request: str) -> str:
            """Delegates to this agent and returns its answer."""
            return str(self.run(request))

        call_subagent.__name__ = _tool_name(self.name)
        call_subagent.__doc__ = (
            self.description or f"Delegates a request to the {self.name} agent."
        )
        return call_subagent

    @contextlib.contextmanager
    def session(self):
        """Enters this agent's session, opening it on first use."""
        from kaggle_benchmarks import actors, chats, contexts

        if self._session is None:
            with chats.new(self.name) as session:
                self._session = session
                if self.instructions:
                    actors.system.send(self.instructions)
                yield session
        else:
            with contexts.enter(chat=self._session):
                yield self._session

    def reset(self) -> None:
        """Drops the session, so the next :meth:`run` starts from nothing."""
        self._session = None

    def run(self, message: str, schema: type[T] = str, **kwargs) -> T:
        """Answers ``message`` in this agent's session, using its own tools."""
        with self.session():
            return self.exchange(
                message,
                schema=schema,
                tools=self.callables or None,
                max_tool_rounds=self.max_tool_rounds,
                **kwargs,
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"tools={len(self.tools)}, subagents={len(self.subagents)})"
        )


def _tool_name(name: str) -> str:
    """Turns an agent name into an identifier a provider will accept."""
    cleaned = re.sub(r"\W+", "_", name).strip("_").lower()
    return f"ask_{cleaned}" if cleaned else "ask_agent"
