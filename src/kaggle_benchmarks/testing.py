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

"""Public test doubles for running benchmarks without a real LLM API.

The everyday tool here is :class:`ScriptedLLM` — an :class:`~kaggle_benchmarks.actors.LLMChat`
that replays a fixed list of responses instead of calling a model. Because it
plugs in at the ``invoke()`` boundary, the responses still flow through the real
``respond()`` / ``prompt()`` path: schema parsing, assertions, chat history and
serialization all run exactly as they would with a live backend — just offline
and deterministically.

    from kaggle_benchmarks import testing

    llm = testing.ScriptedLLM(["Paris"])
    run = my_task.run(llm)          # no API key required
    assert run.passed

Scripted responses are matched to the ``schema=`` a task requests:

* ``schema=str``          → the raw string is returned as-is (``"Paris"``).
* ``schema=int|bool|...`` → a ``TypedResponse`` JSON wrapper, e.g. ``{"value": 1969}``.
* ``schema=dict|dataclass|pydantic`` → JSON matching that schema.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

from kaggle_benchmarks.actors import LLMChat
from kaggle_benchmarks.llm_messages import LLMMessage

__all__ = ["ScriptedLLM"]


class ScriptedLLM(LLMChat):
    """An ``LLMChat`` that replays pre-scripted responses, one per ``invoke``.

    Args:
        responses: The responses to return, in order. Each item is either a
            ``str`` (returned verbatim) or a JSON-serializable object (which is
            ``json.dumps``-ed — convenient for structured-output schemas). An
            ``LLMMessage`` is also accepted (its ``content`` is used).
        name: Display name for the fake model.
        cycle: If ``True``, loop over ``responses`` forever instead of raising
            when they run out (useful for chatrooms / repeated calls).
        support_structured_outputs: Mirrors the real ``LLMChat`` flag. Scripted
            content is parsed either way, so the default matches the base class.

    Attributes:
        invocations: A list of ``(messages, kwargs)`` recorded for every
            ``invoke`` call, so tests can assert on what the harness sent.
    """

    def __init__(
        self,
        responses: list[Any],
        *,
        name: str = "ScriptedLLM",
        cycle: bool = False,
        support_structured_outputs: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            support_structured_outputs=support_structured_outputs,
            **kwargs,
        )
        templates = [self._as_template(r) for r in responses]
        if not templates:
            raise ValueError("ScriptedLLM needs at least one response.")
        self._templates = templates
        self._iter = itertools.cycle(templates) if cycle else iter(templates)
        self.invocations: list[tuple[list, dict]] = []

    def _as_template(self, response: Any) -> LLMMessage:
        """Normalizes a scripted response into an ``LLMMessage`` template.

        An ``LLMMessage`` is kept as-is, preserving ``tool_calls`` /
        ``reasoning_traces`` / ``usage`` for tool loops.
        """
        if isinstance(response, LLMMessage):
            return response
        content = response if isinstance(response, str) else json.dumps(response)
        return LLMMessage(sender=self, content=content)

    def _clone(self, template: LLMMessage) -> LLMMessage:
        """Returns a fresh copy of a template so replays never mutate the source.

        ``respond()`` rewrites ``.content`` to the parsed schema value and sets
        the sender, so returning the stored object directly would corrupt it for
        a later replay (e.g. ``cycle=True``).
        """
        tool_calls = getattr(template, "tool_calls", None)
        return LLMMessage(
            content=template.content,
            sender=self,
            reasoning_traces=getattr(template, "reasoning_traces", None),
            tool_calls=list(tool_calls) if tool_calls else None,
            usage=getattr(template, "usage", None),
            is_visible_to_llm=getattr(template, "is_visible_to_llm", True),
        )

    @classmethod
    def from_contents(cls, contents: list[str], **kwargs) -> "ScriptedLLM":
        """Builds a ``ScriptedLLM`` from a list of raw string responses."""
        return cls(list(contents), **kwargs)

    @classmethod
    def from_contents_data(cls, contents: list[Any], **kwargs) -> "ScriptedLLM":
        """Builds a ``ScriptedLLM`` from objects serialized to JSON responses."""
        return cls([json.dumps(c) for c in contents], **kwargs)

    def invoke(self, messages, tools=None, **kwargs):  # noqa: D102 (see LLMChat)
        self.invocations.append((messages, kwargs))
        try:
            template = next(self._iter)
        except StopIteration:
            raise AssertionError(
                f"ScriptedLLM {self.name!r} ran out of scripted responses "
                f"({len(self._templates)} provided). Add more responses or "
                f"pass cycle=True."
            ) from None
        return self._clone(template)
