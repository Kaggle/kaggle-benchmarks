# Copyright 2025 Kaggle Inc.
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

"""The ``Message`` type: the full concrete conversation message.

``Message`` is a :class:`kaggle_benchmarks.core.BaseMessage` (itself an
``Event``) that adds LLM-oriented state — visibility, streaming, token usage,
reasoning traces and tool calls. ``BaseMessage`` and ``Event`` are re-exported
here for convenience; use them for typing when you only need ``content``.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import Any, Iterable, TypeVar

import pydantic

from kaggle_benchmarks import events
from kaggle_benchmarks.core import Actor, BaseMessage, Event, Status

__all__ = ["BaseMessage", "Chunk", "Event", "Message", "T"]

T = TypeVar("T")
Chunk = TypeVar("Chunk")


class Message(BaseMessage[T]):
    """A message sent by an ``Actor`` into a chat."""

    def __init__(
        self,
        content: T,
        sender: Actor,
        _status: Status = Status.SUCCESS,
        is_visible_to_llm: bool = True,
        _meta: dict[str, Any] | None = None,
        *,
        id: str | None = None,
    ):
        super().__init__(content, sender=sender, status=_status, id=id)
        # Controls 1) visibility to the LLM and 2) inclusion in the
        # conversation's protobuf JSON output. Defaults to True.
        self.is_visible_to_llm = is_visible_to_llm
        self._meta: dict[str, Any] = {} if _meta is None else _meta

    @property
    def reasoning_traces(self):
        # TODO: Remove this _meta workaround once invoke() returns LLMMessage
        # directly (PR #115 added this path). LLMMessage.reasoning_traces
        # will be the canonical field.
        return self._meta.get("reasoning_traces")

    @property
    def tool_calls(self):
        return self._meta.get("tool_calls")

    @property
    def usage(self):
        """Token usage and cost metadata for this message."""
        from kaggle_benchmarks.usage import Usage

        return Usage(
            input_tokens=self._meta.get("input_tokens"),
            output_tokens=self._meta.get("output_tokens"),
            input_tokens_cost_nanodollars=self._meta.get(
                "input_tokens_cost_nanodollars"
            ),
            output_tokens_cost_nanodollars=self._meta.get(
                "output_tokens_cost_nanodollars"
            ),
            total_backend_latency_ms=self._meta.get("total_backend_latency_ms"),
        )

    @property
    def payload(self) -> str | list[dict]:
        if hasattr(self.content, "get_payload"):
            return self.content.get_payload()
        if isinstance(self.content, pydantic.BaseModel):
            return self.content.model_dump_json()
        if dataclasses.is_dataclass(self.content) and not isinstance(
            self.content, type
        ):
            return json.dumps(dataclasses.asdict(self.content))
        if "raw_content" in self._meta:
            return self._meta["raw_content"]
        if hasattr(self.content, "__dict__"):
            warnings.warn(
                f"Object of type {type(self.content)} (value: {self.content}) lacks a proper serialization method."
                "Falling back to `__dict__`."
            )
            return json.dumps(self.content.__dict__)
        return self.text

    def __panel__(self):
        from kaggle_benchmarks.ui import panel

        return panel.render_message(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.__panel__()._repr_mimebundle_(include, exclude)

    def to_dict(self):
        """Returns a dictionary representation of the message."""
        return dict(sender={"id": self.sender.id}, content=self.payload)

    def __eq__(self, other):
        # Defining __eq__ makes Message unhashable (Python sets __hash__ = None),
        # matching the old dataclass behavior.
        return (
            isinstance(other, Message)
            and self.sender == other.sender
            # Should work with arbitrary types like pd.DataFrame
            and self.content is other.content
        )

    def _update_from_tool_call_chunk(self, tc_chunk: Any):
        tool_calls = self._meta.setdefault("tool_calls", [])

        while len(tool_calls) <= tc_chunk.index:
            tool_calls.append(
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            )

        current_call = tool_calls[tc_chunk.index]

        if tc_chunk.id:
            current_call["id"] = tc_chunk.id
        if name := getattr(tc_chunk.function, "name", None):
            current_call["function"]["name"] = name
            # TODO(b/452943306): Some models incorrectly send multiple tool calls using
            # the same `index` (e.g., all `index=0`) instead of incrementing it.
            # This reset prevents arguments from different tools from being
            # concatenated into a single, invalid JSON string.
            # This can be removed once the upstream model/API bug is fixed.
            current_call["function"]["arguments"] = ""
        if args := getattr(tc_chunk.function, "arguments", None):
            current_call["function"]["arguments"] += args
        if getattr(tc_chunk, "thought_signature", None) is not None:
            current_call["_thought_signature"] = tc_chunk.thought_signature
        if getattr(tc_chunk, "thought", None) is not None:
            current_call["_thought"] = tc_chunk.thought

    def stream(self, content: Iterable[Chunk]):
        """Streams content into the message, updating its content and metadata.

        Args:
            content: An iterable of chunks. Chunks can be strings or objects
                with `content` and `meta` attributes.
        """
        events.manager.dispatch("start_streaming", self)
        for chunk in content:
            chunk_content = (
                chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
            )
            self.content += chunk_content
            # Token metrics of stream should be reported as of the final chunk, not summed across all chunks.
            if hasattr(chunk, "meta"):
                self._meta.update(chunk.meta)
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tool_call_chunk in chunk.tool_calls:
                    self._update_from_tool_call_chunk(tool_call_chunk)
            events.manager.dispatch("new_chunk", self, chunk)

        # Now that all argument chunks have arrived, the JSON is parseable.
        self._finalize_tool_calls()
        events.manager.dispatch("end_content", self)

    def _finalize_tool_calls(self):
        """Converts accumulated _meta['tool_calls'] dicts to ToolInvocation.

        Must run after the stream loop completes — partial JSON can't be
        parsed mid-stream.
        """
        from kaggle_benchmarks.tools import base as tool_utils

        raw_calls = self._meta.get("tool_calls")
        if not raw_calls:
            return
        self._meta["tool_calls"] = [
            tool_utils.ToolInvocation.from_api_dict(tc) if isinstance(tc, dict) else tc
            for tc in raw_calls
        ]
