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

import dataclasses
import json
import warnings
from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar

from kaggle_benchmarks import events, utils

if TYPE_CHECKING:
    from kaggle_benchmarks import actors


T = TypeVar("T")
Chunk = TypeVar("Chunk")


@dataclasses.dataclass
class Message(Generic[T]):
    content: T
    sender: "actors.Actor"
    _status: utils.Status = utils.Status.SUCCESS
    # Controls 1) visibility to the LLM and 2) inclusion in the
    # conversation's protobuf JSON output. Defaults to True.
    is_visible_to_llm: bool = True
    _meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value: utils.Status):
        events.manager.dispatch("message_update", self, value)
        self._status = value

    @property
    def text(self):
        return str(self.content)

    @property
    def tool_calls(self):
        return self._meta.get("tool_calls")

    @property
    def payload(self) -> str | list[dict]:
        if hasattr(self.content, "get_payload"):
            return self.content.get_payload()
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
        from kaggle_benchmarks import ui

        return ui.panel.render_message(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.__panel__()._repr_mimebundle_(include, exclude)

    def to_dict(self):
        """Returns a dictionary representation of the message."""
        return dict(sender={"id": self.sender.id}, content=self.payload)

    def __str__(self, indent: str = "") -> str:
        return f"{indent}{self.sender.avatar} [{self.sender.name}]: {self.text}"

    def __eq__(self, other):
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

    def stream(self, content: Iterable[Chunk]):
        """Streams content into the message, updating its content and metadata.

        Args:
            content: An iterable of chunks. Chunks can be strings or objects
                with `content` and `meta` attributes.
        """
        from kaggle_benchmarks import events

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

        events.manager.dispatch("end_content", self)
