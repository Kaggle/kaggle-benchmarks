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

"""Core classes and functions for representing and managing chats."""

import contextlib
import dataclasses
import functools
import uuid
from typing import Any, Iterator, Self

from kaggle_benchmarks import actors, events, utils
from kaggle_benchmarks.messages import Message


@dataclasses.dataclass
class Chat:
    """Represents a thread of messages in a chat."""

    history: list[Message | Self] = dataclasses.field(default_factory=list)
    name: str = "chat"
    id: str = ""
    sender: actors.Actor = actors.system  # added to mach Message's structural type

    _status: utils.Status = utils.Status.PENDING

    def __post_init__(self):
        if not self.id:
            # Generate a unique ID based on the name and a UUID
            self.id = f"{self.name}-{uuid.uuid4().hex[:8]}"

    @property
    def messages(self) -> list[Message]:
        return [m for m in self.history if isinstance(m, Message)]

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        events.manager.dispatch("chat_update", self, value)
        self._status = value

    def __panel__(self):
        from kaggle_benchmarks.ui import panel

        return panel.render_chat_as_step(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        from kaggle_benchmarks.ui import panel

        return panel.render_chat(self)._repr_mimebundle_(include, exclude)

    @events.manager.event_dispatcher("new_message")
    def append(self, item: Message | Self) -> Message | Self:
        """Adds a message to the chat."""
        self.history.append(item)
        return item

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the thread."""
        return dict(messages=[obj.to_dict() for obj in self.history], name=self.name)

    def __str__(self, indent: str = "") -> str:
        messages = "\n".join(m.__str__(indent + "  ") for m in self.history)
        return f"{indent}🧵{self.name or ''}:\n{messages}"


class GoldfishChat(Chat):
    """A chat that keeps only the last message (useful for interactive mode)."""

    @events.manager.event_dispatcher("new_message")
    def append(self, item: Message | Self) -> Message | Self:
        self.history = [item]
        return item


def get_current_chat():
    """Returns the current chat."""
    from kaggle_benchmarks import contexts

    return contexts.get_current().chat


def send(message: Message | Chat) -> Message | Chat:
    """A shortcut to send a message to the current chat."""
    return get_current_chat().append(message)


def emits_message(func):
    @functools.wraps(func)
    def wrapper(self, *args, is_visible_to_llm=True, **kwargs):
        chat = get_current_chat()
        events.manager.dispatch("awaiting_message", chat, self)
        result = func(self, *args, **kwargs)
        if isinstance(result, Message):
            result.is_visible_to_llm = is_visible_to_llm
            if result not in chat.history:
                chat.append(result)
        else:
            chat.append(
                Message(result, sender=self, is_visible_to_llm=is_visible_to_llm)
            )
        return result

    return wrapper


@contextlib.contextmanager
def new(
    name: str = "Chat",
    system_instructions: str | None = None,
    orphan: bool = False,
) -> Iterator[Chat]:
    """Creates a new chat thread within a context manager."""
    from kaggle_benchmarks import contexts

    with contexts.enter(chat=Chat(name=name)) as ctx:
        if system_instructions:
            actors.system.send(system_instructions)

        if not orphan and ctx.parent:
            ctx.parent.chat.append(ctx.chat)

        yield ctx.chat


@contextlib.contextmanager
def fork(name: str = "Fork"):
    parent = get_current_chat()

    with new(name) as chat:
        chat.history.extend(parent.messages)
        yield chat
