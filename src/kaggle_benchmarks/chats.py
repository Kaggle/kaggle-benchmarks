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

"""The ``Chat`` type and helpers for creating and managing chats.

``Chat`` is a concrete :class:`kaggle_benchmarks.core.Session` — an ordered
list of events (messages and nested chats). ``Session`` is re-exported here
for convenience, and ``Message`` is re-exported so that ``chats.Message``
keeps resolving.
"""

from __future__ import annotations

import contextlib
import functools
import uuid
from collections.abc import Iterable
from typing import Iterator

from kaggle_benchmarks import actors, events
from kaggle_benchmarks.core import Actor, Event, Session, Status
from kaggle_benchmarks.messages import Message
from kaggle_benchmarks.usage import Usage

__all__ = [
    "Chat",
    "GoldfishChat",
    "Message",
    "Session",
    "emits_message",
    "fork",
    "get_current_chat",
    "last_reasoning_traces",
    "new",
    "send",
]


class Chat(Session):
    """Represents a thread of messages in a chat."""

    # Name dispatched by Event.status setter when this chat's status changes.
    _status_event = "chat_update"

    def __init__(
        self,
        *,
        history: Iterable[Event] = (),
        name: str = "chat",
        sender: Actor | None = None,
        status: Status = Status.PENDING,
        id: str | None = None,
    ):
        super().__init__(
            history=history,
            name=name,
            # A chat carries a sender so it matches Message's structural type
            # when it sits (nested) in another chat's history.
            sender=actors.system if sender is None else sender,
            status=status,
            id=id,
        )
        # Preserve the historical name-derived id format ("<name>-<hex8>") for
        # backward compatibility (e.g. serialized conversation ids). Event's
        # __init__ assigns a bare uuid; override it when no explicit id is given.
        if id is None:
            self.id = f"{self.name}-{uuid.uuid4().hex[:8]}"

    @property
    def messages(self) -> list[Message]:
        return [m for m in self.history if isinstance(m, Message)]

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the chat.

        Emits the legacy ``messages`` key alongside the ``history`` key so that
        consumers written before the Event refactor keep working.
        """
        serialized = [obj.to_dict() for obj in self.history]
        return dict(messages=serialized, history=serialized, name=self.name)

    @property
    def usage(self) -> Usage:
        """Aggregated token usage and cost metadata across all assistant messages."""
        total = Usage()
        for m in self.messages:
            if m.sender.role == "assistant":
                total = total + m.usage
        return total

    def __panel__(self):
        from kaggle_benchmarks.ui import panel

        return panel.render_chat_as_step(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        from kaggle_benchmarks.ui import panel

        return panel.render_chat(self)._repr_mimebundle_(include, exclude)


class GoldfishChat(Chat):
    """A chat that keeps only the last message (useful for interactive mode)."""

    @events.manager.event_dispatcher("new_event", "new_message")
    def append(self, item: Event) -> Event:
        self.history = [item]
        return item


def get_current_chat():
    """Returns the current chat."""
    from kaggle_benchmarks import contexts

    return contexts.get_current().chat


def last_reasoning_traces() -> str | None:
    """Returns the reasoning traces from the last message in the current chat."""
    messages = get_current_chat().messages
    return messages[-1].reasoning_traces if messages else None


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
def fork(name: str = "Fork", orphan: bool = False):
    """
    Creates and enters a new chat with the same history as the current one.

    Args:
        name: The name of the new chat thread.
        orphan: Controls whether the new chat will appear in the parent chat's history.
    """
    parent = get_current_chat()
    messages = parent.messages

    with new(name, orphan=orphan) as chat:
        chat.history.extend(messages)
        yield chat
