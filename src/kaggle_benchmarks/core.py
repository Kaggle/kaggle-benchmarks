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

"""Core base classes: ``Actor``, ``Event``, ``BaseMessage``, and ``Session``.

These are the "core objects" the rest of the library is built on:

* ``Actor``       — something that participates in a conversation and produces
                    events. Concrete actors (``Tool``, ``system``, ``user`` …)
                    live in :mod:`kaggle_benchmarks.actors.base`.
* ``Event``       — a single item in a ``Session``'s history. Carries a
                    ``sender``, a ``status`` and an ``id``.
* ``BaseMessage`` — a minimal ``Event`` that carries ``content``. It is kept
                    deliberately small (no usage/reasoning/tool-call fields) so
                    it can be used as a lightweight type. The full concrete
                    ``Message`` (in :mod:`kaggle_benchmarks.messages`) subclasses
                    it and is what the library builds at runtime.
* ``Session``     — an ordered list of ``Event`` objects. Because a ``Session``
                    is itself an ``Event``, a session's history can hold both
                    messages and nested sessions. ``Chat`` (in
                    :mod:`kaggle_benchmarks.chats`) is the concrete session type.

Keeping the bases here lets them reference each other directly without the
circular imports that arise when the concrete ``Message``/``Chat``/``Actor``
types import one another. This module imports only leaf modules at load time
(``events`` for the dispatcher); the concrete ``Message`` is imported lazily
inside the methods that build one (and under ``TYPE_CHECKING`` for annotations).

``Status`` (the lifecycle enum) also lives here; ``utils`` re-exports it for
backward compatibility.
"""

from __future__ import annotations

import enum
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from kaggle_benchmarks.events import manager

if TYPE_CHECKING:
    from kaggle_benchmarks.messages import Message

T = TypeVar("T")
Role = Literal["system", "user", "assistant", "developer", "tool"]


class Status(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Event:
    """A single item in a ``Session``'s history — a message or a nested session.

    Carries a ``sender``, a ``status`` and a unique ``id``. Subclasses set
    ``_status_event`` to the name dispatched on the shared event manager when
    their status changes (``Message`` uses ``"message_update"``; ``Chat`` uses
    ``"chat_update"``).
    """

    # Event name dispatched when ``status`` is assigned. Overridden by subclasses.
    _status_event: str = "event_update"

    def __init__(
        self,
        *,
        sender: Actor,
        status: Status = Status.SUCCESS,
        id: str | None = None,
    ):
        self.sender = sender
        self._status = status
        self.id = uuid.uuid4().hex if id is None else id

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value: Status):
        manager.dispatch(self._status_event, self, value)
        self._status = value

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the event.

        Concrete subclasses (``Message``, ``Session``) override this.
        """
        raise NotImplementedError

    def __str__(self, indent: str = "") -> str:
        raise NotImplementedError


class BaseMessage(Event, Generic[T]):
    """A minimal ``Event`` carrying ``content``.

    Deliberately small — it does not know about token usage, reasoning traces
    or tool calls. The concrete :class:`kaggle_benchmarks.messages.Message`
    subclasses this and adds those. Use ``BaseMessage`` (or ``Event``) for
    typing when you only need ``content``/``sender``/``status``.
    """

    _status_event: str = "message_update"

    def __init__(
        self,
        content: T,
        *,
        sender: Actor,
        status: Status = Status.SUCCESS,
        id: str | None = None,
    ):
        super().__init__(sender=sender, status=status, id=id)
        self.content = content

    @property
    def text(self) -> str:
        return str(self.content)

    def __str__(self, indent: str = "") -> str:
        return f"{indent}{self.sender.avatar} [{self.sender.name}]: {self.text}"


class Actor:
    def __init__(
        self,
        name: str | None = None,
        role: Role = "assistant",
        avatar: str = "",
        id: str | None = None,
    ):
        self.name = type(self).__name__ if name is None else name
        self.id = id or name
        self.role = role
        self.avatar = avatar

    def send(
        self, message: T | Message[T], is_visible_to_llm: bool = True
    ) -> Message[T]:
        return self._send(message, is_visible_to_llm)

    def _send(self, message: T | Message[T], is_visible_to_llm: bool) -> Message[T]:
        from kaggle_benchmarks import chats
        from kaggle_benchmarks.messages import Message

        if isinstance(message, Message):
            message.is_visible_to_llm = is_visible_to_llm
        else:
            message = Message(
                sender=self, content=message, is_visible_to_llm=is_visible_to_llm
            )

        chats.send(message)
        return message

    def stream(self, content: Iterable[str]) -> Message[str]:
        from kaggle_benchmarks.messages import Message

        msg = Message(content="", sender=self, _status=Status.RUNNING)
        self.send(msg)
        msg.stream(content)
        msg.status = Status.SUCCESS
        return msg

    def as_dict(self) -> dict[str, str]:
        """Returns a dictionary representation of the agent."""
        return dict(id=self.id, name=self.name, role=self.role, avatar=self.avatar)

    def __repr__(self) -> str:
        cls = type(self)
        return f"{cls.__name__}(name={self.name!r}, avatar={self.avatar!r})"

    def __str__(self) -> str:
        return f"{self.avatar} {self.name}"


class Session(Event):
    """An ordered list of ``Event`` objects; the base class for ``Chat``.

    A ``Session`` is itself an ``Event``, so a session's ``history`` can
    contain both messages and nested sessions.
    """

    _status_event: str = "session_update"

    def __init__(
        self,
        *,
        history: Iterable[Event] = (),
        name: str | None = None,
        sender: Actor,
        status: Status = Status.SUCCESS,
        id: str | None = None,
    ):
        super().__init__(sender=sender, status=status, id=id)
        self.history = list(history)
        self.name = type(self).__name__ if name is None else name

    @manager.event_dispatcher("new_event")
    def append(self, item: Event) -> Event:
        """Adds an event to the session."""
        self.history.append(item)
        return item

    @property
    def events(self) -> list[Event]:
        """All events in this session's history."""
        return [e for e in self.history if isinstance(e, Event)]

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the session."""
        return dict(history=[obj.to_dict() for obj in self.history], name=self.name)

    def __str__(self, indent: str = "") -> str:
        events = "\n".join(m.__str__(indent + "  ") for m in self.history)
        return f"{indent}🧵{self.name or ''}:\n{events}"
