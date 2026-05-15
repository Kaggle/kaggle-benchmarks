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
from kaggle_benchmarks.usage import Usage


@dataclasses.dataclass
class Chat:
    """Represents a thread of messages in a chat."""

    history: list[Message | Self] = dataclasses.field(default_factory=list)
    name: str = "chat"
    _id_suffix: str = dataclasses.field(
        default_factory=lambda: uuid.uuid4().hex[:8], init=False
    )
    sender: actors.Actor = actors.system  # added to mach Message's structural type

    _status: utils.Status = utils.Status.PENDING

    @property
    def id(self) -> str:
        return f"{self.name}-{self._id_suffix}"

    @property
    def messages(self) -> list[Message]:
        return [m for m in self.history if isinstance(m, Message)]

    @property
    def usage(self) -> Usage:
        """Aggregated token usage and cost metadata across all assistant messages."""
        total = Usage()
        for m in self.messages:
            if m.sender.role == "assistant":
                total = total + m.usage
        return total

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


class ChatRoom(Chat):
    """A multi-agent conversation room with perspective-aware history.

    ChatRoom extends Chat to support multiple participants (LLMs and code-driven
    Actors) conversing in a shared space. Each participant sees a projected view
    of the conversation history where their own messages appear as "assistant"
    and peers' messages appear as "user" with name prefixes.

    IMPORTANT: Perspective projection uses object identity (Python ``is``)
    to determine message ownership. All participants must be distinct
    object instances — even if backed by the same LLM model. Use separate
    ``ModelProxy()`` calls to create per-participant instances.

    Usage:
        room = ChatRoom(participants=[alice, bob], system_prompt="Debate AI.")
        with room:
            room.post("Topic: AI safety")
            alice.talk()
            bob.talk()
    """

    def __init__(
        self,
        participants: list["actors.Actor"],
        system_prompt: str = "",
        name: str = "Room",
        _parent_room: "ChatRoom | None" = None,
    ):
        super().__init__(name=name)
        self.participants = list(participants)
        self.system_prompt = system_prompt
        self._parent_room = _parent_room
        self._registered_with_parent = False
        self._cm_stack: list[contextlib.AbstractContextManager] = []

        # Narrator actor for room.post() messages — uses the room's name
        # so the LLM sees consistent "[Moderator]: ..." messages matching
        # the roster's "Note on ..." instruction.
        self._narrator = actors.Actor(name=name, role="user", avatar="📢")

    @contextlib.contextmanager
    def enter(self):
        """Enter this room as the active chat context.

        Returns a fresh context manager each time — safe to call in loops.
        This is the primitive; ``__enter__``/``__exit__`` delegate here.
        """
        from kaggle_benchmarks import contexts

        # First-time parent registration
        if not self._registered_with_parent:
            try:
                parent_chat = get_current_chat()
                if parent_chat and self not in parent_chat.history:
                    parent_chat.append(self)
                self._registered_with_parent = True
            except (LookupError, AttributeError):
                pass

        with contexts.enter(chat=self):
            yield self

    def __enter__(self):
        cm = self.enter()
        result = cm.__enter__()
        self._cm_stack.append(cm)
        return result

    def __exit__(self, *exc):
        cm = self._cm_stack.pop()
        return cm.__exit__(*exc)

    def post(self, message: str, visible_to=None) -> Message:
        """Post a narrator message to the room.

        Uses the room's ``_narrator`` actor (whose name matches the room name)
        so the LLM sees consistent sender identity.

        Args:
            message: The message text.
            visible_to: Optional list of Actors who can see this message.
                If None, all participants can see it.
        """
        msg = Message(sender=self._narrator, content=message)
        if visible_to is not None:
            msg._meta["visible_to"] = visible_to
        self.append(msg)
        return msg

    def _build_roster(self, viewer: "actors.Actor") -> str:
        """Build the participant roster description for the viewer.

        Lists peer names only — ``system_prompt`` is never exposed to
        prevent secret role leakage in games like Werewolf.
        """
        peers = [p for p in self.participants if p is not viewer]
        lines = [f"You are {viewer.name}."]
        if peers:
            lines.append("Other participants in this conversation:")
            for p in peers:
                lines.append(f"- {p.name}")
        lines.append("")
        lines.append(
            "Messages from other participants are prefixed with their name,"
            " e.g., [Bob]: ..."
        )
        lines.append("Your messages appear without a prefix.")
        lines.append("")
        lines.append(
            f'Note on "{self.name}": Messages from "{self.name}" are'
            " system/narrator instructions, not from another player."
        )
        return "\n".join(lines)

    def _build_system_prompt(self, viewer: "actors.Actor") -> str:
        """Build the full system prompt for a viewer.

        Concatenates: roster + --- + room prompt + --- + personal prompt.
        """
        parts = [self._build_roster(viewer)]
        if self.system_prompt:
            parts.append(self.system_prompt)
        personal = getattr(viewer, "system_prompt", None)
        if personal:
            parts.append(personal)
        return "\n---\n".join(parts)

    def _get_root_room(self) -> "ChatRoom":
        """Walk the _parent_room chain to find the topmost room."""
        room = self
        while room._parent_room is not None:
            room = room._parent_room
        return room

    def _build_perspective(
        self, viewer: "actors.Actor", _recursive: bool = False
    ) -> list[Message]:
        """Project the ground-truth history into a viewer's perspective.

        Recursively resolves nested ChatRoom subchannels to interleave private
        discussions for members, while keeping non-members blind.
        - Viewer's own messages → sender with role="assistant"
        - Everyone else's messages → sender with role="user", name-prefixed
        - Messages with visible_to that exclude the viewer are filtered out.
        - Messages from private channels are tagged with context (e.g. "[Bob (private: Night Chat)]: ...")
        """
        # If this is a child private room and we are called at the top-level,
        # delegate perspective building to the topmost root room so that
        # public history and other authorized channels are chronologically
        # interleaved.
        if self._parent_room is not None and not _recursive:
            return self._get_root_room()._build_perspective(viewer)

        projected = []
        for item in self.history:
            if isinstance(item, ChatRoom):
                # Recursively project and interleave nested rooms
                if viewer in item.participants:
                    projected.extend(item._build_perspective(viewer, _recursive=True))
            elif isinstance(item, Message):
                # Standard framework visibility check
                if not item.is_visible_to_llm:
                    continue

                # Visibility filtering on individual messages.
                visible = item._meta.get("visible_to")
                if visible is not None and viewer not in visible:
                    continue

                if item.sender is viewer:
                    projected.append(
                        Message(
                            sender=actors.Actor(
                                name=viewer.name,
                                role="assistant",
                                avatar=viewer.avatar,
                            ),
                            content=item.content,
                        )
                    )
                else:
                    name = item.sender.name
                    avatar = item.sender.avatar

                    # If this message is inside a private child room, tag it
                    is_child_room = self._parent_room is not None
                    # Don't redundantly tag the room's own narrator messages
                    if is_child_room and item.sender is not self._narrator:
                        content = f"[{name} (private: {self.name})]: {item.content}"
                    else:
                        content = f"[{name}]: {item.content}"

                    projected.append(
                        Message(
                            sender=actors.Actor(name=name, role="user", avatar=avatar),
                            content=content,
                        )
                    )
        return projected

    def private_channel(
        self, participants: list["actors.Actor"], name: str = "Private Channel"
    ) -> "ChatRoom":
        """Create a child ChatRoom visible only to the specified participants.

        Any messages posted inside the child channel are interleaved into the
        parent room's ground-truth log with restricted visibility.

        Raises ValueError if any participant is not a member of this room.
        """
        unknown = [p for p in participants if p not in self.participants]
        if unknown:
            names = ", ".join(p.name for p in unknown)
            raise ValueError(
                f"Private channel participants must be members of the parent "
                f"room. Unknown: {names}"
            )
        return ChatRoom(participants=participants, name=name, _parent_room=self)

    def run(self, rounds=1, order="round_robin", stop_when=None):
        """Automated turn loop (Phase 3 — not yet implemented)."""
        raise NotImplementedError(
            "room.run() is planned for Phase 3. Use explicit talk() loops for now."
        )
