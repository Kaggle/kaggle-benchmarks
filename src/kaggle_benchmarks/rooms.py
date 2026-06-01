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

"""Multi-agent conversation rooms with perspective-aware history.

This module provides :class:`ChatRoom` for multi-participant conversations and
:class:`Participant` — a lightweight identity wrapper that binds a backing actor
(often a shared LLM) with per-room state (name, avatar, system_prompt).

Key design principle: **the room owns all per-participant customizations**.  The
backing LLM/Actor object is shared and stateless.  No cloning occurs.

Usage::

    room = ChatRoom(system_prompt="A debate on AI safety.")
    alice = room.add_participant(llm, name="Alice", system_prompt="Argue FOR.")
    bob = room.add_participant(llm, name="Bob", system_prompt="Argue AGAINST.")

    with room:
        room.post("Topic: AI safety")
        alice.reply()
        bob.reply()
"""

import contextlib
from typing import TYPE_CHECKING, Literal, TypeVar

from kaggle_benchmarks import actors
from kaggle_benchmarks.chats import Chat, get_current_chat
from kaggle_benchmarks.messages import Message

if TYPE_CHECKING:
    from kaggle_benchmarks.actors.llms import LLMChat

T = TypeVar("T")


class Participant:
    """Lightweight identity for an actor in a ChatRoom.

    Wraps a backing actor (often a shared LLM) with per-room identity
    (name, avatar) and per-room state (system_prompt).  The backing
    actor is never cloned.

    Users obtain ``Participant`` instances from
    :meth:`ChatRoom.add_participant` — they are not constructed directly.

    Interaction methods:

    - :meth:`reply` — LLM-generated response (delegates to room).
    - :meth:`say` — scripted message (delegates to room).
    - :meth:`prompt` — single-turn LLM call *outside* a room context
      (delegates to the backing actor).
    """

    def __init__(
        self,
        actor: "actors.Actor",
        *,
        name: str,
        avatar: str,
        role: str,
        system_prompt: str | None = None,
    ):
        self.actor = actor
        self.name = name
        self.id = name
        self.avatar = avatar
        self.role = role
        self.system_prompt = system_prompt

    def reply(self, schema: type[T] = str, tools=None, **kwargs) -> T:
        """Generate an LLM response in the active ChatRoom.

        This is a **ChatRoom-only** method for LLM participants. It builds a
        perspective-projected history for this participant, invokes the
        underlying LLM, and appends the generated response to the room's
        ground-truth log.

        Unlike :meth:`say`, which posts scripted content, this method
        autonomously generates a response based on the conversation the
        participant can see.

        Args:
            schema: Output schema type for structured output.
            tools: Reserved for future tool support inside rooms.
            **kwargs: Additional keyword arguments forwarded to respond().

        Returns:
            The LLM-generated response content (parsed to ``schema`` type).

        Raises:
            RuntimeError: If called outside of an active ChatRoom context.
        """
        room = get_current_chat()
        if not isinstance(room, ChatRoom):
            raise RuntimeError(
                "Participant.reply() must be called within an active ChatRoom context."
            )
        return room._generate_reply(self, schema=schema, tools=tools, **kwargs)

    def prompt(self, *args, **kwargs):
        """Single-turn LLM call — delegates to the backing actor.

        This is a convenience proxy so that code which obtained a
        ``Participant`` from :meth:`ChatRoom.add_participant` can still call
        ``participant.prompt(...)`` outside the room context (e.g. for
        post-room judge evaluation).

        Must NOT be called inside an active ChatRoom context.
        """
        return self.actor.prompt(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Participant(name={self.name!r}, actor={self.actor!r})"

    def __str__(self) -> str:
        return f"{self.avatar} {self.name}"


class ChatRoom(Chat):
    """A multi-agent conversation room with perspective-aware history.

    ChatRoom extends Chat to support multiple participants (LLMs) conversing
    in a shared space. Each participant sees a projected view of the conversation
    history where their own messages appear as "assistant" and peers' messages
    appear as "user" with name prefixes.

    Two primitives drive all interaction:

    - ``room.post(msg)`` — narrator/system announcements (game rules, topics).
    - ``participant.reply()`` — LLM generates a response based on its perspective.

    After exiting the room context, access ``room.messages`` for the full
    ground-truth transcript to perform evaluation or assertions.

    Usage::

        room = ChatRoom(system_prompt="Debate AI.")
        alice = room.add_participant(llm, name="Alice", system_prompt="I am Alice.")
        bob = room.add_participant(llm, name="Bob", system_prompt="I am Bob.")

        with room:
            room.post("Topic: AI safety")
            alice.reply()
            bob.reply()

        # Post-room evaluation on the transcript
        for msg in room.messages:
            assert len(msg.content) > 0
    """

    def __init__(
        self,
        system_prompt: str = "",
        name: str = "Room",
        _parent_room: "ChatRoom | None" = None,
    ):
        super().__init__(name=name)
        self.participants: list[Participant] = []

        self.system_prompt = system_prompt
        self._parent_room = _parent_room
        self._registered_with_parent = False
        # Per-instance; safe as long as each thread creates its own ChatRoom.
        self._cm_stack: list[contextlib.AbstractContextManager] = []
        # Cache for synthetic Actor objects used in perspective projection.
        # Avoids creating O(n) ephemeral actors per _build_perspective() call.
        self._synthetic_actors: dict[tuple[str, str, str], "actors.Actor"] = {}

        # Narrator actor for room.post() messages — uses the room's name
        # so the LLM sees consistent "[Moderator]: ..." messages matching
        # the roster's "Note on ..." instruction.
        # Uses role="user" because LLM APIs require user/assistant alternation.
        self._narrator = actors.Actor(name=name, role="user", avatar="📢")

    def add_participant(
        self,
        participant: "LLMChat",
        *,
        name: str | None = None,
        avatar: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> "Participant":
        """Register an LLM as a participant in this room.

        Returns a :class:`Participant` handle — a lightweight identity wrapper
        around the (shared) LLM.  No cloning occurs; the same LLM can be
        registered in multiple rooms or as multiple participants (with distinct
        names) without interference.

        Optional keyword arguments ``name``, ``avatar``, and ``system_prompt``
        override the corresponding attributes on the participant.

        Raises:
            TypeError: If ``participant`` is not an :class:`LLMChat` instance.
            ValueError: If a participant with the same name already exists.
        """
        from kaggle_benchmarks.actors.llms import LLMChat

        if not isinstance(participant, LLMChat):
            raise TypeError(
                f"add_participant() requires an LLMChat instance, got "
                f"{type(participant).__name__}. For scripted messages, use "
                f"room.post() instead."
            )

        effective_name = name if name is not None else participant.name

        # Reject duplicate names (would break perspective projection).
        if any(existing.name == effective_name for existing in self.participants):
            raise ValueError(
                f"Participant name '{effective_name}' is already taken by an "
                f"existing participant. Each participant must have a unique "
                f"name for correct perspective projection."
            )

        p = Participant(
            actor=participant,
            name=effective_name,
            avatar=avatar if avatar is not None else participant.avatar,
            role=participant.role,
            system_prompt=system_prompt,
        )

        for k, v in kwargs.items():
            if not hasattr(p, k):
                raise AttributeError(
                    f"Participant {p.name!r} has no attribute {k!r}. Check for typos."
                )
            setattr(p, k, v)

        self.participants.append(p)

        return p

    # ── Room Orchestration ──────────────────────────────────────────────

    def _generate_reply(
        self, participant: Participant, schema: type[T] = str, tools=None, **kwargs
    ) -> T:
        """Build perspective and invoke the backing LLM.

        This is the internal method that :meth:`Participant.reply` delegates to.
        It constructs the perspective-projected history and system prompt for
        the participant, calls the backing LLM's ``respond()`` method, and
        fixes up the message sender to be the ``Participant`` (not the backing
        LLM).
        """
        if tools:
            raise NotImplementedError(
                "Tool support inside ChatRoom.reply() is planned for a future "
                "release. As a workaround, use an orphan chats.new() "
                "side-chat for tool calls."
            )

        system = self._build_system_prompt(participant)
        perspective = self._build_perspective(participant)

        response = participant.actor.respond(
            system=system, schema=schema, input_messages=perspective, **kwargs
        )
        # Fix up sender: respond() creates the message with sender=backing_llm,
        # but we need it to be the Participant so perspective projection
        # (which uses `is` identity) works correctly on subsequent turns.
        response.sender = participant
        return response.content

    # ── Context Management ──────────────────────────────────────────────

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
            except LookupError:
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

    # ── Narrator ────────────────────────────────────────────────────────

    def post(
        self, message: str, visible_to: list[Participant] | None = None
    ) -> Message:
        """Post a system/narrator directive to the room.

        Messages posted via ``post()`` come from the room's internal narrator
        — an actor that is **not** a registered participant. The roster
        explicitly tells LLMs to treat these messages as system instructions
        rather than peer speech, reducing the chance of LLMs "arguing back"
        at directives.

        Use ``post()`` for structural directives: phase transitions, rules, topic prompts.
        Example: ``room.post("--- Phase 2: Rebuttals ---")``

        Args:
            message: The message text.
            visible_to: Optional list of participants who can see this message.
                If None (default), all participants can see it. Useful for
                sending private instructions to specific participants without
                creating a full ``private_channel()``.
        """
        msg = Message(sender=self._narrator, content=message)
        if visible_to is not None:
            msg._meta["visible_to"] = visible_to
        self.append(msg)
        return msg

    # ── Perspective & System Prompt ─────────────────────────────────────

    def _build_roster(self, viewer: Participant) -> str:
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
            "Messages prefixed with '[private: Channel Name]' or tagged with "
            "'(private: Channel Name)' are from a private channel visible only "
            "to its members. Other participants cannot see these messages. "
            "When you see a private channel prompt, respond to it directly "
            "— do not continue the public conversation."
        )
        lines.append("")
        lines.append(
            f'Note on "{self.name}": Messages from "{self.name}" are'
            " system/narrator instructions, not from another player."
        )
        return "\n".join(lines)

    def _build_system_prompt(self, viewer: Participant) -> str:
        """Build the full system prompt for a viewer.

        Concatenates: roster + --- + room prompt + --- + personal prompt.
        """
        parts = [self._build_roster(viewer)]
        if self.system_prompt:
            parts.append(self.system_prompt)
        if viewer.system_prompt:
            parts.append(viewer.system_prompt)
        return "\n---\n".join(parts)

    def _get_root_room(self) -> "ChatRoom":
        """Walk the _parent_room chain to find the topmost room."""
        room = self
        while room._parent_room is not None:
            room = room._parent_room
        return room

    def _build_perspective(
        self, viewer: Participant, _recursive: bool = False
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
                    sender = self._get_synthetic_actor(
                        viewer.name, "assistant", viewer.avatar
                    )
                    content = item.content
                else:
                    name = item.sender.name
                    is_child_room = self._parent_room is not None
                    if is_child_room:
                        if item.sender is self._narrator:
                            content = f"[private: {self.name}]: {item.content}"
                        else:
                            content = f"[{name} (private: {self.name})]: {item.content}"
                    else:
                        content = f"[{name}]: {item.content}"
                    sender = self._get_synthetic_actor(name, "user", item.sender.avatar)

                projected.append(Message(sender=sender, content=content))
        return projected

    def _get_synthetic_actor(
        self,
        name: str,
        role: Literal["system", "user", "assistant", "developer", "tool"],
        avatar: str,
    ) -> "actors.Actor":
        """Return a cached synthetic Actor for perspective projection."""
        key = (name, role, avatar)
        if key not in self._synthetic_actors:
            self._synthetic_actors[key] = actors.Actor(
                name=name, role=role, avatar=avatar
            )
        return self._synthetic_actors[key]

    # ── Private Channels ────────────────────────────────────────────────

    def private_channel(
        self, participants: list[Participant], name: str = "Private Channel"
    ) -> "ChatRoom":
        """Create a child ChatRoom visible only to the specified participants.

        Any messages posted inside the child channel are interleaved into the
        parent room's ground-truth log with restricted visibility.

        Nesting beyond two levels is supported — enter each channel within
        its parent's context.

        Raises ValueError if any participant is not a member of this room.
        """
        unknown = [p for p in participants if p not in self.participants]
        if unknown:
            names = ", ".join(p.name for p in unknown)
            raise ValueError(
                f"Private channel participants must be members of the parent "
                f"room. Unknown: {names}"
            )
        if name == self.name:
            raise ValueError(
                f"Private channel name must differ from parent room name "
                f"'{self.name}' to avoid narrator identity confusion."
            )

        existing_channel_names = [
            item.name for item in self.history if isinstance(item, ChatRoom)
        ]
        if name in existing_channel_names:
            raise ValueError(
                f"A private channel named '{name}' already exists in this room."
            )

        seen = set()
        for p in participants:
            if p in seen:
                raise ValueError(
                    f"Duplicate participant '{p.name}' in private channel."
                )
            seen.add(p)

        child_room = ChatRoom(name=name, _parent_room=self)
        child_room.participants = list(participants)
        return child_room

    def __repr__(self) -> str:
        names = [p.name for p in self.participants]
        return f"ChatRoom(name={self.name!r}, participants={names})"
