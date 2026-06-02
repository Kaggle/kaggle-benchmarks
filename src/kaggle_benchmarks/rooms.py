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

"""Multi-agent conversation rooms with perspective-aware history.

This module provides :class:`ChatRoom` for multi-participant conversations and
:class:`Participant` — a lightweight identity wrapper that binds a backing LLM
(often shared across participants) with per-room state (name, avatar,
system_prompt).

Key design principle: **the room owns all per-participant customizations**.  The
backing LLM is shared and stateless.  No cloning occurs.

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
    """Lightweight identity for an LLM in a ChatRoom.

    Wraps a backing :class:`LLMChat` (often shared across participants)
    with per-room identity (name, avatar) and per-room state
    (system_prompt).  The backing LLM is never cloned.

    Users obtain ``Participant`` instances from
    :meth:`ChatRoom.add_participant` — they are not constructed directly.

    Interaction methods:

    - :meth:`reply` — LLM-generated response (delegates to room).

    For one-shot LLM calls outside any room (e.g. post-room judge
    evaluation), call the backing model directly: ``llm.prompt(...)``.
    A `Participant` is a room-scoped identity; outside the room there is
    no per-participant state to apply.

    Removal is hard-delete by design: :meth:`ChatRoom.remove_participant`
    drops the participant from the active roster and blocks further
    :meth:`reply`, while historical messages remain attributed to them
    in the transcript. If reversible deactivation is ever needed (e.g.
    AFK chat presence, temporary muting), add an `active: bool` field
    here and gate `ChatRoom._build_roster` and `_generate_reply` on it
    — that extension is additive and does not break removal semantics.
    """

    def __init__(
        self,
        llm: "LLMChat",
        *,
        name: str,
        avatar: str,
        system_prompt: str | None = None,
    ):
        self.llm: "LLMChat" = llm
        self.name = name
        self.id = name
        self.avatar = avatar
        self.system_prompt = system_prompt

    @property
    def role(self) -> str:
        return self.llm.role

    def reply(self, schema: type[T] = str, tools=None, **kwargs) -> T:
        """Generate an LLM response in the active ChatRoom.

        This is a **ChatRoom-only** method for LLM participants. It builds a
        perspective-projected history for this participant, invokes the
        underlying LLM, and appends the generated response to the room's
        ground-truth log.

        Unlike ``room.post()``, which posts scripted content from the narrator,
        this method autonomously generates a response based on the conversation
        the participant can see.

        Args:
            schema: Output schema type for structured output. When set, the
                response is shown to peers as ``str(content)`` — override
                ``__str__`` on the schema if you need a different on-the-wire
                representation (e.g. to hide a private field).
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

    def __repr__(self) -> str:
        return f"Participant(name={self.name!r}, llm={self.llm!r})"

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
        name: str = "Narrator",
        narrator_avatar: str = "📢",
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
        self._narrator = actors.Actor(name=name, role="user", avatar=narrator_avatar)

    def add_participant(
        self,
        participant: "LLMChat",
        *,
        name: str | None = None,
        avatar: str | None = None,
        system_prompt: str | None = None,
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

        # Intentional: scripted/code-driven peers should be added via a future
        # add_scripted() / ScriptedParticipant sibling, not by re-accepting Actor
        # here. Routing scripted content through Actor reintroduces the
        # "peer-and-narrator-at-once" roster ambiguity that this refactor fixed.
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
            llm=participant,
            name=effective_name,
            avatar=avatar if avatar is not None else participant.avatar,
            system_prompt=system_prompt,
        )

        self.participants.append(p)

        return p

    def remove_participant(self, participant: "Participant") -> None:
        """Remove a participant from this room's active roster.

        After removal:

        - The participant no longer appears in other participants'
          rosters (so surviving LLMs are not told a dead/departed
          peer is still in the room).
        - Calling :meth:`Participant.reply` raises ``RuntimeError`` —
          the room no longer accepts turns from this identity.
        - Historical messages stay in the transcript, still attributed
          to the participant's name and avatar. The Participant object
          remains referenced via ``Message.sender`` for those messages.

        Removal does **not** cascade to child private channels — if
        the participant is a member of an active
        :meth:`private_channel`, remove them from that channel
        explicitly. Keeping cascade out of the framework matches the
        "one knob per call" design of the rest of the room API.

        Raises:
            ValueError: If ``participant`` is not currently a member.
        """
        if participant not in self.participants:
            raise ValueError(
                f"Participant {participant.name!r} is not a member of this room."
            )
        self.participants.remove(participant)

    # ── Room Orchestration ──────────────────────────────────────────────

    def _generate_reply(
        self, participant: Participant, schema: type[T] = str, tools=None, **kwargs
    ) -> T:
        """Build perspective and invoke the backing LLM.

        This is the internal method that :meth:`Participant.reply` delegates to.
        It constructs the perspective-projected history and system prompt for
        the participant, then calls the backing LLM's ``respond()`` method
        with ``sender=participant`` so the message is created with the correct
        identity from the start (important for streaming UIs and event
        subscribers that observe messages before this method returns).
        """
        if tools:
            raise NotImplementedError(
                "Tool support inside ChatRoom.reply() is planned for a future "
                "release. As a workaround, use an orphan chats.new() "
                "side-chat for tool calls."
            )

        if participant not in self.participants:
            raise RuntimeError(
                f"Participant {participant.name!r} was removed from this room "
                f"and can no longer reply. Re-add via add_participant() if you "
                f"intend to bring them back (which creates a new identity)."
            )

        system = self._build_system_prompt(participant)
        perspective = self._build_perspective(participant)

        response = participant.llm.respond(
            system=system,
            schema=schema,
            input_messages=perspective,
            sender=participant,
            **kwargs,
        )
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
        # Front-load narrator identity so the LLM knows who "[Narrator]: ..."
        # messages come from before it sees peer names.
        lines.append(
            f'Messages from "{self.name}" are system/narrator instructions,'
            " not from another participant."
        )
        lines.append("")
        if peers:
            lines.append("Other participants in this conversation:")
            for p in peers:
                lines.append(f"- {p.name}")
            lines.append("")
        lines.append(
            "Messages from other participants are prefixed with their name,"
            " e.g., [Name]: ..."
        )
        lines.append("Your messages appear without a prefix.")
        lines.append(
            "When you reply, output only your message content. Do not prefix "
            "your response with your name, brackets, or any attribution label "
            "— the system adds attribution automatically."
        )
        lines.append("")
        lines.append(
            "Messages prefixed with '[private: Channel Name]' or tagged with "
            "'(private: Channel Name)' are from a private channel visible only "
            "to its members. Other participants cannot see these messages. "
            "When the most recent message is a private-channel directive, your "
            "reply addresses only that directive. Do not respond to public-room "
            "topics in the same turn."
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
                # Both filters must pass for the viewer to see the message:
                # - is_visible_to_llm: framework-wide hide flag (e.g. error
                #   messages kept for the transcript but never shown to LLMs).
                # - _meta["visible_to"]: ChatRoom-specific per-audience list,
                #   set by room.post(visible_to=...). None means "everyone".
                if not item.is_visible_to_llm:
                    continue
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
        self, participants: list[Participant], *, name: str
    ) -> "ChatRoom":
        """Create a child ChatRoom visible only to the specified participants.

        Any messages posted inside the child channel are interleaved into the
        parent room's ground-truth log with restricted visibility.

        ``name`` is required and keyword-only — it appears in LLM-visible
        tags like ``[private: <name>]: ...``, so it should be semantically
        meaningful to the participants (e.g. ``"Wolf Night Chat"``,
        ``"Team Briefing"``), not a generic placeholder.

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
