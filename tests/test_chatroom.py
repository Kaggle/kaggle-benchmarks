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

import pytest

from kaggle_benchmarks import actors, chats
from kaggle_benchmarks.messages import Message
from kaggle_benchmarks.rooms import ChatRoom, Participant
from tests.mocks import MockedChat

# ── Public API surface ──


def test_chatroom_and_participant_are_top_level_exports():
    """ChatRoom and Participant must be reachable as kbench.ChatRoom /
    kbench.Participant, for ergonomic parity with kbench.Actor / kbench.LLMChat.
    Locks the re-export so a future refactor of __init__.py doesn't silently
    break the public surface."""
    import kaggle_benchmarks as kbench

    assert kbench.ChatRoom is ChatRoom
    assert kbench.Participant is Participant


# ── Context Manager ──


def test_chatroom_rejects_duplicate_participants():
    """Same LLMChat added twice without name override is rejected (name collision)."""
    alice = MockedChat.from_contents(["x"], name="Alice")

    room = ChatRoom()
    room.add_participant(alice)
    with pytest.raises(ValueError, match="already taken"):
        room.add_participant(alice)


def test_add_participant_same_llm_creates_distinct_participants():
    """Same LLMChat object added with different names yields two distinct Participants."""
    shared = MockedChat.from_contents(["x"], name="Model", cycle=True)
    room = ChatRoom()
    alice = room.add_participant(shared, name="Alice")
    bob = room.add_participant(shared, name="Bob")

    assert alice is not bob
    assert isinstance(alice, Participant)
    assert isinstance(bob, Participant)
    # Both wrap the same backing LLM
    assert alice.llm is shared
    assert bob.llm is shared
    assert alice.name == "Alice"
    assert bob.name == "Bob"
    assert len(room.participants) == 2


def test_add_participant_same_llm_independent_perspectives():
    """Participants from the same LLM have correctly separated perspectives."""
    shared = MockedChat.from_contents(["hello", "world"], name="Model", cycle=True)
    room = ChatRoom()
    alice = room.add_participant(shared, name="Alice")
    bob = room.add_participant(shared, name="Bob")

    with room:
        room.post("Topic")
        alice.reply()
        bob.reply()

    # Messages attributed to correct participant
    assert room.messages[1].sender is alice
    assert room.messages[2].sender is bob

    # Alice's perspective: her msg as assistant, Bob's as user
    alice_p = room._build_perspective(alice)
    assert alice_p[1].sender.role == "assistant"
    assert "[Bob]:" in alice_p[2].content

    # Bob's perspective: his msg as assistant, Alice's as user
    bob_p = room._build_perspective(bob)
    assert "[Alice]:" in bob_p[1].content
    assert bob_p[2].sender.role == "assistant"


def test_add_participant_rejects_duplicate_name():
    """Different LLMs with the same effective name are rejected."""
    a = MockedChat.from_contents(["x"], name="Alice")
    b = MockedChat.from_contents(["y"], name="Bob")
    room = ChatRoom()
    room.add_participant(a)
    with pytest.raises(ValueError, match="already taken"):
        room.add_participant(b, name="Alice")


def test_add_participant_plain_actor_rejected():
    """Plain Actor instance added to room is rejected with TypeError."""
    game = actors.Actor(name="Game")
    room = ChatRoom()
    with pytest.raises(TypeError, match="requires an LLMChat instance"):
        room.add_participant(game)


def test_chatroom_sets_active_context():
    """with room: must make the ChatRoom the active chat."""
    alice = MockedChat.from_contents(["hi"], name="Alice", cycle=True)
    room = ChatRoom()
    alice = room.add_participant(alice)
    with room:
        assert chats.get_current_chat() is room
    assert chats.get_current_chat() is not room


# ── remove_participant() ──


def test_remove_participant_excludes_from_roster():
    """Removed participant must not appear in surviving members' rosters."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice", cycle=True)
    bob_mock = MockedChat.from_contents(["x"], name="Bob")
    charlie_mock = MockedChat.from_contents(["x"], name="Charlie")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    bob = room.add_participant(bob_mock)
    room.add_participant(charlie_mock)

    room.remove_participant(bob)

    with room:
        room.post("kickoff")
        alice.reply()
    _, kwargs = alice_mock.invocations[0]
    system = kwargs["system"]
    assert "Charlie" in system
    assert "Bob" not in system


def test_remove_participant_preserves_historical_messages():
    """Removed participant's prior messages stay in the transcript and
    remain visible in other members' perspectives, attributed to them."""
    alice_mock = MockedChat.from_contents(["from Alice"], name="Alice", cycle=True)
    bob_mock = MockedChat.from_contents(["from Bob"], name="Bob")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    bob = room.add_participant(bob_mock)

    with room:
        room.post("kickoff")
        bob.reply()
    room.remove_participant(bob)

    # Bob's message is still in the room
    assert any(m.sender is bob for m in room.messages)

    # And it shows up in Alice's perspective with the [Bob]: prefix
    perspective = room._build_perspective(alice)
    assert any("[Bob]: from Bob" in m.content for m in perspective)


def test_remove_participant_blocks_further_reply():
    """A removed participant's .reply() must raise."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice", cycle=True)
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    room.remove_participant(alice)

    with room:
        with pytest.raises(RuntimeError, match="was removed from this room"):
            alice.reply()


def test_remove_participant_twice_raises():
    """Double-remove raises ValueError — silently swallowing would hide bugs."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    room.remove_participant(alice)
    with pytest.raises(ValueError, match="not a member"):
        room.remove_participant(alice)


def test_remove_participant_unknown_raises():
    """Removing a Participant that was never added raises ValueError."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom()
    room.add_participant(alice_mock)
    outsider = Participant(alice_mock, name="Ghost", avatar="")
    with pytest.raises(ValueError, match="not a member"):
        room.remove_participant(outsider)


def test_remove_participant_does_not_cascade_to_private_channel():
    """Removal from parent room is intentionally non-cascading; the
    participant remains a member of any active private channel until
    explicitly removed there too. This documents the contract."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    bob_mock = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    bob = room.add_participant(bob_mock)

    channel = room.private_channel([alice, bob], name="Side")
    room.remove_participant(bob)

    assert bob in channel.participants
    assert bob not in room.participants


def test_remove_then_readd_same_name_creates_distinct_identity():
    """Reusing a name after remove_participant produces a NEW Participant.

    This pins the current "resurrection-as-amnesia" behavior so a future
    change to restrict name reuse would notice this test and need to
    decide intentionally. The two Participants share a name (so peer-side
    projection tags both as [Alice]) but are distinct Python objects with
    independent avatars and system_prompts.
    """
    alice_mock = MockedChat.from_contents(["before", "after"], name="Alice", cycle=True)
    room = ChatRoom()

    p1 = room.add_participant(alice_mock, name="Alice", avatar="👩")
    with room:
        room.post("kickoff")
        p1.reply()  # "before"
    room.remove_participant(p1)

    # Same name allowed again
    p2 = room.add_participant(alice_mock, name="Alice", avatar="🧙")
    with room:
        p2.reply()  # "after" — room already has history from above

    # Distinct objects with different avatars
    assert p1 is not p2
    assert p1.avatar == "👩"
    assert p2.avatar == "🧙"

    # Both messages stay in the transcript, attributed to their respective
    # Participant identities. messages[0] is the kickoff post.
    assert room.messages[1].sender is p1
    assert room.messages[2].sender is p2

    # p1 can no longer reply (was removed); p2 can
    with room:
        with pytest.raises(RuntimeError, match="was removed"):
            p1.reply()


# ── reply() Primitive ──


def test_llmchat_reply_appends_to_room():
    alice_mock = MockedChat.from_contents(["I agree!"], name="Alice", cycle=True)
    bob_mock = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    room.add_participant(bob_mock)

    with room:
        room.post("Topic: AI safety")
        result = alice.reply()

    assert result == "I agree!"
    assert len(room.messages) == 2
    assert room.messages[1].sender.name == "Alice"
    assert room.messages[1].content == "I agree!"


def test_participant_reply_in_empty_room_raises():
    """Calling reply() before anything has been posted is a usage bug.

    Some providers (e.g. Gemini via Vertex AI) reject requests whose
    message list contains only a system prompt with
    "at least one contents field is required". We fail fast at the room
    layer with an actionable error pointing the user at room.post(...).
    """
    alice_mock = MockedChat.from_contents(["should-not-be-called"], name="Alice")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)

    with room:
        with pytest.raises(RuntimeError, match=r"room\.post"):
            alice.reply()


def test_participant_reply_outside_room_raises():
    """Participant.reply() must be called inside a ChatRoom context."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    with pytest.raises(RuntimeError, match="ChatRoom context"):
        alice.reply()


def test_llm_prompt_inside_chatroom_raises():
    """LLMChat.prompt() must not be called inside an active ChatRoom —
    it bypasses perspective projection and corrupts the room transcript."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice", cycle=True)
    room = ChatRoom()
    room.add_participant(alice_mock)
    with room:
        with pytest.raises(RuntimeError, match="cannot be called inside"):
            alice_mock.prompt("hello")


def test_participant_has_no_prompt_method():
    """Participant is a room-scoped identity; prompt() would silently
    drop per-participant state. Removed intentionally — callers should
    use llm.prompt() directly for one-shot calls outside a room."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    assert not hasattr(alice, "prompt")


# ── Perspective Projection ──


def test_perspective_does_not_mutate_originals():
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    original = Message(sender=alice, content="original")
    room.history = [original]

    perspective = room._build_perspective(alice)
    assert perspective[0] is not original
    assert original.sender is alice


def test_perspective_filters_invisible_messages():
    """Messages with is_visible_to_llm=False are not shown in perspective."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    bob_mock = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    bob = room.add_participant(bob_mock)

    room.history = [
        Message(sender=alice, content="I can be seen"),
        Message(sender=alice, content="I am hidden", is_visible_to_llm=False),
    ]

    perspective = room._build_perspective(bob)
    contents = [m.content for m in perspective]
    assert any("I can be seen" in c for c in contents)
    assert not any("I am hidden" in c for c in contents)


def test_perspective_preserves_avatars():
    """Synthetic actors in perspective should retain original avatars."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    bob_mock = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom()
    alice = room.add_participant(alice_mock, avatar="🐺")
    bob = room.add_participant(bob_mock, avatar="🧙")

    room.history = [
        Message(sender=alice, content="hello"),
        Message(sender=bob, content="world"),
    ]

    perspective = room._build_perspective(alice)
    assert perspective[0].sender.avatar == "🐺"
    assert perspective[1].sender.avatar == "🧙"


# ── System Prompt ──


def test_system_prompt_composition():
    """System prompt contains roster, room instructions, personal prompt,
    narrator note, but NOT peers' system_prompts."""
    alice_mock = MockedChat.from_contents(["hello"], name="Alice")
    bob_mock = MockedChat.from_contents(["hi"], name="Bob")

    room = ChatRoom(
        system_prompt="A game of Werewolf.",
        name="Moderator",
    )
    alice = room.add_participant(alice_mock, system_prompt="SECRET: You are a werewolf")
    room.add_participant(bob_mock, system_prompt="SECRET: You are a villager")

    with room:
        room.post("kickoff")
        alice.reply()

    _, kwargs = alice_mock.invocations[0]
    system = kwargs["system"]

    # Identity & roster
    assert "You are Alice" in system
    assert "Bob" in system
    # Room instructions
    assert "A game of Werewolf" in system
    # Personal prompt included
    assert "SECRET: You are a werewolf" in system
    # Peer's secret NOT leaked
    assert "SECRET: You are a villager" not in system
    # Narrator identity (front-loaded in roster)
    assert 'Messages from "Moderator" are system/narrator instructions' in system


def test_post_sender_matches_room_name():
    """room.post() sender name matches the room's name."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(name="Moderator")
    room.add_participant(alice_mock)
    with room:
        msg = room.post("Hello")
    assert msg.sender.name == "Moderator"


# ── Visibility & Private Channels ──


def test_visible_to_filters_messages():
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    bob_mock = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    bob = room.add_participant(bob_mock)

    with room:
        room.post("Public message")
        room.post("Secret for Alice", visible_to=[alice])

    assert len(room._build_perspective(alice)) == 2
    assert len(room._build_perspective(bob)) == 1


def test_visibility_filters_compose_with_and():
    """A message that fails EITHER filter is hidden. Pins the AND semantics
    so a future refactor can't accidentally turn it into OR (which would
    leak hidden messages to the audience listed in visible_to)."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)

    msg = Message(sender=alice, content="hidden by is_visible_to_llm")
    msg.is_visible_to_llm = False
    msg._meta["visible_to"] = [alice]  # would otherwise allow Alice to see it
    room.history = [msg]

    # Alice is in visible_to BUT is_visible_to_llm is False — still hidden.
    assert room._build_perspective(alice) == []


def test_private_channel_isolation_and_parent_visibility():
    """Covers: parent history visible from child, private content hidden
    from non-members, private messages tagged with channel name."""
    alice_mock = MockedChat.from_contents(["wolf plan", "I'm innocent!"], name="Alice")
    bob_mock = MockedChat.from_contents(["ack"], name="Bob")
    charlie_mock = MockedChat.from_contents(["suspicious"], name="Charlie")

    room = ChatRoom(
        system_prompt="Werewolf game.",
        name="Moderator",
    )
    alice = room.add_participant(alice_mock)
    bob = room.add_participant(bob_mock)
    charlie = room.add_participant(charlie_mock)

    with room:
        room.post("Day starts.")
        # Wolves coordinate privately
        wolf_chat = room.private_channel([alice, bob], name="Wolf Night")
        with wolf_chat:
            wolf_chat.post("Pick a victim.")
            alice.reply()  # "wolf plan"
            bob.reply()  # "ack"

        # Public day phase
        alice.reply()  # "I'm innocent!"
        charlie.reply()  # "suspicious"

    # Charlie (non-member) must NOT see wolf chat content
    charlie_msgs, _ = charlie_mock.invocations[0]
    charlie_content = " ".join(m.content for m in charlie_msgs)
    assert "wolf plan" not in charlie_content
    assert "Pick a victim" not in charlie_content
    assert "Day starts" in charlie_content

    # Bob (member) should see parent history + tagged private messages
    bob_msgs, _ = bob_mock.invocations[0]
    bob_content = " ".join(m.content for m in bob_msgs)
    assert "Day starts" in bob_content  # parent visible from child
    private_tagged = [m for m in bob_msgs if "private: Wolf Night" in m.content]
    assert len(private_tagged) > 0  # Alice's msg tagged with channel


def test_private_channel_validates_participants():
    """private_channel() rejects participants not in the parent room."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    bob_mock = MockedChat.from_contents(["x"], name="Bob")
    outsider_mock = MockedChat.from_contents(["x"], name="Outsider")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    room.add_participant(bob_mock)

    outsider_participant = Participant(outsider_mock, name="Outsider", avatar="")
    with pytest.raises(ValueError, match="Unknown: Outsider"):
        room.private_channel([alice, outsider_participant], name="Bad Channel")


def test_private_channel_rejects_same_name_as_parent():
    """private_channel() rejects name matching the parent room's name."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    bob_mock = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(name="GameRoom")
    alice = room.add_participant(alice_mock)
    room.add_participant(bob_mock)

    with pytest.raises(ValueError, match="must differ from parent room name"):
        room.private_channel([alice], name="GameRoom")


def test_private_channel_requires_name():
    """private_channel() requires a keyword-only name argument — the name
    appears in LLM-visible '[private: ...]' tags, so a generic default
    would degrade prompt quality silently."""
    alice_mock = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom()
    alice = room.add_participant(alice_mock)

    with pytest.raises(TypeError):
        room.private_channel([alice])  # missing required name=


def test_nested_channels_three_levels():
    """Channel-in-channel: root -> team -> leader, with correct visibility."""
    alice_mock = MockedChat.from_contents(["intel", "report"], name="Alice")
    bob_mock = MockedChat.from_contents(["ack"], name="Bob")
    charlie_mock = MockedChat.from_contents(["cover"], name="Charlie")

    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    bob = room.add_participant(bob_mock)
    charlie = room.add_participant(charlie_mock)

    team = room.private_channel([alice, bob], name="Team")

    with room:
        room.post("Public")
        charlie.reply()  # "cover" — public

        with team:
            team.post("Team only")
            alice.reply()  # "intel"

            leader = team.private_channel([alice], name="Leader")
            with leader:
                leader.post("Leader only")
                alice.reply()  # "report"

    # Alice (leader) saw all 3 levels
    alice_msgs_r2, _ = alice_mock.invocations[1]
    alice_content = " ".join(m.content for m in alice_msgs_r2)
    assert "Public" in alice_content
    assert "Team only" in alice_content

    # Charlie sees only public
    charlie_p = room._build_perspective(charlie)
    charlie_content = [m.content for m in charlie_p]
    assert any("Public" in c for c in charlie_content)
    assert not any("Team only" in c for c in charlie_content)
    assert not any("Leader only" in c for c in charlie_content)


def test_reentrant_chatroom_loop():
    """Same private channel entered multiple times in a loop."""
    alice_mock = MockedChat.from_contents(["r1", "r2", "r3"], name="Alice", cycle=True)
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    channel = room.private_channel([alice], name="Night")

    with room:
        for i in range(3):
            with channel:
                channel.post(f"Night {i}")
                alice.reply()

    night_posts = [m for m in channel.messages if "Night" in m.content]
    assert len(night_posts) == 3
    alice_msgs = [m for m in channel.messages if m.sender is alice]
    assert len(alice_msgs) == 3


# ── _meta ──


def test_meta_chat_points_to_room():
    alice_mock = MockedChat.from_contents(["response"], name="Alice", cycle=True)
    room = ChatRoom()
    alice = room.add_participant(alice_mock)
    with room:
        room.post("kickoff")
        alice.reply()
    # messages[0] is the kickoff post; messages[1] is alice's reply.
    assert room.messages[1]._meta.get("chat") is room


class MockedLLMResponseChat(actors.LLMChat):
    """Mock returning LLMResponse (not LLMMessage) to exercise the code
    path where _meta gets mutable values like tool_calls."""

    def __init__(self, responses, name="Mock", cycle=False, **kwargs):
        super().__init__(name=name, **kwargs)

        self._responses = (
            __import__("itertools").cycle(responses) if cycle else iter(responses)
        )
        self.invocations = []

    @classmethod
    def from_contents(cls, contents, cycle=False, **kwargs):
        from kaggle_benchmarks.actors.llms import LLMResponse

        responses = [
            LLMResponse(
                content=c,
                reasoning_traces=None,
                tool_calls=None,
                meta={"input_tokens": 10, "output_tokens": 5},
            )
            for c in contents
        ]
        return cls(responses, cycle=cycle, **kwargs)

    def invoke(self, messages, **kwargs):
        self.invocations.append((messages, kwargs))
        return next(self._responses)


def test_meta_llm_response_path():
    """Verify _meta contains LLMResponse.meta fields (usage tokens etc.)
    when the LLMResponse code path is used."""
    alice_mock = MockedLLMResponseChat.from_contents(
        ["hello"], name="Alice", cycle=True
    )
    room = ChatRoom()
    alice = room.add_participant(alice_mock)

    with room:
        room.post("kickoff")
        alice.reply()

    # messages[0] is the kickoff post; messages[1] is alice's reply.
    room_msg = room.messages[1]
    assert "input_tokens" in room_msg._meta
    assert room_msg._meta["input_tokens"] == 10
    assert room_msg._meta.get("chat") is room


# ── Sender Identity at Event Time ──


def test_sender_is_participant_at_event_time_llm_message():
    """Message sender must be the Participant (not the backing LLMChat)
    at the moment new_message fires — not patched after the fact."""
    from kaggle_benchmarks import events

    alice_mock = MockedChat.from_contents(["hi"], name="Alice", cycle=True)
    room = ChatRoom()
    alice = room.add_participant(alice_mock, name="AliceP", avatar="👩")

    captured_senders = []

    class SenderCapture:
        def new_message(self, chat, message):
            captured_senders.append(message.sender)

    capture = SenderCapture()
    events.manager.bind(capture)
    try:
        with room:
            room.post("Topic")
            alice.reply()
    finally:
        events.manager.unbind(capture)

    # Find the reply message (not the post)
    reply_senders = [
        s for s in captured_senders if getattr(s, "name", None) == "AliceP"
    ]
    assert len(reply_senders) == 1
    assert reply_senders[0] is alice  # Participant, not backing LLMChat


def test_sender_is_participant_at_event_time_llm_response():
    """Same as above but for the LLMResponse code path."""
    from kaggle_benchmarks import events

    alice_mock = MockedLLMResponseChat.from_contents(["hi"], name="Alice", cycle=True)
    room = ChatRoom()
    alice = room.add_participant(alice_mock, name="AliceP", avatar="👩")

    captured_senders = []

    class SenderCapture:
        def new_message(self, chat, message):
            captured_senders.append(message.sender)

    capture = SenderCapture()
    events.manager.bind(capture)
    try:
        with room:
            room.post("Topic")
            alice.reply()
    finally:
        events.manager.unbind(capture)

    reply_senders = [
        s for s in captured_senders if getattr(s, "name", None) == "AliceP"
    ]
    assert len(reply_senders) == 1
    assert reply_senders[0] is alice


# ── Comprehensive Perspective Verification ──


def test_three_player_perspective():
    """Verify exact messages, roles, prefixes, and system prompts that
    each of 3 LLMs receives across 2 rounds (6 invocations total)."""
    alice_mock = MockedChat.from_contents(["Alice R1", "Alice R2"], name="Alice")
    bob_mock = MockedChat.from_contents(["Bob R1", "Bob R2"], name="Bob")
    charlie_mock = MockedChat.from_contents(
        ["Charlie R1", "Charlie R2"], name="Charlie"
    )

    room = ChatRoom(
        system_prompt="A three-way discussion.",
        name="Moderator",
    )
    alice = room.add_participant(alice_mock, system_prompt="I am Alice")
    bob = room.add_participant(bob_mock, system_prompt="I am Bob")
    charlie = room.add_participant(charlie_mock, system_prompt="I am Charlie")

    with room:
        room.post("Topic: Testing")
        alice.reply()
        bob.reply()
        charlie.reply()
        alice.reply()
        bob.reply()
        charlie.reply()

    assert len(room.messages) == 7

    # ── Alice R1: sees [post] ──
    a1_msgs, a1_kw = alice_mock.invocations[0]
    assert len(a1_msgs) == 1
    assert a1_msgs[0].sender.role == "user"
    assert "[Moderator]: Topic: Testing" in a1_msgs[0].content
    assert "You are Alice" in a1_kw["system"]
    assert "I am Alice" in a1_kw["system"]
    assert "I am Bob" not in a1_kw["system"]

    # ── Alice R2: sees [post, self(assistant), Bob, Charlie] ──
    a2_msgs, _ = alice_mock.invocations[1]
    assert len(a2_msgs) == 4
    assert a2_msgs[1].sender.role == "assistant"
    assert a2_msgs[1].content == "Alice R1"
    assert "[Alice]:" not in a2_msgs[1].content
    assert "[Bob]: Bob R1" in a2_msgs[2].content
    assert "[Charlie]: Charlie R1" in a2_msgs[3].content

    # ── Bob R1: sees [post, Alice R1] ──
    b1_msgs, b1_kw = bob_mock.invocations[0]
    assert len(b1_msgs) == 2
    assert b1_msgs[1].sender.role == "user"
    assert "[Alice]: Alice R1" in b1_msgs[1].content
    assert "I am Bob" in b1_kw["system"]
    assert "I am Alice" not in b1_kw["system"]

    # ── Bob R2: sees [post, Alice R1, self(assistant), Charlie R1, Alice R2] ──
    b2_msgs, _ = bob_mock.invocations[1]
    assert len(b2_msgs) == 5
    assert b2_msgs[2].sender.role == "assistant"
    assert b2_msgs[2].content == "Bob R1"
    assert "[Bob]:" not in b2_msgs[2].content

    # ── Charlie R2: sees all 6 prior messages ──
    c2_msgs, _ = charlie_mock.invocations[1]
    assert len(c2_msgs) == 6
    assert c2_msgs[3].sender.role == "assistant"
    assert c2_msgs[3].content == "Charlie R1"
    assert "[Alice]: Alice R2" in c2_msgs[4].content
    assert "[Bob]: Bob R2" in c2_msgs[5].content
