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
from kaggle_benchmarks.chats import ChatRoom
from kaggle_benchmarks.messages import Message
from tests.mocks import MockedChat

# ── Context Manager ──


def test_chatroom_sets_active_context():
    """with room: must make the ChatRoom the active chat."""
    alice = MockedChat.from_contents(["hi"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])
    with room:
        assert chats.get_current_chat() is room
    assert chats.get_current_chat() is not room


# ── talk() Primitives ──


def test_llmchat_talk_appends_to_room():
    alice = MockedChat.from_contents(["I agree!"], name="Alice", cycle=True)
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    with room:
        room.post("Topic: AI safety")
        result = alice.talk()

    assert result == "I agree!"
    assert len(room.messages) == 2
    assert room.messages[1].sender.name == "Alice"
    assert room.messages[1].content == "I agree!"


def test_actor_talk_appends_to_room():
    game = actors.Actor(name="Game")
    alice = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(participants=[game, alice])

    with room:
        game.talk("Board: X|O|_")

    assert len(room.messages) == 1
    assert room.messages[0].sender is game
    assert room.messages[0].content == "Board: X|O|_"


def test_llmchat_talk_outside_room_raises():
    alice = MockedChat.from_contents(["x"], name="Alice")
    with pytest.raises(RuntimeError, match="ChatRoom context"):
        alice.talk()


def test_actor_talk_outside_room_raises():
    game = actors.Actor(name="Game")
    with pytest.raises(RuntimeError, match="ChatRoom context"):
        game.talk("hello")


def test_actor_talk_without_message_raises_typeerror():
    """Actor.talk() requires a message argument."""
    game = actors.Actor(name="Game")
    room = ChatRoom(participants=[game])
    with room:
        with pytest.raises(TypeError):
            game.talk()


def test_prompt_inside_chatroom_raises():
    """prompt() must not be called inside an active ChatRoom."""
    alice = MockedChat.from_contents(["x"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])
    with room:
        with pytest.raises(RuntimeError, match="cannot be called inside"):
            alice.prompt("hello")


# ── Perspective Projection ──


def test_perspective_does_not_mutate_originals():
    alice = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(participants=[alice])
    original = Message(sender=alice, content="original")
    room.history = [original]

    perspective = room._build_perspective(alice)
    assert perspective[0] is not original
    assert original.sender is alice


def test_perspective_filters_invisible_messages():
    """Messages with is_visible_to_llm=False are not shown in perspective."""
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

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
    alice = MockedChat.from_contents(["x"], name="Alice")
    alice.avatar = "🐺"
    bob = MockedChat.from_contents(["x"], name="Bob")
    bob.avatar = "🧙"
    room = ChatRoom(participants=[alice, bob])

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
    alice = MockedChat.from_contents(["hello"], name="Alice")
    alice.system_prompt = "SECRET: You are a werewolf"
    bob = MockedChat.from_contents(["hi"], name="Bob")
    bob.system_prompt = "SECRET: You are a villager"

    room = ChatRoom(
        participants=[alice, bob],
        system_prompt="A game of Werewolf.",
        name="Moderator",
    )

    with room:
        alice.talk()

    _, kwargs = alice.invocations[0]
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
    # Narrator note
    assert 'Note on "Moderator"' in system


def test_system_prompt_via_from_contents():
    alice = MockedChat.from_contents(["x"], name="Alice", system_prompt="argues FOR")
    assert alice.system_prompt == "argues FOR"


def test_post_sender_matches_room_name():
    """room.post() sender name matches the room's name."""
    alice = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(participants=[alice], name="Moderator")
    with room:
        msg = room.post("Hello")
    assert msg.sender.name == "Moderator"


# ── Visibility & Private Channels ──


def test_visible_to_filters_messages():
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    with room:
        room.post("Public message")
        room.post("Secret for Alice", visible_to=[alice])

    assert len(room._build_perspective(alice)) == 2
    assert len(room._build_perspective(bob)) == 1


def test_private_channel_isolation_and_parent_visibility():
    """Covers: parent history visible from child, private content hidden
    from non-members, private messages tagged with channel name."""
    alice = MockedChat.from_contents(["wolf plan", "I'm innocent!"], name="Alice")
    bob = MockedChat.from_contents(["ack"], name="Bob")
    charlie = MockedChat.from_contents(["suspicious"], name="Charlie")

    room = ChatRoom(
        participants=[alice, bob, charlie],
        system_prompt="Werewolf game.",
        name="Moderator",
    )

    with room:
        room.post("Day starts.")
        # Wolves coordinate privately
        wolf_chat = room.private_channel([alice, bob], name="Wolf Night")
        with wolf_chat:
            wolf_chat.post("Pick a victim.")
            alice.talk()  # "wolf plan"
            bob.talk()  # "ack"

        # Public day phase
        alice.talk()  # "I'm innocent!"
        charlie.talk()  # "suspicious"

    # Charlie (non-member) must NOT see wolf chat content
    charlie_msgs, _ = charlie.invocations[0]
    charlie_content = " ".join(m.content for m in charlie_msgs)
    assert "wolf plan" not in charlie_content
    assert "Pick a victim" not in charlie_content
    assert "Day starts" in charlie_content

    # Bob (member) should see parent history + tagged private messages
    bob_msgs, _ = bob.invocations[0]
    bob_content = " ".join(m.content for m in bob_msgs)
    assert "Day starts" in bob_content  # parent visible from child
    private_tagged = [m for m in bob_msgs if "private: Wolf Night" in m.content]
    assert len(private_tagged) > 0  # Alice's msg tagged with channel


def test_private_channel_validates_participants():
    """private_channel() rejects participants not in the parent room."""
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    outsider = MockedChat.from_contents(["x"], name="Outsider")
    room = ChatRoom(participants=[alice, bob])

    with pytest.raises(ValueError, match="Unknown: Outsider"):
        room.private_channel([alice, outsider], name="Bad Channel")


def test_private_channel_rejects_same_name_as_parent():
    """private_channel() rejects name matching the parent room's name."""
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob], name="GameRoom")

    with pytest.raises(ValueError, match="must differ from parent room name"):
        room.private_channel([alice], name="GameRoom")


def test_nested_channels_three_levels():
    """Channel-in-channel: root -> team -> leader, with correct visibility."""
    alice = MockedChat.from_contents(["intel", "report"], name="Alice")
    bob = MockedChat.from_contents(["ack"], name="Bob")
    charlie = MockedChat.from_contents(["cover"], name="Charlie")

    room = ChatRoom(participants=[alice, bob, charlie])
    team = room.private_channel([alice, bob], name="Team")

    with room:
        room.post("Public")
        charlie.talk()  # "cover" — public

        with team:
            team.post("Team only")
            alice.talk()  # "intel"

            leader = team.private_channel([alice], name="Leader")
            with leader:
                leader.post("Leader only")
                alice.talk()  # "report"

    # Alice (leader) saw all 3 levels
    alice_msgs_r2, _ = alice.invocations[1]
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
    alice = MockedChat.from_contents(["r1", "r2", "r3"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])
    channel = room.private_channel([alice], name="Night")

    with room:
        for i in range(3):
            with channel:
                channel.post(f"Night {i}")
                alice.talk()

    night_posts = [m for m in channel.messages if "Night" in m.content]
    assert len(night_posts) == 3
    alice_msgs = [m for m in channel.messages if m.sender is alice]
    assert len(alice_msgs) == 3


# ── _meta ──


def test_meta_chat_points_to_room():
    alice = MockedChat.from_contents(["response"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])
    with room:
        alice.talk()
    assert room.messages[0]._meta.get("chat") is room


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
    alice = MockedLLMResponseChat.from_contents(["hello"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])

    with room:
        alice.talk()

    room_msg = room.messages[0]
    assert "input_tokens" in room_msg._meta
    assert room_msg._meta["input_tokens"] == 10
    assert room_msg._meta.get("chat") is room


# ── Comprehensive Perspective Verification ──


def test_three_player_perspective():
    """Verify exact messages, roles, prefixes, and system prompts that
    each of 3 LLMs receives across 2 rounds (6 invocations total)."""
    alice = MockedChat.from_contents(["Alice R1", "Alice R2"], name="Alice")
    alice.system_prompt = "I am Alice"
    bob = MockedChat.from_contents(["Bob R1", "Bob R2"], name="Bob")
    bob.system_prompt = "I am Bob"
    charlie = MockedChat.from_contents(["Charlie R1", "Charlie R2"], name="Charlie")
    charlie.system_prompt = "I am Charlie"

    room = ChatRoom(
        participants=[alice, bob, charlie],
        system_prompt="A three-way discussion.",
        name="Moderator",
    )

    with room:
        room.post("Topic: Testing")
        alice.talk()
        bob.talk()
        charlie.talk()
        alice.talk()
        bob.talk()
        charlie.talk()

    assert len(room.messages) == 7

    # ── Alice R1: sees [post] ──
    a1_msgs, a1_kw = alice.invocations[0]
    assert len(a1_msgs) == 1
    assert a1_msgs[0].sender.role == "user"
    assert "[Moderator]: Topic: Testing" in a1_msgs[0].content
    assert "You are Alice" in a1_kw["system"]
    assert "I am Alice" in a1_kw["system"]
    assert "I am Bob" not in a1_kw["system"]

    # ── Alice R2: sees [post, self(assistant), Bob, Charlie] ──
    a2_msgs, _ = alice.invocations[1]
    assert len(a2_msgs) == 4
    assert a2_msgs[1].sender.role == "assistant"
    assert a2_msgs[1].content == "Alice R1"
    assert "[Alice]:" not in a2_msgs[1].content
    assert "[Bob]: Bob R1" in a2_msgs[2].content
    assert "[Charlie]: Charlie R1" in a2_msgs[3].content

    # ── Bob R1: sees [post, Alice R1] ──
    b1_msgs, b1_kw = bob.invocations[0]
    assert len(b1_msgs) == 2
    assert b1_msgs[1].sender.role == "user"
    assert "[Alice]: Alice R1" in b1_msgs[1].content
    assert "I am Bob" in b1_kw["system"]
    assert "I am Alice" not in b1_kw["system"]

    # ── Bob R2: sees [post, Alice R1, self(assistant), Charlie R1, Alice R2] ──
    b2_msgs, _ = bob.invocations[1]
    assert len(b2_msgs) == 5
    assert b2_msgs[2].sender.role == "assistant"
    assert b2_msgs[2].content == "Bob R1"
    assert "[Bob]:" not in b2_msgs[2].content

    # ── Charlie R2: sees all 6 prior messages ──
    c2_msgs, _ = charlie.invocations[1]
    assert len(c2_msgs) == 6
    assert c2_msgs[3].sender.role == "assistant"
    assert c2_msgs[3].content == "Charlie R1"
    assert "[Alice]: Alice R2" in c2_msgs[4].content
    assert "[Bob]: Bob R2" in c2_msgs[5].content
