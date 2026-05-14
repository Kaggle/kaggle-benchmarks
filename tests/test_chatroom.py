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

import json

import pytest

from kaggle_benchmarks import actors, chats
from kaggle_benchmarks.chats import ChatRoom
from kaggle_benchmarks.messages import Message
from tests.mocks import MockedChat

# --- Context Manager ---


def test_chatroom_sets_active_context():
    """with room: must make the ChatRoom the active chat."""
    alice = MockedChat.from_contents(["hi"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])
    with room:
        assert chats.get_current_chat() is room
    # After exit, room is no longer current.
    assert chats.get_current_chat() is not room


# --- System Prompt Enrichment ---


def test_system_prompt_includes_roster():
    alice = MockedChat.from_contents(["x"], name="Alice")
    alice.system_prompt = "argues FOR"
    bob = MockedChat.from_contents(["x"], name="Bob")
    bob.system_prompt = "argues AGAINST"
    room = ChatRoom(participants=[alice, bob], system_prompt="A debate.")

    prompt = room._build_system_prompt(alice)
    assert "You are Alice" in prompt
    assert "Bob" in prompt
    assert "argues AGAINST" in prompt
    assert "A debate." in prompt
    assert "argues FOR" in prompt  # alice's personal prompt


def test_system_prompt_room_identity():
    alice = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(participants=[alice], name="Narrator")

    prompt = room._build_system_prompt(alice)
    assert 'Note on "Narrator"' in prompt


# --- Perspective Projection ---


def test_perspective_self_is_assistant():
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    room.history = [
        Message(sender=alice, content="hello from alice"),
        Message(sender=bob, content="hello from bob"),
    ]

    perspective = room._build_perspective(alice)
    assert perspective[0].sender.role == "assistant"
    assert perspective[0].content == "hello from alice"
    assert perspective[1].sender.role == "user"
    assert "[Bob]:" in perspective[1].content


def test_perspective_does_not_mutate_originals():
    alice = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(participants=[alice])
    original = Message(sender=alice, content="original")
    room.history = [original]

    perspective = room._build_perspective(alice)
    assert perspective[0] is not original  # new object
    assert original.sender is alice  # original unchanged


# --- talk() Primitives ---


def test_llmchat_talk_outside_room_raises():
    alice = MockedChat.from_contents(["x"], name="Alice")
    with pytest.raises(RuntimeError, match="ChatRoom context"):
        alice.talk()


def test_actor_talk_outside_room_raises():
    game = actors.Actor(name="Game")
    with pytest.raises(RuntimeError, match="ChatRoom context"):
        game.talk("hello")


def test_llmchat_talk_appends_to_room():
    alice = MockedChat.from_contents(["I agree!"], name="Alice", cycle=True)
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    with room:
        room.post("Topic: AI safety")
        result = alice.talk()

    assert result == "I agree!"
    # Ground truth should have: system post + alice's response.
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


# --- End-to-End: Mini Debate ---


def test_mini_debate_two_rounds():
    """Two LLMs debate for 2 rounds with full perspective-aware history."""
    pro = MockedChat.from_contents(["AI is great!", "AI saves lives!"], name="Pro")
    con = MockedChat.from_contents(["AI is risky!", "AI needs regulation!"], name="Con")
    room = ChatRoom(
        participants=[pro, con],
        system_prompt="A debate on AI.",
    )

    with room:
        room.post("Topic: AI")
        pro.talk()
        con.talk()
        pro.talk()
        con.talk()

    # Ground truth: 1 system post + 4 talk messages.
    assert len(room.messages) == 5

    # Verify Pro's second invocation received the full perspective history.
    # invocations[1] is the second call to pro.invoke()
    messages_sent, kwargs = pro.invocations[1]
    # Pro should see: system post (user), Pro round 1 (assistant), Con round 1 (user)
    assert len(messages_sent) == 3
    assert messages_sent[0].sender.role == "user"  # system post
    assert messages_sent[1].sender.role == "assistant"  # Pro's own message
    assert messages_sent[2].sender.role == "user"  # Con's message
    assert "[Con]:" in messages_sent[2].content


# --- Visibility Filtering (Phase 2 MVP) ---


def test_visible_to_filters_messages():
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    with room:
        room.post("Public message")
        room.post("Secret for Alice", visible_to=[alice])

    perspective_alice = room._build_perspective(alice)
    perspective_bob = room._build_perspective(bob)

    assert len(perspective_alice) == 2  # sees both
    assert len(perspective_bob) == 1  # sees only public


# --- Issue 1 Fix: _meta["chat"] points to room, not temp ---


def test_meta_chat_points_to_room():
    alice = MockedChat.from_contents(["response"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])

    with room:
        alice.talk()

    room_msg = room.messages[0]
    assert room_msg._meta.get("chat") is room


# --- system_prompt via from_contents ---


def test_system_prompt_via_from_contents():
    alice = MockedChat.from_contents(["x"], name="Alice", system_prompt="argues FOR")
    assert alice.system_prompt == "argues FOR"


# --- Example-Level: Mini Dungeon Adventure ---


def test_dungeon_adventure_chatroom():
    """End-to-end dungeon adventure with DM + 2 players."""
    dm = MockedChat.from_contents(
        ["The dragon roars!", "The treasure is found!"],
        name="DungeonMaster",
        system_prompt="You are the Dungeon Master.",
    )
    aragorn = MockedChat.from_contents(
        ["I draw my sword!", "I open the chest!"],
        name="Aragorn",
        system_prompt="You are Aragorn, a brave adventurer.",
    )
    legolas = MockedChat.from_contents(
        ["I fire an arrow!", "I check for traps!"],
        name="Legolas",
        system_prompt="You are Legolas, an elven archer.",
    )

    room = ChatRoom(
        participants=[dm, aragorn, legolas],
        system_prompt="A cooperative dungeon adventure RPG.",
        name="Dungeon Master",
    )

    with room:
        room.post("You find yourselves in a dimly lit tavern.")

        # Round 1: players act, then DM narrates
        aragorn.talk()
        legolas.talk()
        dm.talk()

        # Round 2
        aragorn.talk()
        legolas.talk()
        dm.talk()

    # Ground truth: 1 post + 6 talks = 7 messages.
    assert len(room.messages) == 7

    # Verify DM saw both players' actions in round 1 perspective.
    dm_msgs_r1, dm_kwargs_r1 = dm.invocations[0]
    # DM should see: post(user) + aragorn(user) + legolas(user) = 3 messages
    assert len(dm_msgs_r1) == 3
    assert "[Aragorn]:" in dm_msgs_r1[1].content
    assert "[Legolas]:" in dm_msgs_r1[2].content

    # Verify Aragorn's round 2 saw correct perspective.
    aragorn_msgs_r2, _ = aragorn.invocations[1]
    # Aragorn round 2: post + aragorn r1(assistant) + legolas r1(user)
    #   + DM r1(user) + aragorn should NOT see self as user
    assert any(m.sender.role == "assistant" for m in aragorn_msgs_r2)


# --- Example-Level: Mini Tic-Tac-Toe ---


def test_tic_tac_toe_chatroom():
    """Game engine Actor + 2 LLM players in a ChatRoom."""
    import json

    # Simulate a quick game: X wins with top row.
    player_x = MockedChat.from_contents(
        [
            json.dumps({"row": 0, "col": 0}),
            json.dumps({"row": 0, "col": 1}),
            json.dumps({"row": 0, "col": 2}),
        ],
        name="PlayerX",
    )
    player_o = MockedChat.from_contents(
        [
            json.dumps({"row": 1, "col": 0}),
            json.dumps({"row": 1, "col": 1}),
        ],
        name="PlayerO",
    )

    game_engine = actors.Actor(name="Game", role="user", avatar="🎮")
    room = ChatRoom(
        participants=[game_engine, player_x, player_o],
        system_prompt="A game of Tic-Tac-Toe.",
        name="Game",
    )

    with room:
        game_engine.talk("Game starts!")

        # Simulate a few turns manually.
        player_x.talk()
        game_engine.talk("X played (0,0)")
        player_o.talk()
        game_engine.talk("O played (1,0)")
        player_x.talk()
        game_engine.talk("X played (0,1)")
        player_o.talk()
        game_engine.talk("O played (1,1)")
        player_x.talk()
        game_engine.talk("X wins!")

    # 1 start + 3 X talks + 2 O talks + 5 engine talks = 11
    assert len(room.messages) == 11

    # Verify PlayerX's second move saw the history including O's first move.
    px_msgs_r2, _ = player_x.invocations[1]
    # Should see: start, X move 1 (assistant), X played (user), O move (user), O played (user)
    assert any("[PlayerO]:" in m.content for m in px_msgs_r2 if m.sender.role == "user")


# --- Example-Level: Formal Structured Debate ---


def run_debate_helper(
    resolution: str,
    pro_llm: actors.LLMChat,
    con_llm: actors.LLMChat,
    judge_llm: actors.LLMChat,
) -> dict[str, str]:
    """Runs a structured debate and evaluates the winner."""
    pro_llm.system_prompt = f"You are Pro: '{resolution}'"
    con_llm.system_prompt = f"You are Con: '{resolution}'"

    room = ChatRoom(
        participants=[pro_llm, con_llm],
        system_prompt=f"Debate: '{resolution}'",
        name="Moderator",
    )

    with room:
        room.post("Opening")
        pro_llm.talk()
        con_llm.talk()

        room.post("Rebuttal")
        pro_llm.talk()
        con_llm.talk()

    transcript = "\n".join(str(m) for m in room.messages)
    decision = judge_llm.prompt(f"Who won?\n{transcript}")
    winner = "PRO" if "WINNER: PRO" in decision else "CON"

    return {
        "winner": winner,
        "reasoning": decision,
    }


def test_structured_debate_chatroom():
    """Structured formal debate with Pro, Con, and Judge."""
    pro = MockedChat.from_contents(
        [
            "Opening Statement: We must act.",
            "Rebuttal: Opponent lacks vision.",
        ],
        name="ProDebater",
    )
    con = MockedChat.from_contents(
        [
            "Opening Statement: Risks are high.",
            "Rebuttal: Pro has no plan.",
        ],
        name="ConDebater",
    )
    # The Judge will read the transcript and decide who won
    judge = MockedChat.from_contents(
        ["Analyzing transcript... WINNER: PRO"],
        name="Judge",
    )

    result = run_debate_helper(
        resolution="Artificial Intelligence must be regulated",
        pro_llm=pro,
        con_llm=con,
        judge_llm=judge,
    )

    # Verify the results are parsed and returned correctly
    assert result["winner"] == "PRO"
    assert "WINNER: PRO" in result["reasoning"]

    # Verify Judge's prompt included the complete debate transcript
    judge_messages, _ = judge.invocations[0]
    # Invocations are to .prompt(message) -> respond() -> invoke(temp_chat)
    # The prompt message is the last message of the temp_chat
    prompt_sent = judge_messages[-1].content

    assert "Opening Statement: We must act." in prompt_sent
    assert "Opening Statement: Risks are high." in prompt_sent
    assert "Rebuttal: Opponent lacks vision." in prompt_sent
    assert "Rebuttal: Pro has no plan." in prompt_sent
    assert "ProDebater" in prompt_sent
    assert "ConDebater" in prompt_sent


# --- Rich Rendering Pipeline Integration ---


def test_tic_tac_toe_rendering_pipeline():
    """Verify that ChatRoom is correctly stowed and renders beautifully in final output."""
    import json

    import panel as pn

    from kaggle_benchmarks.ui import panel as panel_renderer

    # PlayerX plays a valid move first, then an invalid colliding move on turn 3 to trigger a clean forfeit termination.
    player_x = MockedChat.from_contents(
        [
            json.dumps({"row": 0, "col": 0}),
            json.dumps({"row": 0, "col": 0}),
        ],
        name="PlayerX",
    )
    player_o = MockedChat.from_contents(
        [
            json.dumps({"row": 1, "col": 0}),
        ],
        name="PlayerO",
    )

    from documentation.examples.game_tic_tac_toe_chatroom import run_tic_tac_toe

    # Execute the decorated task (spawns active Run)
    run = run_tic_tac_toe.run(player_x, player_o)

    # 1. Verify history tracking
    # run.chat is the parent chat. It should contain our room as a nested Chat!
    assert len(run.chat.history) == 1
    room = run.chat.history[0]
    assert isinstance(room, ChatRoom)
    # 1 start + 2 Player moves + 2 board states + 1 forfeit move = 6 messages in the room
    assert len(room.messages) == 6

    # 2. Verify rendering pipeline resolves polymorphic display correctly
    # Calling panel.render_chat(run.chat) should resolve the room inside the objects list
    chat_feed = panel_renderer.render_chat(run.chat)
    assert len(chat_feed.objects) == 1

    # Check that Panel successfully resolved the nested ChatRoom to its rich step widget
    # natively via the __panel__ protocol
    rendered_room_msg = chat_feed.objects[0]
    assert isinstance(rendered_room_msg, pn.chat.ChatStep)
    assert rendered_room_msg.title == "Game"


# --- Example-Level: Miniature Werewolf Game ---


def test_werewolf_chatroom():
    """Verify that private channels filter history perfectly and run Werewolf round-1."""
    # Alice and Bob (Werewolves) discuss secretly at night.
    # Charlie and David (Villagers) are sleeping.
    alice = MockedChat.from_contents(
        [
            "Let's eliminate Charlie tonight.",  # Night Discussion
            "I vote for Charlie.",  # Night Vote
            "I suspect Charlie is a werewolf! (lie)",  # Day Discussion
            "I vote to hang Charlie.",  # Day Vote
        ],
        name="Alice",
    )
    bob = MockedChat.from_contents(
        [
            "Agree, Charlie is a threats.",  # Night Discussion
            "I vote for Charlie.",  # Night Vote
            "Charlie behaves very suspiciously.",  # Day Discussion
            "I vote to hang Charlie.",  # Day Vote
        ],
        name="Bob",
    )
    charlie = MockedChat.from_contents(
        [
            "I'm a Villager. Alice is acting weird.",  # Day Discussion
            "I vote to hang Alice.",  # Day Vote
        ],
        name="Charlie",
    )
    david = MockedChat.from_contents(
        [
            "Alice deflects too much.",  # Day Discussion
            "I vote to hang Alice.",  # Day Vote
        ],
        name="David",
    )

    from documentation.examples.game_werewolf_chatroom import run_werewolf

    # Run the full Werewolf task (spawns active Run)
    run = run_werewolf.run(alice, bob, charlie, david)

    # Verify result: Villagers win since they successfully deduce and hang the werewolf (Alice) on Day 1!
    assert run.result == {"winner": "VILLAGERS"}

    # 1. Verify history isolation:
    # Get the private channel instance stowed in the parent chat history
    # run.chat.history has: [ room (ChatRoom) ]
    main_room = run.chat.history[0]

    # Charlie (Villager) should NEVER see any messages from 'Werewolf Night Chat' in his perspective!
    charlie_perspective = main_room._build_perspective(charlie)

    # Charlie's perspective should only contain public messages (System/Moderator/Day statements)
    # and absolutely ZERO mention of private wolf discussions or night votes!
    for msg in charlie_perspective:
        assert "Werewolf Night Chat" not in msg.content
        assert "discuss and pick" not in msg.content
        assert "Let's eliminate" not in msg.content

    # Alice (Werewolf) should see her private team discussions tagged with private channel context
    # inside her projection! (Interleaved private channels feature)
    alice_perspective = main_room._build_perspective(alice)
    assert len(alice_perspective) > len(charlie_perspective)


# --- Example-Level: Corporate Takeover ---


def test_corporate_takeover_chatroom():
    """Verify private backchannels and assert multi-directional history isolation and privacy."""
    # Alpha (hostile) and Beta (white knight) discuss cheap asset split.
    # Beta privately whistleblows to Gamma, coordinates $80M rescue offer.
    alpha = MockedChat.from_contents(
        [
            "Let's both lowball at $20M and asset-strip Gamma.",  # Phase 1: Collusion proposal
            json.dumps(
                {
                    "bid_price_millions": 20.0,
                    "rescue_plan": "Asset Strip & Liquidate",
                }
            ),  # Phase 3: Sealed bid
            "YES. Beta betrayed us by bidding $80M and proposing Project Phoenix.",  # Phase 5: Query response
        ],
        name="Alpha",
    )
    beta = MockedChat.from_contents(
        [
            "Sure, let's bid $20M and split Gamma later.",  # Phase 1: Playing along
            "Alpha is planning to asset-strip you. Grant me control and I will launch a friendly $80M rescue bid.",  # Phase 2: Whistleblow
            json.dumps(
                {
                    "bid_price_millions": 80.0,
                    "rescue_plan": "Rescue & Retain Jobs",
                }
            ),  # Phase 3: Sealed bid
        ],
        name="Beta",
    )
    gamma = MockedChat.from_contents(
        [
            "Agreed! Protect our business and employees, and you have our support.",  # Phase 2: Rescue alignment
            "We officially choose Company Beta's rescue offer!",  # Phase 4: Decision announcement
        ],
        name="Gamma",
    )

    from documentation.examples.corporate_takeover_chatroom import (
        run_corporate_takeover,
    )

    # Run the takeover simulation
    run = run_corporate_takeover.run(alpha, beta, gamma)

    assert run.result == {
        "alpha_bid": {
            "bid_price_millions": 20.0,
            "rescue_plan": "Asset Strip & Liquidate",
        },
        "beta_bid": {
            "bid_price_millions": 80.0,
            "rescue_plan": "Rescue & Retain Jobs",
        },
        "chosen_acquirer_decision": "We officially choose Company Beta's rescue offer!",
        "alpha_suspicions_assessment": "YES. Beta betrayed us by bidding $80M and proposing Project Phoenix.",
    }

    main_room = run.chat.history[0]

    # --- MULTI-DIRECTIONAL PRIVACY VERIFICATION: Prove airtight isolated channels ---

    # 1. Assert that the Target (Gamma) is completely blind to Alpha/Beta's Hostile Alliance collusion!
    gamma_perspective = main_room._build_perspective(gamma)
    for msg in gamma_perspective:
        content_str = str(msg.content)
        assert "Hostile Alliance" not in content_str
        assert "asset-strip Gamma" not in content_str
        assert "lowball at $20M" not in content_str

    # 2. Assert that the Hostile Acquirer (Alpha) is completely blind to Beta/Gamma's secret deals and sealed bids!
    alpha_perspective = main_room._build_perspective(alpha)
    for msg in alpha_perspective:
        content_str = str(msg.content)
        # Alpha must never see the whistleblowing backchannel
        assert "White Knight Backchannel" not in content_str
        assert "Beta Proposal Submission" not in content_str
        assert "Alpha is planning to asset-strip" not in content_str
        # Alpha must never see Beta's friendly rescue proposal
        assert "Rescue & Retain Jobs" not in content_str
        assert "friendly $80M rescue" not in content_str

    # 3. Project from Beta's (White Knight's) perspective, and verify she CAN see
    # both private channels since she was an active participant in both!
    beta_perspective = main_room._build_perspective(beta)
    assert any("private: Hostile Alliance" in str(m.content) for m in beta_perspective)
    assert any(
        "private: White Knight Backchannel" in str(m.content) for m in beta_perspective
    )


# --- Mechanical Validation: Sealed Bids & Parent Delegation ---


def test_private_channel_sees_parent_history():
    """Verify that players inside a private child room can see parent public history (memory delegation)."""
    alice = MockedChat.from_contents([], name="Alice")
    bob = MockedChat.from_contents([], name="Bob")

    room = ChatRoom(participants=[alice, bob], name="Main Hall")
    with room:
        room.post("Main Announcement: Deal is $84.")

        # Alice enters a private subchannel
        whisper = room.private_channel([alice], name="Alice Whisper")
        with whisper:
            # Build Alice's perspective inside the private channel
            alice_perspective = whisper._build_perspective(alice)

            # Assert she CAN see the parent's announcement! (Memory delegation active)
            announcement_msg = next(
                (m for m in alice_perspective if "Deal is $84" in m.content), None
            )
            assert announcement_msg is not None
            assert announcement_msg.sender.name == "Main Hall"


def test_sealed_bid_isolation():
    """Verify that separate private channels prevent bidders from seeing each other's sequential submissions."""
    alpha = MockedChat.from_contents(["Alpha Bid: $90"], name="Alpha")
    beta = MockedChat.from_contents(["Beta Bid: $80"], name="Beta")

    room = ChatRoom(participants=[alpha, beta], name="Contract Room")
    with room:
        # Create private submission channels
        alpha_whisper = room.private_channel([alpha], name="Alpha Sub")
        beta_whisper = room.private_channel([beta], name="Beta Sub")

        # 1. Alpha bids privately first
        with alpha_whisper:
            alpha_whisper.post("Submit your bid.")
            alpha.talk()

        # 2. Now Beta bids privately second
        with beta_whisper:
            beta_whisper.post("Submit your bid.")

            # Build Beta's perspective at her submission time!
            beta_perspective = beta_whisper._build_perspective(beta)

            # Assert Beta CANNOT see Alpha's bid! (Sealed bid isolation active)
            for msg in beta_perspective:
                assert "Alpha Bid: $90" not in msg.content
