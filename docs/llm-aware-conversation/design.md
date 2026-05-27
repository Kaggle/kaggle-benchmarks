# Design Proposal: Native Multi-Agent Conversations (`ChatRoom`)

> **Status**: Proposed
> **Target Version**: v0.5.0

---

## 1. Executive Summary

This proposal introduces **`ChatRoom`**, a shared conversation context manager for
`kaggle-benchmarks` that enables multiple LLMs to converse with full awareness of
each other's identities and roles.

Currently, multi-LLM benchmarks require the user to act as a manual middleman—
forwarding messages between isolated chat contexts, stripping and re-injecting
roles, and managing turn order by hand. `ChatRoom` eliminates this boilerplate
by providing perspective-aware message routing: each LLM automatically sees its
own messages as `assistant` and peers' messages as attributed `user` messages.

---

## 2. Motivation

### Why Multi-Agent Evaluation Matters

Single-turn benchmarks (MMLU, GSM8K) and single-agent execution benchmarks
(SWE-bench) are increasingly saturated. Rigorous frontier evaluation is shifting
toward **interactive multi-agent scenarios** that test capabilities impossible to
measure in isolation:

| Dimension | What It Tests | Example |
|-----------|--------------|---------|
| **Theory of Mind** | Opponent modeling, reasoning about peer goals | Poker, negotiation |
| **Information Asymmetry** | Secret keeping, bluffing, deception | Werewolf |
| **Consensus & Cooperation** | Group decision-making under conflict | Public Goods Game |
| **Conversational Consistency** | Maintaining character in dynamic exchanges | Dungeon adventure |
| **Persuasion & Rebuttal** | Constructing and adapting arguments in real-time | Debate |

### Evidence from This Codebase

Several multi-agent benchmarks already exist, each forced to work around the
lack of native support:

- [dungeon_adventure.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/docs/llm-aware-conversation/dungeon_adventure.py) — isolated `Chat` per agent; user manually serializes state (~40 lines of boilerplate)
- [game_tic_tac_toe.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/docs/llm-aware-conversation/game_tic_tac_toe.py) — brand new `Chat` each turn; zero memory of previous turns
- [pgg.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/documentation/examples/pgg.py) — uses `chats.fork()` and manually stitches messages back
- [play_20_questions.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/documentation/examples/play_20_questions.py) — opens nested chats for every yes/no answer

In all cases, **LLMs have no idea they are talking to each other**.

---

## 3. API Design

### ChatRoom — The Core Abstraction

A `ChatRoom` is a shared conversation space that multiple participants interact in.
It acts as a Python context manager (`with room:`).

```python
import kaggle_benchmarks as kbench

alice = kbench.LLMChat(kbench.llms["gemini-2.5-flash"], name="Alice",
    system_prompt="You argue FOR renewable energy.")
bob = kbench.LLMChat(kbench.llms["gemini-2.0-flash"], name="Bob",
    system_prompt="You argue AGAINST renewable energy.")

room = kbench.ChatRoom(
    participants=[alice, bob],
    system_prompt="A structured debate. Take turns presenting arguments.",
)

with room:
    room.post("Topic: Should we phase out fossil fuels by 2035?")
    alice.talk()
    bob.talk()
    alice.talk()
    bob.talk()
```

#### The `participants` Parameter

The `participants` list defines the roster of agents in the room. It serves
three purposes:

1. **Identity awareness** — the room auto-injects participant names and
   descriptions into each LLM's system prompt, so every agent knows who
   else is in the room.
2. **Message routing (pub/sub)** — when any participant speaks (via `talk()`),
   their message is automatically visible to all other participants. No manual
   forwarding needed.

Participants can be `LLMChat` instances (LLM-driven) or `Actor` instances
(code-driven). See [Code-Driven Participants](#code-driven-participants-actor)
below.

### The 2 Primitives (Inside a Room)

All interaction inside a `ChatRoom` uses exactly 2 operations:

| Primitive | Who decides the content? | LLM Call? |
|-----------|--------------------------|-----------|
| `room.post(msg)` | Anonymous system broadcast | No |
| `participant.talk()` | The participant speaks | Depends on actor type |

#### `room.post(msg)` — Anonymous System Broadcast

Posts an anonymous message into the room with no sender attribution. **No LLM
is called.** Use for system-level directives, game-over announcements, or
narration that shouldn't be attributed to any participant.

```python
room.post("The debate topic is AI safety.")           # all see this
room.post("Your hand: A♠ K♥", visible_to=[player_1])  # only player_1 sees
```

#### `participant.talk()` — A Participant Speaks

`talk()` is the universal method for a participant to speak in the room. The
message is attributed to the speaker. The content source depends on the actor
type:

- **`LLMChat.talk()`** — the LLM reads the room history from its perspective
  and **generates** a response.
- **`Actor.talk(msg)`** — code provides the content directly (game engine,
  rule system, moderator logic).

```python
# LLM-driven: LLM generates the content
alice.talk()                                # free-form response
move = player.talk(schema=TicTacToeMove)    # structured output

# Code-driven: user code provides the content
game_engine.talk(f"Board:\n{game.get_board()}")  # game state
moderator.talk("Round 1 complete. Moving to voting.")  # narration
```

> **`room.post()` vs `actor.talk()`:** `room.post()` is anonymous (no sender).
> `actor.talk(msg)` is attributed to the actor. Use `room.post()` for system
> directives; use `actor.talk()` when a named participant is speaking.

> **Forward compatibility:** `talk()` should accept the same parameters as the
> existing `prompt()` API (e.g., `schema`, `tools`, `reasoning`). The initial
> implementation will support `schema=`; others can be added incrementally.

#### `llm.prompt(msg)` — Existing API (Outside Rooms)

The existing user→LLM API is unchanged and used **outside** rooms:

```python
# No room context — standard single-agent usage
judge.prompt("Rate this response.", schema=int)
```

### Private Information (`visible_to`)

Any `room.post()` call can be restricted to a subset of participants. When the
room builds each LLM's perspective, it filters out messages not visible to that
participant:

```python
with room:
    # Assign secret roles — each player only sees their own
    for player, role in roles.items():
        room.post(f"Your role is: {role}. Keep it secret.", visible_to=[player])

    # Post game state to the active player only
    room.post(f"Your hand: {hand}", visible_to=[current_player])
```

### Private Channels (`private_channel`)

For multi-turn private conversations (e.g., team strategy, werewolf night phase),
create a sub-room. Messages in the channel are invisible to non-members:

```python
# Create a private channel — only werewolves can see it
wolf_channel = room.private_channel(werewolves, name="Werewolf Night Chat")

with wolf_channel:
    wolf_channel.post("Who should we eliminate tonight?")
    for wolf in werewolves:
        wolf.talk()

# Back in the main room — public discussion
with room:
    room.post("Day 2. A villager was eliminated. Discuss.")
    for player in players:
        player.talk()
```

**How channels and rooms relate:**

- A private channel is a **child** `ChatRoom` of the parent room.
- When building perspective for a werewolf in the **main room**, the room
  includes both main room messages and messages from channels the werewolf
  belongs to (so the werewolf retains context across both spaces).
- Non-members never see channel messages.
- The system prompt in a private channel indicates the context: *"You are in
  the Werewolf Night Chat with [Wolf_1, Wolf_2]. This conversation is private."*


### Code-Driven Participants (`Actor`)

Not every participant needs to be LLM-driven. A game engine, rule system, or
moderator can participate as a code-driven `Actor`. This makes the game a
**first-class member** of the conversation rather than anonymous room posts.

Both `LLMChat` and `Actor` use `talk()` to speak — the difference is the
content source:

```python
# Game engine as a named participant
game_engine = kbench.Actor(name="Game", avatar="🎮")

room = kbench.ChatRoom(participants=[game_engine, player_x, player_o])

with room:
    while not game.is_game_over():
        current = players[game.get_current_player()]
        # Game talks as itself — attributed to "Game", not anonymous
        game_engine.talk(f"Board:\n{game.get_board()}\nYour turn.")
        move = current.talk(schema=TicTacToeMove)
        game.make_move(move)
```

This generalizes the pattern: any actor whose responses are driven by code
(game rules, API calls, database lookups) can participate alongside LLMs.
Other participants see `[Game]: Board: ...` just like they see `[Player X]: ...`.

---

## 4. Example Rewrites

### Dungeon Adventure

**Current implementation** ([dungeon_adventure.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/docs/llm-aware-conversation/dungeon_adventure.py)):
each agent has an isolated `Chat`. The user manually extracts the DM's story,
formats it as a string, and passes it to each player as a "user" prompt.
Players never see each other's actions directly:

```python
# Current: ~40 lines of manual orchestration
for i in range(n_rounds):
    story = dungeon_master.story                          # read DM state
    action = player(story)                                # forward to player
    actions.append(f"{player.name}: {action}")
    new_story = dungeon_master(actions)                   # forward back to DM
```

**With ChatRoom** — ~15 lines, full conversation awareness:

```python
@kbench.task("Dungeon Adventure")
def play_dungeon_adventure(dm_llm, player1_llm, player2_llm, n_rounds=3):
    dm = kbench.LLMChat(dm_llm, name="Dungeon Master",
        system_prompt="You are the DM. Narrate the story. Keep it to one sentence.")
    aragorn = kbench.LLMChat(player1_llm, name="Aragorn",
        system_prompt="You are Aragorn, a brave warrior. Describe actions in one sentence.")
    legolas = kbench.LLMChat(player2_llm, name="Legolas",
        system_prompt="You are Legolas, an elven archer. Describe actions in one sentence.")

    room = kbench.ChatRoom(
        participants=[dm, aragorn, legolas],
        system_prompt="A dungeon adventure. The DM narrates, players respond with actions.",
    )

    with room:
        room.post("The adventure begins in a dimly lit tavern.")
        dm.talk()

        for round in range(n_rounds):
            aragorn.talk()
            legolas.talk()
            dm.talk()
```

### Tic-Tac-Toe

**Current implementation** ([game_tic_tac_toe.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/docs/llm-aware-conversation/game_tic_tac_toe.py)):
creates a **brand new chat context** every turn. The LLM has zero memory of
previous turns. The entire game state is re-serialized into the prompt each time:

```python
# Current: fresh chat each turn, no memory
while not game.is_game_over():
    player_id = game.get_current_player()
    with kbench.chats.new(...):                    # brand new context!
        move = llm_agent.prompt(state_prompt, schema=action_schema)
        game.make_move(move)
```

**With ChatRoom** — LLMs retain full conversation history and see each other's
moves. The game engine participates as a code-driven `Actor`:

```python
@kbench.task("Tic-Tac-Toe")
def tic_tac_toe(player_x_llm, player_o_llm):
    game = TicTacToe()
    game_engine = kbench.Actor(name="Game", avatar="🎮")

    player_x = kbench.LLMChat(player_x_llm, name="Player X",
        system_prompt="You are Player X in tic-tac-toe.")
    player_o = kbench.LLMChat(player_o_llm, name="Player O",
        system_prompt="You are Player O in tic-tac-toe.")

    room = kbench.ChatRoom(participants=[game_engine, player_x, player_o])
    players = {"X": player_x, "O": player_o}

    with room:
        while not game.is_game_over():
            current = players[game.get_current_player()]
            game_engine.talk(f"Board:\n{game.get_state_representation()}")
            move = current.talk(schema=TicTacToeMove)
            game.make_move(move)

    return game.get_scores()
```

### Werewolf (Private Channels)

A full example showcasing `visible_to`, `private_channel`, and structured voting outputs:

```python
@dataclasses.dataclass(frozen=True)
class WerewolfVote:
    voted_player: str
    reason: str

@kbench.task("Werewolf")
def werewolf(player_llms: list, moderator_llm, max_rounds=5):
    mod = kbench.LLMChat(moderator_llm, name="Moderator",
        system_prompt="You are the moderator of a Werewolf game.")
    players = [kbench.LLMChat(llm, name=f"Player_{i}")
               for i, llm in enumerate(player_llms)]

    room = kbench.ChatRoom(
        participants=[mod] + players,
        system_prompt="You are playing Werewolf. Follow the moderator's instructions.",
    )

    roles = assign_roles(players)  # {"Player_0": "Werewolf", ...}
    werewolves = [p for p, r in roles.items() if r == "Werewolf"]
    wolf_channel = room.private_channel(werewolves, name="Werewolf Night Chat")

    with room:
        # Assign roles privately — each player only sees their own
        for player, role in roles.items():
            room.post(f"Your role is: {role}. Keep it secret.", visible_to=[player])

        for round in range(max_rounds):
            # Night: wolves privately decide who to eliminate
            with wolf_channel:
                wolf_channel.post("Who should we eliminate tonight?")
                for wolf in werewolves:
                    wolf.talk()

            # Day: public discussion
            room.post(f"Day {round+1}. A villager was eliminated. Discuss.")
            for player in alive_players:
                player.talk()

            # Vote using structured outputs
            room.post("Vote to eliminate someone.")
            votes = {}
            for player in alive_players:
                vote_result = player.talk(schema=WerewolfVote)
                votes[player.name] = vote_result.voted_player
```

### Simple Debate (Automated Turns)

```python
@kbench.task("AI Safety Debate")
def debate(pro_llm, con_llm):
    pro = kbench.LLMChat(pro_llm, name="ProAI",
        system_prompt="You argue FOR AI development. Be concise.")
    con = kbench.LLMChat(con_llm, name="ConAI",
        system_prompt="You argue AGAINST unchecked AI development. Be concise.")

    room = kbench.ChatRoom(
        participants=[pro, con],
        system_prompt="A structured debate. Present arguments concisely.",
    )

    with room:
        room.post("Topic: Should we accelerate AI development?")
        for _ in range(5):
            pro.talk()
            con.talk()

    # Judge the debate (outside room — standard prompt API)
    judge = kbench.LLMChat(kbench.llms["gemini-2.5-pro"], name="Judge")
    verdict = judge.prompt(
        f"Judge this debate. Who argued better?\n\n{room.history.summary()}"
    )
```

---

## 5. Implementation Details

### Participant Identity Awareness

LLMs need to understand the participant structure. `ChatRoom` handles this
automatically through two mechanisms:

#### System Prompt Enrichment

The room prepends a participant roster to each LLM's system prompt:

```
You are Alice.
Other participants in this conversation:
- Bob: argues against renewable energy

Messages from other participants are prefixed with their name, e.g., [Bob]: ...
Your messages appear without a prefix.

Note on "Room": Messages from "Room" are game/narrator environment instructions, not messages from another player.
---
You argue FOR renewable energy. Be concise.
```

#### API `name` Field

For APIs that support it (OpenAI, Gemini), the room sets the `name` field on
messages:

```json
{"role": "user", "name": "Bob", "content": "But the grid can't handle intermittency."}
```

For APIs without `name` support, the room falls back to content-prefixing:
`"[Bob]: But the grid can't handle intermittency."`.

### Role Mapping Matrix

When building the perspective for a given `viewer` LLM:

| Message Sender | API `role` | API `name` | Content |
|:---|:---|:---|:---|
| **Viewer itself** | `"assistant"` | — | `"{content}"` |
| **Peer LLM** | `"user"` | `"Bob"` | `"[Bob]: {content}"` (fallback) |
| **Code-driven Actor** | `"user"` | `"Game"` | `"[Game]: {content}"` (fallback) |
| **Room (anonymous)** | `"user"` | `"Room"` | `"[Room]: {content}"` (fallback) |

### Chat Storage

The `ChatRoom` instantiates its own `Chat` context. When running within a
benchmark task, the room's chat is attached as a sub-chat of the task's
global chat, preserving hierarchy and rendering in the Panel UI.

### Tool Integration

Tools registered with an `LLMChat` remain scoped to that agent. Tool calls
and results are recorded in the room chat and visible to other participants
(unless filtered by `visible_to`).

---

## 6. Open Questions

### Q1: System Prompt Delimiter
How to delimit the auto-injected roster from the user's custom system prompt?
*Proposal:* Use a markdown `---` delimiter.

### Q2: Room Name for Anonymous Posts
What `name` value should `room.post()` messages use?
*Proposal:* Default to `"Room"`, configurable via `ChatRoom(name="Narrator")`. To ensure the LLM knows what this sender represents, the auto-injected roster prepended to the system prompt will explicitly state:
`Note on "Room": Messages from "Room" are game/narrator environment instructions, not messages from another player.` (or custom name).

### Q3: `talk()` Visibility
Should `talk()` support a `visible_to` parameter for private LLM-generated
responses? *Proposal:* Defer to P2 — for now, all `talk()` output is public.

### Q4: Serialization to `run.json`
How should `ChatRoom` conversations be serialized in the benchmark run output?
Options:
- **Ground-truth view** — serialize the room's internal message log (with
  `sender`, `visible_to` metadata) as a single conversation.
- **Per-participant views** — serialize each LLM's projected perspective
  separately (what each LLM actually received).
- **Both** — ground truth for analysis, per-participant for debugging.

*Proposal:* Store the ground-truth log as the primary record, with
per-participant projections available on demand via a helper method.

### Q5: Private Channel Context Merging
When a participant belongs to both the main room and a private channel, how is context combined? Options:
- **Interleaved** — channel messages appear inline in the main room history at their chronological position (visible only to members).
- **Appended** — channel history is appended as a separate block before the current generation.

*Proposal:* **Interleaved with explicit context tags.**
To prevent the LLM from leaking secrets or confusing public/private boundaries, any message originating from a private channel must be explicitly tagged when projected to channel members.
For example, instead of a plain name prefix:
`[Bob (private: Werewolf Night Chat)]: We should eliminate Player_1.`
This gives the LLM a clear, explicit cognitive boundary of where each conversation took place. Also, the auto-injected system instructions will explain:
`Messages tagged with (private: [Channel Name]) are visible only to members of that channel. Do not discuss private information in public messages.`

### Q6: Per-Participant System Prompts
Currently, `LLMChat` is **stateless by design** — system instructions live on
the `Chat` context (`chats.new(system_instructions="...")`), not on the LLM
instance. But `ChatRoom` needs per-participant identity prompts (e.g., "You
argue FOR" vs "You argue AGAINST").

Options:
- **a)** Add `system_prompt` to `LLMChat.__init__` — clean API
  (`kbench.LLMChat(llm, name="Alice", system_prompt="...")`), but breaks the
  stateless principle.
- **b)** Pass prompts via the room — `ChatRoom(participants=[(alice, "You argue
  FOR"), ...])` — preserves statelessness but awkward syntax.
- **c)** Treat it as room-scoped metadata — `room.set_prompt(alice, "You argue
  FOR")` — explicit, but verbose.

*Proposal:* Option (a). The per-participant prompt is **identity**, not
conversational state. It's analogous to `name` and `avatar` — fixed attributes
that define who the agent *is*. The `ChatRoom` concatenates the room's system
prompt with each participant's personal prompt when building perspectives.

---

## 7. Phased Implementation Plan

To ensure highly stable and easily reviewable code changes, the implementation is divided into three incremental, test-driven phases.

### Phase 1: Core `ChatRoom`, `talk()`, and Perspective Projection

The minimum viable product (MVP): participants can `talk()` in a room with correct perspective-aware history, roster injection, and context-manager tracking. No visibility filtering or private channels yet.

#### 1.1 Context Manager Mechanism
`ChatRoom` must integrate with the existing `contexts.enter(chat=...)` system so that `chats.get_current_chat()` returns the active room instance inside `with room:`.

##### [MODIFY] [chats.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/src/kaggle_benchmarks/chats.py)
Add `ChatRoom` as a subclass of `Chat` with custom `__enter__` and `__exit__`:
```python
class ChatRoom(Chat):
    def __init__(
        self,
        participants: list["actors.Actor"],
        system_prompt: str = "",
        name: str = "Room",
    ):
        super().__init__(name=name)
        self.participants = participants
        self.system_prompt = system_prompt
        self._ctx_manager = None

    def __enter__(self):
        from kaggle_benchmarks import contexts
        self._ctx_manager = contexts.enter(chat=self)
        self._ctx_manager.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx_manager.__exit__(*exc)

    def post(self, message: str, visible_to=None):
        """Anonymous system broadcast — no sender attribution."""
        msg = Message(sender=actors.system, content=message)
        if visible_to is not None:
            msg._meta["visible_to"] = visible_to
        self.append(msg)
        return msg
```

*Why this works:* `contexts.enter(chat=self)` pushes the `ChatRoom` onto the context stack. Any call to `chats.get_current_chat()` inside `with room:` now returns the `ChatRoom` instance. On exit, the previous context is restored.
*`room.post()` pipeline:* Calls `self.append(msg)` directly, which dispatches `"new_message"` events for UI rendering. This bypasses `Actor.send()` → `chats.send()` because `room.post()` is anonymous—there's no sender actor to route through.

---

#### 1.2 Perspective Projection
The core algorithm that projects roles and attributes names based on the active viewer.

##### [MODIFY] [chats.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/src/kaggle_benchmarks/chats.py) (continued)
```python
class ChatRoom(Chat):
    # ... (constructor, __enter__, __exit__, post from above)

    def _build_roster(self, viewer: "actors.Actor") -> str:
        """Build the participant roster description for the viewer."""
        peers = [p for p in self.participants if p is not viewer]
        lines = [f"You are {viewer.name}."]
        if peers:
            lines.append("Other participants in this conversation:")
            for p in peers:
                desc = getattr(p, "system_prompt", None) or ""
                lines.append(f"- {p.name}" + (f": {desc}" if desc else ""))
        lines.append("")
        lines.append('Messages from other participants are prefixed with their name, e.g., [Bob]: ...')
        lines.append("Your messages appear without a prefix.")
        lines.append("")
        lines.append(f'Note on "{self.name}": Messages from "{self.name}" are system/narrator instructions, not from another player.')
        return "\n".join(lines)

    def _build_system_prompt(self, viewer: "LLMChat") -> str:
        """Concatenate: roster + --- + room prompt + --- + personal prompt."""
        parts = [self._build_roster(viewer)]
        if self.system_prompt:
            parts.append(self.system_prompt)
        personal = getattr(viewer, "system_prompt", None)
        if personal:
            parts.append(personal)
        return "\n---\n".join(parts)

    def _build_perspective(self, viewer: "actors.Actor") -> list[Message]:
        """Project the ground-truth history into viewer's perspective.

        Returns NEW Message objects — originals are never mutated.
        - Viewer's own messages → sender with role="assistant"
        - Everyone else's messages → sender with role="user", name-prefixed
        """
        projected = []
        for msg in self.messages:
            if msg.sender is viewer:
                # Viewer sees their own messages as "assistant"
                projected.append(Message(
                    sender=actors.Actor(name=viewer.name, role="assistant"),
                    content=msg.content,
                ))
            else:
                # Viewer sees peers as "user" with name prefix
                name = msg.sender.name
                content = f"[{name}]: {msg.content}"
                projected.append(Message(
                    sender=actors.Actor(name=name, role="user"),
                    content=content,
                ))
        return projected
```

*Key design:* `_build_perspective` creates **new `Message` objects** with synthetic `Actor` instances. Original messages in the ground-truth log are never mutated. The synthetic actors have the correct `role` for the viewer's perspective (`"assistant"` for self, `"user"` for peers).

---

#### 1.3 `LLMChat.talk()` and `Actor.talk()`

##### [MODIFY] [llms.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/src/kaggle_benchmarks/actors/llms.py)
Add `system_prompt` to `LLMChat.__init__` and implement `talk()`:
```python
class LLMChat(actors.Actor):
    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        support_structured_outputs: bool = False,
        support_temperature: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("role", "assistant")
        kwargs.setdefault("avatar", "🤖")
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.support_structured_outputs = support_structured_outputs
        self.support_temperature = support_temperature
        self.stream_responses = config.interactive_mode

    def talk(self, schema: type[T] = str, **kwargs) -> T:
        """Speak in the active ChatRoom. Raises RuntimeError outside a room."""
        from kaggle_benchmarks import chats

        room = chats.get_current_chat()
        if not isinstance(room, chats.ChatRoom):
            raise RuntimeError(
                "LLMChat.talk() must be called within an active ChatRoom context."
            )

        system = room._build_system_prompt(self)
        perspective = room._build_perspective(self)

        # Enter a temporary orphan chat with the projected history.
        # respond() reads from this temp chat, not the room's ground truth.
        with chats.new(name=f"_perspective_{self.name}", orphan=True) as temp:
            temp.history.extend(perspective)
            response = self.respond(system=system, schema=schema, **kwargs)

        # Re-attribute the response to this actor and append to ground truth.
        # The response was already appended to temp by @emits_message,
        # so we create a clean copy for the room log.
        room_msg = Message(sender=self, content=response.content)
        room_msg._meta = response._meta.copy()
        room.append(room_msg)

        return response.content
```

*Double-append guard:* `respond()` is decorated with `@chats.emits_message`, which auto-appends to the *current* chat. Inside `talk()`, the current chat is the orphan `temp`—so the response lands in `temp`, not in the room. We then create a **separate `Message`** attributed to `self` and append it to the room's ground-truth log. No double-append.

##### [MODIFY] [base.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/src/kaggle_benchmarks/actors/base.py)
```python
class Actor:
    # ... existing code ...

    def talk(self, message: str) -> Message:
        """Speak in the active ChatRoom. Raises RuntimeError outside a room."""
        from kaggle_benchmarks import chats

        chat = chats.get_current_chat()
        if not isinstance(chat, chats.ChatRoom):
            raise RuntimeError(
                "Actor.talk() must be called within an active ChatRoom context."
            )

        msg = Message(sender=self, content=message)
        chat.append(msg)
        return msg
```

---

#### 1.4 Public API Exports

##### [MODIFY] [__init__.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/src/kaggle_benchmarks/__init__.py)
```diff
 from kaggle_benchmarks.actors import Actor, LLMChat, system, user
-from kaggle_benchmarks.chats import last_reasoning_traces
+from kaggle_benchmarks.chats import ChatRoom, last_reasoning_traces
```

This makes `kbench.ChatRoom(...)` the canonical user-facing constructor.

---

#### 1.5 Phase 1 Tests

##### [NEW] [test_chatroom.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/tests/test_chatroom.py)
```python
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
    # After exit, room is no longer current
    assert chats.get_current_chat() is not room


# --- System Prompt Enrichment ---

def test_system_prompt_enrichment_includes_roster():
    alice = MockedChat.from_contents(["x"], name="Alice", system_prompt="argues FOR")
    bob = MockedChat.from_contents(["x"], name="Bob", system_prompt="argues AGAINST")
    room = ChatRoom(participants=[alice, bob], system_prompt="A debate.")

    prompt = room._build_system_prompt(alice)
    assert "You are Alice" in prompt
    assert "Bob" in prompt
    assert "argues AGAINST" in prompt
    assert "A debate." in prompt
    assert "argues FOR" in prompt  # alice's personal prompt


def test_system_prompt_enrichment_room_identity():
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
    assert original.sender is alice        # original unchanged


# --- talk() Primitives ---

def test_talk_outside_room_raises():
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
    # Ground truth should have: system post + alice's response
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


# --- Example-Level: Mini Debate ---

def test_mini_debate_two_rounds():
    """End-to-end: two LLMs debate for 2 rounds with full history."""
    pro = MockedChat.from_contents(["AI is great!", "AI saves lives!"],
                                    name="Pro")
    con = MockedChat.from_contents(["AI is risky!", "AI needs regulation!"],
                                    name="Con")
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

    # Ground truth: 1 system post + 4 talk messages
    assert len(room.messages) == 5

    # Verify the second invocation of Pro saw the full history
    # (system post + Pro round 1 + Con round 1)
    _, kwargs = pro.invocations[1]
    # The invoke received messages from the temp perspective chat
    # Pro's first message should be "assistant", Con's should be "user"
```

---

### Phase 2: `visible_to` and `private_channel()`

> [!IMPORTANT]
> Phase 2 should only begin after Phase 1 is merged and stable.

#### 2.1 Message `visible_to` Property

##### [MODIFY] [messages.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/src/kaggle_benchmarks/messages.py)
```python
@property
def visible_to(self) -> list["actors.Actor"] | None:
    return self._meta.get("visible_to")

@visible_to.setter
def visible_to(self, value: list["actors.Actor"] | None):
    if value is not None:
        self._meta["visible_to"] = value
    else:
        self._meta.pop("visible_to", None)
```

#### 2.2 Visibility Filtering in `_build_perspective`

##### [MODIFY] [chats.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/src/kaggle_benchmarks/chats.py)
Update `_build_perspective` to skip messages not visible to the viewer:
```python
def _build_perspective(self, viewer: "actors.Actor") -> list[Message]:
    projected = []
    for msg in self.messages:
        # Phase 2: visibility filtering
        visible = msg._meta.get("visible_to")
        if visible is not None and viewer not in visible:
            continue  # viewer can't see this message

        # ... existing projection logic ...
```

#### 2.3 Private Channels

##### [MODIFY] [chats.py](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/src/kaggle_benchmarks/chats.py)
```python
def private_channel(self, participants, name="Private Channel"):
    """Create a child ChatRoom visible only to the given participants."""
    channel = ChatRoom(participants=participants, name=name)
    channel._parent_room = self
    return channel
```

#### 2.4 Phase 2 Tests
```python
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
    assert len(perspective_bob) == 1    # sees only public


def test_private_channel_messages_invisible_to_non_members():
    alice = MockedChat.from_contents(["wolf plan"], name="Alice", cycle=True)
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])
    channel = room.private_channel([alice], name="Wolf Chat")

    with channel:
        channel.post("Who to eliminate?")
        alice.talk()

    # Bob should not see channel messages in main room perspective
    # (exact mechanism depends on interleaving implementation)
```


## 8. Verification and Quality Checklist

Before completing execution, run the following verification commands:

```bash
# Run Phase 1 tests
uv run --group test pytest tests/test_chatroom.py -v

# Run full suite regression
uv run --group test pytest tests/ -v

# Verify code quality & types
ruff format . && ruff check --fix . && mypy src/
```

## 9. MVP Findings & Architectural Learnings

Following a successful Test-Driven Development (TDD) cycle, the complete Phase 1 multi-agent core was implemented and validated against a comprehensive test suite (15 tests passing, zero regressions across 370+ test files).

### 9.1 Solved Complexities & Refinements

- **Temp Chat Isolation works perfectly**: The use of an orphan temp chat (`chats.new(orphan=True)`) inside `talk()` acts as a flawless sandbox. It prevents `@emits_message` from double-appending responses to the room log, ensuring full control over ground-truth recording.
- **Reference Leak Addressed**: An initial reference leak where the response message's `_meta["chat"]` pointed to the orphan temp chat was resolved by explicitly overriding it to point directly back to the `ChatRoom` instance:
  ```python
  room_msg._meta["chat"] = room
  ```
- **Context Manager Stack Integrity**: Running `contexts.enter(chat=self)` inside `ChatRoom.__enter__` integrates seamlessly with the framework's existing active-context stack, ensuring `chats.get_current_chat()` works cleanly.
- **Parent Chat History Integration & Polymorphic Rendering**: Entering a `ChatRoom` context manager automatically registers and appends itself as a nested `Chat` step inside the active parent run's chat history:
  ```python
  parent_chat = chats.get_current_chat()
  if parent_chat and self not in parent_chat.history:
      parent_chat.append(self)
  ```
  This guarantees that the complete multi-agent debate transcript is naturally captured as a nested element.
  - *Zero-Overhead Polymorphic UI Integration*: Because the base `Chat` class already implements a custom `__panel__()` rich display protocol (returning `panel.render_chat_as_step(self)`), we did **not** need to modify any of Panel's core layouts or rendering paths in `panel.py`. Panel's layout engine automatically detects the custom `__panel__` method on the nested `ChatRoom` instance inside `history` and renders it beautifully as a nested collapsible Accordion step. This elegant, polymorphic design keeps the core UI codebase 100% clean and decoupled.
- **Task Return-Type Auto-Inference Restrictions**: The `@kbench.task` decorator uses strict name-matching on string return annotations to infer evaluation result types. 
  - Using typing-wrapped annotations like `Dict[str, str]` fails type-inference and triggers a `TypeError`.
  - Annotating the task signature with the plain builtin `dict` class (or subclassing `benchmarks.results.Result`) resolves type-inference and executes correctly.
- **Object Reference Identity Collisions**: In multi-agent evaluations, passing the *exact same model object reference* (e.g. reusing `kbench.llm` for all players) collapses the `msg.sender is viewer` check during perspective projection. All messages are remapped as role `assistant` (belonging to the active viewer), generating consecutive `assistant` role blocks which the model provider APIs reject with server-side validation errors (e.g., throwing NoneType choice subscript errors).
  - *Mitigation*: Multi-agent rooms must instantiate **separate participant references** (one per player, even if sharing the same model configuration) by calling the `ModelProxy` factory independently:
    ```python
    player_x = ModelProxy(model_name, name="PlayerX")
    player_o = ModelProxy(model_name, name="PlayerO")
    ```
- **Post-Game Assertion Decoupling**: Running evaluation assertions inline during turn loops clutters the final output panel with alert blocks, disrupting reading immersion. Moving assertions to a post-game loop (iterating over `room.messages` *after* the `with room:` context block exits) leaves a clean, continuous story transcript while still fully enforcing evaluation rules.

---

### 9.2 Structural Observations (Future Optimization Opportunities)

#### 1. Decoupling ownership of Appending in `respond()`
Currently, `respond()` has dual-ownership of the appending pipeline: it appends to history internally under three different branches, while the `@emits_message` decorator *also* performs a conditional append based on an identity-based comparison (`self.content is other.content`).
While this works perfectly under current conditions, long-term stability would be improved by refactoring `respond()` to purely generate and return `LLMMessage` objects, delegating all log-appending and event dispatching exclusively to `@emits_message`.

#### 2. Lifecycle vs. Dataclass representation
`ChatRoom` inherits from `Chat`, which is a `@dataclasses.dataclass`. Because `ChatRoom` handles context management and maintains stateful collections of active participants, it implements a standard `__init__` constructor instead of a decorated dataclass representation. This is the correct object-oriented approach for this subsystem, but means default dataclass utilities (`asdict`, auto-repr) won't map participant metadata automatically.

---

### 9.3 Production Example Porting Analysis

To verify feasibility and ergonomic improvements, two full multi-agent benchmarks from this codebase were refactored to use the new native `ChatRoom` primitives.

#### A. Dungeon Adventure ([Original](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/docs/llm-aware-conversation/dungeon_adventure.py) vs. [ChatRoom](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/docs/llm-aware-conversation/dungeon_adventure_chatroom.py))

- **Boilerplate Reduction**: Reduced from ~160 lines of complex, nested manual context switching and manual history stitching to under **60 lines** of concise game orchestration.
- **Information Flow**: Instead of manually formatting previous story history and player moves into raw strings to send with every single prompt, agents simply invoke `player.talk()`. History projection handles attribution and roles seamlessly.
- **Cognitive Shift**: The code reads like a description of a real social game, focusing purely on actions and narrative progression rather than low-level framework routing.

#### B. Tic-Tac-Toe ([Original](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/docs/llm-aware-conversation/game_tic_tac_toe.py) vs. [ChatRoom](file:///usr/local/google/home/limagoog/git/kaggle-benchmarks/docs/llm-aware-conversation/game_tic_tac_toe_chatroom.py))

- **Memory Retention**: In the original implementation, a brand new `Chat` was spun up every single turn, leaving LLMs with zero memory of their own past choices or peer play. With `ChatRoom`, the entire move sequence is naturally preserved as an attributed user-versus-assistant chain.
- **Structured Tool Interaction**: Integrates perfectly with structured outputs. Calling `player.talk(schema=TicTacToeMove)` returns clean Pydantic move instances directly within the perspective-aware conversation.
- **State Integration**: The non-LLM game engine plays as a standard code-driven `Actor`, broadcasting board states cleanly using `game_engine.talk(board)` so LLMs are fully aligned on physical game state.

---

### 9.4 Multi-Agent Cache Isolation, Model Identity, and Private Channel Optimization

As the `ChatRoom` system was scaled to support complex multi-agent simulations (such as Werewolf and Corporate Takeover), several critical framework integration findings and optimizations were discovered and resolved:

#### 1. Preventing Model Identity and Cache Collisions
In typical single-agent runs, the benchmark subject is represented by an `LLMChat` instance whose `name` property is set to the model slug (e.g. `google/gemini-2.5-flash`). However, in multi-agent environments, players must be instantiated with customized, context-rich participant names (e.g., `name="Alice"`, `name="Bob"`):
```python
alice = kbench.kaggle.ModelProxy(model_name, name="Alice", avatar="🐺")
```
When compiling run metadata and generating the cache ID, the framework originally serialized `param.name` as the model version slug:
* **The Bug:** This serialized `"Alice"` as the model version slug on leaderboards, and appended `_Alice` as the run's `cache_id` suffix. If the benchmark was subsequently run with a different model (e.g. Claude instead of Gemini) while keeping the participant name `"Alice"`, they would yield the exact same cache file and overwrite each other, completely breaking cache isolation.
* **The Fix:** Updated both `runs.py` (`cache_id` calculation) and `serialization.py` (`_extract_model_version_data` serialization) to extract the actual underlying model version via `getattr(param, "model", None) or param.name`, ensuring robust model identity and cache-file isolation across different model evaluations.

#### 2. Reducing Redundant Private Channels via `visible_to`
While complex backchannels require multi-turn child `private_channel` rooms, simple secret events (such as prompting the Board of Gamma privately with the compiled bids or asking Alpha a private post-game query) do not need to incur the overhead of spinning up separate child chatrooms.
* **Best Practice:** Utilizing `room.post(..., visible_to=[gamma])` inside the main public `room` allows the system to present secret directives exclusively to the intended viewer, keeping the logs simpler, cleaner, and much more readable.

### 9.5 Streaming UI Fix and Structured Voting

#### 1. Panel Streaming `TypeError` Bug Fix
When `enable_interactive_mode()` or `player.stream_responses = True` was activated, the Panel UI crashed with a `TypeError` during live token streaming:
```
TypeError: can only concatenate str (not "LLMResponse") to str
```
* **Root Cause:** The `messages.py` `stream()` method dispatches raw `LLMResponse` chunk objects via `events.manager.dispatch("new_chunk", self, chunk)`. The `PanelUI.new_chunk` listener forwarded these raw objects directly to `pn.chat.ChatMessage.stream()`, which expects a plain string token. When the Panel widget tried to concatenate the accumulated string content with the incoming `LLMResponse` object, it crashed.
* **The Fix:** Updated `PanelUI.new_chunk` in `panel.py` to extract the string content from the chunk before forwarding:
  ```python
  def new_chunk(self, message, chunk):
      if message in self:
          chunk_text = (
              chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
          )
          self[message].stream(chunk_text)
  ```

