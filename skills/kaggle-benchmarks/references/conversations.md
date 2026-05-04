# Conversation Management

## Default: Automatic History

Within a task, `llm.prompt()` calls share history:

```python
@kbench.task()
def multi_turn(llm):
    llm.prompt("My favorite color is blue.")
    response = llm.prompt("What's my favorite color?")
    kbench.assertions.assert_contains_regex(r"(?i)blue", response)
```

## `chats.new()` — Isolated Conversation

Creates a clean conversation (no shared history):

```python
with kbench.chats.new("evaluation") as chat:
    judge_llm.prompt("Rate this response...")  # Clean slate
```

Parameters:
```python
kbench.chats.new(
    name="chat_name",                    # Display name
    system_instructions="You are ...",   # Optional system prompt
    orphan=False,                        # If True, don't nest in parent chat history
)
```

## `chats.fork()` — Copy Current History

Creates a new conversation starting with the current chat's history (the original chat is unaffected):

```python
# Build up some context
llm.prompt("My name is Alice and I'm a data scientist.")
llm.prompt("I work on NLP projects.")

# Branch the conversation — fork has full history, original continues separately
with kbench.chats.fork("hypothesis") as branch:
    # This prompt sees "Alice" + "NLP" context
    response = llm.prompt("Given my background, suggest a research topic.")
    # Anything said here does NOT affect the original conversation

# Back in original — still only has the two original messages
response = llm.prompt("What's my name?")  # Still remembers "Alice"
```

## `contexts.enter()` — Multi-Agent

For complex multi-agent scenarios:

```python
from kaggle_benchmarks import chats, contexts

agent_a_chat = chats.Chat(name="Agent A")
agent_b_chat = chats.Chat(name="Agent B")

with contexts.enter(chat=agent_a_chat):
    response_a = llm_a.prompt("Agent A's prompt...")

with contexts.enter(chat=agent_b_chat):
    response_b = llm_b.prompt("Agent B's prompt...")
```

## Choosing Conversation Strategy

| Scenario | Method |
|----------|--------|
| Default multi-turn | Automatic — just call `llm.prompt()` repeatedly |
| Judge evaluation | `chats.new("judge")` — no history leakage |
| System instructions for a section | `chats.new(system_instructions="...")` |
| Continue with shared history | `chats.fork("branch")` |
| Multiple agents with separate histories | `contexts.enter(chat=...)` |
