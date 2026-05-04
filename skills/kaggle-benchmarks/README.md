# Kaggle Benchmarks Skill

This skill teaches AI coding agents how to write benchmark tasks using the `kaggle_benchmarks` Python library.

## Structure

```
skills/kaggle-benchmarks/
├── SKILL.md                           # Entry point — quick reference, rules, common mistakes
├── README.md                          # This file
└── references/
    ├── tasks_and_running.md           # Import styles, task definition, .run(), .evaluate()
    ├── llm_interaction.md             # llm.prompt(), structured output, multimodal, reasoning
    ├── assertions.md                  # All assertions, LLM-as-judge, custom assertions
    ├── conversations.md               # chats.new(), chats.fork(), contexts
    ├── tools.md                       # Python execution, web/HTML, custom tools
    ├── advanced.md                    # Model loading, dataset eval, testing, env vars
    └── examples.md                    # 9 complete benchmark patterns (A-I)
```

## Installation

Point your AI coding agent to read `SKILL.md`. The agent will follow references to `references/` files as needed.

**Gemini CLI** — Copy to the auto-discovery location:
```bash
cp -r skills/kaggle-benchmarks ~/.gemini/skills/kaggle-benchmarks
```

**Claude Code** — Add to your `CLAUDE.md`:
```
@skills/kaggle-benchmarks/SKILL.md
```

**Other tools** — Ask your agent to read `skills/kaggle-benchmarks/SKILL.md`.

## Testing

See `skill_tests/agent_test_scenarios.md` for 54 validated test scenarios covering all skill patterns.
