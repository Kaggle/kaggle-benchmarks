# Issac's agent benchmark tasks

These are executable Kaggle Benchmark tasks for comparing agents in the same
harness. They are separate from the `kaggle_benchmarks` library implementation.

## Clay agent wafer routing

`clay_agent_wafer_routing.py` gives an agent sixteen visible integer linear
equations. For each equation, the agent must call a typed
`submit_integer(value)` tool exactly once. The tool confirms only that the value
was recorded; it never reveals correctness. The evaluator scores the captured
action outside the model conversation.

The primary result is exact action accuracy. Logs also contain family-segmented
accuracy and hash-bound Sacred Egg timer receipts. The timers reuse the
injected-clock, coarse-window, and one-use hatch semantics of the local Sacred
Egg system. They are public deadline receipts, not authentication or encryption.

Local setup and Kaggle execution:

```powershell
kaggle b init -y --env-file .env --example-file dev/issdandavis/starter_task.py
kaggle b t push clay-agent-wafer-routing -f agent_tasks/clay_agent_wafer_routing.py --wait
kaggle b t run clay-agent-wafer-routing -m google/gemini-3-flash-preview -m deepseek-ai/deepseek-v3.2 --wait
kaggle b t download clay-agent-wafer-routing -o downloads/clay-agent-wafer-routing
```

The task remains private after push. Publishing is a separate operation.

