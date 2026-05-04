# Model Loading, Dataset Evaluation, and Publishing

## Model Loading

### Three approaches:

```python
# 1. Default model (Preferred — lets Kaggle platform manage model selection)
kbench.llm          # Default model
kbench.judge_llm    # Judge model

# 2. Named model from available models
kbench.llms["google/gemini-2.5-flash"]
kbench.llms["meta/llama-3.1-70b"]

# 3. Direct ModelProxy (for explicit API control)
from kaggle_benchmarks.kaggle import model_proxy
llm = model_proxy.ModelProxy(model="google/gemini-2.5-flash", api="genai")
llm = model_proxy.ModelProxy(model="google/gemini-2.5-flash", api="openai")
```

**When to use which:**
- **`kbench.llm`**: Default choice — portable across Kaggle's "Add Models" feature
- **`kbench.llms["..."]`**: When you need a specific model (e.g., vision, judge)
- **`ModelProxy`**: When you need to specify the API backend (genai vs openai)

---

## Dataset Evaluation

```python
import pandas as pd

@kbench.task()
def qa_task(llm, question, answer) -> bool:
    response = llm.prompt(question)
    return answer.lower() in response.lower()

df = pd.DataFrame([
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
])

# Task parameter names must match DataFrame column names
results = qa_task.evaluate(llm=[kbench.llm], evaluation_data=df)
print(results.as_dataframe())
```

### Caching

```python
with kbench.client.enable_cache():
    results = my_task.evaluate(llm=[kbench.llm], evaluation_data=df)
```

---

## Publishing to Leaderboard

```python
# Final cell of Kaggle notebook:
%choose my_main_task
```

Currently only one task per notebook for leaderboards.

---

## Testing Your Tasks

### Running with uv

```bash
source ~/ws/uv/bin/activate

uv pip install -e .
uv run python documentation/examples/simple_task.py
uv run --group test pytest tests/ -v
```

### Environment Variables

- `MODEL_PROXY_URL` — Model proxy endpoint
- `MODEL_PROXY_API_KEY` — API key
- `KBENCH_EXECUTION_MODE` — `testing` for test mode
- `KBENCH_UI_MODE` — `panel`, `console`, or `none`

### MockedChat for Unit Tests

```python
from tests.mocks import MockedChat

mock = MockedChat(responses=["Paris", "42"])
response1 = mock.prompt("Capital of France?")  # Returns "Paris"
response2 = mock.prompt("What is 6*7?")         # Returns "42"

# Verify what was sent
assert mock.invocations[0].messages[0].content == "Capital of France?"
```
