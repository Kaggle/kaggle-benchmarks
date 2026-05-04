# Tools

## Python Code Execution — Two Approaches

**Approach A: Extract + Run (Preferred for code generation tasks)**
```python
response = llm.prompt("Write Python to calculate factorial of 10.")
code = kbench.tools.python.extract_code(response)
result = kbench.tools.python.script_runner.run_code(code)
kbench.assertions.assert_contains_regex("3628800", result.stdout)
kbench.assertions.assert_empty(result.stderr.strip(), "No errors expected")

# For programs that read stdin:
result = kbench.tools.python.script_runner.run_code(code, input="test input\n")
```

**Approach B: IPythonREPL (for expression evaluation)**
```python
repl = kbench.tools.python.IPythonREPL()
output = repl.invoke("2 + 2", is_visible_to_llm=False)
kbench.assertions.assert_equal(4, float(output.output))
```

## Web/HTML Testing

```python
with kbench.tools.web.Browser() as browser:
    html_code = kbench.tools.web.extract_html(response)
    snapshot = browser.take_snapshot(html_code, wait_before=5000, full_page=True)
    # snapshot.html — rendered HTML
    # snapshot.logs — console logs
```

## Custom Function Tools

Define plain Python functions with type hints and docstrings. Pass them via `tools=`.

```python
def run_simple_calculator(a: float, b: float, operator: str) -> float:
    """Calculates the result of an arithmetic operation. Supported operators: + - * /"""
    if operator == "+": return a + b
    if operator == "-": return a - b
    if operator == "*": return a * b
    if operator == "/": return a / b
    raise ValueError(f"Unknown operator: {operator}")

@kbench.task()
def calc_task(llm):
    response = llm.prompt("What is 50 plus 25?", tools=[run_simple_calculator])
    kbench.assertions.assert_contains_regex(r"75", response)
```

**Multiple tools — LLM selects the right one:**
```python
def add_tool(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

def multiply_tool(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

@kbench.task()
def multi_tool_task(llm):
    response = llm.prompt(
        "What is 12 multiplied by 34?",
        tools=[add_tool, multiply_tool],
    )
    kbench.assertions.assert_contains_regex(r"408", response)
```

**Tool error handling — tools can raise exceptions:**
```python
def flaky_tool() -> str:
    """This tool always fails with an error."""
    raise ValueError("Tool execution failed.")

@kbench.task()
def error_handling_task(llm):
    response = llm.prompt("Call the flaky_tool and report what happens.", tools=[flaky_tool])
    kbench.assertions.assert_contains_regex(r"(?i)error|failed", response)
```

> **Note:** Automatic tool calling is currently only supported via the `genai` API.
> For `openai` API, tools must be called manually (see `use_calculator_tool.py`).
>
> **Note:** `kbench.assertions.assert_tool_was_invoked(fn)` appears in golden tests
> but may not be available in all versions of the library. Check before using.
