# Kaggle Benchmarks

`kaggle-benchmarks` is a Python library designed to help you rigorously evaluate AI models on tasks that matter to you. It provides a structured framework for defining tasks, interacting with models, and asserting the correctness of their outputs.

This is especially useful for:

- **Reproducibility:** Capture the exact inputs, outputs, and model interactions for later review.
- **Complex Evaluations:** Go beyond simple string matching to test for code execution, tool use, and multi-turn conversational capabilities.
- **Rapid Prototyping:** Quickly test a model’s capabilities on a new, creative task you’ve designed.


- [Features](#features)
- [Getting Started](#getting-started)
  - [On Kaggle](#on-kaggle)
  - [Local Development](#local-development)
- [Usage](#usage)
- [Supported Models](#supported-models)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Define Custom Tasks**: Easily define evaluation tasks using a simple `@kbench.task` decorator.
- **Interact with Multiple LLMs**: Programmatically interact with and compare various large language models.
- **Structured & Multimodal I/O**: Go beyond plain text. Get structured `dataclass` or `pydantic` objects from models and provide image inputs.
- **Tool Use**: Empower models with tools, including a built-in Python interpreter to execute code.
- **Robust Assertions**: Use a rich set of built-in assertions or create your own to validate model outputs.
- **Dataset Evaluation**: Run benchmarks over entire datasets (e.g., pandas DataFrames) to get aggregate performance metrics.

## Getting Started

### On Kaggle

The easiest way to use `kaggle-benchmarks` is directly within a Kaggle notebook.

**Prerequisites**: A Kaggle account.

**Installation**: No installation is needed! For early access, simply navigate to **https://www.kaggle.com/benchmarks/tasks/new**. This will create a new Kaggle notebook with the library and its dependencies pre-installed and ready to use.

**Data Usage and Leaderboard Generation**: When running in Kaggle notebook, each benchmark task outputs a `task file` and associated `run files`. These files are used to build the benchmark entity and display its results on a Kaggle leaderboard. An example can be seen on the [ICML 2025 Experts Leaderboard](https://www.kaggle.com/benchmarks/kaggle/icml-2025-experts).

### Local Development

For local development, you will need to configure your environment to use the Kaggle Model Proxy.

**Prerequisites**:
- Python 3.11+
- Git
- [`uv`](https://docs.astral.sh/uv)

**Installation & Configuration**:
1.  Clone the repository:
    ```bash
    git clone https://github.com/Kaggle/kaggle-benchmarks.git
    cd kaggle-benchmarks
    ```
2.  Create a virtual environment and install dependencies using `uv`:
    ```bash
    # Create and activate the virtual environment
    uv venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

    # Install dependencies
    uv pip install -e .
    ```
3.  Obtain a Kaggle `MODEL_PROXY_API_KEY` for access.
4.  Create a `.env` file in the project root and add your configuration:
    ```env
    MODEL_PROXY_URL=https://mp-staging.kaggle.net/models/openapi
    MODEL_PROXY_API_KEY={your_token}
    LLM_DEFAULT=google/gemini-2.5-flash
    LLM_DEFAULT_EVAL=openai/gpt-4o
    LLMS_AVAILABLE=anthropic/claude-sonnet-4,google/gemini-2.5-flash,meta/llama-3.1-70b,openai/gpt-4o
    PYTHONPATH=src
    ```
    - `LLM_DEFAULT`: Sets the model identifier for `kbench.llm`, the default model used for running tasks.
    - `LLM_DEFAULT_EVAL`: Sets the model identifier for `kbench.judge_llm`, which is typically a model used for evaluation or judging the outputs of other models.
    - `LLMS_AVAILABLE`: A comma-separated list of models authorized for use by your proxy token.

    **Note**: The `LLM_DEFAULT`, `LLM_DEFAULT_EVAL`, and `LLMS_AVAILABLE` variables depend on the models authorized by your proxy token.

## Usage

Here is a simple example of a benchmark that asks a model a riddle and checks its answer.

```python
import kaggle_benchmarks as kbench

@kbench.task(name="simple_riddle")
def solve_riddle(llm, riddle: str, answer: str):
    """Asks a riddle and checks for a keyword in the answer."""
    response = llm.prompt(riddle)

    # Assert that the model's response contains the answer, ignoring case.
    kbench.assertions.assert_contains_regex(
        f"(?i){answer}", response, expectation="LLM should give the right answer."
    )

# Execute the task
solve_riddle.run(
    llm=kbench.llm, # Uses the default LLM
    riddle="What gets wetter as it dries?",
    answer="Towel",
)
```

For a detailed walkthrough of the library's features, check out our documentation:
- [Quick Start Guide](quick_start.md)
- [User Guide](user_guide.md)

## Supported Models

This library supports a wide range of models available through Kaggle's backend. The exact models available to you depend on your environment (Kaggle Notebook vs. local proxy token).

## Contributing

Contributions are welcome! Please refer to our [Contribution Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
