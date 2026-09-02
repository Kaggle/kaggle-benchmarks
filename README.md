# Kaggle Benchmarks

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Kaggle/kaggle-benchmarks)

`kaggle-benchmarks` is a Python library designed to help you rigorously evaluate AI models on tasks that matter to you. It provides a structured framework for defining tasks, interacting with models, and asserting the correctness of their outputs. The fastest way for you to write a task and run it on Kaggle is to point your AI coding agent to this [write-kaggle-benchmarks skill](https://github.com/Kaggle/kaggle-skills/blob/main/write-kaggle-benchmarks/SKILL.md).

This is especially useful for:

- **Reproducibility:** Capture the exact inputs, outputs, and model interactions for later review.
- **Complex Evaluations:** Go beyond simple string matching to test for code execution, tool use, and multi-turn conversational capabilities.
- **Rapid Prototyping:** Quickly test a model’s capabilities on a new, creative task you’ve designed.


- [Features](#features)
- [Getting Started](#getting-started)
  - [On Kaggle](#on-kaggle)
  - [Local Development](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/local_development.md)
- [Usage](#usage)
- [Supported Models](#supported-models)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Define Custom Tasks**: Easily define evaluation tasks using a simple `@kbench.task` decorator.
- **Interact with Multiple LLMs**: Programmatically interact with and compare various large language models.
- **Structured & Multimodal I/O**: Go beyond plain text. Get structured `dataclass` or `pydantic` objects from models and provide image, audio, and video inputs.
- **Tool Use**: Empower models with tools, including a built-in Python interpreter to execute code.
- **Robust Assertions**: Use a rich set of built-in assertions or create your own to validate model outputs.
- **Dataset Evaluation**: Run benchmarks over entire datasets (e.g., pandas DataFrames) to get aggregate performance metrics.

## Getting Started

### On Kaggle

The easiest way to use `kaggle-benchmarks` is directly within a Kaggle notebook.

**Prerequisites**: A Kaggle account.

**Installation**: No installation is needed! For early access, simply navigate to **https://www.kaggle.com/benchmarks/tasks/new**. This will create a new Kaggle notebook with the library and its dependencies pre-installed and ready to use.

**Data Usage and Leaderboard Generation**: When running in Kaggle notebook, each benchmark task outputs a `task file` and associated `run files`. These files are used to build the benchmark entity and display its results on a Kaggle leaderboard. An example can be seen on the [ICML 2025 Experts Leaderboard](https://www.kaggle.com/benchmarks/kaggle/icml-2025-experts).


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
    llm=kbench.llm,  # Uses the default LLM
    riddle="What gets wetter as it dries?",
    answer="Towel",
)
```

For a detailed walkthrough of the library's features, check out our documentation:
- [Quick Start Guide](quick_start.md)
- [User Guide](user_guide.md)
- [Cookbook](cookbook.md)

## Supported Models

This library supports a wide range of models available through Kaggle's backend. The exact models available to you depend on your environment (Kaggle Notebook vs. local proxy token).

## Contributing

Contributions are welcome! Please refer to our [Contribution Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
