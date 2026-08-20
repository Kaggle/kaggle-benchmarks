# AGENTS.md - AI Assistant Guide for kaggle-benchmarks

## Project Context

`kaggle-benchmarks` is a Python library for rigorously evaluating LLMs on custom tasks using decorators, assertions, and tool-augmented interactions. **Tech Stack:** Python 3.11+, uv (package manager), pytest, ruff, mypy, Protocol Buffers.

## High-Level Architecture

This is a **library-first codebase** organized by functional concerns:

- **`src/kaggle_benchmarks/`** - Core library implementing the benchmark framework
  - Top-level modules define primitives: tasks, assertions, clients, messages, results, runs
  - Subdirectories provide specialized subsystems: `actors/` (LLM interaction), `tools/` (Python interpreter, web search), `envs/` (execution environments), `kaggle/` (platform integration), `ui/` (Panel-based interfaces)
- **`tests/`** - Pytest test suite mirroring `src/` structure
- **`research_benchmarks/`** - Reference implementations of academic benchmarks (MathVista, SimpleQA)
- **`documentation/`** - Quarto-based docs with executable examples in `examples/`
- **`protos/`** - Protocol Buffer schemas for serialization
- **`cicd/`** - Docker and CI/CD scripts

**Mental Model:** Users write decorated functions (`@kbench.task`) that prompt LLMs and assert outputs. The library handles orchestration, caching, serialization, and UI rendering.

## Sources of Truth (The Map)

### Configuration & Environment
- **Environment variables** → Read `.env` file format in `README.md`
- **Execution modes** → `src/kaggle_benchmarks/_config.py` defines `ExecutionMode` enum and `Config` dataclass
- **Package metadata** → `pyproject.toml` (dependencies, version, tool configs)

### Core API Surface
- **Public exports** → `src/kaggle_benchmarks/__init__.py` defines what users import
- **Task/benchmark decorators** → `src/kaggle_benchmarks/tasks.py` (`@task`, `@benchmark`)
- **Assertions** → `src/kaggle_benchmarks/assertions.py` (all `assert_*` functions)
- **LLM clients** → `src/kaggle_benchmarks/clients.py` (client abstraction and resolution)

### Subsystems
- **Core objects** → `src/kaggle_benchmarks/core.py` (base classes `Actor`, `Event`, `BaseMessage`, `Session` + the `Status` enum; concrete types are `Message` in `messages.py`, `Chat`/`GoldfishChat` in `chats.py`, and the actors in `actors/base.py`. These are plain classes, not dataclasses.)
- **Actor system** → `src/kaggle_benchmarks/actors/` (`LLMChat` in `llms.py`; concrete `Tool`/`system`/`user` actors in `base.py`; `Actor` base class in `core.py`)
- **Tools** → `src/kaggle_benchmarks/tools/` (Python interpreter, web search)
- **Execution environments** → `src/kaggle_benchmarks/envs/` (local, docker)
- **Kaggle integration** → `src/kaggle_benchmarks/kaggle/` (model loading, serialization)
- **UI components** → `src/kaggle_benchmarks/ui/` (Panel-based rendering)

### Testing & Quality
- **Test fixtures** → `tests/conftest.py`
- **Pre-commit hooks** → `.pre-commit-config.yaml` (ruff, addlicense)
- **Type checking config** → `pyproject.toml` `[tool.mypy]` section

### Documentation
- **User-facing guides** → `documentation/quick_start.qmd`, `documentation/user_guide.qmd`
- **Example code** → `documentation/examples/*.py`

## Critical Implementation Rules

1. **Never manually edit generated files** - Files matching `**/*_pb2.py` are auto-generated from `protos/`. Run `cd protos && ./build.sh` to regenerate.

2. **Use `uv` for all dependency operations** - Not `pip`. Commands: `uv pip install`, `uv run --group <group> <command>`. Dependency groups defined in `pyproject.toml` lines 29-72.

## Operational Commands

### Setup
```bash
# Create virtual environment and install
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .

# Install dev dependencies
uv pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
uv run --group test pytest tests

# Run specific test file
uv run --group test pytest tests/test_assertions.py

# Run with verbose output
uv run --group test pytest tests -v
```

### Code Quality
```bash
# Format code
ruff format .

# Lint and auto-fix
ruff check --fix .

# Type check
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Protocol Buffers
```bash
# Rebuild protobuf definitions (required after editing protos/)
cd protos && ./build.sh
```

### Docker
```bash
# Build, run, or start Jupyter
cd cicd
./build.sh   # Build image
./run.sh     # Run container
./jupyter.sh # Start Jupyter server
```

## Agent Skills

- **Writing Benchmarks**: See `skills/kaggle-benchmarks/SKILL.md` for detailed patterns and examples on writing benchmark tasks using the `kaggle_benchmarks` library.

