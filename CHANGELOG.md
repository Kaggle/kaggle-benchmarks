# Changelog

## v0.5.0 (May 1st, 2026)

* **Console UI output:** Added a new `ConsoleUI` for terminal/script environments with color, quiet mode, and structured run output. Auto-detects host environment (terminal vs notebook) and binds the appropriate UI handler. (#145, #149)
* **Reasoning traces:** Added `reasoning` and `include_thoughts` as explicit `prompt()` parameters for accessing LLM thinking/reasoning output. (#133)
* **PanelUI concurrency hardening:** Added defensive `in self` guards to all PanelUI event handlers (`new_chunk`, `end_content`, `end_run`, `new_run`, `new_tool_call`) to prevent `KeyError` crashes under `evaluate(n_jobs > 1)`. (#150, #152)
* **Thread-safe PanelUI state management:** Replaced shared `depth` counter with per-thread `threading.local()` so concurrent `evaluate(n_jobs > 1)` workers track nesting independently. Snapshot `EventManager.dispatch()` listener list to prevent `RuntimeError` during self-unbind. (#154)
* **Configurable task limits:** Made `task.name` and `task.description` max-length limits environment-variable driven (`KAGGLE_BENCHMARK_MAX_NAME_LENGTH`, `KAGGLE_BENCHMARK_MAX_DESCRIPTION_LENGTH`). (#134)
* **Nested evaluate safety:** Coerce nested `evaluate()` `max_attempts` to 1 with a warning to prevent exponential retry blowup. (#143)
* **Developer experience:** Added development and code review guides. (#141) Dropped noisy log when no `.env` file is found. (#144) Log effective `.env` path on load. (#140)

## v0.4.0 (Apr 24th, 2026)

* Added audio and video modality input support, broadening multimodal evaluation capabilities.
* Introduced a dedicated serialization module with Pydantic model support for messages.
* Refactored `task_autopilot` to use `LLMChat` and added support for `LLMMessages` to be returned from `invoke`.
* Added `orphan` argument to `chats.fork` for more flexible conversation branching.
* Added `flags` parameter to `assert_not_contains_regex` assertion.
* Fixed `seed` parameter exclusion for non-supporting models.
* Improved developer experience: added default `.env.example`, Kaggle ModelProxy authentication warnings, and dependabot configuration.
* Supply chain hardening with `uv exclude-newer`.
* Expanded test coverage: added more golden tests for tools and reduced test reliance on `LLMChat` implementation details.

## v0.3.0 (Mar 26th, 2026)

* Enhanced structured output and tool integration: Standardized response parsing and established a robust framework for LLM tool-use.
* Integrated usage metrics: Implemented cost and latency tracking, surfaced directly within Chat and Message objects.
* Established golden tests: Added an end-to-end testing suite with "golden" output verification across multiple environments.
* Expanded documentation: Updated guides and added a new example cookbook.
* General stability improvements: Resolved various bugs related to serialization, session handling, and environment configuration.

## v0.2.0 (Nov 19th, 2025)

* Initial public release.
* Core implementation.
* Basic documentation and usage examples.
* CI/CD pipeline for publishing to PyPI.
