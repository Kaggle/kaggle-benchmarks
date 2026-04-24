# Changelog

## Next Release

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
