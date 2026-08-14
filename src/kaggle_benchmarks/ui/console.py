# Copyright 2025 Kaggle Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import sys
import textwrap
import threading

from kaggle_benchmarks import assertions, chats, core

_DEFAULT_MIN_WIDTH = 40
_DEFAULT_MAX_WIDTH = 120
_FALLBACK_WIDTH = 80


class _Colors:
    GREEN = "\033[32m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


class _NoColors:
    GREEN = ""
    RED = ""
    CYAN = ""
    YELLOW = ""
    BOLD = ""
    DIM = ""
    RESET = ""


def _supports_color(stream) -> bool:
    """Auto-detect whether the given stream supports ANSI colors.

    Disables color when NO_COLOR is set (https://no-color.org), forces it on
    when FORCE_COLOR is set, otherwise enables only for TTYs.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


class ConsoleUI:
    def __init__(
        self,
        tab_size: int = 2,
        quiet: bool = False,
        color: bool | None = None,
        width: int | None = None,
        min_width: int = _DEFAULT_MIN_WIDTH,
        max_width: int = _DEFAULT_MAX_WIDTH,
        output=None,
    ):
        self.depth = 0
        self.tab_size = tab_size
        self.quiet = quiet
        self._output = output or sys.stdout
        # Auto-detect color from TTY if not explicitly set
        self.color = _supports_color(self._output) if color is None else color
        # If width is None, auto-detect from terminal size on each access,
        # clamped to [min_width, max_width]. Otherwise use the fixed value.
        self._fixed_width = width
        self.min_width = min_width
        self.max_width = max_width
        self._lock = threading.Lock()
        self._c = _Colors() if self.color else _NoColors()
        self._in_run = False
        self._run_depth = 0
        # Tracks message ids whose body was already rendered via new_chunk
        # events; new_event skips the eager body print for these so the
        # text isn't duplicated. Stores id(message) to avoid pinning Message
        # objects in memory.
        self._streamed_messages: set[int] = set()

    @property
    def width(self) -> int:
        if self._fixed_width is not None:
            return self._fixed_width
        try:
            cols = shutil.get_terminal_size(fallback=(_FALLBACK_WIDTH, 24)).columns
        except (OSError, ValueError):
            cols = _FALLBACK_WIDTH
        return max(self.min_width, min(self.max_width, cols))

    # -- Internal helpers --

    def _print(self, *args, **kwargs):
        kwargs.setdefault("file", self._output)
        indent = " " * (self._run_depth * self.tab_size) if self._run_depth > 0 else ""
        with self._lock:
            if indent and args:
                # Indent every line of multi-line content
                text = str(args[0])
                indented = "\n".join(
                    indent + line if line else line for line in text.splitlines()
                )
                print(indented, *args[1:], **kwargs)
            else:
                print(*args, **kwargs)

    def _header_bar(self, char="="):
        return char * self.width

    def _colorize(self, text, color_code):
        if not self.color:
            return text
        return f"{color_code}{text}{self._c.RESET}"

    def _format_usage(self, usage):
        parts = []
        if usage.input_tokens is not None:
            parts.append(f"Input: {usage.input_tokens}")
        if usage.output_tokens is not None:
            parts.append(f"Output: {usage.output_tokens}")
        if usage.total_cost_nanodollars is not None:
            cost_dollars = usage.total_cost_nanodollars / 1e9
            parts.append(f"Cost: ${cost_dollars:.6f}")
        if usage.total_backend_latency_ms is not None:
            parts.append(f"Latency: {usage.total_backend_latency_ms}ms")
        if not parts:
            return ""
        return " · ".join(parts)

    def _format_assertion_table(self, assertion_results):
        if not assertion_results:
            return ""

        c = self._c
        line_width = 6
        result_width = 8  # "✅ PASS" / "❌ FAIL"
        effective_width = self.width - self._run_depth * self.tab_size
        expect_width = effective_width - line_width - result_width - 4  # 2x2 gaps
        # Floor at 10 so textwrap.wrap doesn't blow up (raises ValueError on
        # width<=0) when the terminal is narrow or run depth is deep.
        expect_width = max(expect_width, 10)
        sep_line = self._colorize("-" * effective_width, c.DIM)

        lines = []
        lines.append("")
        lines.append(sep_line)
        lines.append(f"{'Line':>{line_width}}  {'Expectation':<{expect_width}}  Result")
        lines.append(sep_line)

        for result in assertion_results:
            expectation = result.expectation or ""
            line_no = ""
            if result.details:
                ln = result.details.get("line_number")
                if ln is not None:
                    line_no = str(ln)

            assert_type = ""
            if result.details:
                assert_type = result.details.get("assertion_type", "")
            if assert_type:
                expectation = (
                    f"{assert_type}: {expectation}" if expectation else assert_type
                )

            if result.passed:
                status = self._colorize("✅ Pass", c.GREEN)
            else:
                status = self._colorize("❌ Fail", c.RED)

            wrapped = textwrap.wrap(expectation, width=expect_width) or [""]

            line_no_cell = self._colorize(f"{line_no:>{line_width}}", c.BOLD)
            lines.append(f"{line_no_cell}  {wrapped[0]:<{expect_width}}  {status}")
            for continuation in wrapped[1:]:
                lines.append(f"{'':>{line_width}}  {continuation:<{expect_width}}")

        lines.append(sep_line)
        return "\n".join(lines)

    # -- Quiet mode handlers (original behavior) --

    def _quiet_new_chat(self, chat):
        print(
            chat.__str__(indent=" " * (self.depth * self.tab_size)),
            end="",
            file=self._output,
        )
        self.depth += 1

    def _quiet_end_chat(self, chat):
        self.depth -= 1

    def _quiet_new_event(self, chat, message):
        if isinstance(message, chats.Chat):
            return
        # If this message was streamed, chunks already rendered the body.
        if id(message) in self._streamed_messages:
            return
        print(
            message.__str__(indent=" " * (self.depth * self.tab_size)),
            end="" if message.status == core.Status.RUNNING else "\n",
            file=self._output,
        )

    def _quiet_new_chunk(self, message, token):
        print(token, end="", file=self._output)

    def _quiet_end_content(self, message):
        print(file=self._output)

    # -- Event handlers --

    def new_run(self, run):
        if self.quiet:
            return
        c = self._c
        self._in_run = True
        if self._run_depth == 0:
            bar = self._colorize(self._header_bar("="), c.BOLD)
            title = self._colorize(f"TASK: {run.task.name}: {run.id}", c.BOLD)
            self._print(bar)
            self._print(title)
            self._print(bar)
        else:
            bar = self._header_bar("-")
            self._print(bar)
            self._print(f"SUBTASK: {run.task.name}: {run.id}")
            self._print(bar)
        self._run_depth += 1

    def end_run(self, run):
        if self.quiet:
            return
        c = self._c
        self._run_depth -= 1

        # Assertion table
        if run.assertion_results:
            table = self._format_assertion_table(run.assertion_results)
            self._print(table)

        # Metrics + Result, grouped together
        if run.chat and run.chat.usage:
            usage_str = self._format_usage(run.chat.usage)
            if usage_str:
                self._print(f"\n{self._colorize('METRICS:', c.BOLD)}  {usage_str}")

        # Result or error
        if run.status == core.Status.FAILED:
            error_msg = run.error_message or "Unknown Error"
            self._print(self._colorize(f"ERROR:    {error_msg}", c.RED))
        else:
            result_str = run.format_result().strip()
            self._print(f"{self._colorize('RESULT:', c.BOLD)}   {result_str}")

        # Closing bar
        if self._run_depth == 0:
            self._print(self._colorize(self._header_bar("="), c.BOLD))
        else:
            self._print(self._header_bar("-"))
        self._print("")

        if self._run_depth == 0:
            self._in_run = False

    def new_chat(self, chat):
        if self.quiet:
            self._quiet_new_chat(chat)
            return
        if not self._in_run:
            self._quiet_new_chat(chat)
            return
        # Inside a run: only print nested subchats
        self.depth += 1
        if self.depth > 2:
            self._print(self._colorize(f"\n[CHAT: {chat.name}]", self._c.DIM))

    def end_chat(self, chat):
        if self.quiet:
            self._quiet_end_chat(chat)
            return
        self.depth -= 1

    def _format_content(self, content):
        """Format message content as readable text, handling non-string types."""
        if isinstance(content, str):
            text = content
        elif hasattr(content, "model_dump"):
            text = json.dumps(content.model_dump(), default=str)
        else:
            text = str(content)
        # Wrap long lines to fit terminal width (account for run-depth indent)
        max_width = self.width - self._run_depth * self.tab_size
        wrapped_lines = []
        for line in text.splitlines():
            if len(line) > max_width:
                wrapped_lines.extend(textwrap.wrap(line, width=max_width) or [""])
            else:
                wrapped_lines.append(line)
        return "\n".join(wrapped_lines)

    def new_event(self, chat, message):
        if self.quiet:
            self._quiet_new_event(chat, message)
            return
        if isinstance(message, chats.Chat):
            return
        if not self._in_run:
            self._quiet_new_event(chat, message)
            return

        # Skip assertion result messages -- they're shown in the assertion
        # table. Everything else falls through to the role-based rendering
        # below; in particular, messages with is_visible_to_llm=False (e.g.
        # tool/debug output) are still useful to surface in the console.
        if isinstance(message.content, assertions.AssertionResult):
            return

        c = self._c
        role = message.sender.role
        name = message.sender.name
        text = self._format_content(message.content)
        # Print body unless chunks already streamed it. Header still prints
        # so the [ROLE] label appears above the streamed tokens regardless
        # of whether append happened before or after stream().
        print_body = bool(text) and id(message) not in self._streamed_messages

        if role == "user":
            self._print(self._colorize("\n[PROMPT]", c.CYAN))
            if print_body:
                self._print(text)
        elif role == "assistant":
            self._print(self._colorize(f"\n[RESPONSE: {name}]", c.CYAN))
            if print_body:
                self._print(text)
        elif role == "system":
            self._print(self._colorize("\n[SYSTEM]", c.DIM))
            if print_body:
                self._print(text)
        elif role == "tool":
            self._print(self._colorize(f"\n[TOOL: {name}]", c.DIM))
            if print_body:
                self._print(text)
        else:
            self._print(f"\n[{name}]")
            if print_body:
                self._print(text)

    def new_chunk(self, message, chunk):
        self._streamed_messages.add(id(message))
        if self.quiet:
            self._quiet_new_chunk(message, chunk)
            return
        chunk_text = chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
        with self._lock:
            print(chunk_text, end="", flush=True, file=self._output)

    def end_content(self, message):
        if self.quiet:
            self._quiet_end_content(message)
            return
        with self._lock:
            print(file=self._output)
        # Per-message metrics intentionally omitted: end_run prints the
        # aggregate METRICS line from chat.usage. Printing here would
        # double up for assistant-role messages.
