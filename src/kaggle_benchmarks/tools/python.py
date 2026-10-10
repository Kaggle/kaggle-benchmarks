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

import queue
import re
import time
from dataclasses import dataclass
from typing import Literal

import jupyter_client

from kaggle_benchmarks import actors, chats, envs, utils


def run(code: str, env: envs.Environment, input: str | None = None) -> envs.RunResult:
    return env.run(["python", "-c", code], input=input)


def extract_code(text: str, all_blocks: bool = False) -> str:
    """Extract Python code block(s) from markdown text.

    Args:
        text: Text potentially containing markdown Python code blocks.
        all_blocks: If True, extract and concatenate all ```python blocks.
                   If False (default), extract only the first block.

    Returns:
        Extracted Python code, or the original text if no code blocks found.
    """
    if "```python" in text:
        return utils.extract_code_block(
            text, name="python", greedy=False, all_blocks=all_blocks
        ).strip()
    return text


def markdown_code(string: str | None, kind: str = "", header: str = "") -> str:
    if string:
        return f"{header}\n```{kind}\n{string}\n```"
    return ""


ansi_styles = {
    "30": "color: black;",
    "31": "color: red;",
    "32": "color: green;",
    "33": "color: yellow;",
    "34": "color: blue;",
    "35": "color: magenta;",
    "36": "color: cyan;",
    "37": "color: white;",
    "90": "color: gray;",
    "40": "background-color: black;",
    "41": "background-color: red;",
    "42": "background-color: green;",
    "43": "background-color: yellow;",
    "44": "background-color: blue;",
    "45": "background-color: magenta;",
    "46": "background-color: cyan;",
    "47": "background-color: white;",
}


def replace_ansi_colors(text: str) -> str:
    ansi_regex = re.compile(r"\x1b\[([0-9;]+)m")

    def replace_ansi(match):
        codes = match.group(1).split(";")
        styles = " ".join(ansi_styles[code] for code in codes if code in ansi_styles)
        return f'<span style="{styles}">'

    result = ansi_regex.sub(replace_ansi, text)
    result = result.replace("\x1b[0m", "</span>")

    return result


@dataclass
class ScriptOutput(envs.RunResult):
    code: str | None = None

    def _repr_markdown_(self):
        return f"""
<details>
<summary>Code:</summary>

{markdown_code(self.code, kind="python")}
</details>

{markdown_code(self.stdout, header="Stdout:")}
{markdown_code(self.stderr, header="Stderr:")}
"""


class ScriptRunner(actors.Actor):
    def __init__(self):
        super().__init__(name="Python", role="tool", avatar="🐍")

    @chats.emits_message
    def run_code(self, code: str, input: str | None = None) -> ScriptOutput:
        code = extract_code(code)
        out = run(code, envs.current, input=input)
        return ScriptOutput(
            code=code,
            stdout=out.stdout,
            stderr=out.stderr,
            exit_code=out.exit_code,
        )

    def respond(self) -> ScriptOutput:
        chat = chats.get_current_chat()
        return self.run_code(chat.messages[-1].text)


script_runner = ScriptRunner()


@dataclass
class CellOutput:
    status: Literal["ok", "error"]
    output: str | None = None
    stdout: str | None = None
    traceback: str | None = None

    def _repr_markdown_(self) -> str:
        return f"""{"✔️" if self.status == "ok" else "❌"}

{markdown_code(replace_ansi_colors(self.traceback or self.stdout or ""))}
{markdown_code(self.output, header="Out:")}
</div>"""


class IPythonREPL(actors.Actor):
    def __init__(self):
        super().__init__(avatar="🐍", role="tool")
        self.km = jupyter_client.KernelManager()  # type: ignore
        self.km.start_kernel()
        self.client = self.km.client()
        self.client.start_channels()
        self.client.wait_for_ready()

    @chats.emits_message
    def run_code(
        self, code: str, timeout: int = -1, input: str | None = None
    ) -> CellOutput:
        code = extract_code(code)
        msg_id = self.client.execute(code)

        if input:
            self.client.input(input)

        iopub_timeout = None if timeout <= 0 else float(timeout)
        msgs = list(self._collect_iopub_messages(msg_id, timeout=iopub_timeout))

        result = None
        for msg in msgs:
            if msg["msg_type"] == "execute_result":
                result = msg["content"]["data"]["text/plain"]
                break

        traceback = "\n".join(
            "\n".join(m["content"]["traceback"])
            for m in msgs
            if m["msg_type"] == "error"
        ).strip()

        # Clean up the shell channel for this execution to prevent message leakage
        while True:
            try:
                shell_msg = self.client.get_shell_msg(timeout=0.1)
                if shell_msg.get("parent_header", {}).get("msg_id") == msg_id:
                    break
            except queue.Empty:
                break

        status = "error" if traceback else "ok"

        return CellOutput(
            status=status,
            output=result,
            stdout="\n".join(
                m["content"]["text"] for m in msgs if m["msg_type"] == "stream"
            ),
            traceback=traceback or None,
        )

    def _collect_iopub_messages(self, msg_id: str, timeout: float | None = None):
        start_time = time.time()
        while True:
            poll_timeout = 1.0
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    break
                poll_timeout = min(poll_timeout, remaining)

            try:
                msg = self.client.get_iopub_msg(timeout=poll_timeout)
            except queue.Empty:
                if timeout is not None and (time.time() - start_time) >= timeout:
                    break
                if not self.km.is_alive():
                    raise RuntimeError("Kernel died during execution")
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            yield msg

            if (
                msg.get("msg_type") == "status"
                and msg.get("content", {}).get("execution_state") == "idle"
            ):
                break

    def respond(self, **kwargs) -> CellOutput:
        chat = chats.get_current_chat()
        message = chat.messages[-1]
        code = extract_code(message.text)
        return self.run_code(code, **kwargs)

    def invoke(self, message: str, **kwargs) -> CellOutput:
        is_visible_to_llm = kwargs.get("is_visible_to_llm", True)
        self.send(
            f"▶️{self.avatar}\n```python\n{message}\n```",
            is_visible_to_llm=is_visible_to_llm,
        )
        return self.respond(**kwargs)

    def stop(self):
        self.km.shutdown_kernel()
