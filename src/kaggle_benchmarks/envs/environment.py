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

"""
Defines the core `Environment` protocol and related classes for interacting with
different execution environments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class RunResult:
    """Represents the result of running a command."""

    exit_code: int
    stdout: str
    stderr: str

    def __str__(self):
        return f"RunResult [{self.exit_code}]\nStdout: {self.stdout}\nStderr: {self.stderr}"


class Environment(Protocol):
    """Protocol for interacting with an execution environment."""

    def run(self, command: str | list[str], input: str | None = None) -> RunResult: ...
    def __getitem__(self, path: str | Path) -> str: ...
    def __setitem__(self, path: str | Path, content: str): ...
