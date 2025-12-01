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

import os
import tempfile
from pathlib import Path


class TemporalDirectoryMixin:
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)

    def _check_path(self, path: str | Path):
        if os.path.isabs(path):
            raise ValueError(f"Absolute paths are not supported: {path}")

    def __getitem__(self, path: str | Path) -> str:
        """Read the content of a file in the temporary directory."""
        with self.open(path, "r") as file:
            return file.read()

    def __setitem__(self, path: str | Path, content: str):
        """Read the content of a file in the temporary directory."""
        self._check_path(path)

        full_path = self.directory / path
        full_path.parent.mkdir(exist_ok=True)

        with self.open(path, "w") as file:
            file.write(content)

    def open(self, path: str | Path, mode: str = "r"):
        self._check_path(path)

        full_path = self.directory / path
        return open(full_path, mode=mode)
