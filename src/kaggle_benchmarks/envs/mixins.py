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

    def _resolve_checked_path(self, path: str | Path) -> Path:
        """Resolve ``path`` relative to :attr:`directory` and verify it stays inside.

        Rejects absolute paths and any relative path that escapes the
        temporary directory via ``..`` traversal (see
        Kaggle/kaggle-benchmarks#159).
        """
        if os.path.isabs(path):
            raise ValueError(f"Absolute paths are not supported: {path}")

        base = self.directory.resolve()
        candidate = (base / path).resolve()

        if candidate != base and base not in candidate.parents:
            raise ValueError(f"Path escapes temporary directory: {path}")

        return candidate

    def __getitem__(self, path: str | Path) -> str:
        """Read the content of a file in the temporary directory."""
        with self.open(path, "r") as file:
            return file.read()

    def __setitem__(self, path: str | Path, content: str):
        """Write ``content`` to a file in the temporary directory."""
        full_path = self._resolve_checked_path(path)
        full_path.parent.mkdir(exist_ok=True)

        with open(full_path, "w") as file:
            file.write(content)

    def open(self, path: str | Path, mode: str = "r"):
        full_path = self._resolve_checked_path(path)
        return open(full_path, mode=mode)
