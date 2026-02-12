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
import shlex
import subprocess

from kaggle_benchmarks.envs import environment, mixins


class LocalEnvironment(mixins.TemporalDirectoryMixin):
    def __init__(self):
        super().__init__()
        self.original_dir = os.getcwd()

    def run(
        self, command: str | list[str], input: str | None = None
    ) -> environment.RunResult:
        """Runs a shell command in the temporary directory."""
        if isinstance(command, str):
            command = shlex.split(command)

        result = subprocess.run(
            command,
            shell=False,
            input=input,
            cwd=self.temp_dir.name,
            capture_output=True,
            text=True,
        )
        return environment.RunResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def close(self):
        os.chdir(self.original_dir)
        self.temp_dir.cleanup()

    def __enter__(self):
        os.chdir(self.directory)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
