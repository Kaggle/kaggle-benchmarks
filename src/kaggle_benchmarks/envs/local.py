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
import subprocess
import warnings

from kaggle_benchmarks.envs import environment, mixins


class InternalUnsafeLocalEnvironment(mixins.TemporalDirectoryMixin):
    """An **unsandboxed** local execution environment. Internal use only.

    .. warning::

        **This class provides NO sandboxing or isolation.** Commands execute with
        the full privileges of the host process. Specifically:

        - Path traversal (e.g. ``../``) can escape the temporary directory and
          read/write anywhere the current user has access.

        **Do not use this with untrusted or adversarial model outputs.** For
        isolated execution, use :class:`DockerEnvironment` instead.

        This class exists to support internal tooling (e.g. the Python script
        runner and Browser tool). External SDK users should not instantiate it
        directly.
    """

    def __init__(self, *, _internal: bool = False):
        super().__init__()
        self.original_dir = os.getcwd()
        if not _internal:
            warnings.warn(
                "InternalUnsafeLocalEnvironment provides NO sandboxing or "
                "isolation — commands run with full host privileges. "
                "This class is intended for internal use only. "
                "For isolated execution with untrusted models, use "
                "DockerEnvironment instead.",
                UserWarning,
                stacklevel=2,
            )

    def run(
        self, command: list[str], input: str | None = None
    ) -> environment.RunResult:
        """Runs a command in the temporary directory.

        ``command`` must be a list of argument strings (``argv`` form). String
        commands are rejected to avoid a shell-injection foot-gun when callers
        interpolate untrusted input. To intentionally invoke a shell, pass
        ``["bash", "-c", "<script>"]`` explicitly.

        .. warning::
            No sandboxing is applied. The command has full access to the host
            filesystem and network.
        """
        if isinstance(command, str):
            raise TypeError(
                "command must be a list[str], not str. "
                "Pass arguments as a list (e.g. ['echo', 'hi']). "
                "To run a shell pipeline, use ['bash', '-c', '<script>']."
            )

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
