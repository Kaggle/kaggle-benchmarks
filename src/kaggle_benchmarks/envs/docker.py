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

import shlex

import docker
import docker.errors
import docker.models
import docker.models.containers
import docker.types

from kaggle_benchmarks.envs import environment, mixins


def available():
    try:
        docker.from_env()
        return True
    except docker.errors.DockerException:
        return False


class DockerEnvironment(mixins.TemporalDirectoryMixin):
    def __init__(
        self,
        image,
        mounts: dict[str, str] | None = None,
        working_dir: str = "/tmp",
        client: docker.DockerClient | None = None,
        **kwargs,
    ):
        super().__init__()
        self.image = image
        self.working_dir = working_dir

        self.mounts = [
            docker.types.Mount(target, source)
            for target, source in (mounts or {}).items()
        ]

        # mount working dir to local temporary folder for easier filesystem operations
        self.mounts.append(
            docker.types.Mount(
                target=str(self.directory), source=str(self.working_dir), type="bind"
            )
        )

        if client is None:
            client = docker.from_env()
        self.client = client

        self.container: docker.models.containers.Container | None = None
        self.params = kwargs

    def __enter__(self):
        if not self.image.startswith("sha256:"):
            self.client.images.pull(self.image)

        self.start_container()
        return self

    def start_container(self):
        self.container = self.client.containers.create(
            self.image,
            tty=True,
            mounts=self.mounts,
            detach=False,
            working_dir=self.working_dir,
            **self.params,
        )
        self.container.start()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.container:
            self.container.stop()
            self.container.remove()

    def run(
        self, command: str | list[str], input: str | None = None
    ) -> environment.RunResult:
        """Runs a command in the container."""
        if self.container is None:
            raise RuntimeError("Container was not started")

        if input is not None:
            raise NotImplementedError("input is not supported")

        if not isinstance(command, str):
            command = shlex.join(command)

        result = self.container.exec_run(
            ["sh", "-c", command], demux=True, workdir=self.working_dir
        )
        return environment.RunResult(
            result.exit_code,
            stdout=(result.output[0] or b"").decode(),
            stderr=(result.output[1] or b"").decode(),
        )
