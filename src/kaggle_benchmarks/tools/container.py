import io
import logging
import posixpath
import tarfile
import time

import docker

from kaggle_benchmarks import actors

logger = logging.getLogger(__name__)


class DockerContainer(actors.Actor):
    def __init__(self, image: str):
        super().__init__(name="DockerContainer", role="tool", avatar="🐳")
        self.image = image
        self.client = docker.from_env()
        self.container = None

    def __enter__(self):
        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            logger.info(f"Image {self.image} not found. Pulling...")
            self.client.images.pull(self.image)

        self.container = self.client.containers.run(
            self.image, command="tail -f /dev/null", detach=True, tty=True
        )

        start_time = time.time()
        while self.container.status != "running":
            self.container.reload()
            if time.time() - start_time > 10:
                try:
                    self.container.stop()
                    self.container.remove()
                except Exception:
                    pass
                raise TimeoutError("Container failed to start within 10 seconds")
            time.sleep(0.1)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.container:
            try:
                self.container.stop(timeout=1)
                self.container.remove()
            except docker.errors.APIError:
                pass

    def run_command(self, command: str, workdir: str | None = None) -> str:
        """Runs a command and returns the decoded string output."""
        if not self.container:
            raise RuntimeError("Container not started")

        exit_code, output = self.container.exec_run(command, workdir=workdir)

        result = output.decode("utf-8", errors="replace").strip()

        in_dir = f" in `{workdir}`" if workdir else ""
        self.send(f"Run `{command}`{in_dir}", is_visible_to_llm=False)

        if exit_code != 0:
            return f"Error (Exit Code {exit_code}):\n{result}"
        return result

    def write_text_file(self, path: str, content: str):
        """Writes string content to a file inside the container."""
        if not self.container:
            raise RuntimeError("Container not started")

        dir_name = posixpath.dirname(path)
        file_name = posixpath.basename(path)

        if not dir_name:
            dir_name = "/"

        self.container.exec_run(f'mkdir -p "{dir_name}"')

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = content.encode("utf-8")
            tarinfo = tarfile.TarInfo(name=file_name)
            tarinfo.size = len(data)
            tarinfo.mtime = time.time()
            tar.addfile(tarinfo, io.BytesIO(data))

        tar_stream.seek(0)

        self.container.put_archive(dir_name, tar_stream)

        self.send(
            f"Write `{path}` with content length {len(str(content))}",
            is_visible_to_llm=False,
        )

    def read_text_file(self, path: str) -> str:
        """Reads a file from the container and returns it as a string."""
        if not self.container:
            raise RuntimeError("Container not started")

        self.send(f"Read `{path}`", is_visible_to_llm=False)

        try:
            stream, _ = self.container.get_archive(path)

            file_obj = io.BytesIO()
            for chunk in stream:
                file_obj.write(chunk)
            file_obj.seek(0)

            with tarfile.open(fileobj=file_obj, mode="r") as tar:
                members = tar.getmembers()
                if not members:
                    return f"Error: File {path} is empty or invalid tar."

                member = members[0]

                f = tar.extractfile(member)
                if f is None:
                    return f"Error: Could not extract file {path}."

                return f.read().decode("utf-8", errors="replace")

        except docker.errors.NotFound:
            return f"Error: File {path} not found."
        except Exception as e:
            return f"Error reading file {path}: {str(e)}"
