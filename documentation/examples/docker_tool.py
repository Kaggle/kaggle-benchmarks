# %%
import io
import os
import tarfile
import textwrap
import time
from dataclasses import dataclass
from typing import Optional

import docker

import kaggle_benchmarks as kbench
from kaggle_benchmarks import actors, chats


class DockerContainer(actors.Actor):
    def __init__(self, image: str):
        super().__init__(name="DockerContainer", role="tool", avatar="🐳")
        self.image = image
        self.client = docker.from_env()
        self.container = None

    def __enter__(self):
        # Check if image exists locally to avoid unnecessary pulls
        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            print(f"Image {self.image} not found. Pulling...")
            self.client.images.pull(self.image)

        # Use a keep-alive command so the container doesn't exit immediately
        # detach=True starts it in the background
        # auto_remove=True cleans up if it crashes, but we handle cleanup in __exit__
        self.container = self.client.containers.run(
            self.image, command="tail -f /dev/null", detach=True
        )

        # Wait for 'running' status with a timeout to prevent infinite loops
        start_time = time.time()
        while self.container.status != "running":
            self.container.reload()
            if time.time() - start_time > 10:
                raise TimeoutError("Container failed to start within 10 seconds")
            time.sleep(0.1)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
            except docker.errors.NotFound:
                pass  # Already removed

    @chats.emits_message
    def run_command(self, command: str, workdir: str = "/") -> str:
        """Runs a command and returns the decoded string output."""
        if not self.container:
            raise RuntimeError("Container not started")

        exit_code, output = self.container.exec_run(command, workdir=workdir)

        # Decode bytes to string for easier handling
        result = output.decode("utf-8").strip()

        if exit_code != 0:
            return f"Error (Exit Code {exit_code}):\n{result}"
        return result

    def write_file(self, path: str, content: str):
        """Writes string content to a file inside the container."""
        if not self.container:
            raise RuntimeError("Container not started")

        # Docker put_archive expects a tar stream
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = content.encode("utf-8")
            tarinfo = tarfile.TarInfo(name=os.path.basename(path))
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))

        tar_stream.seek(0)

        # Put the tarball into the directory containing the target file
        dir_name = os.path.dirname(path)
        if not dir_name:
            dir_name = "/"

        self.container.put_archive(dir_name, tar_stream)

    def read_file(self, path: str) -> str:
        """Reads a file from the container and returns it as a string."""
        if not self.container:
            raise RuntimeError("Container not started")

        try:
            # get_archive returns (stream, stat)
            stream, stat = self.container.get_archive(path)

            # Read the stream into bytes
            file_obj = io.BytesIO()
            for chunk in stream:
                file_obj.write(chunk)
            file_obj.seek(0)

            # Extract the actual file content from the tar wrapper
            with tarfile.open(fileobj=file_obj, mode="r") as tar:
                # We expect a single file, so we take the first member
                member = tar.next()
                return tar.extractfile(member).read().decode("utf-8")

        except docker.errors.NotFound:
            return f"Error: File {path} not found."


# %%
# Simple sanity checks
with DockerContainer(image="alpine") as container:
    r = container.run_command('/bin/sh -c "cd home && ls -al"')
    print(r)
    r = container.write_file(path="/a.txt", content="hello world")
    print(r)
    r = container.run_command("sed -i 's/hello/hi/g' /a.txt")
    print(r)
    r = container.read_file(path="/a.txt")
    print(r)

# %%

# class ActionType(str, Enum):
#     READ_FILE = "read_file"
#     WRITE_FILE = "write_file"
#     RUN_COMMAND = "run"
#     FINISHED = "finish"


@dataclass
class LLMAction:
    action: str
    reasoning: str
    path: Optional[str] = None
    content: Optional[str] = None
    command: Optional[str] = None
    workdir: Optional[str] = None


# %%



@kbench.task()
def debug_py_code(llm, max_attempts=5):
    buggy_py_code = textwrap.dedent("""
        fruits = [
          'apple',
          'orange'
          'banana',
          'peach',
          'avocado'
        ]

        print(len(fruits)) # shoudl be 5
    """)
    code_path = "/working/fruits.py"
    code_dir = os.path.dirname(code_path)

    agent_prompt = textwrap.dedent(f"""
    You are an expert Python debugging assistant.
    GOAL: Locate and fix bugs in all Python files within the directory: `{code_dir}`.

    process:
    1. EXPLORE: Start by listing files to understand the directory structure.
    2. ANALYZE: Read files to understand the code and identify errors.
    3. FIX: Create or overwrite files with fixed code.
    4. VERIFY: You may run scripts to verify fixes if needed.

    RESPONSE FORMAT INSTRUCTIONS:
    You must respond STRICTLY with a valid JSON object.
    - NO Markdown formatting (do not use ```json).
    - NO text before or after the JSON.
    - The JSON must strictly follow this schema:

    {{
      "reasoning": "A brief explanation of why you are taking this action and what you expect to find.",
      "action": "read_file" | "write_file" | "run" | "submit",
      "path": "string path (required for read_file/write_file, else null)",
      "content": "string file content (required for write_file, else null)",
      "command": "string shell command (required for run, else null)"
    }}

    # Important rule:
    Remeber the valid valudes for "action" can only be one of "read_file", "write_file", "run" and "submit"
    but nothing else.

    EXAMPLES:

    ## Action: Debug the current folder.
    Assistant: {{
      "reasoning": "I need to see what files are in the directory to identify targets for debugging.",
      "action": "run",
      "command": "ls -R",
      "path": null,
      "content": null
    }}

    ## Action: I see a bug in main.py.
    Assistant: {{
      "reasoning": "I will read main.py to understand the logic error reported.",
      "action": "read_file",
      "path": "main.py",
      "command": null,
      "content": null
    }}

    ## Action: The error is a missing import in main.py.
    Assistant: {{
      "reasoning": "I will rewrite main.py to include the missing 'import os' statement.",
      "action": "write_file",
      "path": "main.py",
      "content": "import os\\n\\nprint(os.getcwd())",
      "command": null
    }}

    ## Action: Done or cannot continue
    Assistant: {{
      "reasoning": "I have successfully finished the job.",
      "action": "submit",
      "path": null,
      "content": null,
      "command": null
    }}
    """).strip()

    with DockerContainer(image="python:3.11-slim") as container:
        # setup
        container.run_command("mkdir -p /working")
        container.write_file(path=code_path, content=buggy_py_code)

        kbench.user.send(agent_prompt, is_visible_to_llm=True)
        response = ""
        for _ in range(max_attempts):
            llm_action = llm.prompt(
                textwrap.dedent(f"""Here is the response from your previous action (can be empty):

                ```
                {response}
                ```

                What's your next action?
                """),
                schema=LLMAction,
            )
            try:
                match llm_action:
                    case LLMAction(action="read_file", path=p) if p:
                        print(f"Reading from file: {p}")
                        response = container.read_file(path=p)

                    case LLMAction(action="write_file", path=p, content=c):
                        print(f"Writing to {p} with content length {len(str(c))}")
                        response = container.write_file(path=p, content=c)

                    case LLMAction(action="run", command=cmd, workdir=wd):
                        print(f"Executing: {cmd} in {wd or 'current dir'}")
                        response = container.run_command(command=cmd, workdir=wd)

                    case LLMAction(action="submit"):
                        print("Task completed.")
                        break
                    case _:
                        print(f"Unknown or invalid action received: {llm_action}")
                        response = f"Unknown or invalid action received: {llm_action}"
            except Exception as e:
                response = f"error: {e}"

        # Test fixed code
        fixed_code = kbench.tools.python.extract_code(
            container.read_file(path=code_path)
        )
        fixed_code_output = kbench.tools.python.script_runner.run_code(fixed_code)
        kbench.assertions.assert_equal(
            "5",
            fixed_code_output.stdout.strip(),
            expectation="The fixed code should output 5, the correct length of the list.",
        )


# %%
debug_py_code.run(kbench.llm, max_attempts=10)
