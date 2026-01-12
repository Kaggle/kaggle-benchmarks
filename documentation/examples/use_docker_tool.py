# %%
import os
import textwrap
from dataclasses import dataclass
from typing import Optional

import kaggle_benchmarks as kbench


# %%
@dataclass
class LLMAction:
    action: str
    reasoning: str
    path: Optional[str] = None
    content: Optional[str] = None
    command: Optional[str] = None
    workdir: Optional[str] = None


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

        print(len(fruits)) # should be 5
    """)
    code_path = "/working/fruits.py"
    code_dir = os.path.dirname(code_path)

    system_prompt = textwrap.dedent(f"""
    You are an expert Python debugging assistant.
    GOAL: Locate and fix bugs in all Python files within the directory: `{code_dir}`.

    PROCESS:
    1. EXPLORE: List files to understand the directory structure.
    2. ANALYZE: Read files to understand the code and identify errors.
    3. FIX: Create or overwrite files with fixed code.
    4. VERIFY: Run scripts to verify fixes if needed.

    RESPONSE FORMAT:
    You must respond with a JSON object following this schema:
    {{
      "reasoning": "Explanation of the action",
      "action": "read_file" | "write_file" | "run" | "submit",
      "path": "File path (required for read_file/write_file)",
      "content": "File content (required for write_file)",
      "command": "Shell command (required for run)",
      "workdir": "Working directory (optional for run)"
    }}

    EXAMPLES:
    {{
      "reasoning": "List files to see what is in the directory.",
      "action": "run",
      "command": "ls -R",
      "workdir": "/"
    }}

    {{
      "reasoning": "Read main.py to find bugs.",
      "action": "read_file",
      "path": "/absolute/path/to/main.py"
    }}

    {{
      "reasoning": "Fix the missing import in main.py.",
      "action": "write_file",
      "path": "/absolute/path/to/main.py",
      "content": "import os\\n\\nprint(os.getcwd())"
    }}

    {{
      "reasoning": "All bugs fixed.",
      "action": "submit",
    }}
    """).strip()

    prompt_template = textwrap.dedent("""
        Here is the response from your previous action (can be empty):

        ```
        {response}
        ```

        What's your next action?
    """).strip()

    # Initiate a container session for LLM to interact with
    with kbench.tools.container.DockerContainer(image="python:3.11-slim") as container:
        # Setup
        container.run_command("mkdir -p /working")
        container.write_text_file(path=code_path, content=buggy_py_code)

        # LLM debugging
        kbench.user.send(system_prompt, is_visible_to_llm=True)
        response = ""
        for _ in range(max_attempts):
            llm_action = llm.prompt(
                message=prompt_template.format(response=response),
                schema=LLMAction,
            )
            try:
                match llm_action:
                    case LLMAction(action="read_file", path=p) if p:
                        print(f"Reading from file: `{p}`")
                        response = container.read_text_file(path=p)

                    case LLMAction(action="write_file", path=p, content=c) if (
                        p and c is not None
                    ):
                        print(f"Writing to `{p}` with content length {len(c)}")
                        container.write_text_file(path=p, content=c)
                        response = "File written successfully."

                    case LLMAction(action="run", command=cmd, workdir=wd) if cmd:
                        print(f"Executing: {cmd} in working directory: {wd}")
                        response = container.run_command(command=cmd, workdir=wd)

                    case LLMAction(action="submit"):
                        print("Task submitted.")
                        break
                    case _:
                        print(f"Unknown or invalid action received: {llm_action}")
                        response = f"Unknown or invalid action received: {llm_action}"
            except Exception as e:
                response = f"error: {e}"

        # Test fixed code
        fixed_code = kbench.tools.python.extract_code(
            container.read_text_file(path=code_path)
        )
        kbench.assertions.assert_contains_regex(
            r"(['\"])orange\1\s*,",
            fixed_code,
            "'orange' is followed by a comma in fixed code.",
        )
        output = container.run_command("python /working/fruits.py").strip()
        kbench.assertions.assert_equal(
            "5", output, "Running fixed code should print 5."
        )


# %%
debug_py_code.run(kbench.llm, max_attempts=10)

# %%
