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

from IPython import core


def autopilot(line):
    """
    Creates a UI to ask a question, then generates
    the Python code based on user inputs.
    """
    import ipywidgets as widgets
    from IPython.display import display

    from kaggle_benchmarks import utils

    # --- UI Layout Definitions ---
    input_layout = widgets.Layout(width="95%", height="80px")
    code_display_layout = widgets.Layout(width="95%", height="200px")  # Taller height
    description_style = {"description_width": "initial"}

    # --- UI Components ---
    title = widgets.HTML("<h3>📝 Auto-Code Generator</h3>")

    task_box = widgets.Textarea(
        value="",
        placeholder="Describe your task e.g., What is Kaggle?",
        description="Task Description:",
        layout=input_layout,
        style=description_style,
    )

    assertion_box = widgets.Textarea(
        value="",
        placeholder='Describe what you want to assert e.g., "data science" in answer.',
        description="Assertion Criteria:",
        layout=input_layout,
        style=description_style,
    )

    generate_button = widgets.Button(
        description="Generate Code",
        button_style="success",
        tooltip="Click to generate the code",
        icon="play",
    )

    output_area = widgets.Output()

    code_display = widgets.Textarea(
        value="",
        placeholder="Generated code will appear here...",
        description="Generated Code:",
        disabled=True,
        layout=code_display_layout,
        style=description_style,
    )

    def on_button_clicked(b):
        output_area.clear_output()

        task_description = task_box.value
        if not task_description.strip():
            with output_area:
                print("⚠️ Please enter a task description.")
            return

        assertion_description = assertion_box.value
        if not assertion_description.strip():
            with output_area:
                print("⚠️ Please enter an assertion description.")
            return

        # --- Update UI to show loading state ---
        generate_button.disabled = True
        generate_button.description = "Generating..."
        generate_button.icon = "spinner"
        with output_area:
            print("⏳ Please wait, generating code...")

        try:
            # --- Call the potentially slow function ---
            generated_code = utils.task_autopilot(
                task_description, assertion_description
            )

            # --- Update UI with the result ---
            code_display.value = generated_code
            output_area.clear_output()  # Clear the "waiting" message
            with output_area:
                print("✅ Code generated successfully.")

        except Exception as e:
            # --- Handle potential errors from the function ---
            output_area.clear_output()
            with output_area:
                print(f"❌ An error occurred: {e}")

        finally:
            # --- Restore the button to its original state ---
            generate_button.disabled = False
            generate_button.description = "Generate Code"
            generate_button.icon = "play"

    generate_button.on_click(on_button_clicked)

    # Display all UI components in a vertical box
    ui_container = widgets.VBox(
        [title, task_box, assertion_box, generate_button, code_display, output_area]
    )
    display(ui_container)


def choose(line):
    """
    IPython line magic to choose a benchmark task and remove all other
    *run.json and *task.json files from /kaggle/working.
    """
    import re
    from pathlib import Path

    def _normalize_name(name: str) -> str:
        """Replaces special characters such as spaces and slashes in a name."""
        name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return re.sub(r"[^a-zA-Z0-9_.-]", "", name)

    param_str = line.strip()
    if not param_str:
        print("Usage: %choose <benchmark_task_name_or_object>")
        return

    ipython_shell = core.getipython.get_ipython()
    if ipython_shell is None:
        print("Error: This magic must be run in an IPython environment.")
        return

    task_name = None
    task_obj = ipython_shell.user_ns.get(param_str)

    if (
        task_obj
        and hasattr(task_obj, "name")
        and isinstance(getattr(task_obj, "name"), str)
    ):
        # It's a valid variable in the namespace that has a .name attribute
        task_name = task_obj.name
    else:
        # It's not a known variable, so treat the input as the task name string
        task_name = param_str

    normalized_task_name = _normalize_name(task_name)

    working_dir = Path("/kaggle/working")
    if not working_dir.is_dir():
        print(f"Directory not found: {working_dir}")
        return

    all_run_files = list(working_dir.glob("*.run.json"))
    all_task_files = list(working_dir.glob("*.task.json"))

    files_to_keep = set()

    task_files_to_keep = [
        f for f in all_task_files if f.name == f"{normalized_task_name}.task.json"
    ]
    files_to_keep.update(task_files_to_keep)

    matching_run_files = sorted(
        (
            f
            for f in all_run_files
            if re.fullmatch(f"{normalized_task_name}.*\\.run\\.json", f.name)
        ),
        key=lambda f: f.stat().st_mtime,
    )
    if matching_run_files:
        files_to_keep.add(matching_run_files[-1])  # Use the last run file

    all_files_to_scan = set(all_run_files + all_task_files)

    for f in all_files_to_scan:
        if f not in files_to_keep:
            try:
                f.unlink()
            except OSError as e:
                print(f"Error removing file {f.name}: {e}")
        else:
            print(f"Kept: {f.name}")


ip = core.getipython.get_ipython()
# Register the magic if we're in an interactive IPython environment.
if ip is not None:
    ip.register_magic_function(autopilot, "line")
    ip.register_magic_function(choose, "line")
