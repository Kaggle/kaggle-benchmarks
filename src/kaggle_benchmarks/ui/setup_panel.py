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

import panel as pn
from panel import theme

from kaggle_benchmarks._config import ExecutionMode, config


def _is_vscode_kernel() -> bool:
    """Detects if the kernel is hosted by the VSCode Jupyter extension.

    VSCode injects these env vars into the kernel process; JupyterLab does not.
    """
    return "VSCODE_PID" in os.environ or "VSCODE_IPC_HOOK_CLI" in os.environ

# css overrides to make the HoloViz output work with Kaggle theming.
global_css = """
:root {
    --design-primary-color: var(--kaggle-theme-design-primary-color);
    --design-primary-text-color: var(--kaggle-theme-design-primary-text-color);
    --design-secondary-color: var(--kaggle-theme-design-secondary-color);
    --design-secondary-text-color: var(--kaggle-theme-design-secondary-text-color);
    --design-background-color: var(--kaggle-theme-design-background-color);
    --design-background-text-color: var(--kaggle-theme-design-background-text-color);
    --design-surface-color: var(--kaggle-theme-design-surface-color);
    --design-surface-text-color: var(--kaggle-theme-design-surface-text-color);
}
"""

bg_color = "--kaggle-theme-design-primary-color"
text_color = "--kaggle-theme-design-primary-text-color"

stylesheet = f"""
div.message, div.avatar {{
    background-color: color-mix(in lch, var({bg_color}), white 12%) !important;
    color: var({text_color});
}}

.name {{
    color: var({text_color});
}}

div.card {{
    background-color: color-mix(in lch, var({bg_color}), white 5%) !important;
    color: var({text_color});
}}

div.card-header, div.accordion-header, div.json-formatter-row  {{
    background-color: color-mix(in lch, var({bg_color}), white 10%) !important;
    color: var({text_color});
}}

.bk-tab {{
    color: var({text_color}) !important;
}}

.bk-tab.bk-active {{
    background-color: var(--kaggle-theme-design-secondary-text-color) !important;
    color: var(--kaggle-theme-design-secondary-color) !important;
}}

@media (hover: hover) and (pointer: fine) {{
    .tabulator-col-content {{
        background-color: var({bg_color})
    }}
}}

.accordion-header {{
    text-align: left;
}}
"""


class CustomDesign(theme.Material):
    modifiers = {pn.viewable.Viewable: {"stylesheets": [theme.Inherit, stylesheet]}}


if (
    config.execution_mode == ExecutionMode.RUN
    or config.execution_mode == ExecutionMode.DOC
):
    kwargs = {}
elif _is_vscode_kernel():
    # The VSCode Jupyter renderer doesn't register the @bokeh/jupyter_bokeh
    # widget JS, so the BokehModel ipywidget that comms="vscode"/"ipywidgets"
    # produce can't be rendered. Force comms="default" to bypass ipywidget
    # rendering and use Bokeh's inline JS path. Must be passed explicitly:
    # Panel's _detect_comms otherwise auto-overrides to "vscode" when
    # VSCODE_PID is set.
    kwargs = dict(comms="default")
elif config.execution_mode == ExecutionMode.DEV:
    kwargs = dict(comms="vscode")
else:
    kwargs = dict(comms="ipywidgets")

pn.extension("tabulator", design=CustomDesign, global_css=[global_css], **kwargs)
