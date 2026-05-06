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

"""Tests for Config.apply() auto-detect behavior.

Notebook kernels (Jupyter, Kaggle, VSCode notebook) MUST NOT auto-bind
PanelUI: large benchmarks crash the kernel under live streaming. Users
opt in via enable_interactive_mode() / INTERACTIVE_UI=True. Terminals
keep the auto-on ConsoleUI introduced in #149.
"""

import pytest

from kaggle_benchmarks import _config, events
from kaggle_benchmarks._config import ExecutionMode, HostEnvironment
from kaggle_benchmarks.ui import console as console_ui


@pytest.fixture
def patch_host_env(monkeypatch):
    """Return a setter that overrides _config.detect_host_environment."""

    def _set(env: HostEnvironment):
        monkeypatch.setattr(_config, "detect_host_environment", lambda: env)

    return _set


@pytest.mark.parametrize(
    "host_env",
    [HostEnvironment.JUPYTER, HostEnvironment.VSCODE_NOTEBOOK],
)
def test_notebook_kernel_does_not_autobind_panel_ui(cfg, patch_host_env, host_env):
    """In any notebook kernel, default apply() leaves no handler bound."""
    patch_host_env(host_env)
    cfg.execution_mode = ExecutionMode.NOTEBOOK
    cfg.interactive_mode = False
    cfg.console_mode = False
    cfg.ui_handler = None
    events.manager.listeners = []

    cfg.apply()

    assert cfg.ui_handler is None
    assert events.manager.listeners == []


def test_notebook_kernel_enable_interactive_mode_binds_panel_ui(cfg, patch_host_env):
    """enable_interactive_mode() in a notebook kernel must bind PanelUI."""
    from kaggle_benchmarks.ui import panel as panel_ui

    patch_host_env(HostEnvironment.JUPYTER)
    cfg.execution_mode = ExecutionMode.NOTEBOOK
    cfg.interactive_mode = False
    cfg.console_mode = False
    cfg.ui_handler = None
    events.manager.listeners = []

    cfg.enable_interactive_mode()

    assert isinstance(cfg.ui_handler, panel_ui.PanelUI)
    assert cfg.ui_handler in events.manager.listeners


def test_terminal_default_autobinds_console_ui(cfg, patch_host_env):
    """In a terminal host env, default apply() still auto-binds ConsoleUI."""
    patch_host_env(HostEnvironment.TERMINAL)
    cfg.execution_mode = ExecutionMode.NOTEBOOK
    cfg.interactive_mode = False
    cfg.console_mode = False
    cfg.ui_handler = None
    events.manager.listeners = []

    cfg.apply()

    assert isinstance(cfg.ui_handler, console_ui.ConsoleUI)
    assert cfg.ui_handler in events.manager.listeners


@pytest.mark.parametrize(
    "host_env",
    [
        HostEnvironment.TERMINAL,
        HostEnvironment.JUPYTER,
        HostEnvironment.VSCODE_NOTEBOOK,
    ],
)
def test_testing_mode_binds_nothing(cfg, patch_host_env, host_env):
    """ExecutionMode.TESTING never auto-binds a handler regardless of host."""
    patch_host_env(host_env)
    cfg.execution_mode = ExecutionMode.TESTING
    cfg.interactive_mode = False
    cfg.console_mode = False
    cfg.ui_handler = None
    events.manager.listeners = []

    cfg.apply()

    assert cfg.ui_handler is None
    assert events.manager.listeners == []
