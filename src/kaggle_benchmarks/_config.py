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

import dataclasses
import enum
import logging
import os
from pathlib import Path
from typing import Any, Self, overload

import dotenv
import panel as pn

base_dir = Path(__file__).parent.parent.parent

_logger = logging.getLogger(__name__)
if _dotenv_path := dotenv.find_dotenv():
    _logger.info("Loading environment variables from %s", _dotenv_path)
    dotenv.load_dotenv(_dotenv_path, override=True)

logger = logging.getLogger(__name__)


def string_to_bool(s: str) -> bool:
    """Converts a string to a boolean, handling various truthy values."""
    return s.lower() in ("true", "1", "t", "y", "yes")


@overload
def _parse_int_env(name: str) -> int | None: ...
@overload
def _parse_int_env(name: str, default: int) -> int: ...
def _parse_int_env(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"Ignoring non-integer value for {name}={raw!r}; using {default}."
        )
        return default


class ExecutionMode(enum.Enum):
    NOTEBOOK = 0
    INTERACTIVE = 0
    RUN = 1
    TRIAL = 1
    TESTING = 2
    DEV = 3
    DOC = 4


@dataclasses.dataclass()
class Config:
    execution_mode: ExecutionMode = dataclasses.field(
        default_factory=lambda: ExecutionMode[os.getenv("BENCHMARK_MODE", "NOTEBOOK")]
    )

    enable_caching: bool = dataclasses.field(
        default_factory=lambda: string_to_bool(
            os.environ.get("ENABLE_LOCAL_CACHING", "False")
        )
    )

    cache_directory: Path = dataclasses.field(
        default_factory=lambda: Path(
            os.environ.get("CACHE_DIRECTORY", base_dir / ".cache")
        )
    )

    cache_timeout_seconds: int = dataclasses.field(
        default_factory=lambda: _parse_int_env(
            "CACHE_TIMEOUT_SECONDS", 7 * 24 * 60 * 60
        )
    )

    interactive_mode: bool = dataclasses.field(
        default_factory=lambda: string_to_bool(
            os.environ.get("INTERACTIVE_UI", "False")
        )
    )
    ui_handler: Any = None

    _ui_theme: str = dataclasses.field(
        default_factory=lambda: os.environ.get("UI_THEME", "default")
    )
    suppress_ui_warnings: bool = dataclasses.field(
        default_factory=lambda: string_to_bool(
            os.environ.get("SUPPRESS_UI_WARNINGS", "true")
        )
    )
    continue_with_exceptions: bool = dataclasses.field(
        default_factory=lambda: (
            os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch"
            or string_to_bool(os.environ.get("CONTINUE_WITH_EXCEPTIONS", "False"))
        )
    )
    render_subruns: bool = dataclasses.field(
        default_factory=lambda: string_to_bool(os.environ.get("RENDER_SUBRUNS", "True"))
    )

    # Maximum length the host platform allows for `@kbench.task(...)` `name`
    # and `description` arguments. The Kaggle notebook runtime sets these to
    # match its backend column widths so users get fast feedback before a
    # wasted run; non-Kaggle users leave them unset (`None`) for no limit.
    task_name_max_length: int | None = dataclasses.field(
        default_factory=lambda: _parse_int_env("KAGGLE_BENCHMARK_MAX_NAME_LENGTH")
    )
    task_description_max_length: int | None = dataclasses.field(
        default_factory=lambda: _parse_int_env(
            "KAGGLE_BENCHMARK_MAX_DESCRIPTION_LENGTH"
        )
    )

    show_message_details: bool = False

    def apply(self) -> Self:
        pn.config.theme = self.ui_theme  # type: ignore

        if self.suppress_ui_warnings:
            logger = logging.getLogger("bokeh")
            logger.setLevel(logging.ERROR)
        if self.interactive_mode:
            from kaggle_benchmarks import events, ui

            if self.ui_handler is None:
                self.ui_handler = ui.panel.PanelUI()
                events.manager.bind(self.ui_handler)

        return self

    @property
    def ui_theme(self):
        return self._ui_theme

    @ui_theme.setter
    def ui_theme(self, value):
        self._ui_theme = value
        self.apply()

    def enable_dark_mode(self):
        self.ui_theme = "dark"

    def disable_dark_mode(self):
        self.ui_theme = "default"

    def enable_interactive_mode(self):
        self.interactive_mode = True
        self.apply()

    def disable_tqdm(self):
        return self.execution_mode != ExecutionMode.NOTEBOOK


config = Config()
