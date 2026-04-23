# Copyright 2026 Kaggle Inc.
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
import warnings
from datetime import datetime, timezone
from pathlib import Path

_BASE_DIR = Path(__file__).parent.parent.parent.parent
_KAGGLE_AUTH_COMMAND = f"cd {_BASE_DIR} && kaggle benchmarks auth"


def assert_kaggle_auth_exists(
    url: str | None = None,
    api_key: str | None = None,
    raise_on_error: bool = False,
) -> None:
    """Warn (or raise) if required auth env vars are missing."""
    resolved_url = url or os.getenv("MODEL_PROXY_URL")
    resolved_api_key = api_key or os.getenv("MODEL_PROXY_API_KEY")

    missing = []
    if not resolved_url:
        missing.append("MODEL_PROXY_URL")
    if not resolved_api_key:
        missing.append("MODEL_PROXY_API_KEY")
    if not missing:
        return

    missing_list = "\n".join(f"  - {v}" for v in missing)
    separator = "-" * len(_KAGGLE_AUTH_COMMAND)
    msg = (
        f"\n\nMissing environment variables for Kaggle authentication:\n\n{missing_list}\n\n"
        f"Authenticate by running:\n{separator}\n{_KAGGLE_AUTH_COMMAND}\n{separator}\n"
    )

    if raise_on_error:
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=2)


def assert_kaggle_auth_valid(
    expiry_time: str | None = None,
) -> None:
    """Warn if the auth token has expired."""
    resolved_expiry_time = expiry_time or os.getenv("MODEL_PROXY_EXPIRY_TIME")

    # Kaggle notebook-based flows don't set MODEL_PROXY_EXPIRY_TIME, so skip the check.
    if not resolved_expiry_time:
        return

    try:
        expiry = datetime.fromisoformat(resolved_expiry_time)
    except (ValueError, TypeError):
        return

    if expiry <= datetime.now(timezone.utc):
        separator = "-" * len(_KAGGLE_AUTH_COMMAND)
        warnings.warn(
            "\n\nKaggle authentication has expired. Re-authenticate by running:\n"
            f"{separator}\n{_KAGGLE_AUTH_COMMAND}\n{separator}\n",
            stacklevel=2,
        )
