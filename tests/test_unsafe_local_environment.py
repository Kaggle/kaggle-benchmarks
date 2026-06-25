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

import warnings

import pytest

from kaggle_benchmarks.envs.local import InternalUnsafeLocalEnvironment


class TestInternalUnsafeLocalEnvironmentWarning:
    """Tests for the runtime safety warning on InternalUnsafeLocalEnvironment."""

    def test_warns_by_default(self):
        """External callers (default _internal=False) should see a UserWarning."""
        with pytest.warns(UserWarning, match="NO sandboxing"):
            env = InternalUnsafeLocalEnvironment()
        env.close()

    def test_no_warning_when_internal(self):
        """Internal callers passing _internal=True should not see a warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env = InternalUnsafeLocalEnvironment(_internal=True)
        env.close()

    def test_warning_message_mentions_docker(self):
        """The warning should guide users toward DockerEnvironment."""
        with pytest.warns(UserWarning, match="DockerEnvironment"):
            env = InternalUnsafeLocalEnvironment()
        env.close()

    def test_still_functional_regardless_of_internal_flag(self):
        """Both _internal=True and _internal=False produce a working environment."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for internal in (True, False):
                with InternalUnsafeLocalEnvironment(_internal=internal) as env:
                    result = env.run(["echo", "hello"])
                    assert result.exit_code == 0
                    assert result.stdout.strip() == "hello"

    def test_run_rejects_string_command(self):
        """String commands must be rejected to prevent shell-injection foot-guns."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with InternalUnsafeLocalEnvironment(_internal=True) as env:
                with pytest.raises(TypeError, match="list"):
                    env.run("echo hello; touch /tmp/should_not_exist")

    @pytest.mark.parametrize(
        "command",
        [
            ("echo", "hi"),
            (x for x in ["echo", "hi"]),
        ],
        ids=["tuple", "generator"],
    )
    def test_run_rejects_non_list_command(self, command):
        """Non-list command containers (tuple, generator, etc.) must be rejected
        even though subprocess.run would accept them. The list[str] requirement
        is the contract — narrowing it to "any iterable" would let callers
        forget the security rationale and start passing whatever they have."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with InternalUnsafeLocalEnvironment(_internal=True) as env:
                with pytest.raises(TypeError, match="list"):
                    env.run(command)

    def test_run_list_args_neutralize_shell_metacharacters(self, tmp_path):
        """Shell metacharacters interpolated into a list argument must be
        treated as literal text, not interpreted by a shell.

        Regression test for Kaggle/kaggle-benchmarks#159: this is the actual
        security guarantee the list[str] requirement provides — even if a
        caller does ``env.run(["echo", untrusted_input])``, the metacharacters
        in ``untrusted_input`` cannot spawn additional commands on the host.
        """
        marker = tmp_path / "injection_marker"
        malicious = f"hello; touch {marker}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with InternalUnsafeLocalEnvironment(_internal=True) as env:
                result = env.run(["echo", malicious])

        assert result.exit_code == 0
        assert malicious in result.stdout
        assert not marker.exists(), "shell metacharacters were interpreted"

    def test_run_explicit_shell_escape_hatch_still_works(self):
        """Callers that explicitly opt into a shell (the documented migration
        path for old string usage) must still be able to use shell features."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with InternalUnsafeLocalEnvironment(_internal=True) as env:
                result = env.run(["bash", "-c", "echo one two three | wc -w"])

        assert result.exit_code == 0
        assert result.stdout.strip() == "3"
