import os
import pytest
from kaggle_benchmarks.envs.local import LocalEnvironment

def test_shell_injection_prevention():
    """Test that shell injection is prevented."""
    env = LocalEnvironment()
    try:
        # Attempt to create a file using shell injection
        # If shell=True, this would create 'injected.txt'
        cmd = "echo vulnerable; touch injected.txt"

        result = env.run(cmd)

        # Verify the command executed (echo vulnerable) but the injection failed
        # shlex.split("echo vulnerable; touch injected.txt") -> ['echo', 'vulnerable;', 'touch', 'injected.txt']
        # So echo will print "vulnerable; touch injected.txt"

        assert "vulnerable" in result.stdout
        assert "injected.txt" in result.stdout

        # Verify injected file does not exist
        injected_file = os.path.join(env.temp_dir.name, "injected.txt")
        assert not os.path.exists(injected_file), "Shell injection successful! File created."

    finally:
        env.close()

def test_valid_commands_string():
    """Test that valid commands still work when passed as string."""
    env = LocalEnvironment()
    try:
        result = env.run("echo hello world")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello world"
    finally:
        env.close()

def test_valid_commands_list():
    """Test that valid commands still work when passed as list."""
    env = LocalEnvironment()
    try:
        result = env.run(["echo", "hello", "world"])
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello world"
    finally:
        env.close()

def test_quoted_arguments():
    """Test that quoted arguments are handled correctly by shlex."""
    env = LocalEnvironment()
    try:
        # "echo 'hello world'" -> ['echo', 'hello world']
        result = env.run("echo 'hello world'")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello world"
    finally:
        env.close()
