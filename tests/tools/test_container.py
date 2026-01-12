import io
import tarfile
from unittest.mock import MagicMock, patch

import docker
import pytest

from kaggle_benchmarks.tools.container import DockerContainer


@pytest.fixture
def mock_docker_client():
    """Mocks the docker.from_env() client."""
    with patch("docker.from_env") as mock_env:
        yield mock_env


@pytest.fixture
def container_actor(mock_docker_client):
    """
    Returns an instance of DockerContainer with a mocked client
    and a mocked 'send' method.
    """
    actor = DockerContainer(image="python:3.9-slim")

    # Mock the internal docker client container object
    actor.container = MagicMock()
    actor.container.status = "running"

    # Mock the 'send' method inherited from actors.Actor
    # so we can verify log messages are sent correctly
    actor.send = MagicMock()

    return actor


# --- Context Manager Tests ---


def test_enter_pulls_image_if_missing(mock_docker_client):
    """Test that image is pulled if ImageNotFound is raised."""
    mock_client_instance = mock_docker_client.return_value
    mock_client_instance.images.get.side_effect = docker.errors.ImageNotFound("Missing")

    mock_container = MagicMock()
    mock_container.status = "running"
    mock_client_instance.containers.run.return_value = mock_container

    actor = DockerContainer("test-image")
    # We must mock send here too since __init__ runs before we can mock it on the instance
    actor.send = MagicMock()

    with actor:
        pass

    mock_client_instance.images.pull.assert_called_once_with("test-image")
    mock_client_instance.containers.run.assert_called_with(
        "test-image", command="tail -f /dev/null", detach=True, tty=True
    )


def test_enter_timeouts_if_not_running(mock_docker_client):
    """Test that TimeoutError is raised if container never reaches 'running' status."""
    mock_client_instance = mock_docker_client.return_value
    mock_container = MagicMock()
    mock_container.status = "created"  # Never becomes "running"
    mock_client_instance.containers.run.return_value = mock_container

    actor = DockerContainer("test-image")

    # Mock time.time to simulate passage of time (0s -> 11s)
    with (
        patch("time.time", side_effect=[0, 11, 12, 13]),
        patch("time.sleep", return_value=None),
    ):
        with pytest.raises(TimeoutError, match="within 10 seconds"):
            with actor:
                pass

    # Verify cleanup attempts
    mock_container.stop.assert_called()
    mock_container.remove.assert_called()


# --- run_command Tests ---


def test_run_command_success(container_actor):
    """Test standard command execution logs correctly."""
    # Setup mock return
    container_actor.container.exec_run.return_value = (0, b"Hello World\n")

    result = container_actor.run_command("echo Hello")

    # Verify result
    assert result == "Hello World"

    # Verify docker call (workdir defaults to None)
    container_actor.container.exec_run.assert_called_with("echo Hello", workdir=None)

    # Verify internal logging via self.send
    container_actor.send.assert_called_with("Run `echo Hello`", is_visible_to_llm=False)


def test_run_command_with_workdir(container_actor):
    """Test run_command with a specific working directory."""
    container_actor.container.exec_run.return_value = (0, b"")

    container_actor.run_command("ls", workdir="/app")

    # Verify docker call
    container_actor.container.exec_run.assert_called_with("ls", workdir="/app")

    # Verify logging format for workdir
    container_actor.send.assert_called_with(
        "Run `ls` in `/app`", is_visible_to_llm=False
    )


def test_run_command_failure(container_actor):
    """Test that non-zero exit codes are formatted as errors."""
    container_actor.container.exec_run.return_value = (1, b"Syntax Error")

    result = container_actor.run_command("bad_cmd")

    assert "Error (Exit Code 1)" in result
    assert "Syntax Error" in result


# --- write_text_file Tests ---


def test_write_text_file_logic(container_actor):
    """Test that file writing handles paths and sends logs."""
    path = "my dir/file.txt"
    content = "Hello"

    container_actor.write_text_file(path, content)

    # Verify directory creation handles spaces (the quote fix)
    container_actor.container.exec_run.assert_called_with('mkdir -p "my dir"')

    # Verify put_archive called with correct dir
    container_actor.container.put_archive.assert_called_once()
    args, _ = container_actor.container.put_archive.call_args
    assert args[0] == "my dir"

    # Verify logging
    container_actor.send.assert_called_with(
        f"Write `{path}` with content length {len(content)}", is_visible_to_llm=False
    )


# --- read_text_file Tests ---


def test_read_text_file_success(container_actor):
    """Test successful file read."""
    # Create valid tar bytes
    data = b"File Content"
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        info = tarfile.TarInfo(name="test.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    tar_stream.seek(0)

    # Mock get_archive
    container_actor.container.get_archive.return_value = (tar_stream, {})

    content = container_actor.read_text_file("test.txt")

    assert content == "File Content"
    container_actor.send.assert_called_with("Read `test.txt`", is_visible_to_llm=False)


def test_read_text_file_not_found(container_actor):
    """Test error handling for missing files."""
    container_actor.container.get_archive.side_effect = docker.errors.NotFound(
        "Missing"
    )

    result = container_actor.read_text_file("ghost.txt")

    assert "Error: File ghost.txt not found" in result
    container_actor.send.assert_called_with("Read `ghost.txt`", is_visible_to_llm=False)


def test_read_text_file_empty_tar(container_actor):
    """Test handling of empty tar archive."""
    empty_tar = io.BytesIO()
    with tarfile.open(fileobj=empty_tar, mode="w") as _:
        pass
    empty_tar.seek(0)

    container_actor.container.get_archive.return_value = (empty_tar, {})

    result = container_actor.read_text_file("empty.txt")

    assert "Error: File empty.txt is empty or invalid tar" in result


# --- Edge Case & Error Handling Tests ---


def test_methods_raise_runtime_error_if_not_started(mock_docker_client):
    """Test that methods raise RuntimeError if container is not running."""
    actor = DockerContainer("test-image")

    with pytest.raises(RuntimeError, match="Container not started"):
        actor.run_command("ls")

    with pytest.raises(RuntimeError, match="Container not started"):
        actor.write_text_file("file.txt", "content")

    with pytest.raises(RuntimeError, match="Container not started"):
        actor.read_text_file("file.txt")


def test_exit_suppresses_api_error(container_actor):
    """Test that __exit__ suppresses docker APIErrors during cleanup."""
    container_actor.container.stop.side_effect = docker.errors.APIError("Docker error")

    # Should not raise exception
    container_actor.__exit__(None, None, None)

    container_actor.container.stop.assert_called()


def test_run_command_decoding_error(container_actor):
    """Test that invalid UTF-8 output is handled gracefully."""
    # \x80 is invalid in UTF-8
    container_actor.container.exec_run.return_value = (0, b"Invalid \x80 byte")

    result = container_actor.run_command("cmd")

    # errors="replace" usually inserts the replacement character  (U+FFFD)
    assert "Invalid  byte" in result or "Invalid" in result
