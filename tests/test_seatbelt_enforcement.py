"""End-to-end enforcement tests against the real macOS Seatbelt sandbox.

These tests run real commands under ``/usr/bin/sandbox-exec`` with no mocking of
OS boundaries. They skip on platforms without Seatbelt unless
``LOCALMCP_REQUIRE_SEATBELT=1`` is set, in which case collection fails loudly so
a CI job dedicated to enforcement cannot pass with every test skipped.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from localmcp.sandbox import CommandResult, RootAccess, SandboxProfile, SandboxRoot
from localmcp.sandbox.seatbelt import MacOSSandbox, _developer_directory, _developer_installation

_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_SEATBELT_AVAILABLE = sys.platform == "darwin" and _SANDBOX_EXEC.is_file()

if not _SEATBELT_AVAILABLE:
    _reason = f"macOS Seatbelt unavailable (platform={sys.platform}, {_SANDBOX_EXEC} present={_SANDBOX_EXEC.is_file()})"
    if os.environ.get("LOCALMCP_REQUIRE_SEATBELT") == "1":
        pytest.fail(f"LOCALMCP_REQUIRE_SEATBELT=1 but {_reason}", pytrace=False)
    pytest.skip(_reason, allow_module_level=True)

pytestmark = pytest.mark.seatbelt

_NC = "/usr/bin/nc"


@dataclass(frozen=True)
class Workspace:
    """Host directories used as sandbox roots plus a sibling outside every root."""

    read_write: Path
    read_only: Path
    outside: Path

    def profile(
        self, *, denied_paths: tuple[Path, ...] = (), network: bool = False, dev_tools: bool = False
    ) -> SandboxProfile:
        return SandboxProfile(
            roots=(
                SandboxRoot(self.read_write, RootAccess.READ_WRITE),
                SandboxRoot(self.read_only, RootAccess.READ_ONLY),
            ),
            denied_paths=denied_paths,
            network=network,
            dev_tools=dev_tools,
        )


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    base = tmp_path.resolve()
    directories = Workspace(base / "rw", base / "ro", base / "outside")
    for directory in (directories.read_write, directories.read_only, directories.outside):
        directory.mkdir()
    return directories


async def _run(
    profile: SandboxProfile, command: str, *, timeout_seconds: float = 10, max_output_bytes: int = 1024 * 1024
) -> CommandResult:
    sandbox = MacOSSandbox(profile, timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)
    return await sandbox.run(command)


def _assert_denied(result: CommandResult, *, leaked: str | None = None) -> None:
    assert result.exit_code != 0, result
    assert not result.timed_out
    if leaked is not None:
        assert leaked not in result.stdout
        assert leaked not in result.stderr


# Filesystem: baseline allowances


async def test_reads_file_in_read_only_root(workspace: Workspace) -> None:
    (workspace.read_only / "data.txt").write_text("read-only-content\n")

    result = await _run(workspace.profile(), f"cat {workspace.read_only / 'data.txt'}")

    assert result.exit_code == 0, result
    assert result.stdout == "read-only-content\n"


async def test_writes_file_in_read_write_root_visible_on_host(workspace: Workspace) -> None:
    result = await _run(workspace.profile(), "printf written > created.txt")

    assert result.exit_code == 0, result
    assert (workspace.read_write / "created.txt").read_text() == "written"


async def test_runs_with_first_root_as_working_directory(workspace: Workspace) -> None:
    result = await _run(workspace.profile(), "pwd")

    assert result.exit_code == 0, result
    assert result.stdout.strip() == str(workspace.read_write)


# Filesystem: write denials


async def test_write_to_read_only_root_is_denied(workspace: Workspace) -> None:
    target = workspace.read_only / "blocked.txt"

    result = await _run(workspace.profile(), f"printf nope > {target}")

    _assert_denied(result)
    assert not target.exists()


async def test_modifying_existing_file_in_read_only_root_is_denied(workspace: Workspace) -> None:
    target = workspace.read_only / "existing.txt"
    target.write_text("original")

    result = await _run(workspace.profile(), f"printf changed > {target} || rm -f {target}")

    _assert_denied(result)
    assert target.read_text() == "original"


async def test_write_outside_every_root_is_denied(workspace: Workspace) -> None:
    target = workspace.outside / "escaped.txt"

    result = await _run(workspace.profile(), f"printf nope > {target}")

    _assert_denied(result)
    assert not target.exists()


async def test_write_via_tmp_is_denied(workspace: Workspace) -> None:
    marker = f"localmcp-seatbelt-{os.getpid()}-{time.monotonic_ns()}"
    target = Path("/private/tmp") / marker

    result = await _run(workspace.profile(), f"printf nope > {target}")

    _assert_denied(result)
    assert not target.exists()


# Filesystem: read denials


async def test_read_outside_every_root_is_denied(workspace: Workspace) -> None:
    secret = workspace.outside / "secret.txt"
    secret.write_text("outside-secret-value")

    result = await _run(workspace.profile(), f"cat {secret}")

    _assert_denied(result, leaked="outside-secret-value")


async def test_listing_directory_outside_every_root_is_denied(workspace: Workspace) -> None:
    (workspace.outside / "hidden-name.txt").write_text("x")

    result = await _run(workspace.profile(), f"ls {workspace.outside}")

    _assert_denied(result, leaked="hidden-name.txt")


async def test_listing_real_home_directory_is_denied(workspace: Workspace) -> None:
    home = Path.home().resolve()
    if not home.is_dir() or any(home.is_relative_to(root) for root in (workspace.read_write, workspace.read_only)):
        pytest.skip("home directory unavailable or overlaps a sandbox root")

    result = await _run(workspace.profile(), f"ls -a {home}")

    _assert_denied(result)
    assert result.stdout == ""


async def test_symlink_in_root_to_file_outside_roots_cannot_be_read(workspace: Workspace) -> None:
    secret = workspace.outside / "secret.txt"
    secret.write_text("symlinked-secret-value")
    (workspace.read_write / "link.txt").symlink_to(secret)

    result = await _run(workspace.profile(), "cat link.txt")

    _assert_denied(result, leaked="symlinked-secret-value")


async def test_symlink_in_root_to_directory_outside_roots_cannot_be_listed(workspace: Workspace) -> None:
    (workspace.outside / "secret.txt").write_text("symlinked-dir-secret")
    (workspace.read_write / "linkdir").symlink_to(workspace.outside, target_is_directory=True)

    result = await _run(workspace.profile(), "cat linkdir/secret.txt")

    _assert_denied(result, leaked="symlinked-dir-secret")


async def test_sandbox_created_symlink_cannot_escape_roots(workspace: Workspace) -> None:
    secret = workspace.outside / "secret.txt"
    secret.write_text("created-link-secret")

    result = await _run(workspace.profile(), f"ln -s {secret} made.txt && cat made.txt")

    _assert_denied(result, leaked="created-link-secret")


# Filesystem: denied paths inside roots


@pytest.fixture
def denied_layout(workspace: Workspace) -> tuple[Path, Path, Path]:
    """Return (denied file, file inside denied directory, allowed sibling) inside the read-write root."""
    denied_file = workspace.read_write / "credentials.txt"
    denied_file.write_text("denied-file-secret")
    denied_directory = workspace.read_write / "private"
    denied_directory.mkdir()
    nested = denied_directory / "key.txt"
    nested.write_text("denied-dir-secret")
    allowed = workspace.read_write / "public.txt"
    allowed.write_text("public-content")
    return denied_file, nested, allowed


@pytest.fixture
def denied_profile(workspace: Workspace, denied_layout: tuple[Path, Path, Path]) -> SandboxProfile:
    denied_file, nested, _ = denied_layout
    return workspace.profile(denied_paths=(denied_file, nested.parent))


async def test_denied_file_inside_root_cannot_be_read(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, f"cat {denied_layout[0].name}")

    _assert_denied(result, leaked="denied-file-secret")


async def test_file_inside_denied_directory_cannot_be_read(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, "cat private/key.txt")

    _assert_denied(result, leaked="denied-dir-secret")


async def test_denied_directory_cannot_be_listed(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, "ls private")

    _assert_denied(result, leaked="key.txt")


async def test_sibling_of_denied_paths_remains_readable(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, f"cat {denied_layout[2].name}")

    assert result.exit_code == 0, result
    assert result.stdout == "public-content"


async def test_symlink_to_denied_file_cannot_be_read(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, f"ln -s {denied_layout[0]} alias.txt && cat alias.txt")

    _assert_denied(result, leaked="denied-file-secret")


async def test_hard_link_to_denied_file_cannot_be_read(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, f"ln {denied_layout[0].name} hard.txt && cat hard.txt")

    _assert_denied(result, leaked="denied-file-secret")


@pytest.mark.parametrize(
    ("command", "leaked"),
    [
        pytest.param("mv credentials.txt renamed.txt && cat renamed.txt", "denied-file-secret", id="denied-file"),
        pytest.param("mv private exposed && cat exposed/key.txt", "denied-dir-secret", id="denied-directory"),
        pytest.param(
            "mv private/key.txt moved.txt && cat moved.txt", "denied-dir-secret", id="file-in-denied-directory"
        ),
    ],
)
async def test_renaming_denied_path_in_read_write_root_does_not_expose_it(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path], command: str, leaked: str
) -> None:
    result = await _run(denied_profile, command)

    _assert_denied(result, leaked=leaked)


async def test_renaming_ancestor_of_denied_path_does_not_expose_it(workspace: Workspace) -> None:
    denied = workspace.read_write / "config" / "secrets" / "token.txt"
    denied.parent.mkdir(parents=True)
    denied.write_text("nested-secret")

    result = await _run(workspace.profile(denied_paths=(denied,)), "mv config exposed && cat exposed/secrets/token.txt")

    _assert_denied(result, leaked="nested-secret")


async def test_paths_unrelated_to_denied_paths_can_be_renamed(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, "mv public.txt renamed.txt && mkdir d && mv d e && cat renamed.txt")

    assert result.exit_code == 0, result
    assert result.stdout == "public-content"


# Network


@pytest.fixture
def listening_port() -> Iterator[int]:
    """A loopback TCP listener; the kernel backlog completes handshakes without accept()."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        yield server.getsockname()[1]


@pytest.mark.skipif(not Path(_NC).is_file(), reason="/usr/bin/nc unavailable")
async def test_outbound_connection_denied_without_network(workspace: Workspace, listening_port: int) -> None:
    result = await _run(workspace.profile(network=False), f"{_NC} -n -z -w 2 127.0.0.1 {listening_port}")

    _assert_denied(result)


@pytest.mark.skipif(not Path(_NC).is_file(), reason="/usr/bin/nc unavailable")
async def test_outbound_connection_allowed_with_network(workspace: Workspace, listening_port: int) -> None:
    result = await _run(workspace.profile(network=True), f"{_NC} -n -z -w 2 127.0.0.1 {listening_port}")

    assert result.exit_code == 0, result


# Environment


async def test_caller_environment_is_not_inherited(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALMCP_TEST_SECRET", "caller-env-sentinel")

    result = await _run(workspace.profile(), "env")

    assert result.exit_code == 0, result
    assert "LOCALMCP_TEST_SECRET" not in result.stdout
    assert "caller-env-sentinel" not in result.stdout
    environment = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert environment["HOME"] == "/var/empty"
    assert environment["TMPDIR"] == str(workspace.read_write)


# Developer tools

# The /usr/bin shims re-resolve the toolchain on every call, which is slow under the sandbox.
_TOOLCHAIN_TIMEOUT_SECONDS = 30


def _selected_developer() -> Path | None:
    return _developer_directory(os.environ.get("DEVELOPER_DIR"))


@pytest.fixture
def developer() -> Path:
    selected = _selected_developer()
    if selected is None:
        pytest.skip("no Xcode or Command Line Tools selected")
    return selected


@pytest.fixture(params=["host-path", "shim-only-path"])
def git_path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run git tests with the caller's PATH, and with only the /usr/bin shim on PATH."""
    if request.param == "shim-only-path":
        if _selected_developer() is None:
            pytest.skip("no Xcode or Command Line Tools selected")
        monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    elif shutil.which("git") is None:
        pytest.skip("git not on PATH")
    return str(request.param)


def _host_git(cwd: Path, *args: str) -> None:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(cwd),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    if "DEVELOPER_DIR" in os.environ:
        environment["DEVELOPER_DIR"] = os.environ["DEVELOPER_DIR"]
    subprocess.run(
        ["git", "-c", "user.email=t@e.st", "-c", "user.name=Tester", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env=environment,
    )


@pytest.fixture
def seeded_repository(workspace: Workspace) -> Path:
    """A one-commit repository in the read-only root."""
    (workspace.read_only / "file.txt").write_text("first\nsecond\n")
    _host_git(workspace.read_only, "init", "-q")
    _host_git(workspace.read_only, "add", "file.txt")
    _host_git(workspace.read_only, "commit", "-qm", "seed")
    return workspace.read_only


async def test_git_works_inside_read_write_root(workspace: Workspace, git_path: str) -> None:
    result = await _run(
        workspace.profile(dev_tools=True),
        "git init -q && git status --porcelain=v1 --branch",
        timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS,
    )

    assert result.exit_code == 0, result
    assert result.stdout.startswith("## ")
    assert (workspace.read_write / ".git" / "HEAD").is_file()


async def test_git_reads_history_in_read_only_root(
    workspace: Workspace, seeded_repository: Path, git_path: str
) -> None:
    result = await _run(
        workspace.profile(dev_tools=True),
        f"cd {seeded_repository} && git log --oneline && git blame file.txt",
        timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS,
    )

    assert result.exit_code == 0, result
    assert "seed" in result.stdout
    assert "first" in result.stdout and "second" in result.stdout


async def test_git_dispatches_exec_helpers(workspace: Workspace, seeded_repository: Path, git_path: str) -> None:
    # file:// transport execs git-upload-pack from git's exec path, not from PATH.
    result = await _run(
        workspace.profile(dev_tools=True),
        f"git ls-remote file://{seeded_repository} && git clone -q file://{seeded_repository} clone",
        timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS,
    )

    assert result.exit_code == 0, result
    assert "HEAD" in result.stdout
    assert (workspace.read_write / "clone" / "file.txt").read_text() == "first\nsecond\n"


async def test_dev_tools_runs_toolchain_python_through_the_shim(workspace: Workspace, developer: Path) -> None:
    if not (developer / "usr" / "bin" / "python3").is_file():
        pytest.skip("selected developer directory has no python3")

    result = await _run(
        workspace.profile(dev_tools=True),
        "/usr/bin/python3 -c 'import json, sqlite3; print(json.dumps(sqlite3.sqlite_version_info[0]))'",
        timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS,
    )

    assert result.exit_code == 0, result
    assert result.stdout.strip() == "3"


async def test_dev_tools_grants_no_writes_to_the_toolchain(workspace: Workspace, developer: Path) -> None:
    target = _developer_installation(developer) / f"localmcp-seatbelt-{os.getpid()}-{time.monotonic_ns()}"

    result = await _run(workspace.profile(dev_tools=True), f"printf nope > {target}")

    _assert_denied(result)
    assert not target.exists()


async def test_toolchain_is_unreadable_without_dev_tools(workspace: Workspace, developer: Path) -> None:
    result = await _run(workspace.profile(), f"ls {developer / 'usr' / 'bin'}")

    _assert_denied(result)
    assert result.stdout == ""


# Resource limits


async def test_long_running_command_times_out_promptly(workspace: Workspace) -> None:
    started = time.monotonic()

    result = await _run(workspace.profile(), "sleep 30", timeout_seconds=1)

    elapsed = time.monotonic() - started
    assert result.timed_out is True
    assert result.exit_code == -1
    assert "command timed out" in result.stderr
    assert elapsed < 10


async def test_excess_output_is_truncated_to_limit(workspace: Workspace) -> None:
    limit = 4096

    result = await _run(workspace.profile(), "yes localmcp | head -c 1000000", max_output_bytes=limit)

    assert result.truncated is True
    assert result.exit_code == -1
    assert len(result.stdout.encode()) == limit
    assert result.stdout == ("localmcp\n" * limit)[:limit]
    assert "command output exceeded its bounded safety limit" in result.stderr
