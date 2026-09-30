import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from localmcp.sandbox import (
    MAX_COMMAND_CHARACTERS,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
    SandboxRoot,
    seatbelt,
)
from localmcp.sandbox.seatbelt import MacOSSandbox, _developer_directory, _git_exec_path


class _Process:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.pid = 1234
        self.stdout: asyncio.StreamReader | None = None
        self.stderr: asyncio.StreamReader | None = None
        self.communicated = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicated = True
        return b"", b""

    async def wait(self) -> int:
        self.waited = True
        return self.returncode or 0


class _Stream:
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = iter((*chunks, b""))

    async def read(self, _size: int) -> bytes:
        return next(self._chunks)


class _Budget:
    def __init__(self, *claims: tuple[bool, str | None]) -> None:
        self._claims = iter(claims)

    async def claim(self) -> tuple[bool, str | None]:
        return next(self._claims)

    def metadata(self, exhausted_scope: str | None = None) -> dict[str, object]:
        return {"scope": exhausted_scope or "available"}


def test_profile_rejects_read_only_root_nested_in_writable_root(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    with pytest.raises(SandboxError, match="cannot be enforced"):
        MacOSSandbox(
            SandboxProfile(
                (
                    SandboxRoot(tmp_path, RootAccess.READ_WRITE),
                    SandboxRoot(nested, RootAccess.READ_ONLY),
                )
            )
        )


def test_profile_normalizes_and_deduplicates_nested_roots(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(tmp_path, target_is_directory=True)

    sandbox = MacOSSandbox(
        SandboxProfile(
            (
                SandboxRoot(root_alias),
                SandboxRoot(tmp_path),
                SandboxRoot(nested),
            )
        )
    )

    assert sandbox.roots == (SandboxRoot(tmp_path.resolve()),)
    assert sandbox.root == tmp_path.resolve()


def test_denied_path_must_be_inside_a_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"

    with pytest.raises(SandboxError, match="outside"):
        MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),), denied_paths=(outside,)))


def test_compiled_profile_emits_denies_after_every_allow(tmp_path: Path) -> None:
    first_denied = tmp_path / "credentials"
    second_denied = tmp_path / "private" / "token"
    sandbox = MacOSSandbox(
        SandboxProfile(
            (SandboxRoot(tmp_path, RootAccess.READ_WRITE),),
            denied_paths=(first_denied, second_denied),
            network=True,
            ipc=True,
        )
    )

    profile_lines = sandbox._compiled_profile().splitlines()
    allow_positions = [index for index, line in enumerate(profile_lines) if line.startswith("(allow ")]
    denied_positions = [index for index, line in enumerate(profile_lines) if line.startswith("(deny file-read*")]

    assert allow_positions
    assert len(denied_positions) == 2
    assert min(denied_positions) > max(allow_positions)


def test_profile_keeps_credentials_out_of_environment(tmp_path: Path) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)))

    environment = sandbox._environment()
    assert environment["HOME"] == "/var/empty"
    assert "SSH_AUTH_SOCK" not in environment
    assert "ANTHROPIC_API_KEY" not in environment


def test_profile_requires_a_root() -> None:
    with pytest.raises(SandboxError, match="at least one root"):
        SandboxProfile(())


def test_profile_rejects_missing_file_and_conflicting_roots(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(SandboxError, match="does not exist"):
        MacOSSandbox(SandboxProfile((SandboxRoot(missing),)))

    regular_file = tmp_path / "file"
    regular_file.write_text("data")
    with pytest.raises(SandboxError, match="not a directory"):
        MacOSSandbox(SandboxProfile((SandboxRoot(regular_file),)))

    with pytest.raises(SandboxError, match="conflicting access grants"):
        MacOSSandbox(
            SandboxProfile(
                (
                    SandboxRoot(tmp_path, RootAccess.READ_ONLY),
                    SandboxRoot(tmp_path, RootAccess.READ_WRITE),
                )
            )
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 0}, "timeout must be positive"),
        ({"max_output_bytes": 0}, "output limit must be positive"),
        ({"process_label": ""}, "process label"),
        ({"process_label": "bad\0label"}, "process label"),
    ],
)
def test_constructor_rejects_unsafe_limits_and_labels(tmp_path: Path, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(SandboxError, match=message):
        MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)), **kwargs)


def test_denied_paths_are_normalized_and_deduplicated(tmp_path: Path) -> None:
    denied = tmp_path / "private" / ".." / "secret"
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),), denied_paths=(denied, denied)))

    assert sandbox.profile.denied_paths == (tmp_path / "secret",)


def test_nested_parent_root_replaces_redundant_descendant(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(nested), SandboxRoot(tmp_path))))

    # The first root is always retained because it defines cwd; the later parent is
    # also retained so its broader authority is represented explicitly.
    assert sandbox.roots == (SandboxRoot(nested), SandboxRoot(tmp_path))


def test_runtime_paths_include_absolute_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "tool"
    dependency = tmp_path / "library.dylib"
    executable.touch()
    dependency.touch()
    seatbelt._runtime_read_paths.cache_clear()
    monkeypatch.setattr(
        seatbelt.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=f"{executable}:\n\t{dependency} (compatibility)\n\t@rpath/ignored.dylib\n"
        ),
    )

    assert seatbelt._runtime_read_paths(executable) == tuple(sorted((executable, dependency)))
    seatbelt._runtime_read_paths.cache_clear()


@pytest.mark.parametrize("outcome", ["error", "nonzero"])
def test_runtime_paths_fall_back_to_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str) -> None:
    executable = tmp_path / outcome
    executable.touch()
    seatbelt._runtime_read_paths.cache_clear()

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if outcome == "error":
            raise OSError("otool unavailable")
        return subprocess.CompletedProcess(args[0], 1, stdout="")

    monkeypatch.setattr(seatbelt.subprocess, "run", run)
    assert seatbelt._runtime_read_paths(executable) == (executable,)
    seatbelt._runtime_read_paths.cache_clear()


def test_runtime_paths_ignore_missing_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "tool"
    executable.touch()
    missing = tmp_path / "missing.dylib"
    seatbelt._runtime_read_paths.cache_clear()
    monkeypatch.setattr(
        seatbelt.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=f"{executable}:\n\t{missing} (compatibility)\n"
        ),
    )

    assert set(seatbelt._runtime_read_paths(executable)) == {executable, missing}
    seatbelt._runtime_read_paths.cache_clear()


def test_runtime_directories_include_symlink_traversal(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    executable = link / "bin" / "tool"
    seatbelt._runtime_read_directories.cache_clear()

    directories = seatbelt._runtime_read_directories((executable,))

    assert executable.parent in directories
    assert link in directories
    seatbelt._runtime_read_directories.cache_clear()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("error", None),
        ("nonzero", None),
        ("none-owned", None),
        ("finite", 12),
        ("exhausted", None),
        ("infinite", 258),
    ],
)
def test_owned_process_limit_handles_platform_results(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected: int | None
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if mode == "error":
            raise subprocess.SubprocessError
        return subprocess.CompletedProcess(
            args[0],
            1 if mode == "nonzero" else 0,
            stdout="99999\n" if mode == "none-owned" else f"{seatbelt.os.getuid()}\n{seatbelt.os.getuid()}\n",
        )

    monkeypatch.setattr(seatbelt.subprocess, "run", run)
    if mode == "finite":
        monkeypatch.setattr(seatbelt.resource, "getrlimit", lambda _resource: (12, 12))
    elif mode == "exhausted":
        monkeypatch.setattr(seatbelt.resource, "getrlimit", lambda _resource: (2, 2))
    else:
        monkeypatch.setattr(seatbelt.resource, "getrlimit", lambda _resource: (seatbelt.resource.RLIM_INFINITY,) * 2)

    assert seatbelt._owned_process_limit() == expected


@pytest.mark.asyncio
async def test_run_validates_command_and_builds_bounded_shell_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)), timeout_seconds=0.5)
    sandbox._process_limit = 42

    with pytest.raises(SandboxError, match="blank"):
        await sandbox.run(" \t")
    with pytest.raises(SandboxError, match="at most"):
        await sandbox.run("bad\0command")
    with pytest.raises(SandboxError, match="at most"):
        await sandbox.run("x" * (MAX_COMMAND_CHARACTERS + 1))

    captured: list[str] = []

    async def execute(argv: list[str]) -> CommandResult:
        captured.extend(argv)
        return CommandResult(exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox, "_execute", execute)
    result = await sandbox.run("printf ok")

    assert result.stdout == "ok"
    assert "ulimit -t 1 && ulimit -n 256 && ulimit -u 42" in captured[2]
    assert captured[-2:] == [sandbox.process_label, "printf ok"]

    sandbox._process_limit = None
    captured.clear()
    await sandbox.run("true")
    assert "ulimit -u" not in captured[2]


def test_environment_prepends_only_nonstandard_executable_directories(tmp_path: Path) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)))
    sandbox._executables = (Path("/custom/bin/rg"), Path("/custom/bin/git"), Path("/usr/bin/git"))

    assert sandbox._environment()["PATH"] == "/custom/bin:/usr/bin:/bin:/usr/sbin:/sbin"


@pytest.mark.asyncio
async def test_execute_fails_closed_when_seatbelt_is_unavailable(tmp_path: Path) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)))
    sandbox._sandbox_exec = tmp_path / "missing-sandbox-exec"

    with pytest.raises(SandboxError, match="unavailable"):
        await sandbox._execute(["command"])


async def _prepare_execute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, read_output: Any, *, returncode: int | None = 0
) -> tuple[MacOSSandbox, _Process, dict[str, Any]]:
    sandbox = MacOSSandbox(
        SandboxProfile(
            (SandboxRoot(tmp_path, RootAccess.READ_WRITE),),
            denied_paths=(tmp_path / "denied",),
        ),
        timeout_seconds=0.01,
    )
    sandbox_exec = tmp_path / "sandbox-exec"
    sandbox_exec.touch()
    sandbox._sandbox_exec = sandbox_exec
    sandbox._runtime_paths = (tmp_path / "runtime",)
    sandbox._runtime_directories = (tmp_path / "runtime-dir",)
    sandbox._helper_directories = ()
    sandbox._metadata_ancestors = (tmp_path.parent,)
    process = _Process(returncode=returncode)
    captured: dict[str, Any] = {}

    async def create(*argv: str, **kwargs: Any) -> _Process:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(sandbox, "_read_output", read_output)
    return sandbox, process, captured


@pytest.mark.asyncio
async def test_execute_passes_complete_policy_and_returns_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def read_output(process: _Process, stdout: bytearray, stderr: bytearray) -> None:
        stdout.extend(b"output")
        stderr.extend(b"warning")

    sandbox, _process, captured = await _prepare_execute(monkeypatch, tmp_path, read_output, returncode=7)
    result = await sandbox._execute(["/bin/sh", "-c", "exit 7"])

    assert result == CommandResult(exit_code=7, stdout="output", stderr="warning")
    argv = captured["argv"]
    assert f"ROOT_0={tmp_path}" in argv
    assert f"DENIED_PATH_0={tmp_path / 'denied'}" in argv
    assert f"RUNTIME_PATH_0={tmp_path / 'runtime'}" in argv
    assert f"RUNTIME_DIRECTORY_0={tmp_path / 'runtime-dir'}" in argv
    assert f"METADATA_ANCESTOR_0={tmp_path.parent}" in argv
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"]["HOME"] == "/var/empty"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "flag", "message"),
    [
        (TimeoutError(), "timed_out", "command timed out"),
        (
            seatbelt._OutputLimitExceeded(),
            "truncated",
            "command output exceeded its bounded safety limit",
        ),
    ],
)
async def test_execute_kills_process_for_timeout_and_output_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
    flag: str,
    message: str,
) -> None:
    async def read_output(process: _Process, stdout: bytearray, stderr: bytearray) -> None:
        stdout.extend(b"partial")
        stderr.extend(b"existing\n")
        raise failure

    sandbox, process, _captured = await _prepare_execute(monkeypatch, tmp_path, read_output, returncode=None)
    killed: list[_Process] = []

    async def kill(target: _Process) -> None:
        killed.append(target)

    monkeypatch.setattr(sandbox, "_kill", kill)
    result = await sandbox._execute(["command"])

    assert result.exit_code == -1
    assert result.stdout == "partial"
    assert getattr(result, flag) is True
    assert result.stderr == f"existing\n{message}"
    assert killed == [process]


@pytest.mark.asyncio
async def test_execute_kills_and_propagates_unexpected_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def read_output(process: _Process, stdout: bytearray, stderr: bytearray) -> None:
        raise RuntimeError("reader failed")

    sandbox, process, _captured = await _prepare_execute(monkeypatch, tmp_path, read_output, returncode=None)
    killed: list[_Process] = []

    async def kill(target: _Process) -> None:
        killed.append(target)

    monkeypatch.setattr(sandbox, "_kill", kill)
    with pytest.raises(RuntimeError, match="reader failed"):
        await sandbox._execute(["command"])
    assert killed == [process]


@pytest.mark.asyncio
async def test_read_output_collects_streams_and_waits(tmp_path: Path) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)), max_output_bytes=20)
    process = _Process(returncode=0)
    process.stdout = _Stream(b"out", b"put")  # type: ignore[assignment]
    process.stderr = _Stream(b"err")  # type: ignore[assignment]
    stdout = bytearray()
    stderr = bytearray()

    await sandbox._read_output(process, stdout, stderr)  # type: ignore[arg-type]

    assert stdout == b"output"
    assert stderr == b"err"
    assert process.waited is True


@pytest.mark.asyncio
async def test_read_output_enforces_combined_limit_and_cancels_readers(tmp_path: Path) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)), max_output_bytes=4)
    process = _Process(returncode=None)
    process.stdout = _Stream(b"12345")  # type: ignore[assignment]
    stdout = bytearray()
    stderr = bytearray()

    with pytest.raises(seatbelt._OutputLimitExceeded):
        await sandbox._read_output(process, stdout, stderr)  # type: ignore[arg-type]
    assert stdout == b"1234"

    process.stdout = _Stream(b"1234", b"5")  # type: ignore[assignment]
    stdout.clear()
    with pytest.raises(seatbelt._OutputLimitExceeded):
        await sandbox._read_output(process, stdout, stderr)  # type: ignore[arg-type]
    assert stdout == b"1234"


@pytest.mark.asyncio
async def test_kill_terminates_process_group_and_tolerates_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = _Process(returncode=None)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(seatbelt.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    await MacOSSandbox._kill(running)  # type: ignore[arg-type]
    assert killed == [(running.pid, seatbelt.signal.SIGKILL)]
    assert running.communicated is True

    raced = _Process(returncode=None)

    def missing_process(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(seatbelt.os, "killpg", missing_process)
    await MacOSSandbox._kill(raced)  # type: ignore[arg-type]
    assert raced.communicated is True

    finished = _Process(returncode=0)
    monkeypatch.setattr(seatbelt.os, "killpg", lambda _pid, _sig: pytest.fail("must not kill a completed process"))
    await MacOSSandbox._kill(finished)  # type: ignore[arg-type]
    assert finished.communicated is True


@pytest.mark.asyncio
async def test_sandbox_tool_reports_budget_success_and_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results = iter(
        (
            CommandResult(exit_code=0, stdout="ok", stderr=""),
            SandboxError("unavailable"),
        )
    )

    async def run(_self: MacOSSandbox, _command: str) -> CommandResult:
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(MacOSSandbox, "run", run)
    budget = _Budget((False, "operation"), (True, None), (True, None))
    tool = seatbelt.sandbox_tools(
        SandboxProfile((SandboxRoot(tmp_path),)),
        budget=budget,
        exhausted_message="done",
        description="bounded bash",
    )[0]

    exhausted = await tool.ainvoke({"command": "first"})
    success = await tool.ainvoke({"command": "second"})
    failure = await tool.ainvoke({"command": "third"})

    assert tool.description == "bounded bash"
    assert exhausted == {"error": "done", "tool_budget": {"scope": "operation"}}
    assert success["result"] == CommandResult(exit_code=0, stdout="ok", stderr="").model_dump()
    assert success["tool_budget"] == {"scope": "available"}
    assert failure == {
        "error": "Bash failed with SandboxError: unavailable",
        "tool_budget": {"scope": "available"},
    }


def _fake_xcode(tmp_path: Path) -> Path:
    developer = tmp_path / "Xcode.app" / "Contents" / "Developer"
    (developer / "usr" / "bin").mkdir(parents=True)
    (developer / "Toolchains" / "XcodeDefault.xctoolchain" / "usr" / "bin").mkdir(parents=True)
    return developer


def test_developer_directory_prefers_developer_dir_and_accepts_an_app_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    developer = _fake_xcode(tmp_path)
    monkeypatch.setattr(seatbelt.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not probe"))
    _developer_directory.cache_clear()

    assert _developer_directory(str(developer)) == developer.resolve()
    assert _developer_directory(str(tmp_path / "Xcode.app")) == developer.resolve()
    assert _developer_directory(str(tmp_path / "missing")) is None
    assert _developer_directory(str(tmp_path)) is None
    _developer_directory.cache_clear()


@pytest.mark.parametrize("outcome", ["selected", "unselected", "error"])
def test_developer_directory_falls_back_to_xcode_select(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str
) -> None:
    developer = _fake_xcode(tmp_path)
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if outcome == "error":
            raise OSError("missing")
        return subprocess.CompletedProcess(argv, 0 if outcome == "selected" else 2, f"{developer}\n", "")

    monkeypatch.setattr(seatbelt.subprocess, "run", run)
    _developer_directory.cache_clear()

    assert _developer_directory(None) == (developer.resolve() if outcome == "selected" else None)
    # Never xcrun or a /usr/bin shim: those can raise the Command Line Tools install prompt.
    assert calls == [["/usr/bin/xcode-select", "-p"]]
    _developer_directory.cache_clear()


def test_dev_tools_grants_the_xcode_bundle_and_license(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    developer = _fake_xcode(tmp_path).resolve()
    monkeypatch.setattr(seatbelt, "_developer_directory", lambda _value: developer)

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),), dev_tools=True))

    assert sandbox.profile.dev_tools is True
    assert sandbox._developer_installation == developer.parent.parent
    assert sandbox._developer_preferences == (seatbelt._XCODE_LICENSE,)
    assert set(developer.parent.parent.parents) - {Path("/")} <= set(sandbox._metadata_ancestors)
    profile = sandbox._compiled_profile()
    assert '(allow file-read* file-map-executable (subpath (param "DEVELOPER_INSTALLATION")))' in profile
    assert '(allow file-read* (literal (param "DEVELOPER_PREFERENCE_0")))' in profile
    sandbox._executables = ()
    environment = sandbox._environment()
    assert environment["DEVELOPER_DIR"] == str(developer)
    toolchain = developer / "Toolchains" / "XcodeDefault.xctoolchain"
    assert environment["PATH"] == f"{developer}/usr/bin:{toolchain}/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def test_dev_tools_grants_a_command_line_tools_install_as_is(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    developer = tmp_path / "CommandLineTools"
    (developer / "usr" / "bin").mkdir(parents=True)
    monkeypatch.setattr(seatbelt, "_developer_directory", lambda _value: developer)

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),), dev_tools=True))

    assert sandbox._developer_installation == developer
    assert sandbox._developer_preferences == ()
    assert "DEVELOPER_PREFERENCE" not in sandbox._compiled_profile()


def test_dev_tools_rejects_a_toolchain_overlapping_a_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    developer = _fake_xcode(tmp_path).resolve()
    monkeypatch.setattr(seatbelt, "_developer_directory", lambda _value: developer)

    for root in (tmp_path, developer / "usr"):
        with pytest.raises(SandboxError, match="overlaps a sandbox root"):
            MacOSSandbox(SandboxProfile((SandboxRoot(root),), dev_tools=True))


def test_toolchain_is_not_granted_without_dev_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(seatbelt, "_developer_directory", lambda _value: pytest.fail("must not look up"))

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)))

    assert sandbox._developer_installation is None
    assert "DEVELOPER_" not in sandbox._compiled_profile()
    assert "DEVELOPER_DIR" not in sandbox._environment()


def test_git_exec_path_never_runs_the_usr_bin_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seatbelt.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not run the shim"))
    _git_exec_path.cache_clear()

    assert _git_exec_path(Path("/usr/bin/git")) is None
    _git_exec_path.cache_clear()


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [(0, "/opt/git/libexec/git-core\n", Path("/opt/git/libexec/git-core")), (1, "", None), (0, "relative\n", None)],
)
def test_git_exec_path_reports_the_compiled_helper_directory(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str, expected: Path | None
) -> None:
    monkeypatch.setattr(
        seatbelt.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, returncode, stdout, ""),
    )
    _git_exec_path.cache_clear()

    assert _git_exec_path(Path("/opt/git/bin/git")) == expected
    _git_exec_path.cache_clear()


def test_git_helpers_behind_a_symlink_are_granted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Homebrew-style: git's exec path runs through opt/git -> ../Cellar/git/<version>.
    prefix = tmp_path.resolve() / "prefix"
    cellar = prefix / "Cellar" / "git" / "2.0"
    (cellar / "bin").mkdir(parents=True)
    (cellar / "libexec" / "git-core").mkdir(parents=True)
    git = cellar / "bin" / "git"
    git.touch()
    (prefix / "opt").mkdir()
    (prefix / "opt" / "git").symlink_to(cellar, target_is_directory=True)
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(seatbelt.shutil, "which", lambda name: str(git) if name == "git" else None)
    monkeypatch.setattr(seatbelt, "_git_exec_path", lambda _git: prefix / "opt" / "git" / "libexec" / "git-core")

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),)))

    assert sandbox._helper_directories == (cellar / "libexec" / "git-core",)
    assert {prefix / "opt", prefix / "opt" / "git"} <= set(sandbox._metadata_ancestors)
    assert 'file-map-executable (subpath (param "HELPER_DIRECTORY_0"))' in sandbox._compiled_profile()
