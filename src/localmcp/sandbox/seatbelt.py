"""macOS Seatbelt execution for model-controlled commands."""

from __future__ import annotations

import asyncio
import os
import resource
import shutil
import signal
import subprocess
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from localmcp.sandbox.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_MAX_OUTPUT_BYTES,
    MAX_ADDITIONAL_PROCESSES,
    MAX_COMMAND_CHARACTERS,
    MAX_OPEN_FILES,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
    SandboxRoot,
)

DEFAULT_EXHAUSTED_MESSAGE = "Source-tool budget exhausted. Produce the final structured response now."
DEFAULT_TOOL_DESCRIPTION = (
    "Run a command in the policy-bound filesystem roots. Git and ordinary inspection utilities are available. "
    "No credentials or caller environment are inherited."
)

_BASE_PROFILE = """\
(version 1)
(deny default)
(import "system.sb")
(allow process-fork)
(allow process-exec)
(allow file-read*
    (subpath "/bin")
    (subpath "/usr/bin")
    (subpath "/usr/lib")
    (subpath "/System")
    (subpath "/Library/Apple")
    (subpath "/private/var/select"))
"""
_NETWORK_PROFILE = "(allow network-outbound)\n"
_SHARED_MEMORY_IPC_PROFILE = "(allow ipc-posix-shm ipc-sysv-shm)\n"


def _validated_roots(roots: tuple[SandboxRoot, ...]) -> tuple[SandboxRoot, ...]:
    resolved: list[SandboxRoot] = []
    by_path: dict[Path, RootAccess] = {}
    for root in roots:
        try:
            path = root.path.resolve(strict=True)
        except OSError as exc:
            raise SandboxError(f"sandbox root does not exist: {root.path}") from exc
        if not path.is_dir():
            raise SandboxError(f"sandbox root is not a directory: {root.path}")
        previous = by_path.get(path)
        if previous is not None:
            if previous != root.access:
                raise SandboxError(f"sandbox root has conflicting access grants: {path}")
            continue
        by_path[path] = root.access
        resolved.append(SandboxRoot(path, root.access))

    for ancestor in resolved:
        if ancestor.access != RootAccess.READ_WRITE:
            continue
        for descendant in resolved:
            if descendant.access == RootAccess.READ_ONLY and descendant.path.is_relative_to(ancestor.path):
                raise SandboxError(
                    f"read-only root is contained by a read-write root and cannot be enforced: {descendant.path}"
                )

    normalized: list[SandboxRoot] = []
    for index, root in enumerate(resolved):
        if index != 0 and any(
            root.access == existing.access and root.path.is_relative_to(existing.path) for existing in normalized
        ):
            continue
        normalized = [
            existing
            for existing_index, existing in enumerate(normalized)
            if existing_index == 0 or existing.access != root.access or not existing.path.is_relative_to(root.path)
        ]
        normalized.append(root)
    return tuple(normalized)


@lru_cache(maxsize=8)
def _runtime_read_paths(executable: Path) -> tuple[Path, ...]:
    paths = {executable}
    try:
        result = subprocess.run(
            ["/usr/bin/otool", "-L", os.fspath(executable)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return tuple(paths)
    if result.returncode != 0:
        return tuple(paths)
    for line in result.stdout.splitlines()[1:]:
        raw = line.strip().split(" ", 1)[0]
        if not raw.startswith("/"):
            continue
        dependency = Path(raw)
        paths.add(dependency)
        try:
            paths.add(dependency.resolve(strict=True))
        except OSError:
            continue
    return tuple(sorted(paths, key=os.fspath))


@lru_cache(maxsize=8)
def _runtime_read_directories(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    directories = {path.parent for path in paths}
    for path in paths:
        current = Path(path.anchor)
        for part in path.parts[1:-1]:
            current /= part
            if current.is_symlink():
                directories.add(current)
    return tuple(sorted(directories, key=os.fspath))


# Xcode records license acceptance here; the /usr/bin shims refuse to run without reading it.
_XCODE_LICENSE = Path("/Library/Preferences/com.apple.dt.Xcode.plist")


@lru_cache(maxsize=4)
def _developer_directory(developer_dir: str | None) -> Path | None:
    """Return the selected developer directory, or ``None`` when none is usable.

    ``developer_dir`` is the caller's ``DEVELOPER_DIR`` (part of the cache key, so changing
    it is honored). Otherwise the selection comes from ``xcode-select -p``. Neither can
    raise the Command Line Tools install prompt, unlike ``xcrun`` or a ``/usr/bin`` shim.
    """
    value = developer_dir
    if not value:
        try:
            result = subprocess.run(
                ["/usr/bin/xcode-select", "-p"], check=False, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
    if not value:
        return None
    try:
        path = Path(value).resolve(strict=True)
    except OSError:
        return None
    # Like xcrun, accept an Xcode.app bundle path for its Contents/Developer.
    if path.suffix == ".app" and (path / "Contents" / "Developer").is_dir():
        path = path / "Contents" / "Developer"
    return path if (path / "usr" / "bin").is_dir() else None


def _developer_installation(developer: Path) -> Path:
    """Return the tree the toolchain reads from: the whole Xcode.app, or the directory itself.

    Inside Xcode.app the shims also read Contents/Info.plist and load Contents/SharedFrameworks,
    so granting only Contents/Developer is not enough. A Command Line Tools install is self-contained.
    """
    bundle = developer.parent.parent
    if developer.name == "Developer" and developer.parent.name == "Contents" and bundle.suffix == ".app":
        return bundle
    return developer


def _developer_bin_directories(developer: Path) -> tuple[Path, ...]:
    """Return the toolchain's own bin directories, so commands skip the slow, noisy /usr/bin shims."""
    candidates = (
        developer / "usr" / "bin",
        developer / "Toolchains" / "XcodeDefault.xctoolchain" / "usr" / "bin",
    )
    return tuple(candidate for candidate in candidates if candidate.is_dir())


@lru_cache(maxsize=8)
def _git_exec_path(git: Path) -> Path | None:
    """Return the helper directory ``git`` reports, unresolved (it can run through symlinks).

    The /usr/bin shim is never run here: without Command Line Tools it can raise an install prompt.
    """
    if git.is_relative_to("/usr/bin"):
        return None
    try:
        result = subprocess.run(
            [os.fspath(git), "--exec-path"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    if result.returncode != 0 or not value.startswith("/"):
        return None
    return Path(value)


def _path_ancestors(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    ancestors: set[Path] = set()
    for path in paths:
        ancestors.update(parent for parent in path.parents if parent != Path(path.anchor))
    return tuple(sorted(ancestors, key=os.fspath))


def _owned_process_limit() -> int | None:
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "uid="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    owned = sum(line.strip() == str(os.getuid()) for line in result.stdout.splitlines())
    if owned == 0:
        return None
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    proposed = owned + MAX_ADDITIONAL_PROCESSES
    if soft_limit != resource.RLIM_INFINITY:
        proposed = min(proposed, soft_limit)
    return proposed if proposed > owned else None


class _OutputLimitExceeded(Exception):
    pass


class MacOSSandbox:
    """Run a command with no caller environment under immutable Seatbelt authority."""

    def __init__(
        self,
        profile: SandboxProfile,
        *,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        process_label: str = "localmcp-workspace-sandbox",
    ):
        if timeout_seconds <= 0:
            raise SandboxError("sandbox timeout must be positive")
        if max_output_bytes <= 0:
            raise SandboxError("sandbox output limit must be positive")
        if not process_label or "\0" in process_label:
            raise SandboxError("sandbox process label must not be blank or contain NUL bytes")
        self.roots = _validated_roots(profile.roots)
        self.root = self.roots[0].path
        denied_paths: list[Path] = []
        for denied in profile.denied_paths:
            path = Path(os.path.abspath(denied))
            if not any(path.is_relative_to(root.path) for root in self.roots):
                raise SandboxError("sandbox denied path is outside its declared roots")
            if path not in denied_paths:
                denied_paths.append(path)
        self.profile = SandboxProfile(
            self.roots,
            denied_paths=tuple(denied_paths),
            network=profile.network,
            ipc=profile.ipc,
            dev_tools=profile.dev_tools,
        )
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.process_label = process_label
        self._sandbox_exec = Path("/usr/bin/sandbox-exec")
        self._shell = Path("/bin/sh")
        executables = {
            name: Path(value).resolve(strict=True)
            for name in ("rg", "git")
            if (value := shutil.which(name)) is not None
        }
        self._executables = tuple(executables.values())
        self._runtime_paths = tuple(
            sorted(
                {path for executable in self._executables for path in _runtime_read_paths(executable)}, key=os.fspath
            )
        )
        self._runtime_directories = _runtime_read_directories(self._runtime_paths)
        # git dispatches subcommands (clone, ls-remote, ...) by exec'ing helpers from its exec
        # path, which can sit behind a symlink (Homebrew's opt/git); metadata on the unresolved
        # path lets the kernel follow it, and the resolved directory needs read + exec-map.
        git_exec_path = _git_exec_path(executables["git"]) if "git" in executables else None
        self._helper_directories: tuple[Path, ...] = ()
        if git_exec_path is not None:
            try:
                self._helper_directories = (git_exec_path.resolve(strict=True),)
            except OSError:
                git_exec_path = None
        self._developer = _developer_directory(os.environ.get("DEVELOPER_DIR")) if profile.dev_tools else None
        self._developer_installation: Path | None = None
        self._developer_preferences: tuple[Path, ...] = ()
        if self._developer is not None:
            installation = _developer_installation(self._developer)
            if any(
                installation.is_relative_to(root.path) or root.path.is_relative_to(installation) for root in self.roots
            ):
                raise SandboxError(f"developer directory overlaps a sandbox root: {installation}")
            self._developer_installation = installation
            if installation != self._developer:
                self._developer_preferences = (_XCODE_LICENSE,)
        self._process_limit = _owned_process_limit()
        # Grant metadata reads along the resolved toolchain paths so deep developer-dir
        # installs (e.g. inside Xcode.app) stay traversable without opening whole trees.
        self._metadata_ancestors = _path_ancestors(
            tuple(root.path for root in self.roots)
            + self._executables
            + self._helper_directories
            + ((git_exec_path,) if git_exec_path is not None else ())
            + ((self._developer_installation,) if self._developer_installation is not None else ())
            + self._developer_preferences
        )
        # Renaming a denied path, or any directory above it inside a root,
        # would move its contents out from under the read denial.
        self._denied_ancestors = tuple(
            ancestor
            for ancestor in _path_ancestors(self.profile.denied_paths)
            if any(ancestor.is_relative_to(root.path) for root in self.roots)
        )

    async def run(self, command: str) -> CommandResult:
        if not command.strip():
            raise SandboxError("command must not be blank")
        if "\0" in command or len(command) > MAX_COMMAND_CHARACTERS:
            raise SandboxError(f"command must contain at most {MAX_COMMAND_CHARACTERS} characters and no NUL bytes")
        limits = [
            f"ulimit -t {max(1, int(self.timeout_seconds))}",
            f"ulimit -n {MAX_OPEN_FILES}",
        ]
        if self._process_limit is not None:
            limits.append(f"ulimit -u {self._process_limit}")
        return await self._execute(
            [
                os.fspath(self._shell),
                "-c",
                f'{" && ".join(limits)} && exec /bin/sh -c "$1"',
                self.process_label,
                command,
            ],
        )

    async def _execute(self, argv: list[str]) -> CommandResult:
        if not self._sandbox_exec.is_file():
            raise SandboxError("macOS sandbox-exec is unavailable")
        runtime_profile = self._compiled_profile()
        runtime_definitions = [
            value for index, path in enumerate(self._runtime_paths) for value in ("-D", f"RUNTIME_PATH_{index}={path}")
        ]
        runtime_definitions.extend(
            value
            for index, path in enumerate(self._runtime_directories)
            for value in ("-D", f"RUNTIME_DIRECTORY_{index}={path}")
        )
        runtime_definitions.extend(
            value
            for index, path in enumerate(self._helper_directories)
            for value in ("-D", f"HELPER_DIRECTORY_{index}={path}")
        )
        if self._developer_installation is not None:
            runtime_definitions.extend(("-D", f"DEVELOPER_INSTALLATION={self._developer_installation}"))
        runtime_definitions.extend(
            value
            for index, path in enumerate(self._developer_preferences)
            for value in ("-D", f"DEVELOPER_PREFERENCE_{index}={path}")
        )
        runtime_definitions.extend(
            value
            for index, path in enumerate(self._metadata_ancestors)
            for value in ("-D", f"METADATA_ANCESTOR_{index}={path}")
        )
        root_definitions = [
            value for index, root in enumerate(self.roots) for value in ("-D", f"ROOT_{index}={root.path}")
        ]
        denied_path_definitions = [
            value
            for index, path in enumerate(self.profile.denied_paths)
            for value in ("-D", f"DENIED_PATH_{index}={path}")
        ]
        denied_path_definitions.extend(
            value
            for index, path in enumerate(self._denied_ancestors)
            for value in ("-D", f"DENIED_ANCESTOR_{index}={path}")
        )
        process = await asyncio.create_subprocess_exec(
            os.fspath(self._sandbox_exec),
            *root_definitions,
            *denied_path_definitions,
            *runtime_definitions,
            "-p",
            runtime_profile,
            *argv,
            cwd=self.root,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._environment(),
            start_new_session=True,
        )
        stdout = bytearray()
        stderr = bytearray()
        try:
            await asyncio.wait_for(self._read_output(process, stdout, stderr), timeout=self.timeout_seconds)
        except TimeoutError:
            await self._kill(process)
            return CommandResult(
                exit_code=-1,
                stdout=stdout.decode(errors="replace"),
                stderr=self._append_error(stderr, "command timed out"),
                timed_out=True,
            )
        except _OutputLimitExceeded:
            await self._kill(process)
            return CommandResult(
                exit_code=-1,
                stdout=stdout.decode(errors="replace"),
                stderr=self._append_error(stderr, "command output exceeded its bounded safety limit"),
                truncated=True,
            )
        except BaseException:
            await self._kill(process)
            raise
        return CommandResult(
            exit_code=process.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    def _compiled_profile(self) -> str:
        runtime_profile = _BASE_PROFILE
        runtime_profile += "".join(
            f'(allow file-read* (subpath (param "ROOT_{index}")))\n' for index in range(len(self.roots))
        )
        runtime_profile += "".join(
            f'(allow file-write* (subpath (param "ROOT_{index}")))\n'
            for index, root in enumerate(self.roots)
            if root.access == RootAccess.READ_WRITE
        )
        if self.profile.network:
            runtime_profile += _NETWORK_PROFILE
        if self.profile.ipc:
            runtime_profile += _SHARED_MEMORY_IPC_PROFILE
        runtime_profile += "".join(
            f'(allow file-read* file-map-executable (literal (param "RUNTIME_PATH_{index}")))\n'
            for index in range(len(self._runtime_paths))
        )
        runtime_profile += "".join(
            f'(allow file-read* (subpath (param "RUNTIME_DIRECTORY_{index}")))\n'
            for index in range(len(self._runtime_directories))
        )
        # Helper directories hold binaries the tool re-execs (git's git-core), so they need
        # exec-mapping too — a plain read grant would let git start but deny dispatched helpers.
        runtime_profile += "".join(
            f'(allow file-read* file-map-executable (subpath (param "HELPER_DIRECTORY_{index}")))\n'
            for index in range(len(self._helper_directories))
        )
        if self._developer_installation is not None:
            runtime_profile += '(allow file-read* file-map-executable (subpath (param "DEVELOPER_INSTALLATION")))\n'
        runtime_profile += "".join(
            f'(allow file-read* (literal (param "DEVELOPER_PREFERENCE_{index}")))\n'
            for index in range(len(self._developer_preferences))
        )
        runtime_profile += "".join(
            f'(allow file-read-metadata (literal (param "METADATA_ANCESTOR_{index}")))\n'
            for index in range(len(self._metadata_ancestors))
        )
        runtime_profile += "".join(
            f'(deny file-read* (literal (param "DENIED_PATH_{index}")) (subpath (param "DENIED_PATH_{index}")))\n'
            for index in range(len(self.profile.denied_paths))
        )
        runtime_profile += "".join(
            f'(deny file-write-unlink (literal (param "DENIED_PATH_{index}")) '
            f'(subpath (param "DENIED_PATH_{index}")))\n'
            for index in range(len(self.profile.denied_paths))
        )
        runtime_profile += "".join(
            f'(deny file-write-unlink (literal (param "DENIED_ANCESTOR_{index}")))\n'
            for index in range(len(self._denied_ancestors))
        )
        return runtime_profile

    def _environment(self) -> dict[str, str]:
        paths = ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        environment: dict[str, str] = {}
        if self._developer is not None:
            # Pin the selection the grant was computed for, and put the toolchain ahead of the
            # /usr/bin shims, which re-resolve it on every call (slow, and noisy in a sandbox).
            environment["DEVELOPER_DIR"] = os.fspath(self._developer)
            paths[:0] = [os.fspath(path) for path in _developer_bin_directories(self._developer)]
        for executable in self._executables:
            if executable.parent not in {Path(path) for path in paths}:
                paths.insert(0, os.fspath(executable.parent))
        return environment | {
            "HOME": "/var/empty",
            "PATH": os.pathsep.join(paths),
            "LC_ALL": "C",
            "RIPGREP_CONFIG_PATH": "",
            "TMPDIR": os.fspath(self.root),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            # Apple's git ignores GIT_CONFIG_SYSTEM for its bundled share/git-core/gitconfig
            # (full of osxkeychain helpers); NOSYSTEM neutralizes it and keeps git hermetic.
            "GIT_CONFIG_NOSYSTEM": "1",
            # Likewise skip the bundled share/git-core/gitattributes the sandbox can't read.
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }

    async def _read_output(
        self,
        process: asyncio.subprocess.Process,
        stdout: bytearray,
        stderr: bytearray,
    ) -> None:
        total = 0
        lock = asyncio.Lock()

        async def read(stream: asyncio.StreamReader | None, destination: bytearray) -> None:
            nonlocal total
            if stream is None:
                return
            while chunk := await stream.read(64 * 1024):
                async with lock:
                    remaining = self.max_output_bytes - total
                    if remaining <= 0:
                        raise _OutputLimitExceeded
                    destination.extend(chunk[:remaining])
                    total += min(len(chunk), remaining)
                    if len(chunk) > remaining:
                        raise _OutputLimitExceeded

        readers = [asyncio.create_task(read(process.stdout, stdout)), asyncio.create_task(read(process.stderr, stderr))]
        try:
            await asyncio.gather(*readers)
            await process.wait()
        except BaseException:
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            raise

    @staticmethod
    async def _kill(process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.communicate()

    @staticmethod
    def _append_error(stderr: bytearray, message: str) -> str:
        rendered = stderr.decode(errors="replace").rstrip()
        return f"{rendered}\n{message}".lstrip()


class BashInput(BaseModel):
    command: str = Field(min_length=1, max_length=MAX_COMMAND_CHARACTERS)


class ToolCallBudget(Protocol):
    async def claim(self) -> tuple[bool, str | None]: ...

    def metadata(self, exhausted_scope: str | None = None) -> dict[str, object]: ...


def sandbox_tools(
    profile: SandboxProfile,
    *,
    budget: ToolCallBudget,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    exhausted_message: str = DEFAULT_EXHAUSTED_MESSAGE,
    description: str = DEFAULT_TOOL_DESCRIPTION,
    process_label: str = "localmcp-workspace-sandbox",
) -> list[StructuredTool]:
    """Expose Bash under one explicit, immutable Seatbelt profile."""
    sandbox = MacOSSandbox(
        profile,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        process_label=process_label,
    )

    async def invoke(name: str, operation: Callable[[], Awaitable[BaseModel]]) -> dict[str, Any]:
        allowed, exhausted_scope = await budget.claim()
        if not allowed:
            return {"error": exhausted_message, "tool_budget": budget.metadata(exhausted_scope)}
        try:
            result = await operation()
            return {"result": result.model_dump(), "tool_budget": budget.metadata()}
        except Exception as exc:
            return {"error": f"{name} failed with {type(exc).__name__}: {exc}", "tool_budget": budget.metadata()}

    async def bash(command: str) -> dict[str, Any]:
        return await invoke("Bash", lambda: sandbox.run(command))

    return [
        StructuredTool.from_function(
            coroutine=bash,
            name="Bash",
            description=description,
            args_schema=BashInput,
        )
    ]
