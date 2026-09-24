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
        )
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.process_label = process_label
        self._sandbox_exec = Path("/usr/bin/sandbox-exec")
        self._shell = Path("/bin/sh")
        utilities = [shutil.which("rg"), shutil.which("git")]
        self._executables = tuple(Path(value).resolve(strict=True) for value in utilities if value is not None)
        self._runtime_paths = tuple(
            sorted(
                {path for executable in self._executables for path in _runtime_read_paths(executable)}, key=os.fspath
            )
        )
        self._runtime_directories = _runtime_read_directories(self._runtime_paths)
        self._process_limit = _owned_process_limit()
        self._metadata_ancestors = _path_ancestors(tuple(root.path for root in self.roots))

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
        runtime_profile += "".join(
            f'(allow file-read-metadata (literal (param "METADATA_ANCESTOR_{index}")))\n'
            for index in range(len(self._metadata_ancestors))
        )
        runtime_profile += "".join(
            f'(deny file-read* (literal (param "DENIED_PATH_{index}")) (subpath (param "DENIED_PATH_{index}")))\n'
            for index in range(len(self.profile.denied_paths))
        )
        return runtime_profile

    def _environment(self) -> dict[str, str]:
        paths = ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        for executable in self._executables:
            if executable.parent not in {Path(path) for path in paths}:
                paths.insert(0, os.fspath(executable.parent))
        return {
            "HOME": "/var/empty",
            "PATH": os.pathsep.join(paths),
            "LC_ALL": "C",
            "RIPGREP_CONFIG_PATH": "",
            "TMPDIR": os.fspath(self.root),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
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
