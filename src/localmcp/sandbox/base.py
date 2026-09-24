"""Portable contracts for policy-bound command sandboxes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

DEFAULT_COMMAND_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_COMMAND_CHARACTERS = 16_000
MAX_OPEN_FILES = 256
MAX_ADDITIONAL_PROCESSES = 256


class SandboxError(RuntimeError):
    """Raised when a sandbox cannot be constructed or invoked safely."""


class RootAccess(StrEnum):
    """Filesystem authority granted to one sandbox root."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class SandboxRoot:
    """One filesystem root exposed to a sandbox."""

    path: Path
    access: RootAccess = RootAccess.READ_ONLY


@dataclass(frozen=True)
class SandboxProfile:
    """Portable authority requested for a sandboxed command.

    The first root is the process working directory. ``ipc`` permits
    shared-memory IPC only, not sockets or platform service protocols.
    """

    roots: tuple[SandboxRoot, ...]
    denied_paths: tuple[Path, ...] = ()
    network: bool = False
    ipc: bool = False

    def __post_init__(self) -> None:
        if not self.roots:
            raise SandboxError("sandbox profile must declare at least one root")


class CommandResult(BaseModel):
    """Captured result of one bounded command invocation."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


class Sandbox(Protocol):
    """Narrow async interface implemented by platform sandbox backends."""

    async def run(self, command: str) -> CommandResult: ...
