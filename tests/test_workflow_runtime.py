import asyncio
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import localmcp.workflows.runtime as runtime_module
from localmcp.workflows.catalog import OperationRecord, SQLiteOperationCatalog
from localmcp.workflows.runtime import WorkflowRuntime, ordered_runtime_types


class ExampleRuntime(WorkflowRuntime[OperationRecord, SQLiteOperationCatalog], register=False):
    runtime_name = "example"
    state_subdirectory = "example"

    def __init__(self, state_root: Path):
        super().__init__(state_root, trace_namespace="tests")
        self.recoveries = 0

    async def _open_store(self) -> SQLiteOperationCatalog:
        return await SQLiteOperationCatalog.open(self.root / self.operations_database_name)

    def _build_graph(self, checkpointer: Any) -> object:
        return object()

    async def _recover(self) -> None:
        self.recoveries += 1


class LeaseLosingStore:
    def __init__(self) -> None:
        self.renewed = asyncio.Event()
        self.released: list[tuple[str, str]] = []

    async def get(self, operation_id: str) -> object:
        return object()

    async def claim(self, operation_id: str, worker_id: str, *, lease_seconds: float = 90.0) -> bool:
        return True

    async def renew(self, operation_id: str, worker_id: str, *, lease_seconds: float = 90.0) -> bool:
        self.renewed.set()
        return False

    async def release(self, operation_id: str, worker_id: str) -> None:
        self.released.append((operation_id, worker_id))

    async def close(self) -> None:
        return None


class RejectingStore(LeaseLosingStore):
    def __init__(self, *, record: object | None, claimed: bool) -> None:
        super().__init__()
        self.record = record
        self.claimed = claimed
        self.claim_attempts = 0

    async def get(self, operation_id: str) -> object | None:
        return self.record

    async def claim(self, operation_id: str, worker_id: str, *, lease_seconds: float = 90.0) -> bool:
        self.claim_attempts += 1
        return self.claimed


class LeaseRuntime(WorkflowRuntime[object, LeaseLosingStore], register=False):
    runtime_name = "lease"
    state_subdirectory = "lease"
    heartbeat_interval_seconds = 0

    async def _open_store(self) -> LeaseLosingStore:
        return LeaseLosingStore()

    def _build_graph(self, checkpointer: Any) -> object:
        return object()

    async def _recover(self) -> None:
        return None


class RecordingStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def close(self) -> None:
        self.events.append("store.close")


class FailingStartRuntime(WorkflowRuntime[object, Any], register=False):
    runtime_name = "failing-start"
    state_subdirectory = "failing-start"

    def __init__(self, state_root: Path, events: list[str]) -> None:
        super().__init__(state_root)
        self.events = events

    async def _open_store(self) -> RecordingStore:
        self.events.append("store.open")
        return RecordingStore(self.events)

    def _build_graph(self, checkpointer: Any) -> object:
        self.events.append("graph.build")
        return object()

    async def _recover(self) -> None:
        self.events.append("recover")
        raise RuntimeError("recovery failed")

    async def _close_domain_resources(self) -> None:
        self.events.append("domain.close")


class SpanRecorder(AbstractContextManager[None]):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> None:
        self.events.append("span.enter")

    def __exit__(self, *args: object) -> None:
        self.events.append("span.exit")


class TelemetryRecorder:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, str, dict[str, str], str]] = []

    def span(
        self,
        name: str,
        *,
        observation_type: str = "span",
        metadata: dict[str, str] | None = None,
        trace_seed: str | None = None,
        model: str | None = None,
    ) -> AbstractContextManager[None]:
        assert metadata is not None
        assert trace_seed is not None
        self.calls.append((name, observation_type, metadata, trace_seed))
        return SpanRecorder(self.events)


async def test_runtime_starts_and_closes_owned_resources(tmp_path: Path) -> None:
    runtime = ExampleRuntime(tmp_path)

    await runtime.start()
    assert runtime.graph is not None
    assert runtime.store is not None
    assert runtime.recoveries == 1
    assert runtime.root.stat().st_mode & 0o777 == 0o700

    store = runtime.store
    await runtime.start()
    assert runtime.store is store

    await runtime.close()
    assert runtime.graph is None
    assert runtime.store is None


async def test_lease_loss_cancels_owner_execution_and_releases_lease(tmp_path: Path) -> None:
    runtime = LeaseRuntime(tmp_path)
    store = LeaseLosingStore()
    runtime.store = store
    execution_started = asyncio.Event()

    async def execute(record: object) -> None:
        execution_started.set()
        await asyncio.Event().wait()

    owner_task = asyncio.create_task(runtime._run_leased("operation-1", execute))
    await asyncio.wait_for(execution_started.wait(), timeout=1)
    await asyncio.wait_for(store.renewed.wait(), timeout=1)

    with pytest.raises(asyncio.CancelledError):
        await owner_task

    assert store.released == [("operation-1", runtime.worker_id)]


async def test_start_failure_closes_every_acquired_resource(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Checkpointer:
        async def setup(self) -> None:
            events.append("checkpointer.setup")

    class CheckpointerContext:
        async def __aenter__(self) -> Checkpointer:
            events.append("checkpointer.enter")
            return Checkpointer()

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            events.append("checkpointer.exit")

    class Saver:
        @staticmethod
        def from_conn_string(path: str) -> CheckpointerContext:
            events.append("checkpointer.create")
            return CheckpointerContext()

    monkeypatch.setattr(runtime_module, "AsyncSqliteSaver", Saver)
    runtime = FailingStartRuntime(tmp_path, events)

    with pytest.raises(RuntimeError, match="recovery failed"):
        await runtime.start()

    assert events == [
        "store.open",
        "checkpointer.create",
        "checkpointer.enter",
        "checkpointer.setup",
        "graph.build",
        "recover",
        "domain.close",
        "checkpointer.exit",
        "store.close",
    ]
    assert runtime.graph is None
    assert runtime.store is None


async def test_start_preserves_original_failure_when_cleanup_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        runtime_module,
        "log",
        SimpleNamespace(warning=lambda event, **fields: warnings.append((event, fields))),
    )
    runtime = FailingStartRuntime(tmp_path, [])

    async def fail_open() -> RecordingStore:
        raise RuntimeError("open failed")

    async def fail_cleanup() -> None:
        raise LookupError("cleanup failed")

    monkeypatch.setattr(runtime, "_open_store", fail_open)
    monkeypatch.setattr(runtime, "close", fail_cleanup)

    with pytest.raises(RuntimeError, match="open failed"):
        await runtime.start()

    assert warnings == [
        (
            "workflow.runtime_start_cleanup_failed",
            {"runtime": "failing-start", "error_type": "LookupError"},
        )
    ]


async def test_close_attempts_all_cleanup_and_raises_first_failure(tmp_path: Path) -> None:
    events: list[str] = []
    runtime = FailingStartRuntime(tmp_path, events)

    class FailingContext:
        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            events.append("checkpointer.exit")
            raise ValueError("checkpointer failed")

    class FailingStore(RecordingStore):
        async def close(self) -> None:
            events.append("store.close")
            raise OSError("store failed")

    async def domain_close() -> None:
        events.append("domain.close")
        raise RuntimeError("domain failed")

    cancelled = asyncio.Event()

    async def pending() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runtime._close_domain_resources = domain_close
    runtime._checkpointer_cm = FailingContext()
    runtime._checkpointer = object()
    runtime.store = FailingStore(events)
    runtime.graph = object()
    runtime._started = True
    runtime._recovery_task = asyncio.create_task(pending())
    runtime.tasks["operation-1"] = asyncio.create_task(pending())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="domain failed"):
        await runtime.close()

    assert cancelled.is_set()
    assert events == ["domain.close", "checkpointer.exit", "store.close"]
    assert runtime._checkpointer_cm is None
    assert runtime._checkpointer is None
    assert runtime.store is None
    assert runtime.graph is None
    assert runtime.tasks == {}
    assert not runtime._started


async def test_store_requirement_and_task_tracking(tmp_path: Path) -> None:
    runtime = LeaseRuntime(tmp_path)
    with pytest.raises(RuntimeError, match="lease runtime is not initialized"):
        runtime._require_store()

    started = asyncio.Event()
    finish = asyncio.Event()
    calls = 0

    async def work() -> None:
        nonlocal calls
        calls += 1
        started.set()
        await finish.wait()

    runtime._track_task("operation-1", work)
    await started.wait()
    runtime._track_task("operation-1", work)
    assert calls == 1
    finish.set()
    await asyncio.gather(*runtime.tasks.values())
    await asyncio.sleep(0)
    assert runtime.tasks == {}

    runtime._track_task("operation-1", work)
    await asyncio.sleep(0)
    assert calls == 2


@pytest.mark.parametrize(
    ("record", "claimed", "claim_attempts"),
    [(None, True, 0), (object(), False, 1)],
)
async def test_leased_execution_skips_missing_or_unclaimed_operation(
    tmp_path: Path, record: object | None, claimed: bool, claim_attempts: int
) -> None:
    runtime = LeaseRuntime(tmp_path)
    store = RejectingStore(record=record, claimed=claimed)
    runtime.store = store
    executed = False

    async def execute(value: object) -> None:
        nonlocal executed
        executed = True

    await runtime._run_leased("operation-1", execute)

    assert not executed
    assert store.claim_attempts == claim_attempts
    assert store.released == []


async def test_leased_execution_traces_and_releases_after_failure(tmp_path: Path) -> None:
    events: list[str] = []
    telemetry = TelemetryRecorder(events)
    runtime = LeaseRuntime(tmp_path, telemetry=telemetry, trace_namespace="public-tests")
    store = RejectingStore(record="record", claimed=True)
    runtime.store = store

    async def execute(record: object) -> None:
        events.append(f"execute:{record}")
        raise RuntimeError("execution failed")

    with pytest.raises(RuntimeError, match="execution failed"):
        await runtime._run_leased("operation-1", execute)

    assert telemetry.calls == [
        (
            "workflow.lease",
            "agent",
            {"runtime": "lease", "operation_id": "operation-1"},
            "public-tests:lease:operation-1",
        )
    ]
    assert events == ["span.enter", "execute:record", "span.exit"]
    assert store.released == [("operation-1", runtime.worker_id)]


async def test_recovery_loop_logs_failure_retries_and_propagates_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ExampleRuntime(tmp_path)
    runtime.recovery_interval_seconds = 0
    recovered = asyncio.Event()
    attempts = 0
    warnings: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        runtime_module,
        "log",
        SimpleNamespace(warning=lambda event, **fields: warnings.append((event, fields))),
    )

    async def recover() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("transient")
        recovered.set()

    runtime._recover = recover
    task = asyncio.create_task(runtime._recovery_loop())
    await asyncio.wait_for(recovered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert attempts >= 2
    assert warnings == [("workflow.recovery_failed", {"runtime": "example", "error_type": "ValueError"})]


def test_runtime_dependency_ordering() -> None:
    class First(ExampleRuntime, register=False):
        runtime_name = "first"
        state_subdirectory = "first"

    class Second(ExampleRuntime, register=False):
        runtime_name = "second"
        state_subdirectory = "second"
        runtime_dependencies = ("first",)

    assert ordered_runtime_types((Second, First)) == (First, Second)

    class Missing(ExampleRuntime, register=False):
        runtime_name = "missing"
        state_subdirectory = "missing"
        runtime_dependencies = ("absent",)

    with pytest.raises(RuntimeError, match="not registered"):
        ordered_runtime_types((Missing,))

    class Duplicate(ExampleRuntime, register=False):
        runtime_name = "first"
        state_subdirectory = "duplicate"

    with pytest.raises(RuntimeError, match="duplicate workflow runtime name 'first'"):
        ordered_runtime_types((First, Duplicate))

    class Circular(ExampleRuntime, register=False):
        runtime_name = "circular"
        state_subdirectory = "circular"
        runtime_dependencies = ("circular",)

    with pytest.raises(RuntimeError, match="workflow runtime dependency cycle: circular"):
        ordered_runtime_types((Circular,))


def test_registered_runtime_validation_and_listing() -> None:
    class Registered(ExampleRuntime):
        runtime_name = "test-registered-runtime"
        state_subdirectory = "registered"

    try:
        assert Registered in WorkflowRuntime.registered_runtime_types()
        with pytest.raises(TypeError, match="non-empty runtime_name"):

            class MissingName(ExampleRuntime):
                state_subdirectory = "missing-name"

        with pytest.raises(TypeError, match="non-empty state_subdirectory"):

            class MissingStateDirectory(ExampleRuntime):
                runtime_name = "test-missing-state-directory"

        with pytest.raises(RuntimeError, match="already registered"):

            class DuplicateName(ExampleRuntime):
                runtime_name = "test-registered-runtime"
                state_subdirectory = "duplicate"
    finally:
        WorkflowRuntime._registry.pop("test-registered-runtime", None)
