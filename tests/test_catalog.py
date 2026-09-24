from pathlib import Path

import pytest

import localmcp.workflows.catalog as catalog_module
from localmcp.workflows.catalog import OperationCatalogError, OperationNotFoundError, SQLiteOperationCatalog


async def test_catalog_owns_updates_budgets_cancellation_and_recovery(tmp_path: Path) -> None:
    catalog = await SQLiteOperationCatalog.open(tmp_path / "operations.sqlite3")
    try:
        created = await catalog.create(
            operation_id="op-1",
            owner_uid=42,
            label="Example",
            status="queued",
            request={"input": 1},
        )
        assert created.state == {}
        assert created.request == {"input": 1}
        assert await catalog.get("missing") is None
        assert (await catalog.require_owned("op-1", 42)).label == "Example"
        with pytest.raises(OperationNotFoundError):
            await catalog.require_owned("op-1", 7)
        with pytest.raises(OperationNotFoundError):
            await catalog.require_owned("missing", 42)

        assert await catalog.claim_tool_call("op-1", 2) == 1
        assert await catalog.claim_tool_call("op-1", 2) == 0
        assert await catalog.claim_tool_call("op-1", 2) is None

        await catalog.update(
            "op-1",
            status="ready",
            state={"step": 1},
            preview={"revision": 1},
            metadata={"resource": "x"},
        )
        await catalog.request_cancel("op-1", status="cancel_requested")
        updated = await catalog.get("op-1")
        assert updated is not None
        assert updated.status == "cancel_requested"
        assert updated.preview == {"revision": 1}
        assert updated.cancel_requested
        assert [item.id for item in await catalog.recoverable({"completed", "failed"})] == ["op-1"]

        await catalog.update("op-1", preview=None, result={"output": 2}, error="failed")
        updated = await catalog.get("op-1")
        assert updated is not None
        assert updated.preview is None
        assert updated.result == {"output": 2}
        assert updated.error == "failed"

        await catalog.update("op-1", result=None, error=None)
        updated = await catalog.get("op-1")
        assert updated is not None
        assert updated.result is None
        assert updated.error is None
    finally:
        await catalog.close()


async def test_catalog_lease_is_single_owner_and_reentrant(tmp_path: Path) -> None:
    catalog = await SQLiteOperationCatalog.open(tmp_path / "operations.sqlite3")
    try:
        await catalog.create(
            operation_id="op-1",
            owner_uid=1,
            label="Example",
            status="queued",
            request={},
        )
        assert await catalog.claim("op-1", "worker-a")
        assert not await catalog.claim("op-1", "worker-b")
        assert await catalog.renew("op-1", "worker-a")
        assert not await catalog.renew("op-1", "worker-b")
        await catalog.release("op-1", "worker-a")
        assert await catalog.claim("op-1", "worker-b")
    finally:
        await catalog.close()


@pytest.mark.parametrize("table", ["", "operations; DROP TABLE operations", "two words", "1operations"])
def test_catalog_rejects_unsafe_table_names(table: str) -> None:
    with pytest.raises(OperationCatalogError, match="table name is invalid"):
        SQLiteOperationCatalog(None, table=table)  # type: ignore[arg-type]


async def test_catalog_validates_open_and_claim_limits(tmp_path: Path) -> None:
    with pytest.raises(OperationCatalogError, match="busy_timeout_ms must be non-negative"):
        await SQLiteOperationCatalog.open(tmp_path / "operations.sqlite3", busy_timeout_ms=-1)

    catalog = await SQLiteOperationCatalog.open(tmp_path / "operations.sqlite3")
    try:
        with pytest.raises(OperationCatalogError, match="tool-call limit must be positive"):
            await catalog.claim_tool_call("missing", 0)
        assert await catalog.claim_tool_call("missing", 1) is None

        for action in (catalog.claim, catalog.renew):
            with pytest.raises(OperationCatalogError, match="lease_seconds must be positive"):
                await action("missing", "worker", lease_seconds=0)
    finally:
        await catalog.close()


async def test_catalog_cancellation_queries_and_all_recoverable_operations(tmp_path: Path) -> None:
    catalog = await SQLiteOperationCatalog.open(tmp_path / "operations.sqlite3", table="workflow_operations")
    try:
        assert not await catalog.is_cancel_requested("missing")
        await catalog.create(
            operation_id="op-1",
            owner_uid=1,
            label="First",
            status="queued",
            request={},
        )
        await catalog.create(
            operation_id="op-2",
            owner_uid=1,
            label="Second",
            status="completed",
            request={},
        )
        assert not await catalog.is_cancel_requested("op-1")

        await catalog.request_cancel("op-1")

        assert await catalog.is_cancel_requested("op-1")
        assert (await catalog.require_owned("op-1", 1)).status == "queued"
        assert [record.id for record in await catalog.recoverable(set())] == ["op-1", "op-2"]
        assert [record.id for record in await catalog.recoverable({"completed"})] == ["op-1"]
    finally:
        await catalog.close()


async def test_expired_lease_can_be_claimed_by_another_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr(catalog_module.time, "time", lambda: now)
    catalog = await SQLiteOperationCatalog.open(tmp_path / "operations.sqlite3")
    try:
        await catalog.create(
            operation_id="op-1",
            owner_uid=1,
            label="Example",
            status="queued",
            request={},
        )
        assert await catalog.claim("op-1", "worker-a", lease_seconds=10)
        assert not await catalog.claim("op-1", "worker-b", lease_seconds=10)

        now = 111.0

        assert await catalog.claim("op-1", "worker-b", lease_seconds=10)
        record = await catalog.get("op-1")
        assert record is not None
        assert record.lease_owner == "worker-b"
        assert record.lease_expires == 121.0
    finally:
        await catalog.close()
