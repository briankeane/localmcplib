from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any

import pytest
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools.base import ToolResult
from mcp.types import CallToolRequestParams

import localmcp.observability.telemetry as telemetry_module
from localmcp.observability.telemetry import LangfuseTelemetry, LangfuseToolMiddleware

_ENABLED_ENV = {
    "LOCALMCP_LANGFUSE_ENABLED": "true",
    "LANGFUSE_BASE_URL": "https://langfuse.example.com",
    "LANGFUSE_PUBLIC_KEY": "public",
    "LANGFUSE_SECRET_KEY": "secret",
}


async def test_disabled_telemetry_is_noop() -> None:
    telemetry = LangfuseTelemetry({})

    with telemetry.span("example") as observation:
        observation.update(metadata={"safe": True})

    assert telemetry.status.status == "disabled"
    assert (await telemetry.check())["authenticated"] is False
    await telemetry.close()


def test_default_enable_flag_uses_shared_localmcp_name() -> None:
    legacy = LangfuseTelemetry({"LCALMCP_LANGFUSE_ENABLED": "true"})
    assert legacy.status.status == "disabled"

    enabled = LangfuseTelemetry({"LOCALMCP_LANGFUSE_ENABLED": "true"})
    assert enabled.status.status == "configuration_error"
    assert enabled.status.enabled is True
    assert "LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY" in enabled.status.message


def test_payload_capture_is_disabled_by_default_and_explicitly_enabled() -> None:
    default = LangfuseTelemetry({})
    enabled = LangfuseTelemetry({"LOCALMCP_LANGFUSE_CAPTURE_PAYLOADS": "true"})

    assert default.capture_payloads is False
    assert enabled.capture_payloads is True


def test_payload_capture_controls_langfuse_span_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[dict[str, Any]] = []

    class FakeLangfuse:
        def __init__(self, **kwargs: Any):
            clients.append(kwargs)

    modules = {
        "langfuse": SimpleNamespace(Langfuse=FakeLangfuse),
        "langfuse.types": SimpleNamespace(),
    }
    monkeypatch.setattr(telemetry_module.importlib, "import_module", modules.__getitem__)
    LangfuseTelemetry(_ENABLED_ENV)
    LangfuseTelemetry({**_ENABLED_ENV, "LOCALMCP_LANGFUSE_CAPTURE_PAYLOADS": "true"})

    assert callable(clients[0]["mask_otel_spans"])
    assert clients[1]["mask_otel_spans"] is None


class _RecordingObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            kwargs = {**kwargs, "metadata": dict(metadata)}
        self.updates.append(kwargs)


class _RecordingSpan(AbstractContextManager[_RecordingObservation]):
    def __init__(self, observation: _RecordingObservation):
        self.observation = observation

    def __enter__(self) -> _RecordingObservation:
        return self.observation

    def __exit__(self, *_args: Any) -> None:
        return None


class _RecordingTelemetry:
    def __init__(self, *, capture_payloads: bool):
        self.capture_payloads = capture_payloads
        self.observation = _RecordingObservation()
        self.span_call: tuple[tuple[Any, ...], dict[str, Any]] | None = None

    def span(self, *args: Any, **kwargs: Any) -> _RecordingSpan:
        self.span_call = (args, kwargs)
        return _RecordingSpan(self.observation)


@pytest.mark.parametrize("capture_payloads", [False, True])
async def test_tool_middleware_captures_payloads_only_when_enabled(capture_payloads: bool) -> None:
    service = _RecordingTelemetry(capture_payloads=capture_payloads)
    previous = telemetry_module.telemetry()
    telemetry_module.install_telemetry(service)  # type: ignore[arg-type]
    context = MiddlewareContext(
        message=CallToolRequestParams(name="example", arguments={"query": "value"}),
        source="client",
    )

    async def call_next(received: MiddlewareContext[CallToolRequestParams]) -> ToolResult:
        assert received is context
        return ToolResult(structured_content={"answer": 42})

    try:
        result = await LangfuseToolMiddleware().on_call_tool(context, call_next)
    finally:
        telemetry_module.install_telemetry(previous)

    assert result.structured_content == {"answer": 42}
    assert service.span_call == (
        ("mcp.tool.example",),
        {"observation_type": "tool", "metadata": {"tool": "example", "source": "client"}},
    )
    if capture_payloads:
        assert service.observation.updates[0] == {"input": {"query": "value"}}
        assert service.observation.updates[1]["output"]["structured_content"] == {"answer": 42}
    else:
        assert service.observation.updates == []


async def test_tool_payload_serialization_failure_does_not_change_tool_behavior() -> None:
    service = _RecordingTelemetry(capture_payloads=True)
    previous = telemetry_module.telemetry()
    telemetry_module.install_telemetry(service)  # type: ignore[arg-type]
    context = MiddlewareContext(message=CallToolRequestParams(name="example", arguments={}))

    class UnserializableResult:
        def model_dump(self, **_kwargs: Any) -> None:
            raise TypeError("cannot serialize")

    async def call_next(_context: MiddlewareContext[CallToolRequestParams]) -> Any:
        return UnserializableResult()

    try:
        result = await LangfuseToolMiddleware().on_call_tool(context, call_next)
    finally:
        telemetry_module.install_telemetry(previous)

    assert isinstance(result, UnserializableResult)
    assert service.observation.updates == [{"input": {}}]


def test_missing_dependency_is_isolated_and_endpoint_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_dependency(_name: str) -> Any:
        raise ImportError("langfuse is unavailable")

    monkeypatch.setattr(telemetry_module.importlib, "import_module", missing_dependency)
    service = LangfuseTelemetry(
        {
            **_ENABLED_ENV,
            "LANGFUSE_BASE_URL": "https://user:password@langfuse.example.com:8443/private?token=secret",
        }
    )

    assert service.status.status == "dependency_missing"
    assert service.status.base_url == "https://langfuse.example.com:8443"
    assert service.langchain_callback() is None
    assert "password" not in str(service.status.as_dict())
    assert "secret" not in str(service.status.as_dict())


def test_sdk_initialization_failure_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingLangfuse:
        def __init__(self, **_kwargs: Any):
            raise ValueError("secret must not escape")

    modules = {
        "langfuse": SimpleNamespace(Langfuse=FailingLangfuse),
        "langfuse.types": SimpleNamespace(),
    }
    monkeypatch.setattr(telemetry_module.importlib, "import_module", modules.__getitem__)

    service = LangfuseTelemetry(_ENABLED_ENV)

    assert service.status.status == "initialization_error"
    assert service.status.message == "Langfuse SDK initialization failed (ValueError)."
    assert service.langchain_callback() is None


class _ClientSpan(AbstractContextManager[_RecordingObservation]):
    def __init__(self, *, fail_enter: bool = False, fail_exit: bool = False):
        self.observation = _RecordingObservation()
        self.fail_enter = fail_enter
        self.fail_exit = fail_exit
        self.exit_args: tuple[Any, ...] | None = None

    def __enter__(self) -> _RecordingObservation:
        if self.fail_enter:
            raise RuntimeError("span enter failed")
        return self.observation

    def __exit__(self, *args: Any) -> None:
        self.exit_args = args
        if self.fail_exit:
            raise RuntimeError("span exit failed")


class _FakeClient:
    def __init__(
        self,
        *,
        authenticated: bool = True,
        auth_error: Exception | None = None,
        shutdown_error: Exception | None = None,
        span_context: _ClientSpan | None = None,
    ) -> None:
        self.authenticated = authenticated
        self.auth_error = auth_error
        self.shutdown_error = shutdown_error
        self.span_context = span_context or _ClientSpan()
        self.started: list[dict[str, Any]] = []
        self.shutdown_calls = 0

    def auth_check(self) -> bool:
        if self.auth_error is not None:
            raise self.auth_error
        return self.authenticated

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def create_trace_id(self, *, seed: str) -> str:
        return f"trace:{seed}"

    def start_as_current_observation(self, **kwargs: Any) -> _ClientSpan:
        self.started.append(kwargs)
        return self.span_context


def _ready_service(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeClient,
    *,
    callback_handler: Any | None = None,
) -> LangfuseTelemetry:
    modules: dict[str, Any] = {
        "langfuse": SimpleNamespace(Langfuse=lambda **_kwargs: client),
        "langfuse.types": SimpleNamespace(),
    }
    if callback_handler is not None:
        modules["langfuse.langchain"] = SimpleNamespace(CallbackHandler=callback_handler)
    monkeypatch.setattr(telemetry_module.importlib, "import_module", modules.__getitem__)
    return LangfuseTelemetry(_ENABLED_ENV)


def test_span_records_trace_model_metadata_and_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    service = _ready_service(monkeypatch, client)

    with service.span(
        "model.call",
        observation_type="generation",
        metadata={"request_id": "safe"},
        trace_seed="operation-1",
        model="model-a",
    ) as observation:
        observation.update(metadata={"phase": "run"}, usage=1)

    assert client.started == [
        {
            "as_type": "generation",
            "name": "model.call",
            "metadata": {"request_id": "safe"},
            "model": "model-a",
            "trace_context": {"trace_id": "trace:operation-1"},
        }
    ]
    assert client.span_context.observation.updates == [
        {"metadata": {"request_id": "safe", "phase": "run"}, "usage": 1},
        {"metadata": {"request_id": "safe", "phase": "run", "outcome": "success"}},
    ]
    assert client.span_context.exit_args == (None, None, None)


def test_span_preserves_application_exception_and_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    service = _ready_service(monkeypatch, client)
    error = ValueError("application failed")

    with pytest.raises(ValueError, match="application failed"):
        with service.span("operation"):
            raise error

    assert client.span_context.observation.updates == [{"metadata": {"outcome": "error"}}]
    assert client.span_context.exit_args is not None
    assert client.span_context.exit_args[:2] == (ValueError, error)


@pytest.mark.parametrize("failure_point", ["start", "enter"])
def test_span_start_failure_becomes_noop(monkeypatch: pytest.MonkeyPatch, failure_point: str) -> None:
    client = _FakeClient(span_context=_ClientSpan(fail_enter=failure_point == "enter"))
    if failure_point == "start":
        client.start_as_current_observation = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("start failed"))  # type: ignore[method-assign]
    service = _ready_service(monkeypatch, client)

    with service.span("operation") as observation:
        observation.update(metadata={"ignored": True}, output="ignored")

    assert client.span_context.observation.updates == []


def test_span_finish_failure_does_not_change_application_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _ClientSpan(fail_exit=True)
    client = _FakeClient(span_context=context)
    service = _ready_service(monkeypatch, client)

    with service.span("operation"):
        pass

    assert context.observation.updates == [{"metadata": {"outcome": "success"}}]
    assert context.exit_args == (None, None, None)


def test_observation_update_failure_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingObservation(_RecordingObservation):
        def update(self, **_kwargs: Any) -> None:
            raise RuntimeError("export failed")

    context = _ClientSpan()
    context.observation = FailingObservation()
    service = _ready_service(monkeypatch, _FakeClient(span_context=context))

    with service.span("operation", metadata={"safe": True}) as observation:
        observation.update(metadata="invalid", output="ignored")

    assert context.exit_args == (None, None, None)


@pytest.mark.parametrize(
    ("authenticated", "auth_error", "expected_status", "expected_authenticated"),
    [
        (True, None, "ready", True),
        (False, None, "authentication_error", False),
        (False, ConnectionError("offline"), "connection_error", False),
    ],
)
async def test_connectivity_check_isolates_authentication_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    authenticated: bool,
    auth_error: Exception | None,
    expected_status: str,
    expected_authenticated: bool,
) -> None:
    service = _ready_service(
        monkeypatch,
        _FakeClient(authenticated=authenticated, auth_error=auth_error),
    )

    result = await service.check()

    assert result["status"] == expected_status
    assert result["authenticated"] is expected_authenticated
    assert "secret" not in str(result)


@pytest.mark.parametrize("shutdown_error", [None, RuntimeError("shutdown failed")])
async def test_shutdown_is_idempotent_and_failure_is_isolated(
    monkeypatch: pytest.MonkeyPatch, shutdown_error: Exception | None
) -> None:
    client = _FakeClient(shutdown_error=shutdown_error)
    service = _ready_service(monkeypatch, client)

    await service.close()
    await service.close()

    assert client.shutdown_calls == 1
    assert (await service.check())["authenticated"] is False


def test_langchain_callback_reuses_lifecycle_client(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = SimpleNamespace()
    public_keys: list[str] = []

    def callback_handler(*, public_key: str) -> Any:
        public_keys.append(public_key)
        return callback

    client = _FakeClient()
    service = _ready_service(monkeypatch, client, callback_handler=callback_handler)

    assert service.langchain_callback() is callback
    assert public_keys == ["public"]
    assert callback._langfuse_client is client


def test_langchain_callback_failure_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    def callback_handler(*, public_key: str) -> Any:
        raise RuntimeError(public_key)

    service = _ready_service(monkeypatch, _FakeClient(), callback_handler=callback_handler)

    assert service.langchain_callback() is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("file:///tmp/private", "<configured>"),
        ("https://example.com:not-a-port/path", "<configured>"),
        ("https://user:pass@example.com/path?q=secret", "https://example.com"),
    ],
)
def test_safe_endpoint_reports_only_public_origin(value: str | None, expected: str | None) -> None:
    assert telemetry_module._safe_endpoint(value) == expected


def test_payload_suppressor_removes_only_payload_attributes() -> None:
    class Patch:
        def __init__(self, *, delete_attributes: tuple[str, ...]):
            self.delete_attributes = delete_attributes

    class Result:
        def __init__(self, *, span_patches: dict[str, Patch]):
            self.span_patches = span_patches

    suppress = telemetry_module._payload_suppressor(SimpleNamespace(OtelSpanPatch=Patch, MaskOtelSpansResult=Result))
    params = SimpleNamespace(
        spans={
            "generation": SimpleNamespace(
                attributes={
                    "langfuse.trace.input": "private",
                    "gen_ai.output.messages": "private",
                    "model": "safe",
                }
            ),
            "safe": SimpleNamespace(attributes={"model": "safe"}),
        }
    )

    result = suppress(params=params)

    assert result.span_patches.keys() == {"generation"}
    assert result.span_patches["generation"].delete_attributes == (
        "langfuse.trace.input",
        "gen_ai.output.messages",
    )
    assert suppress(params=SimpleNamespace(spans={"safe": params.spans["safe"]})) is None


async def test_tool_exception_is_not_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _RecordingTelemetry(capture_payloads=False)
    previous = telemetry_module.telemetry()
    telemetry_module.install_telemetry(service)  # type: ignore[arg-type]
    context = MiddlewareContext(message=CallToolRequestParams(name="example", arguments=None))

    async def call_next(_context: MiddlewareContext[CallToolRequestParams]) -> ToolResult:
        raise LookupError("tool failed")

    try:
        with pytest.raises(LookupError, match="tool failed"):
            await LangfuseToolMiddleware().on_call_tool(context, call_next)
    finally:
        telemetry_module.install_telemetry(previous)

    assert service.observation.updates == []
