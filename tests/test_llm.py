import logging
import os
from typing import Any

import pytest

from localmcp.llm import (
    DEFAULT_MODEL_REGISTRY,
    ConfiguredModelFactory,
    GatewayEndpoint,
    LLMBackendConfig,
    ModelConfigurationError,
    ModelFactory,
    ModelRegistry,
    ModelSpec,
    NativeModelFactory,
    NativeTransport,
    OpenAICompatibleEndpoint,
    OpenAICompatibleModelFactory,
    OwnedChatModel,
    infer_model_spec,
)


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeChatOpenAI:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs


class AlternateModelFactory:
    def __init__(self, owner: FakeClient) -> None:
        self.owner = owner
        self.created_with: tuple[str, int | None, str | None] | None = None

    def validate(self, model_id: str, *, reasoning_effort: str | None = None) -> ModelSpec:
        del model_id, reasoning_effort
        return ModelSpec()

    def create(
        self,
        model_id: str,
        *,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> OwnedChatModel:
        self.created_with = (model_id, max_output_tokens, reasoning_effort)
        return OwnedChatModel(object(), self.owner)


def create_with(factory: ModelFactory) -> OwnedChatModel:
    return factory.create("native-model", max_output_tokens=321, reasoning_effort="high")


def test_default_catalog_separates_capabilities_from_invocation_policy() -> None:
    gpt = DEFAULT_MODEL_REGISTRY.resolve("gpt-5.6")
    claude = DEFAULT_MODEL_REGISTRY.resolve("claude-opus-4-8")

    assert gpt.supports_reasoning_effort
    assert gpt.default_reasoning_effort is None
    assert claude.native_provider == "anthropic"
    assert claude.temperature == "omit"


async def test_model_factory_protocol_accepts_alternate_backends() -> None:
    client = FakeClient()
    factory = AlternateModelFactory(client)

    async with create_with(factory):
        pass

    assert factory.created_with == ("native-model", 321, "high")
    assert client.closed


async def test_factory_routes_every_model_through_openai_compatible_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    client = FakeClient()
    registry = ModelRegistry(
        {
            "claude-example": ModelSpec(temperature="omit"),
            "gpt-example": ModelSpec(
                temperature="omit",
                supports_reasoning_effort=True,
                default_reasoning_effort="medium",
            ),
        }
    )
    factory = OpenAICompatibleModelFactory(
        registry,
        OpenAICompatibleEndpoint(
            "https://gateway.example/v1",
            default_headers={"x-service-name": "test"},
            keepalive_expiry=1.0,
        ),
        "token",
        http_client_factory=lambda _: client,
    )

    owner = factory.create("claude-example", max_output_tokens=123)
    async with owner as model:
        assert isinstance(model, FakeChatOpenAI)
        assert model.kwargs["model"] == "claude-example"
        assert model.kwargs["max_completion_tokens"] == 123
        assert model.kwargs["default_headers"] == {"x-service-name": "test"}
        assert "temperature" not in model.kwargs
    assert client.closed


async def test_factory_owns_default_httpx2_transport() -> None:
    import httpx2
    from langchain_openai import ChatOpenAI

    factory = OpenAICompatibleModelFactory(
        ModelRegistry({"model": ModelSpec()}),
        OpenAICompatibleEndpoint("https://gateway.invalid/v1"),
        "token",
    )

    owner = factory.create("model")
    client = owner._owner
    assert isinstance(client, httpx2.AsyncClient)
    async with owner as model:
        assert isinstance(model, ChatOpenAI)
    assert client.is_closed


def test_registry_overlay_and_reasoning_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    registry = ModelRegistry({"base": ModelSpec()}).overlay({"other": ModelSpec()})
    factory = OpenAICompatibleModelFactory(
        registry,
        OpenAICompatibleEndpoint("https://gateway.example/v1"),
        "token",
        http_client_factory=lambda _: FakeClient(),
    )

    assert registry.resolve("base") == ModelSpec()
    with pytest.raises(ModelConfigurationError, match="does not support"):
        factory.create("other", reasoning_effort="high")


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("claude-opus-9", ModelSpec(temperature="omit", native_provider="anthropic")),
        ("gpt-7", ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="openai")),
        ("o5-mini", ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="openai")),
        ("gemini-9-pro", ModelSpec(temperature="omit", supports_reasoning_effort=True)),
    ],
)
def test_unregistered_models_resolve_to_inferred_specs(model_id: str, expected: ModelSpec) -> None:
    assert ModelRegistry().resolve(model_id) == expected
    assert infer_model_spec(model_id) == expected


def test_unregistered_model_warns_once_and_blank_ids_are_rejected(caplog: pytest.LogCaptureFixture) -> None:
    registry = ModelRegistry()

    with caplog.at_level(logging.WARNING, logger="localmcp.llm"):
        registry.resolve("warn-once-model")
        registry.resolve("warn-once-model")

    assert [record.getMessage() for record in caplog.records] == [
        'model "warn-once-model" is not in the catalog; using inferred capabilities'
    ]
    with pytest.raises(ModelConfigurationError, match="must not be blank"):
        registry.resolve(" ")


async def test_unregistered_model_passes_through_to_the_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    factory = OpenAICompatibleModelFactory(
        ModelRegistry(),
        OpenAICompatibleEndpoint("https://gateway.example/v1"),
        "token",
        http_client_factory=lambda _: FakeClient(),
    )

    async with factory.create("gpt-7", reasoning_effort="high") as model:
        assert model.kwargs["model"] == "gpt-7"
        assert model.kwargs["reasoning_effort"] == "high"
        assert "temperature" not in model.kwargs


def test_unregistered_models_route_by_inferred_provider() -> None:
    gateway = ConfiguredModelFactory(
        ModelRegistry(),
        LLMBackendConfig(backend="openai_compatible", base_url="https://gateway.example/v1"),
        gateway_api_key="gateway-token",
    )
    native = ConfiguredModelFactory(
        ModelRegistry(),
        LLMBackendConfig(backend="native"),
        anthropic_api_key="anthropic-token",
    )

    assert gateway.registry.resolve("claude-opus-9") == gateway.validate("claude-opus-9")
    assert gateway.validate("claude-opus-9").adapter_dialect == "anthropic"
    assert native.validate("claude-opus-9").native_provider == "anthropic"
    with pytest.raises(ModelConfigurationError, match="does not support reasoning_effort"):
        gateway.validate("claude-opus-9", reasoning_effort="high")
    with pytest.raises(ModelConfigurationError, match='"gemini-9-pro" is available only through a gateway'):
        native.validate("gemini-9-pro")


def test_backend_selection_is_explicit_and_validates_gateway_state() -> None:
    with pytest.raises(ModelConfigurationError, match="requires base_url"):
        LLMBackendConfig(backend="openai_compatible")
    with pytest.raises(ModelConfigurationError, match="does not accept"):
        LLMBackendConfig(backend="native", base_url="https://stale-gateway.example/v1")


async def test_configured_factory_switches_one_model_between_gateway_and_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx2
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    # The Anthropic client reads ANTHROPIC_BASE_URL and friends from the caller's shell.
    for name in [name for name in os.environ if name.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)

    registry = ModelRegistry(
        {
            # The gateway dialect is deliberately independent from the native
            # provider: stock LiteLLM can expose Claude through its OpenAI API.
            "claude-over-openai": ModelSpec(
                temperature="omit",
                native_provider="anthropic",
                dialect="openai",
            ),
            "claude-over-anthropic": ModelSpec(native_provider="anthropic"),
        }
    )
    gateway = ConfiguredModelFactory(
        registry,
        LLMBackendConfig(backend="openai_compatible", base_url="https://litellm.example/v1"),
        gateway_api_key="gateway-token",
    )

    openai_gateway_owner = gateway.create("claude-over-openai")
    assert isinstance(openai_gateway_owner.model, ChatOpenAI)
    assert str(openai_gateway_owner.model.openai_api_base) == "https://litellm.example/v1"
    assert isinstance(openai_gateway_owner._owner, httpx2.AsyncClient)
    async with openai_gateway_owner:
        pass
    assert openai_gateway_owner._owner.is_closed

    anthropic_gateway_owner = gateway.create("claude-over-anthropic")
    assert isinstance(anthropic_gateway_owner.model, ChatAnthropic)
    assert str(anthropic_gateway_owner.model.anthropic_api_url) == "https://litellm.example/v1"
    async with anthropic_gateway_owner:
        pass
    assert anthropic_gateway_owner._owner.is_closed

    native = ConfiguredModelFactory(
        registry,
        LLMBackendConfig(backend="native"),
        anthropic_api_key="anthropic-token",
    )
    native_owner = native.create("claude-over-openai")
    assert isinstance(native_owner.model, ChatAnthropic)
    assert str(native_owner.model.anthropic_api_url) == "https://api.anthropic.com"
    async with native_owner:
        pass
    assert native_owner._owner.is_closed


async def test_native_factory_requires_only_the_selected_provider_key() -> None:
    registry = ModelRegistry(
        {
            "claude": ModelSpec(native_provider="anthropic"),
            "gpt": ModelSpec(native_provider="openai"),
        }
    )
    factory = NativeModelFactory(registry, anthropic_api_key="anthropic-token")

    owner = factory.create("claude")
    with pytest.raises(ModelConfigurationError, match='native openai API key.*"gpt"'):
        factory.create("gpt")

    # Construction is lazy: selecting Claude did not require an OpenAI key.
    assert owner.model is not None
    await owner.__aexit__()


def test_gateway_only_model_fails_before_transport_construction() -> None:
    constructed = False

    def client_factory(*_: object) -> FakeClient:
        nonlocal constructed
        constructed = True
        return FakeClient()

    factory = NativeModelFactory(
        ModelRegistry({"gemini": ModelSpec()}),
        http_client_factory=client_factory,
    )

    with pytest.raises(ModelConfigurationError, match='"gemini" is available only through a gateway'):
        factory.validate("gemini")
    assert not constructed


def test_configured_factory_validates_backend_compatibility_without_constructing_transport() -> None:
    registry = ModelRegistry({"gemini": ModelSpec()})
    gateway = ConfiguredModelFactory(
        registry,
        LLMBackendConfig(backend="openai_compatible", base_url="https://gateway.example/v1"),
        gateway_api_key="gateway-token",
    )
    native = ConfiguredModelFactory(registry, LLMBackendConfig(backend="native"))

    assert gateway.validate("gemini") == ModelSpec()
    with pytest.raises(ModelConfigurationError, match='"gemini" is available only through a gateway'):
        native.validate("gemini")


def test_model_specs_reject_inconsistent_capabilities() -> None:
    with pytest.raises(ModelConfigurationError, match="requires supports_reasoning_effort"):
        ModelSpec(default_reasoning_effort="medium")
    with pytest.raises(ModelConfigurationError, match="only by the OpenAI dialect"):
        ModelSpec(supports_reasoning_effort=True, dialect="anthropic")

    assert ModelSpec(native_provider="anthropic").adapter_dialect == "anthropic"
    assert ModelSpec().adapter_dialect == "openai"


def test_registry_rejects_blank_ids_and_returns_a_defensive_mapping() -> None:
    with pytest.raises(ModelConfigurationError, match="must not be blank"):
        ModelRegistry({" ": ModelSpec()})

    registry = ModelRegistry({"model": ModelSpec()})
    copied = registry.as_mapping()
    assert copied == {"model": ModelSpec()}
    copied.clear()
    assert registry.resolve("model") == ModelSpec()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: GatewayEndpoint(" "),
        lambda: GatewayEndpoint("https://example.test", keepalive_expiry=-1),
        lambda: NativeTransport(keepalive_expiry=-1),
        lambda: LLMBackendConfig(backend="unsupported"),  # type: ignore[arg-type]
        lambda: LLMBackendConfig(backend="openai_compatible", base_url="https://example.test", keepalive_expiry=-1),
    ],
)
def test_transport_configuration_rejects_invalid_values(constructor: Any) -> None:
    with pytest.raises(ModelConfigurationError):
        constructor()


def test_backend_config_exposes_only_the_selected_transport() -> None:
    gateway = LLMBackendConfig(
        backend="openai_compatible",
        base_url="https://gateway.example/v1",
        default_headers={"x-test": "value"},
        verify=False,
        keepalive_expiry=2,
    )
    assert gateway.gateway_endpoint() == GatewayEndpoint(
        "https://gateway.example/v1",
        default_headers={"x-test": "value"},
        verify=False,
        keepalive_expiry=2,
    )
    with pytest.raises(ModelConfigurationError, match="non-native"):
        gateway.native_transport()

    native = LLMBackendConfig(backend="native", verify=False, keepalive_expiry=3)
    assert native.native_transport() == NativeTransport(verify=False, keepalive_expiry=3)
    with pytest.raises(ModelConfigurationError, match="non-openai_compatible"):
        native.gateway_endpoint()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OpenAICompatibleModelFactory(
            ModelRegistry(), OpenAICompatibleEndpoint("https://example.test"), "token", default_max_output_tokens=0
        ),
        lambda: NativeModelFactory(ModelRegistry(), default_max_output_tokens=0),
        lambda: ConfiguredModelFactory(
            ModelRegistry(),
            LLMBackendConfig(backend="openai_compatible", base_url="https://example.test"),
            gateway_api_key="token",
            default_max_output_tokens=0,
        ),
    ],
)
def test_factories_reject_non_positive_default_output_limits(factory: Any) -> None:
    with pytest.raises(ModelConfigurationError, match="must be positive"):
        factory()


def test_create_rejects_non_positive_per_call_output_limit_before_transport() -> None:
    constructed = False

    def client_factory(_: GatewayEndpoint) -> FakeClient:
        nonlocal constructed
        constructed = True
        return FakeClient()

    factory = OpenAICompatibleModelFactory(
        ModelRegistry({"model": ModelSpec()}),
        GatewayEndpoint("https://example.test"),
        "token",
        http_client_factory=client_factory,
    )
    with pytest.raises(ModelConfigurationError, match="must be positive"):
        factory.create("model", max_output_tokens=0)
    assert not constructed


def test_configured_gateway_requires_its_explicit_credential() -> None:
    with pytest.raises(ModelConfigurationError, match="gateway API key is required"):
        ConfiguredModelFactory(
            ModelRegistry(),
            LLMBackendConfig(backend="openai_compatible", base_url="https://example.test"),
        )


def test_anthropic_async_key_provider_is_rejected_before_transport_use() -> None:
    class AwaitableValue:
        def __await__(self) -> Any:
            yield
            return "token"

    def key() -> Any:
        return AwaitableValue()

    factory = NativeModelFactory(
        ModelRegistry({"claude": ModelSpec(native_provider="anthropic")}),
        anthropic_api_key=key,
        http_client_factory=lambda *_: FakeClient(),
    )
    with pytest.raises(ModelConfigurationError, match="synchronous callables"):
        factory.create("claude")
