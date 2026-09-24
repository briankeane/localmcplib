from pathlib import Path

import pytest

from localmcp.config import ConfigDocument
from localmcp.secrets import (
    SecretBackendError,
    SecretCatalog,
    SecretError,
    SecretNotFoundError,
    SecretRef,
    SecretResolver,
    SecretSource,
)


def test_catalog_composes_global_and_server_specific_secrets(tmp_path: Path) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text(
        """
[secrets.shared]
env_vars = ["SHARED"]

[server.alpha.secrets.shared]
env_vars = ["ALPHA_SHARED"]

[server.alpha.secrets.private]
"""
    )
    document = ConfigDocument.load(config_path=path)
    catalog = SecretCatalog.from_config(document, server="alpha")

    assert catalog.require("shared").server is None
    assert catalog.require("shared").env_vars == ("ALPHA_SHARED",)
    assert catalog.require("private").server == "alpha"


def test_catalog_rejects_unsupported_secret_fields(tmp_path: Path) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text(
        """
[secrets.example]
custom_location = "unsupported"
"""
    )
    document = ConfigDocument.load(config_path=path)

    with pytest.raises(SecretError, match="secret example has unsupported fields: custom_location"):
        SecretCatalog.from_config(document, server="alpha")


def test_resolver_uses_environment_before_keyring_and_redacts_repr() -> None:
    calls: list[tuple[str, str]] = []

    def keyring(service: str, account: str) -> str:
        calls.append((service, account))
        return "keyring-value"

    reference = SecretRef(
        name="api-key",
        env_vars=("API_KEY",),
    )
    resolved = SecretResolver({"API_KEY": "env-value"}, keyring_getter=keyring).require(reference)

    assert resolved.reveal() == "env-value"
    assert resolved.source == SecretSource.ENV
    assert "env-value" not in repr(resolved)
    assert calls == []

    fallback = SecretResolver({}, keyring_getter=keyring).require(reference)
    assert fallback.reveal() == "keyring-value"
    assert fallback.source == SecretSource.KEYRING
    assert calls == [("localmcp", "api-key")]

    server_reference = SecretRef(name="private-key", server="alpha")
    SecretResolver({}, keyring_getter=keyring).require(server_reference)
    assert calls[-1] == ("localmcp:alpha", "private-key")


def test_missing_secret_error_contains_locations_but_not_values() -> None:
    reference = SecretRef(server="alpha", name="token", env_vars=("TOKEN",))

    with pytest.raises(SecretNotFoundError, match="alpha/token"):
        SecretResolver({}, keyring_getter=lambda *_: None).require(reference)


def test_keyring_errors_may_fail_closed_or_be_tolerated() -> None:
    reference = SecretRef(server="alpha", name="token")

    def broken(*_: str) -> str:
        raise RuntimeError("backend details")

    with pytest.raises(SecretBackendError, match="RuntimeError"):
        SecretResolver({}, keyring_getter=broken).get(reference)
    assert SecretResolver({}, keyring_getter=broken, tolerate_keyring_errors=True).get(reference) is None


def test_secret_reference_validates_names_and_redacts_string_value() -> None:
    with pytest.raises(SecretError, match="must not be blank"):
        SecretRef(name="token", env_vars=(" ",))
    with pytest.raises(ValueError, match="server must be"):
        SecretRef(name="token", server="two..dots")

    resolved = SecretResolver({"TOKEN": "hunter2"}).require(SecretRef(name="token", env_vars=("TOKEN",)))
    assert str(resolved) == "<redacted>"
    assert "hunter2" not in repr(resolved)


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ("name = 1", "name must be a string"),
        ('env_vars = "TOKEN"', "env_vars must be an array of strings"),
        ('env_vars = ["TOKEN", 1]', "env_vars must be an array of strings"),
    ],
)
def test_catalog_rejects_malformed_declarations(tmp_path: Path, declaration: str, message: str) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text(f"[secrets.example]\n{declaration}\n")

    with pytest.raises(SecretError, match=message):
        SecretCatalog.from_config(ConfigDocument.load(config_path=path))


def test_catalog_disables_declarations_and_returns_defensive_mapping(tmp_path: Path) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text("[secrets.disabled]\ndisabled = true\n[secrets.enabled]\n")
    catalog = SecretCatalog.from_config(ConfigDocument.load(config_path=path))

    assert catalog.get("disabled") is None
    assert catalog.get("enabled") == SecretRef(name="enabled")
    copied = catalog.as_mapping()
    assert copied == {"enabled": SecretRef(name="enabled")}
    with pytest.raises(SecretError, match="declaration not found"):
        catalog.require("missing")


def test_resolver_skips_blank_aliases_and_blank_keyring_values() -> None:
    reference = SecretRef(name="token", env_vars=("FIRST", "SECOND"))
    calls: list[tuple[str, str]] = []

    def blank_keyring(service: str, account: str) -> str:
        calls.append((service, account))
        return " "

    resolver = SecretResolver({"FIRST": " ", "SECOND": ""}, keyring_getter=blank_keyring)
    assert resolver.get(reference) is None
    assert calls == [("localmcp", "token")]
