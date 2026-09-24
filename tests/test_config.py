from pathlib import Path

import pytest

from localmcp.config import ConfigDocument, ConfigError, LocalMCPPaths, merge_mappings, read_toml, validate_identifier


def test_xdg_paths_use_one_shared_document_and_isolated_server_state(tmp_path: Path) -> None:
    paths = LocalMCPPaths.from_environment(
        {
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        },
        home=tmp_path / "ignored",
    )

    assert paths.config_file == tmp_path / "config/localmcp/localmcp.toml"
    assert paths.server_state_dir("server-a") == tmp_path / "state/localmcp/server-a"
    assert paths.server_log_file("server-b") == tmp_path / "state/localmcp/server-b/logs/server-b.log"


def test_relative_xdg_paths_are_ignored(tmp_path: Path) -> None:
    paths = LocalMCPPaths.from_environment(
        {
            "XDG_CONFIG_HOME": "relative/config",
            "XDG_STATE_HOME": "relative/state",
            "XDG_CACHE_HOME": "relative/cache",
        },
        home=tmp_path,
    )

    assert paths.config_file == tmp_path / ".config/localmcp/localmcp.toml"
    assert paths.server_state_dir("server-a") == tmp_path / ".local/state/localmcp/server-a"
    assert paths.server_cache_dir("server-a") == tmp_path / ".cache/localmcp/server-a"


def test_server_view_deep_merges_global_configuration(tmp_path: Path) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text(
        """
schema_version = 1

[llm]
base_url = "https://gateway.example/v1"
model = "global-model"

[limits]
calls = 10

[server.server-a.llm]
model = "server-model"

[server.server-a.limits]
seconds = 30
"""
    )

    config = ConfigDocument.load(config_path=path, missing_ok=False)

    assert config.server_names == ("server-a",)
    assert config.for_server("server-a").values["llm"] == {
        "base_url": "https://gateway.example/v1",
        "model": "server-model",
    }
    assert config.for_server("server-a").values["limits"] == {"calls": 10, "seconds": 30}
    assert "server" not in config.for_server("server-a").values


def test_secret_mappings_share_global_declarations_and_isolate_private_ones(tmp_path: Path) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text(
        """
[secrets.llm]
env_vars = ["SHARED_KEY"]

[server.server-a.secrets.llm]
env_vars = ["SERVER_A_KEY"]

[server.server-a.secrets.private]
env_vars = ["PRIVATE_KEY"]
"""
    )
    config = ConfigDocument.load(config_path=path)

    global_secrets = config.secret_mappings()
    server_secrets = config.secret_mappings("server-a")

    assert global_secrets["llm"] == {"env_vars": ["SHARED_KEY"]}
    assert server_secrets["llm"] == {
        "env_vars": ["SERVER_A_KEY"],
    }
    assert server_secrets["private"] == {"env_vars": ["PRIVATE_KEY"]}
    assert "private" not in global_secrets


def test_invalid_schema_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text("schema_version = 2\n")

    with pytest.raises(ConfigError, match="unsupported localmcp schema_version 2; expected 1"):
        ConfigDocument.load(config_path=path)


def test_legacy_tool_root_without_version_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text('[tool.server-a]\nmode = "legacy"\n')

    with pytest.raises(ConfigError, match="root 'tool' table is no longer supported; use 'server' instead"):
        ConfigDocument.load(config_path=path)


def test_explicit_config_paths_isolate_server_views_and_secret_declarations(tmp_path: Path) -> None:
    first_path = tmp_path / "first.toml"
    second_path = tmp_path / "second.toml"
    first_path.write_text(
        """
[server.app]
boundary = "first"

[server.app.secrets.first_only]
env_vars = ["FIRST_ONLY"]
"""
    )
    second_path.write_text(
        """
[server.app]
boundary = "second"

[server.app.secrets.second_only]
env_vars = ["SECOND_ONLY"]
"""
    )

    first = ConfigDocument.load(config_path=first_path)
    second = ConfigDocument.load(config_path=second_path)

    assert first.path == first_path
    assert second.path == second_path
    assert first.for_server("app").values["boundary"] == "first"
    assert second.for_server("app").values["boundary"] == "second"
    assert set(first.secret_mappings("app")) == {"first_only"}
    assert set(second.secret_mappings("app")) == {"second_only"}


def test_merge_replaces_lists_and_does_not_alias_inputs() -> None:
    first = {"nested": {"keep": 1, "items": [1]}}
    merged = merge_mappings(first, {"nested": {"items": [2]}})
    merged["nested"]["items"].append(3)

    assert merged["nested"]["keep"] == 1
    assert first["nested"]["items"] == [1]


@pytest.mark.parametrize("value", ["", "-leading", "trailing-", "two..dots", "x" * 129])
def test_identifiers_reject_unsafe_path_components(value: str) -> None:
    with pytest.raises(ConfigError, match="server must be"):
        validate_identifier(value, field="server")


def test_missing_and_malformed_documents_fail_predictably(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    assert read_toml(missing) == {}
    with pytest.raises(ConfigError, match="does not exist"):
        read_toml(missing, missing_ok=False)

    malformed = tmp_path / "malformed.toml"
    malformed.write_text("value = [")
    with pytest.raises(ConfigError, match="failed to read"):
        read_toml(malformed)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("schema_version = true\n", "schema_version must be an integer"),
        ('schema_version = "1"\n', "schema_version must be an integer"),
        ('server = "invalid"\n', "server must be a TOML table"),
    ],
)
def test_document_rejects_invalid_root_shapes(tmp_path: Path, contents: str, message: str) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text(contents)

    with pytest.raises(ConfigError, match=message):
        ConfigDocument.load(config_path=path)


def test_server_and_secret_tables_must_be_mappings(tmp_path: Path) -> None:
    path = tmp_path / "localmcp.toml"
    path.write_text('server = { alpha = "invalid" }\nsecrets = { token = "invalid" }\n')
    document = ConfigDocument.load(config_path=path)

    with pytest.raises(ConfigError, match="server.alpha must be a TOML table"):
        document.for_server("alpha")
    with pytest.raises(ConfigError, match="secrets.token must be a TOML table"):
        document.secret_mappings()


def test_server_secret_table_and_names_are_validated(tmp_path: Path) -> None:
    invalid_table = tmp_path / "invalid-table.toml"
    invalid_table.write_text('server = { alpha = { secrets = "invalid" } }\n')
    with pytest.raises(ConfigError, match="server.alpha.secrets must be a TOML table"):
        ConfigDocument.load(config_path=invalid_table).secret_mappings("alpha")

    invalid_name = tmp_path / "invalid-name.toml"
    invalid_name.write_text('secrets = { "two..dots" = {} }\n')
    with pytest.raises(ConfigError, match="secrets name must be"):
        ConfigDocument.load(config_path=invalid_name).secret_mappings()
