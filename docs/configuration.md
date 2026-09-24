# Configuration contract

The default document is `$XDG_CONFIG_HOME/localmcp/localmcp.toml`, falling
back to `~/.config/localmcp/localmcp.toml`. It contains no resolved secret
values. Relative XDG directory values are ignored as required by the XDG Base
Directory specification.

`schema_version` is currently `1`. Every root value other than `server` is a
global default. Selecting MCP server `name` recursively merges `[server.name]`
over those defaults. Tables merge recursively; arrays and scalar values replace
the global value. The former root `tool` spelling is rejected; use `server`.

```toml
schema_version = 1

[observability]
log_level = "WARNING"

[llm]
backend = "openai_compatible"
base_url = "https://gateway.example/v1"

[server.server-a.llm]
backend = "openai_compatible"
```

The effective `server-a` view retains the global `llm.base_url` and inherits
`observability` unchanged. Unknown keys are retained. `STDIOServer` validates
the library-owned `llm`, `observability`, and `secrets` fields. When an
application supplies `config_parser`, it receives the effective `ServerConfig`
with those fields and `schema_version` removed, leaving only application-owned
keys to validate. Without a parser, that filtered `ServerConfig` is passed to
the runtime directly.

## LLM backend selection

Backend selection is explicit and never inferred from the credentials present.
`STDIOServer` validates the effective `[llm]` table and constructs its
`LLMBackendConfig`. The model IDs assigned to application-owned roles do not
change when deployment routing changes.

A gateway deployment uses one endpoint and logical secret:

```toml
[llm]
backend = "openai_compatible"
base_url = "https://gateway.example/v1"

[secrets.llm_api_key]
env_vars = ["GATEWAY_API_KEY"]
```

A native-provider deployment can replace those lines with native routing and
provider credentials:

```toml
[llm]
backend = "native"

[secrets.openai_api_key]
env_vars = ["OPENAI_API_KEY"]

[secrets.anthropic_api_key]
env_vars = ["ANTHROPIC_API_KEY"]
```

`GatewayModelFactory` selects `ChatOpenAI` or `ChatAnthropic` from the model's
independent gateway dialect while sharing the configured endpoint and key. This
allows, for example, an Anthropic-native model to use an OpenAI-dialect LiteLLM
endpoint. `NativeModelFactory`
uses `ModelSpec.native_provider` and never receives a gateway URL. A model with
no native provider is gateway-only. `ConfiguredModelFactory` provides the
stable `ModelFactory.create(model_id, ...)` boundary over both modes. Models
and transports are created lazily for each call and are closed by the returned
async context manager.

Server authors may select a different document at the loading boundary. This
defines which configuration and secret declarations are visible to that
runtime. Secret keyring boundaries are derived from logical server names, not
filesystem paths.

```python
from pathlib import Path

from localmcp.config import ConfigDocument

document = ConfigDocument.load(config_path=Path("/etc/example/localmcp.toml"))
values = document.for_server("server-a").values
```

Omitting `config_path` retains the XDG default described above. The selected
path is available as `document.path`.

## Secret declarations

`[secrets.<logical-name>]` declares a localmcp-wide secret. A matching
`[server.<name>.secrets.<logical-name>]` table overrides lookup metadata while
retaining unspecified global fields. A differently named declaration is
private to that server. Set `disabled = true` in a server declaration to remove
an inherited secret from that server's catalog.

Supported fields are:

- `name`: keyring account, defaulting to the logical name;
- `env_vars`: ordered environment aliases.

Global declarations use keyring service `localmcp`. A declaration introduced
under `[server.<server-name>.secrets]` uses `localmcp:<server-name>`; overriding
a global declaration does not change its global keyring location. Environment
aliases are always checked first; keyring is used only when no environment
alias provides a value.

## State paths

A selected document may configure one or more servers, while state remains
isolated:

- state: `$XDG_STATE_HOME/localmcp/<server-name>/`;
- cache: `$XDG_CACHE_HOME/localmcp/<server-name>/`; and
- log: `$XDG_STATE_HOME/localmcp/<server-name>/logs/<server-name>.log`.
