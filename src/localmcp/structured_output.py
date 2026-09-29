"""Tolerant parsing for native structured output from LangChain agents."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import StructuredOutputValidationError
from pydantic import BaseModel, ValidationError

# One Markdown code fence wrapping the entire response, optionally tagged json.
_JSON_FENCE_PATTERN = re.compile(r"\A\s*```(?:json)?[ \t]*\n(.*)\n[ \t]*```\s*\Z", re.DOTALL | re.IGNORECASE)


def parse_fenced_json[SchemaT: BaseModel](text: str, schema: type[SchemaT]) -> SchemaT | None:
    """Validate ``text`` as ``schema`` after removing one enclosing JSON fence.

    Returns ``None`` when ``text`` is not exactly one fenced block or the
    unwrapped content does not validate.
    """
    match = _JSON_FENCE_PATTERN.fullmatch(text)
    if match is None:
        return None
    try:
        return schema.model_validate_json(match.group(1))
    except ValidationError:
        return None


class FencedJSONOutputMiddleware(AgentMiddleware):
    """Accept ``ProviderStrategy`` output wrapped in one Markdown JSON fence.

    Some gateways drop the provider-side schema constraint, so a model may
    return otherwise valid JSON inside a code fence. The unwrapped text must
    still validate against the same schema; anything else re-raises.
    """

    def __init__(self, schema: type[BaseModel]) -> None:
        super().__init__()
        self._schema = schema

    def _recover(self, exc: StructuredOutputValidationError) -> ModelResponse:
        structured = parse_fenced_json(exc.ai_message.text, self._schema)
        if structured is None:
            raise exc
        return ModelResponse(result=[exc.ai_message], structured_response=structured)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        try:
            return handler(request)
        except StructuredOutputValidationError as exc:
            return self._recover(exc)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        try:
            return await handler(request)
        except StructuredOutputValidationError as exc:
            return self._recover(exc)
