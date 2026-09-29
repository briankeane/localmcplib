"""Structured output for LangChain agents across models and gateways."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse, hook_config
from langchain.agents.middleware.types import PrivateStateAttr
from langchain.agents.structured_output import StructuredOutputError, StructuredOutputValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.runtime import Runtime
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


class SubmitResultError(StructuredOutputError):
    """Raised when a model exhausts its attempts without submitting a valid result."""

    def __init__(self, tool_name: str, attempts: int, ai_message: AIMessage) -> None:
        super().__init__(f"model did not submit a valid {tool_name} result after {attempts} attempts")
        self.tool_name = tool_name
        self.attempts = attempts
        self.ai_message = ai_message


class _SubmitResultState(AgentState[Any]):
    submit_result_attempts: NotRequired[Annotated[int, PrivateStateAttr]]


def _format_validation_error(exc: ValidationError) -> str:
    # Omit input values: they echo the model's own arguments and can be large.
    lines = [
        f"- {'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors(include_url=False, include_input=False)
    ]
    return "\n".join(lines)


class SubmitResultMiddleware[SchemaT: BaseModel](AgentMiddleware[_SubmitResultState, Any, Any]):
    """Collect structured output through an ordinary, unforced submission tool.

    ``ToolStrategy`` forces ``tool_choice``, which some models reject (for
    example with extended thinking), and ``ProviderStrategy`` depends on a
    native schema constraint that gateways may drop or that lets a model answer
    without using its other tools. This middleware instead registers a
    ``tool_name`` tool whose arguments are ``schema`` and leaves ``tool_choice``
    unset, so every tool-capable model can use it.

    A valid submission becomes ``structured_response`` and ends the run. A
    plain-text final answer is answered with a reminder, and an invalid
    submission with its validation errors, before returning to the model. Each
    counts as one attempt; exhausting ``max_attempts`` raises
    :class:`SubmitResultError`. Do not combine this with ``response_format``.
    """

    state_schema = _SubmitResultState

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        tool_name: str = "submit_result",
        description: str = "Submit your final structured result. Call this exactly once, as your last action.",
        max_attempts: int = 5,
    ) -> None:
        super().__init__()
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._schema = schema
        self._tool_name = tool_name
        self._max_attempts = max_attempts
        self._instruction = (
            f"When you are finished, call the `{tool_name}` tool exactly once with your final result. "
            "Do not give your final answer as plain text."
        )
        self.tools: list[BaseTool] = [
            StructuredTool.from_function(
                func=self._unexpected_execution,
                name=tool_name,
                description=description,
                args_schema=schema,
            )
        ]

    @property
    def name(self) -> str:
        return f"{type(self).__name__}[{self._tool_name}]"

    def _unexpected_execution(self, **_: Any) -> str:
        # after_model answers every submission before the tools node runs.
        return f"{self._tool_name} was not processed; call it again as your only tool call."

    def _with_instruction(self, request: ModelRequest) -> ModelRequest:
        system = request.system_message
        if system is None:
            return request.override(system_message=SystemMessage(content=self._instruction))
        if isinstance(system.content, str):
            content: str | list[str | dict[str, Any]] = f"{system.content}\n\n{self._instruction}"
        else:
            content = [*system.content, {"type": "text", "text": self._instruction}]
        return request.override(system_message=SystemMessage(content=content))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._with_instruction(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._with_instruction(request))

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state: _SubmitResultState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        message = state["messages"][-1] if state["messages"] else None
        if not isinstance(message, AIMessage):
            return None
        submissions = [call for call in message.tool_calls if call["name"] == self._tool_name]
        others = [call for call in message.tool_calls if call["name"] != self._tool_name]
        if others and not submissions:
            return None

        if len(submissions) == 1:
            try:
                result = self._schema.model_validate(submissions[0]["args"])
            except ValidationError as exc:
                feedback = f"Invalid {self._tool_name} arguments:\n{_format_validation_error(exc)}"
            else:
                replies = [ToolMessage(content="Result submitted.", tool_call_id=submissions[0]["id"] or "")]
                replies += [
                    ToolMessage(content="Not run: the result was already submitted.", tool_call_id=call["id"] or "")
                    for call in others
                ]
                return {"messages": replies, "structured_response": result, "jump_to": "end"}
        elif submissions:
            feedback = f"Call {self._tool_name} exactly once; this turn called it {len(submissions)} times."
        else:
            feedback = ""

        attempts = state.get("submit_result_attempts", 0) + 1
        if attempts >= self._max_attempts:
            raise SubmitResultError(self._tool_name, attempts, message)
        update: dict[str, Any] = {"submit_result_attempts": attempts}
        if submissions:
            update["messages"] = [
                ToolMessage(content=feedback, tool_call_id=call["id"] or "", status="error") for call in submissions
            ]
            # Other tool calls in the same turn still run; the tools node then returns to the model.
            if not others:
                update["jump_to"] = "model"
        else:
            update["messages"] = [HumanMessage(content=f"You have not submitted a result. {self._instruction}")]
            update["jump_to"] = "model"
        return update

    async def aafter_model(self, state: _SubmitResultState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
