import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, StructuredOutputValidationError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from localmcp.structured_output import (
    FencedJSONOutputMiddleware,
    SubmitResultError,
    SubmitResultMiddleware,
    parse_fenced_json,
)


class Answer(BaseModel):
    value: int


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _agent(text: str):
    model = _FakeChatModel(messages=iter([AIMessage(content=text)]))
    return create_agent(
        model,
        [],
        response_format=ProviderStrategy(Answer),
        middleware=[FencedJSONOutputMiddleware(Answer)],
    )


@pytest.mark.parametrize(
    "text",
    ['{"value": 3}', '```json\n{"value": 3}\n```', '\n```JSON\n{"value": 3}\n```\n', '```\n{"value": 3}\n```'],
)
async def test_plain_and_fenced_json_parse(text):
    result = await _agent(text).ainvoke({"messages": [("user", "go")]})

    assert result["structured_response"] == Answer(value=3)


@pytest.mark.parametrize(
    "text",
    [
        'Here you go:\n```json\n{"value": 3}\n```',
        '```json\n{"value": "three"}\n```',
        '```json\n{"value": 3}\n```\n```json\n{"value": 4}\n```',
    ],
)
async def test_prose_invalid_or_multiple_fences_still_fail(text):
    with pytest.raises(StructuredOutputValidationError):
        await _agent(text).ainvoke({"messages": [("user", "go")]})


def test_sync_invocation_also_recovers():
    result = _agent('```json\n{"value": 3}\n```').invoke({"messages": [("user", "go")]})

    assert result["structured_response"] == Answer(value=3)


def test_parse_fenced_json_requires_a_single_valid_fence():
    assert parse_fenced_json('```json\n{"value": 3}\n```', Answer) == Answer(value=3)
    assert parse_fenced_json('{"value": 3}', Answer) is None
    assert parse_fenced_json('```json\n{"value": []}\n```', Answer) is None


class _RecordingChatModel(GenericFakeChatModel):
    bound: list[dict] = []
    systems: list[str] = []

    def bind_tools(self, tools, **kwargs):
        self.bound.append({"tools": [getattr(tool, "name", None) for tool in tools], **kwargs})
        return self

    def _generate(self, messages, *args, **kwargs):
        self.systems.append(messages[0].text if messages and messages[0].type == "system" else "")
        return super()._generate(messages, *args, **kwargs)


def _submit(args, call_id="s1"):
    return {"name": "submit_result", "args": args, "id": call_id, "type": "tool_call"}


def _submit_agent(turns, *, tools=(), max_attempts=5):
    model = _RecordingChatModel(messages=iter(turns), bound=[], systems=[])
    agent = create_agent(
        model,
        list(tools),
        system_prompt="Base prompt.",
        middleware=[SubmitResultMiddleware(Answer, max_attempts=max_attempts)],
    )
    return agent, model


async def test_submit_result_returns_structured_response_without_forcing_tool_choice():
    agent, model = _submit_agent([AIMessage(content="", tool_calls=[_submit({"value": 3})])])

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert result["structured_response"] == Answer(value=3)
    assert model.bound[0]["tools"] == ["submit_result"]
    assert model.bound[0].get("tool_choice") is None
    assert model.systems[0].startswith("Base prompt.\n\n")
    assert "`submit_result`" in model.systems[0]


async def test_plain_text_answer_is_reminded_then_accepted():
    agent, model = _submit_agent(
        [AIMessage(content="Looks fine to me."), AIMessage(content="", tool_calls=[_submit({"value": 4})])]
    )

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert result["structured_response"] == Answer(value=4)
    reminders = [m for m in result["messages"] if m.type == "human" and "not submitted" in m.text]
    assert len(reminders) == 1
    assert len(model.systems) == 2


async def test_invalid_submission_returns_validation_errors_to_the_model():
    agent, _ = _submit_agent(
        [
            AIMessage(content="", tool_calls=[_submit({"value": "three"}, "s1")]),
            AIMessage(content="", tool_calls=[_submit({"value": 3}, "s2")]),
        ]
    )

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert result["structured_response"] == Answer(value=3)
    error = next(m for m in result["messages"] if m.type == "tool" and m.tool_call_id == "s1")
    assert error.status == "error"
    assert "value:" in error.text
    assert "three" not in error.text


async def test_other_tool_calls_in_an_invalid_turn_still_run():
    ran = []

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        ran.append(query)
        return "42"

    agent, _ = _submit_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup", "args": {"query": "q"}, "id": "l1", "type": "tool_call"},
                    _submit({}, "s1"),
                ],
            ),
            AIMessage(content="", tool_calls=[_submit({"value": 42}, "s2")]),
        ],
        tools=[lookup],
    )

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert ran == ["q"]
    assert result["structured_response"] == Answer(value=42)


async def test_valid_submission_ends_without_running_sibling_tool_calls():
    ran = []

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        ran.append(query)
        return "42"

    agent, _ = _submit_agent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup", "args": {"query": "q"}, "id": "l1", "type": "tool_call"},
                    _submit({"value": 1}, "s1"),
                ],
            )
        ],
        tools=[lookup],
    )

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert ran == []
    assert result["structured_response"] == Answer(value=1)
    assert {m.tool_call_id for m in result["messages"] if m.type == "tool"} == {"l1", "s1"}


async def test_repeated_submissions_in_one_turn_count_as_a_failed_attempt():
    agent, _ = _submit_agent(
        [
            AIMessage(content="", tool_calls=[_submit({"value": 1}, "s1"), _submit({"value": 2}, "s2")]),
            AIMessage(content="", tool_calls=[_submit({"value": 2}, "s3")]),
        ]
    )

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert result["structured_response"] == Answer(value=2)


@pytest.mark.parametrize("max_attempts", [1, 2, 5])
async def test_attempts_are_capped(max_attempts):
    agent, model = _submit_agent([AIMessage(content="no") for _ in range(10)], max_attempts=max_attempts)

    with pytest.raises(SubmitResultError) as exc_info:
        await agent.ainvoke({"messages": [("user", "go")]})

    assert exc_info.value.attempts == max_attempts
    assert len(model.systems) == max_attempts


def test_default_attempt_cap_is_five():
    agent, model = _submit_agent([AIMessage(content="no") for _ in range(10)])

    with pytest.raises(SubmitResultError):
        agent.invoke({"messages": [("user", "go")]})

    assert len(model.systems) == 5


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        SubmitResultMiddleware(Answer, max_attempts=0)
