import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, StructuredOutputValidationError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from localmcp.structured_output import FencedJSONOutputMiddleware, parse_fenced_json


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
