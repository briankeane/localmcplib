import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from localmcp.structured_output import (
    SubmitResultError,
    SubmitResultMiddleware,
)


class Answer(BaseModel):
    value: int


class _RecordingChatModel(GenericFakeChatModel):
    bound: list[dict] = []
    systems: list[str] = []
    requests: list[list] = []

    def bind_tools(self, tools, **kwargs):
        self.bound.append({"tools": [getattr(tool, "name", None) for tool in tools], **kwargs})
        return self

    def _generate(self, messages, *args, **kwargs):
        self.systems.append(messages[0].text if messages and messages[0].type == "system" else "")
        self.requests.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _submit(args, call_id="s1"):
    return {"name": "submit_result", "args": args, "id": call_id, "type": "tool_call"}


def _unparsed(name, call_id, args='{"value": 3'):
    return {"name": name, "args": args, "id": call_id, "error": "invalid JSON", "type": "invalid_tool_call"}


def _submit_agent(turns, *, tools=(), max_attempts=5, checkpointer=None):
    model = _RecordingChatModel(messages=iter(turns), bound=[], systems=[], requests=[])
    agent = create_agent(
        model,
        list(tools),
        system_prompt="Base prompt.",
        middleware=[SubmitResultMiddleware(Answer, max_attempts=max_attempts)],
        checkpointer=checkpointer,
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


def _lookup_tool(ran):
    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        ran.append(query)
        return "42"

    return lookup


async def test_attempts_restart_on_each_run_of_a_checkpointed_thread():
    agent, _ = _submit_agent(
        [
            AIMessage(content="no"),
            AIMessage(content="", tool_calls=[_submit({"value": 1}, "s1")]),
            AIMessage(content="no"),
            AIMessage(content="", tool_calls=[_submit({"value": 2}, "s2")]),
        ],
        max_attempts=2,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}

    first = await agent.ainvoke({"messages": [("user", "go")]}, config)
    second = await agent.ainvoke({"messages": [("user", "again")]}, config)

    assert first["structured_response"] == Answer(value=1)
    assert second["structured_response"] == Answer(value=2)


async def test_unparseable_submission_is_repaired_and_answered():
    agent, model = _submit_agent(
        [
            AIMessage(content="", invalid_tool_calls=[_unparsed("submit_result", "s1")]),
            AIMessage(content="", tool_calls=[_submit({"value": 3}, "s2")]),
        ]
    )

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert result["structured_response"] == Answer(value=3)
    replayed = next(m for m in model.requests[1] if m.type == "ai")
    assert replayed.invalid_tool_calls == []
    assert [(c["name"], c["args"], c["id"]) for c in replayed.tool_calls] == [("submit_result", {}, "s1")]
    errors = [m for m in model.requests[1] if m.type == "tool"]
    assert [(m.tool_call_id, m.status) for m in errors] == [("s1", "error")]
    assert "not a valid JSON object" in errors[0].text
    assert not any(m.type == "human" and "not submitted" in m.text for m in result["messages"])


async def test_unparseable_calls_to_other_tools_are_not_run_and_count_as_attempts():
    ran = []
    agent, model = _submit_agent(
        [AIMessage(content="", invalid_tool_calls=[_unparsed("lookup", f"l{i}", "{")]) for i in range(10)],
        tools=[_lookup_tool(ran)],
        max_attempts=2,
    )

    with pytest.raises(SubmitResultError):
        await agent.ainvoke({"messages": [("user", "go")]})

    assert ran == []
    assert len(model.requests) == 2
    assert all(not m.invalid_tool_calls for m in model.requests[1] if m.type == "ai")


async def test_unparseable_call_beside_a_valid_call_does_not_use_an_attempt():
    ran = []
    agent, _ = _submit_agent(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {"query": "q"}, "id": "l1", "type": "tool_call"}],
                invalid_tool_calls=[_unparsed("lookup", "l2", "{")],
            ),
            AIMessage(content="", tool_calls=[_submit({"value": 42}, "s1")]),
        ],
        tools=[_lookup_tool(ran)],
        max_attempts=1,
    )

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert ran == ["q"]
    assert result["structured_response"] == Answer(value=42)
    answered = {m.tool_call_id: m.status for m in result["messages"] if m.type == "tool"}
    assert answered == {"l1": "success", "l2": "error", "s1": "success"}


async def test_valid_submission_answers_unparseable_siblings():
    agent, _ = _submit_agent(
        [
            AIMessage(
                content="",
                tool_calls=[_submit({"value": 5}, "s1")],
                invalid_tool_calls=[_unparsed("lookup", "l1", "{")],
            )
        ],
        tools=[_lookup_tool([])],
    )

    result = await agent.ainvoke({"messages": [("user", "go")]})

    assert result["structured_response"] == Answer(value=5)
    assert {m.tool_call_id for m in result["messages"] if m.type == "tool"} == {"s1", "l1"}
    assert not any(m.invalid_tool_calls for m in result["messages"] if m.type == "ai")
