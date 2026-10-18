"""Tests for handling truncated tool calls (finish_reason: length) across agents.

Verifies that when token limits or JSON truncation cause parse_error on tool calls,
agents do not execute the call with empty arguments or record ghost/empty findings.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openhack.agents.hunter import HunterAgent
from openhack.agents.validator import ValidatorAgent
from openhack.agents.base import BaseAgent
from openhack.agents.llm import ToolCall, LLMResponse, Message
from openhack.agents.session import Session
from openhack.tools.registry import ToolRegistry


class _TestAgent(BaseAgent):
    name = "test-agent"

    def get_system_prompt(self, context):
        return "Test."


@pytest.fixture
def session(tmp_path):
    return Session(str(tmp_path), event_log_path=tmp_path / "events.jsonl")


@pytest.fixture
def tools(tmp_path, session):
    return ToolRegistry(tmp_path, include_agent_tools=True, session=session)


def test_hunter_handle_report_finding_rejects_empty_args(session, tools):
    llm = MagicMock()
    hunter = HunterAgent(llm, tools, session, vuln_categories=["xss"])
    result = hunter._handle_report_finding({})
    assert result["status"] == "rejected"
    assert "Missing required fields" in result["reason"]
    assert len(hunter.findings) == 0


def test_hunter_handle_report_finding_rejects_missing_file_or_desc(session, tools):
    llm = MagicMock()
    hunter = HunterAgent(llm, tools, session, vuln_categories=["xss"])
    # Missing description
    res1 = hunter._handle_report_finding({"file_path": "app.py"})
    assert res1["status"] == "rejected"
    assert len(hunter.findings) == 0

    # Missing file_path
    res2 = hunter._handle_report_finding({"description": "Some bug"})
    assert res2["status"] == "rejected"
    assert len(hunter.findings) == 0


def test_hunter_handle_report_finding_accepts_valid_finding(session, tools):
    llm = MagicMock()
    hunter = HunterAgent(llm, tools, session, vuln_categories=["xss"])
    result = hunter._handle_report_finding({
        "file_path": "app.py",
        "description": "Cross-site scripting via query param",
        "category": "xss",
    })
    assert result["status"] == "recorded"
    assert len(hunter.findings) == 1
    assert hunter.findings[0]["file_path"] == "app.py"


@pytest.mark.asyncio
async def test_hunter_run_handles_truncated_tool_call(session, tools):
    mock_llm = MagicMock()
    hunter = HunterAgent(mock_llm, tools, session, vuln_categories=["sql_injection"])

    # Turn 1: Model emits a truncated report_finding tool call (hit max_tokens)
    truncated_tc = ToolCall(
        id="call_1",
        name="report_finding",
        arguments={},
        raw_arguments='{"category": "sql_injection", "file_path": "db.py", "descript',
        parse_error="Unterminated string starting at line 1 column 54",
    )
    turn1_resp = LLMResponse(
        content="",
        tool_calls=[truncated_tc],
        finish_reason="length",
    )

    # Turn 2: After getting truncated error, model sends complete report_finding and finish_hunt
    valid_tc = ToolCall(
        id="call_2",
        name="report_finding",
        arguments={
            "file_path": "db.py",
            "description": "SQL injection in query",
            "category": "sql_injection",
        },
        raw_arguments='{"category": "sql_injection", "file_path": "db.py", "description": "SQL injection in query"}',
        parse_error=None,
    )
    finish_tc = ToolCall(
        id="call_3",
        name="finish_hunt",
        arguments={},
        raw_arguments="{}",
        parse_error=None,
    )
    turn2_resp = LLMResponse(
        content="Hunt completed",
        tool_calls=[valid_tc, finish_tc],
        finish_reason="stop",
    )

    mock_llm.chat = AsyncMock(side_effect=[turn1_resp, turn2_resp])

    await hunter.run("Hunt for SQL injection")

    # Only the valid finding should have been recorded, not the truncated one
    assert len(hunter.findings) == 1
    assert hunter.findings[0]["file_path"] == "db.py"

    # Verify that the truncated tool call fed back an error message to the LLM
    tool_result_messages = [m for m in hunter.messages if m.role == "tool"]
    assert any("tool_call_truncated" in m.content for m in tool_result_messages)


def test_validator_handle_validate_finding_rejects_empty_status(session, tools):
    llm = MagicMock()
    validator = ValidatorAgent(llm, tools, session)
    result = validator._handle_validate_finding({})
    assert "error" in result
    assert "Missing required field: 'status'" in result["error"]
    assert len(validator.validated_findings) == 0
    assert len(validator.false_positives) == 0


@pytest.mark.asyncio
async def test_validator_run_handles_truncated_tool_call(session, tools):
    mock_llm = MagicMock()
    validator = ValidatorAgent(mock_llm, tools, session)

    truncated_tc = ToolCall(
        id="call_val_1",
        name="validate_finding",
        arguments={},
        raw_arguments='{"status": "confirmed", "evid',
        parse_error="Unterminated string",
    )
    turn1_resp = LLMResponse(
        content="",
        tool_calls=[truncated_tc],
        finish_reason="length",
    )
    finish_tc = ToolCall(
        id="call_val_2",
        name="finish_validation",
        arguments={},
        raw_arguments="{}",
        parse_error=None,
    )
    turn2_resp = LLMResponse(
        content="",
        tool_calls=[finish_tc],
        finish_reason="stop",
    )

    mock_llm.chat = AsyncMock(side_effect=[turn1_resp, turn2_resp])

    await validator.run("Validate findings")

    # Should not have recorded any false positive due to truncation
    assert len(validator.false_positives) == 0
    assert len(validator.validated_findings) == 0


@pytest.mark.asyncio
async def test_base_agent_handles_truncated_tool_call(session, tools):
    mock_llm = MagicMock()
    agent = _TestAgent(mock_llm, tools, session)

    truncated_tc = ToolCall(
        id="call_base_1",
        name="read_file",
        arguments={},
        raw_arguments='{"path": "secret',
        parse_error="Unterminated string",
    )
    turn1_resp = LLMResponse(
        content="",
        tool_calls=[truncated_tc],
        finish_reason="length",
    )
    finish_tc = ToolCall(
        id="call_base_2",
        name="finish_task",
        arguments={"summary": "done", "reason": "completed", "verification": "checked"},
        raw_arguments='{"summary": "done", "reason": "completed", "verification": "checked"}',
        parse_error=None,
    )
    turn2_resp = LLMResponse(
        content="All done",
        tool_calls=[finish_tc],
        finish_reason="stop",
    )

    mock_llm.chat = AsyncMock(side_effect=[turn1_resp, turn2_resp])

    await agent.run("Do something")

    tool_result_messages = [m for m in agent.messages if m.role == "tool"]
    assert any("tool_call_truncated" in m.content for m in tool_result_messages)
