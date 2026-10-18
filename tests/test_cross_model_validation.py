"""
Tests for P0 Cross-Model Validation, Self-Validation Defense, and Consensus Voting.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from openhack.config import (
    Settings,
    check_self_validation,
    resolve_validator_model,
    settings,
)
from openhack.agents.validator_swarm import ValidatorSwarmAgent
from openhack.prompts.validator import VALIDATOR_PROMPT


class TestCrossModelResolution:
    def test_default_cross_model_mapping_openhack(self):
        # On OpenHack provider, hunter gpt-5.6-sol should cross-validate with kimi-k2.5
        val_model = resolve_validator_model(
            hunter_model="gpt-5.6-sol",
            requested_validator_model=None,
            provider="openhack",
        )
        assert val_model == "kimi-k2.5"

        # Hunter kimi-k2.5 should cross-validate with gpt-5.6-sol
        val_model_k25 = resolve_validator_model(
            hunter_model="kimi-k2.5",
            requested_validator_model=None,
            provider="openhack",
        )
        assert val_model_k25 == "gpt-5.6-sol"

    def test_explicit_validator_model_override_wins(self):
        val_model = resolve_validator_model(
            hunter_model="gpt-5.6-sol",
            requested_validator_model="claude-sonnet-5",
            provider="openhack",
        )
        assert val_model == "claude-sonnet-5"

    def test_self_validation_check_detects_same_model(self):
        msg = check_self_validation("gpt-5.6-sol", "gpt-5.6-sol")
        assert msg is not None
        assert "Self-validation detected" in msg
        assert "gpt-5.6-sol" in msg

        # Case-insensitive match
        msg_upper = check_self_validation("GPT-5.6-SOL", "gpt-5.6-sol")
        assert msg_upper is not None
        assert "Self-validation detected" in msg_upper

    def test_self_validation_check_returns_none_for_different_models(self):
        msg = check_self_validation("gpt-5.6-sol", "kimi-k2.5")
        assert msg is None


class TestValidatorSwarmConsensus:
    @pytest.fixture
    def mock_session(self):
        session = MagicMock()
        session.add_trace = MagicMock()
        session.cancelled = False
        return session

    @pytest.fixture
    def mock_tools(self):
        return MagicMock()

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.model = "gpt-5.6-sol"
        llm.provider = "openhack"
        llm.prompt_cache_key = None
        llm.total_cost = 0.01
        llm.total_tokens = 500
        llm.total_input_tokens = 400
        llm.total_output_tokens = 100
        return llm

    @pytest.mark.asyncio
    async def test_self_validation_warning_logged_when_models_equal(self, mock_llm, mock_tools, mock_session):
        with patch.object(settings, "validator_model_id", "gpt-5.6-sol"), \
             patch.object(settings, "hunter_model_id", "gpt-5.6-sol"):
            swarm = ValidatorSwarmAgent(mock_llm, mock_tools, mock_session)
            await swarm.run("validate", context={"hunter": {"findings": []}})

            # Finding list was empty, so returns early without swarm start
            result = await swarm.run("validate", context={
                "hunter": {"findings": [{"file_path": "auth.py", "category": "xss", "description": "test"}]}
            })

            # Check that warning trace event was recorded
            warning_traces = [
                call.kwargs["content"]
                for call in mock_session.add_trace.call_args_list
                if call.kwargs.get("event_type") == "warning"
            ]
            assert len(warning_traces) >= 1
            assert warning_traces[0]["warning"] == "self_validation_detected"

    @pytest.mark.asyncio
    async def test_strict_majority_2_of_3_confirms_finding(self, mock_llm, mock_tools, mock_session):
        # 3 votes: 2 confirmed, 1 false positive -> should be confirmed
        with patch.object(settings, "validator_votes", 3), \
             patch.object(settings, "max_concurrent_validators", 5):
            swarm = ValidatorSwarmAgent(mock_llm, mock_tools, mock_session)

            candidate_finding = {
                "file_path": "server.py",
                "category": "sqli",
                "description": "SQL injection in query",
            }
            context = {"hunter": {"findings": [candidate_finding]}}

            run_count = 0
            async def mock_validator_run(task, ctx):
                nonlocal run_count
                run_count += 1
                if run_count in (1, 2):
                    return {
                        "validated_findings": [{
                            "original_index": 0,
                            "status": "confirmed",
                            "confidence": "high",
                            "evidence": f"Confirmed in trial {run_count}",
                            "poc": "import requests",
                            "fix": "Use parameterized query",
                        }],
                        "false_positives": [],
                        "type": "validation_complete",
                    }
                else:
                    return {
                        "validated_findings": [],
                        "false_positives": [{
                            "original_index": 0,
                            "status": "false_positive",
                            "confidence": "medium",
                            "evidence": "Uncertain",
                        }],
                        "type": "validation_complete",
                    }

            with patch("openhack.agents.validator_swarm.ValidatorAgent.run", side_effect=mock_validator_run):
                res = await swarm.run("validate", context=context)

            assert len(res["validated_findings"]) == 1
            assert res["validated_findings"][0]["status"] == "confirmed"
            assert len(res["false_positives"]) == 0

    @pytest.mark.asyncio
    async def test_strict_majority_tie_is_rejected_not_confirmed(self, mock_llm, mock_tools, mock_session):
        # 2 votes: 1 confirmed, 1 false positive -> tie must NOT confirm (strict majority required)
        with patch.object(settings, "validator_votes", 2), \
             patch.object(settings, "max_concurrent_validators", 5):
            swarm = ValidatorSwarmAgent(mock_llm, mock_tools, mock_session)

            candidate_finding = {
                "file_path": "api.py",
                "category": "csrf",
                "description": "CSRF on state change",
            }
            context = {"hunter": {"findings": [candidate_finding]}}

            run_count = 0
            async def mock_validator_run(task, ctx):
                nonlocal run_count
                run_count += 1
                if run_count == 1:
                    return {
                        "validated_findings": [{
                            "original_index": 0,
                            "status": "confirmed",
                            "evidence": "Trial 1 thinks confirmed",
                        }],
                        "false_positives": [],
                        "type": "validation_complete",
                    }
                else:
                    return {
                        "validated_findings": [],
                        "false_positives": [{
                            "original_index": 0,
                            "status": "false_positive",
                            "evidence": "Trial 2 found SameSite=Lax cookie",
                        }],
                        "type": "validation_complete",
                    }

            with patch("openhack.agents.validator_swarm.ValidatorAgent.run", side_effect=mock_validator_run):
                res = await swarm.run("validate", context=context)

            # Tie must NOT be confirmed
            assert len(res["validated_findings"]) == 0
            assert len(res["false_positives"]) == 1
            assert res["false_positives"][0]["status"] == "false_positive"


class TestHardenedValidatorPromptChecks:
    def test_prompt_includes_csrf_samesite_check(self):
        assert "SameSite=Lax" in VALIDATOR_PROMPT
        assert "SameSite=Strict" in VALIDATOR_PROMPT
        assert "Set-Cookie" in VALIDATOR_PROMPT

    def test_prompt_includes_xxe_external_entity_check(self):
        assert "XXE" in VALIDATOR_PROMPT
        assert "EXTERNAL entities" in VALIDATOR_PROMPT
        assert "xml2js" in VALIDATOR_PROMPT

    def test_prompt_includes_xss_raw_html_sink_check(self):
        assert "dangerouslySetInnerHTML" in VALIDATOR_PROMPT
        assert "RAW-HTML sink" in VALIDATOR_PROMPT
        assert "auto-escaped" in VALIDATOR_PROMPT

    def test_prompt_includes_framework_default_and_blank_rationale(self):
        assert "Nuxt DevTools" in VALIDATOR_PROMPT or "Nuxt devtools" in VALIDATOR_PROMPT.lower()
        assert "Blank / Vague Rationale" in VALIDATOR_PROMPT

    def test_prompt_defines_false_positive_strictly(self):
        assert "MITIGATED, NOT PRESENT, NOT REACHABLE, or UNEXPLOITABLE" in VALIDATOR_PROMPT
