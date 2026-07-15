"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. Fill it in once your nodes are implemented.

TODO (Task 1.7 / Bonus A):
  - Build fake LLM / retriever / tool objects (no Databricks, no network).
  - Call `build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])`.
  - Invoke it on a combined retrieval+calculation query and assert that a plan was
    produced, both specialists ran, and the final answer surfaced on messages[-1].

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


def test_state_fields():
    from agent.state import AnalystState
    import typing
    hints = typing.get_type_hints(AnalystState, include_extras=True)
    for field in ("messages", "plan", "current_step_index", "step_results", "next_agent", "final_answer"):
        assert field in hints, f"Missing field: {field}"


def test_graph_compiles_offline():
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content='["find revenue"]', tool_calls=[])
    fake_llm.bind_tools.return_value = fake_llm

    fake_doc = MagicMock()
    fake_doc.page_content = "Net revenue was 16.91 trillion."
    fake_doc.metadata = {"source": "annual_report.pdf", "page": "4"}
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = [fake_doc]

    fake_tool = MagicMock()
    fake_tool.name = "calculate"

    from agent.graph import build_graph
    graph = build_graph(llm=fake_llm, retriever=fake_retriever, tools=[fake_tool])
    assert graph is not None


def test_route_from_supervisor():
    from agent.supervisor import route_from_supervisor
    from agent.state import AnalystState

    for agent in ("rag_agent", "mcp_tools", "synthesizer"):
        state = AnalystState(
            messages=[], plan=[], current_step_index=0,
            step_results=[], next_agent=agent, final_answer=""
        )
        assert route_from_supervisor(state) == agent
