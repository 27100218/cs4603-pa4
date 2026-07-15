"""Supervisor node + routing edge (Task 1.3).

TODO:
  - `make_supervisor(llm)`: if current_step_index >= len(plan) -> next_agent =
    'synthesizer'; else classify the current step to 'rag_agent' or 'mcp_tools'.
  - `route_from_supervisor(state)`: return state["next_agent"] for the
    conditional edge.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)

        if idx >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[idx]
        total = len(plan)

        response = llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"Step ({idx + 1}/{total}): {step}"),
        ])

        decision = response.content.strip().lower()

        if RAG in decision or "retriev" in decision or "document" in decision:
            next_agent = RAG
        elif MCP in decision or "tool" in decision or "calculat" in decision:
            next_agent = MCP
        elif SYNTH in decision:
            next_agent = SYNTH
        else:
            next_agent = decision if decision in (RAG, MCP, SYNTH) else RAG

        return {"next_agent": next_agent}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]
