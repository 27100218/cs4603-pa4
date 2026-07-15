"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        user_query = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_query = msg["content"]
                break

        response = llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=f"Query: {user_query}"),
        ])

        raw = response.content.strip()
        try:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            plan = json.loads(match.group() if match else raw)
            if not isinstance(plan, list) or not plan:
                raise ValueError
            plan = [str(s) for s in plan]
        except (json.JSONDecodeError, ValueError, AttributeError):
            plan = [user_query]

        return {"plan": plan, "current_step_index": 0, "step_results": []}

    return planner
