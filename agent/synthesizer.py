"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        user_query = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_query = msg["content"]
                break

        step_results = state.get("step_results", [])
        results_text = "\n".join(step_results) if step_results else "No results collected."

        response = llm.invoke([
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(content=(
                f"Original question: {user_query}\n\n"
                f"Step results:\n{results_text}\n\n"
                "Write the final answer."
            )),
        ])

        answer = response.content.strip()
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    return synthesizer
