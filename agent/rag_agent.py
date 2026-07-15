"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

TODO: Implement `make_rag_agent(retriever, llm)` returning a node that:
  - retrieves top-k chunks for the current step,
  - formats them with [source: file, p.N] citations,
  - extracts a single cited fact via the LLM (or 'not found in documents'),
  - appends the fact to step_results and increments current_step_index.
Reuse `rag/store.py::get_retriever()` so local and deployed retrieval match.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState


def format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "annual_report.pdf")
        page = doc.metadata.get("page", "")
        citation = f"source: {source}" + (f", p.{page}" if page else "")
        parts.append(f"[{i}] ({citation})\n{doc.page_content}")
    return "\n\n".join(parts)


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)

        if idx >= len(plan):
            return {"current_step_index": idx + 1}

        step = plan[idx]
        docs = retriever.invoke(step)

        if not docs:
            result = f"NOT FOUND IN DOCUMENT: {step}"
        else:
            context = format_docs(docs)
            response = llm.invoke([
                SystemMessage(content=RAG_EXTRACT_PROMPT),
                HumanMessage(content=f"Task: {step}\n\nContext:\n{context}"),
            ])
            result = response.content.strip()

        step_results = list(state.get("step_results", []))
        step_results.append(f"Step {idx + 1} (retrieval): {result}")

        return {"step_results": step_results, "current_step_index": idx + 1}

    return rag_agent
