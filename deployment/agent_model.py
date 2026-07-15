"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

import os

import mlflow

_REQUIRED = ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL"]


def _get_answer(state: dict) -> str:
    """Extract final answer text from the graph output state."""
    from langchain_core.messages import AIMessage

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
        if getattr(msg, "type", None) == "ai":
            return msg.content
    return state.get("final_answer", "")


class DocumentAnalystModel(mlflow.pyfunc.ChatModel):
    """Wrap the LangGraph agent as an MLflow ChatModel.

    mlflow.pyfunc.ChatModel is the only model type that Databricks Model
    Serving returns as a single JSON object (not the batch-list wrapper that
    mlflow.langchain produces).  This makes the OpenAI SDK's
    resp.choices[0].message.content work out of the box.

    Initialization is deferred to load_context() so that MLflow can load the
    model artifact without requiring env vars at import time.  Databricks
    injects secret-referenced env vars before load_context() is called.
    """

    def __init__(self):
        self._graph = None

    def load_context(self, context):
        """Build the LangGraph agent after env vars are available."""
        missing = [v for v in _REQUIRED if not os.environ.get(v)]
        if missing:
            raise OSError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Configure them in the endpoint environment_vars (secret references)."
            )

        from agent.graph import build_graph, load_mcp_tools
        from config import get_chat_llm
        from rag.store import get_retriever

        # In the serving container, code_paths land at <model_dir>/code/ so the
        # server is at <model_dir>/code/tools/mcp_server.py.  Locally it is two
        # levels up at <repo_root>/tools/mcp_server.py.
        model_dir = os.path.dirname(os.path.abspath(__file__))
        server_path = os.path.join(model_dir, "code", "tools", "mcp_server.py")
        if not os.path.isfile(server_path):
            server_path = os.path.join(os.path.dirname(model_dir), "tools", "mcp_server.py")

        tools = load_mcp_tools(server_path)
        self._graph = build_graph(llm=get_chat_llm(), retriever=get_retriever(), tools=tools)

    def predict(self, context, messages, params=None):
        from mlflow.types.llm import ChatChoice, ChatCompletionResponse, ChatMessage

        input_msgs = [{"role": m.role, "content": m.content or ""} for m in messages]
        result = self._graph.invoke({"messages": input_msgs})
        answer = _get_answer(result)

        return ChatCompletionResponse(
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=answer),
                    finish_reason="stop",
                )
            ]
        )


mlflow.models.set_model(DocumentAnalystModel())
