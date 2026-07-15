"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""

from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


def _patch_mcp_win_stderr():
    """On Windows+Jupyter, MCP's stdio subprocess creation passes Jupyter's
    OutStream as errlog. OutStream has no fileno(), breaking subprocess.Popen.
    Patch _create_platform_compatible_process to force DEVNULL instead."""
    try:
        import subprocess as _sp

        import mcp.client.stdio as _stdio
        if getattr(_stdio, "_stderr_patched", False):
            return
        _orig = _stdio._create_platform_compatible_process
        async def _patched(command, args, env, errlog, cwd):
            return await _orig(command, args, env, _sp.DEVNULL, cwd)
        _stdio._create_platform_compatible_process = _patched
        _stdio._stderr_patched = True
    except Exception:
        pass


def load_mcp_tools(server_path: str | None = None):
    import concurrent.futures

    from langchain_mcp_adapters.client import MultiServerMCPClient

    _patch_mcp_win_stderr()
    mcp_url = os.environ.get("MCP_SERVER_URL")

    async def _load():
        if mcp_url:
            # Databricks Apps require an OAuth access token, not a plain PAT.
            # MCP_AUTH_TOKEN lets callers supply one (e.g. `databricks auth token`)
            # without disturbing DATABRICKS_TOKEN, which is used elsewhere as a PAT.
            token = os.environ.get("MCP_AUTH_TOKEN") or os.environ.get("DATABRICKS_TOKEN", "")
            client = MultiServerMCPClient({
                "analyst": {
                    "url": f"{mcp_url}/mcp",
                    "transport": "streamable_http",
                    "headers": {"Authorization": f"Bearer {token}"},
                }
            })
        else:
            _path = server_path
            if _path is None:
                _path = os.path.join(os.path.dirname(__file__), "..", "tools", "mcp_server.py")
            client = MultiServerMCPClient({
                "analyst": {
                    "command": sys.executable,
                    "args": [os.path.abspath(_path)],
                    "transport": "stdio",
                }
            })
        return await client.get_tools()

    try:
        return asyncio.run(_load())
    except RuntimeError:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _load()).result()


def make_mcp_node(tools, llm):
    def mcp_tools(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)

        if idx >= len(plan):
            return {"current_step_index": idx + 1}

        step = plan[idx]
        llm_with_tools = llm.bind_tools(tools)

        response = llm_with_tools.invoke([
            SystemMessage(content=MCP_STEP_PROMPT),
            HumanMessage(content=step),
        ])

        if response.tool_calls:
            import concurrent.futures
            tool_map = {t.name: t for t in tools}
            results = []
            for tc in response.tool_calls:
                tool = tool_map.get(tc["name"])
                if tool:
                    try:
                        out = asyncio.run(tool.ainvoke(tc["args"]))
                    except RuntimeError:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            out = pool.submit(asyncio.run, tool.ainvoke(tc["args"])).result()
                    if isinstance(out, list):
                        out = " ".join(
                            item.get("text", str(item)) if isinstance(item, dict) else str(item)
                            for item in out
                        )
                    results.append(str(out))
            result = "; ".join(results) if results else response.content
        else:
            result = response.content.strip()

        step_results = list(state.get("step_results", []))
        step_results.append(f"Step {idx + 1} (calculation): {result}")

        return {"step_results": step_results, "current_step_index": idx + 1}

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    from config import get_chat_llm
    from rag.store import get_retriever

    if llm is None:
        llm = get_chat_llm()
    if retriever is None:
        retriever = get_retriever()
    if tools is None:
        tools = load_mcp_tools()

    builder = StateGraph(AnalystState)
    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node(RAG, make_rag_agent(retriever, llm))
    builder.add_node(MCP, make_mcp_node(tools, llm))
    builder.add_node(SYNTH, make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges("supervisor", route_from_supervisor, {
        RAG: RAG,
        MCP: MCP,
        SYNTH: SYNTH,
    })
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(SYNTH, END)

    return builder.compile()
