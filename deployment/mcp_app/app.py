"""Standalone MCP server as a Databricks App (Bonus C).

Reuses the GIVEN tool definitions from tools/mcp_server.py and serves them
over the streamable-http transport instead of stdio. Decouples the tool server
from the model container so both can scale and deploy independently.

Deploy:
  databricks apps create cs4603-mcp-tools
  databricks apps deploy cs4603-mcp-tools --source-code-path <workspace-path>

Then set MCP_SERVER_URL=https://<app-url> in the serving endpoint environment_vars
and update agent/graph.py to connect over HTTP when MCP_SERVER_URL is set.

NOTE: this app's source-code-path is deployment/mcp_app/ itself (app.yaml must
sit at the root of the synced project per Databricks Apps convention), so it
imports the self-contained mcp_server.py copy in this same directory rather
than reaching out to tools/mcp_server.py in the repo root.
"""

from __future__ import annotations

import os

from mcp_server import mcp

if __name__ == "__main__":
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport="streamable-http")
