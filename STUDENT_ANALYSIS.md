# CS4603 PA4 — Document Analyst (Student Submission)

> This is your **submission file**. `README.md` is the assignment spec — this document is where you write up your work.
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment spec.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

`.env` values used for this submission (redacted): `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_MODEL`,
`UC_CATALOG=cs4603`, `UC_SCHEMA=default`, `SECRET_SCOPE=cs4603-deploy`, `SERVING_ENDPOINT_NAME=yahya-document-analyst`.

## Running locally

The corpus (`data/annual_report.pdf`) was ingested once from a Databricks notebook into the
Vector Search index configured by `VECTOR_SEARCH_ENDPOINT` / `VECTOR_SEARCH_INDEX`. The graph
was built and exercised in `pa4.ipynb` (Part 1, Task 1.7 cells) with three test queries:

| Query | Answer produced |
|-------|-----------------|
| "What was the net income in 2023?" | ¥1,107 billion, source: annual_report.pdf |
| "What is 15% of 2.4 billion?" | 360,000,000 (3.6e8) |
| "What was the net revenue in 2023, and what would it be after 10% growth for 3 years?" | ¥16,910 billion → ¥22,507.21 billion (16,910 × 1.1³) |

See `pa4.ipynb` cells under "Task 1.7 — Wire the Full Graph" for full outputs and the
step-by-step execution trace for the combined query.

## Deployment

**Part 2 (manual MLflow, `deployment/deploy.py`):**
- Model logged with `mlflow.pyfunc.save_model` (models-from-code) using `deployment/agent_model.py`,
  registered to `cs4603.default.yahya_document_analyst` in Unity Catalog.
- Served on endpoint `yahya-document-analyst` (workload size Small, `scale_to_zero_enabled=True`),
  with `DATABRICKS_HOST` / `DATABRICKS_TOKEN` / `DATABRICKS_MODEL` injected as secret refs from the
  `cs4603-deploy` secret scope.
- Verified live via curl and the OpenAI SDK (`pa4.ipynb`, Task 2.4 cells) — both returned correct,
  sourced answers, e.g. "The net income in 2023 was ¥1,107 billion... (Source: dbfs:/Volumes/cs4603/default/pa4/annual_report.pdf)."

**Bonus B (`agents.deploy()`, `deployment/deploy_agents.py`):**
- Registered a second run of the same model to the same UC name, deployed with `agents.deploy()`
  to endpoint `agents_cs4603-default-yahya_document_analyst`, which also spun up a Review App.
- **Bug found and fixed:** the first deploy attempt (v17) reached `READY` but failed to load —
  `agents.deploy()` was called without the `environment_vars` argument, so the container never
  received `DATABRICKS_HOST`/`TOKEN`/`MODEL` and `load_context()` raised `OSError: Missing required
  environment variables`. Fixed by passing the same secret-ref `environment_vars` dict used in
  `deploy.py`, redeployed as version 18, which loaded and served correctly.
- **Quota note:** the free-tier workspace only supports one endpoint's worth of provisioned
  concurrency at a time. `agents.deploy()` does a blue-green update that briefly needs 2x capacity,
  so the existing `yahya-document-analyst` endpoint was deleted before running `deploy_agents.py`,
  then recreated afterward with `deploy.py`.
- **Evidence the `agents_` endpoint actually responds**, tested directly against
  `agents_cs4603-default-yahya_document_analyst` after the `environment_vars` fix and redeploy:
  ```
  200
  {"choices": [{"message": {"role": "assistant", "content": "The net income in 2023 was
  ¥1,107 billion, as reported in the annual report (source:
  dbfs:/Volumes/cs4603/default/pa4/annual_report.pdf)."}, "index": 0, "finish_reason": "stop"}],
  "object": "chat.completion", "created": 1784115484, "id": "56fbe7f0-ab43-482e-8fdd-e44e22d419bf"}
  ```
- Review App: `https://dbc-717990d0-8dd5.cloud.databricks.com/ml/review-v2/9e6406ef59e54840a8b35cdd39cd07b3/chat`.
  Submitted 3 queries there (same three as the local test set) with feedback:
  - Net income query: response "The net income in 2023 was ¥1,107 billion... (Source:
    dbfs:/Volumes/cs4603/default/pa4/annual_report.pdf, page 2)." — thumbs up (accurate, relevant,
    well structured, cited source + page).
  - 15%-of-2.4-billion query: response walked through "2.4 billion" as if it needed retrieval,
    logged "Step 1 retrieval returned NOT FOUND IN DOCUMENT," then computed 2.4e9 × 0.15 = 3.6e8
    anyway — thumbs down (poorly structured, too long). The final number (360,000,000) is correct,
    but the planner routed an unnecessary retrieval step for a self-contained numeric query that
    never needed the document at all.
  - Combined revenue-growth query: response gave net revenue ¥16,910 billion (correct, sourced),
    correctly derived the compound factor (1.1)³ = 1.331, then reported the product as
    ¥22,521.41 billion — thumbs down (inaccurate). The correct product is
    16,910 × 1.331 = ¥22,507.21 billion, so the arithmetic error is in the calculation step itself,
    not retrieval or routing.

**Bonus C (standalone MCP server, `deployment/mcp_app/`):**
- Deployed `tools/mcp_server.py`'s tool definitions as a separate Databricks App (`cs4603-mcp-tools`)
  serving over `streamable-http` instead of stdio.
- **Bugs found and fixed getting this working:**
  1. `app.yaml`'s `command` pointed at `deployment/mcp_app/app.py`, but `app.yaml` must sit at the
     root of the deployed source path (Databricks Apps convention), so the source-code-path is
     `deployment/mcp_app/` itself and the command needed to be the relative `app.py`.
  2. `app.py` imported `tools.mcp_server` via a `sys.path` hack three directories up to the repo
     root, but only `deployment/mcp_app/` is synced to the app, so that import would fail at
     runtime. Fixed by keeping a self-contained copy, `deployment/mcp_app/mcp_server.py` (identical
     content to the GIVEN `tools/mcp_server.py`), and importing locally.
  3. There was no `requirements.txt`, so the app's environment wouldn't have had the `mcp` package.
     Added `deployment/mcp_app/requirements.txt` with `mcp>=1.0.0`.
  4. `FastMCP.run()` does not accept `host`/`port` kwargs in the installed SDK version; they're set
     via `mcp.settings.host` / `mcp.settings.port` before calling `run(transport="streamable-http")`.
- **Auth finding:** the assignment's example code (and the original `agent/graph.py`) uses the
  workspace `DATABRICKS_TOKEN` (a personal access token) as the bearer token for the remote MCP
  server. In practice this returns `401` — Databricks Apps require an OAuth access token
  (e.g. from `databricks auth token`), not a plain PAT. A token-exchange attempt using the PAT as
  `subject_token` against `/oidc/v1/token` also failed (`invalid token for subject_token_type`).
  `agent/graph.py`'s `load_mcp_tools()` was updated to check an `MCP_AUTH_TOKEN` env var first,
  falling back to `DATABRICKS_TOKEN` unchanged, so a properly-scoped OAuth token can be supplied
  without disturbing the PAT used elsewhere (LLM calls, Vector Search). This was verified locally
  (see evidence below) rather than wired into the live serving endpoints, since doing so in
  production would require provisioning a service principal with `CAN USE` on the app and M2M
  OAuth — a larger change than the assignment's stated proof requirement calls for.
- **Evidence (works → stopped → fails → restarted):**
  ```
  # Working: calculation via the remote MCP server
  15% of 2.4 billion = 360,000,000 (3.6e8)

  # databricks apps stop cs4603-mcp-tools
  "compute_status": {"state": "STOPPED"}

  # Same query, same code, app stopped:
  httpx.HTTPStatusError: Server error '503 Service Unavailable' for url
  'https://cs4603-mcp-tools-7474660495876993.aws.databricksapps.com/mcp'

  # databricks apps start cs4603-mcp-tools
  "compute_status": {"state": "ACTIVE"}, deployment "state": "IN_PROGRESS"

  # databricks apps get cs4603-mcp-tools (confirmed after restart)
  "active_deployment": {"status": {"message": "App started successfully", "state": "SUCCEEDED"}}
  "app_status": {"message": "App has status: App is running", "state": "RUNNING"}
  ```

## Design decisions

- **Model type:** `mlflow.pyfunc.ChatModel` instead of `mlflow.langchain`, so Databricks Model
  Serving returns a single JSON object rather than the `[output]` batch-list `mlflow.langchain`
  produces — needed for the OpenAI SDK's `resp.choices[0].message.content` to work directly.
- **Lazy initialization:** `DocumentAnalystModel.__init__` does nothing; all setup (env var
  validation, building the LangGraph agent, connecting the retriever and MCP tools) happens in
  `load_context()`, since Databricks injects secret-referenced env vars only after the artifact is
  loaded into the container, not at import time.
- **Windows path patching:** `mlflow.pyfunc.save_model` resolves `python_model="agent_model.py"` to
  an absolute Windows path in the saved `MLmodel` file. Both `deploy.py` and `deploy_agents.py`
  patch this back to the bare filename with a regex before uploading, so the container (Linux)
  never sees an unusable Windows path.
- **Graph architecture:** planner → supervisor → {rag_agent | mcp_tools} → synthesizer, looping the
  supervisor per plan step so retrieval and calculation steps are handled by focused specialists
  rather than one general-purpose agent.

---

## Analysis Questions

### Task 1.2 — Planner
1. What happens when the planner produces steps that depend on each other (e.g., step 3 needs the result of step 1)? How does your architecture handle this?
   - The plan runs sequentially through the supervisor loop. Each step appends its result to
     `step_results` before the next step runs, so by the time step 3 executes, step 1's result is
     already there. The synthesizer sees the full `step_results` list and reasons across all of them
     to produce the final answer. The architecture does not do explicit value injection between
     steps, but that's fine because the synthesizer is what ties everything together.
2. Would a replanning step after each execution improve or hurt performance for this use case? Justify with an example.
   - It would hurt more than help here. Financial document queries are predictable: once you know
     you need to retrieve revenue and then compute growth, the plan doesn't change mid-execution.
     Adding a replanner call after every step means an extra LLM invocation per step, extra latency,
     extra cost, and no real benefit for the straightforward queries this system handles. The one
     case where replanning helps is when a step returns "NOT FOUND IN DOCUMENT" and a smarter
     follow-up query might succeed, but the synthesizer already handles that gracefully by
     acknowledging the gap, so replanning isn't worth it here.

### Task 1.3 — Supervisor
1. Your supervisor makes a routing decision per step. What is the failure mode if it misroutes? How would you detect and recover from a misroute?
   - If the supervisor sends a retrieval step to `mcp_tools`, the LLM tries to call a math tool on
     something like "find Meridian's net revenue." The tool either errors or returns nonsense, and
     that bad result gets appended to `step_results`. The synthesizer then has to work with
     corrupted input, and the symptom is a wrong or hallucinated final answer. To detect this, log
     the `next_agent` decision alongside the step text — if `mcp_tools` is chosen for a step
     containing "find," "look up," or "from the report," that's a misroute signal. Recovery: default
     to `rag_agent` when the LLM response doesn't cleanly resolve to one of the three valid routes.
     (A concrete example of the reverse misroute — a self-contained numeric question routed through
     an unnecessary retrieval step — showed up in the Bonus B Review App feedback; see Deployment
     above.)
2. Compare this supervisor pattern with a single ReAct agent that has access to all tools. When is the supervisor pattern worth the added complexity?
   - A ReAct agent decides which tool to call at every step by reading the full conversation
     history. It's simpler and handles unpredictable queries naturally. The supervisor pattern
     costs more in complexity but pays off through separation of concerns: each specialist has a
     focused system prompt tuned for its job, and routing decisions are explicit and auditable. The
     problem with one agent doing everything is tight coupling — you keep rebuilding the same tool
     logic again and again. The supervisor decouples the specialists. For a financial analyst system
     where you want to tune retrieval independently of calculation and need a clear audit trail, the
     supervisor is worth it.

### Task 1.4 — RAG Agent
1. The RAG agent retrieves for a single decomposed step, not the full user query. How does this affect retrieval quality compared to retrieving for the original question?
   - Retrieval quality improves. A full query like "What was the net revenue in 2023, and what
     would it be after 10% growth for 3 years?" is ambiguous as a single embedding — the vector
     index might pull chunks about growth calculations instead of the revenue figure. When the
     planner decomposes it into "Find Meridian's net revenue for fiscal year 2023," the embedding is
     focused and retrieves the right chunk. This is the core advantage of plan-and-execute: see the
     full picture first, break it into atomic steps, then execute each one cleanly.
2. If the planner produces a vague step like "find relevant financial data," how would you improve the retrieval query before sending it to the vector store?
   - Add a query-rewriting step inside the RAG agent: before calling the retriever, give the LLM
     the vague step plus the original user question and ask it to produce a specific retrieval
     query. For example, "find relevant financial data" plus a question about revenue becomes "net
     revenue fiscal year 2023 Meridian Motor Corporation annual report." Vague instructions to the
     retriever give vague results, so making the query precise before embedding it is the right
     approach.

### Task 2.1 — Model Definition
1. Why does `models-from-code` require a self-contained file? What breaks if you reference external state (e.g., a database running only on your laptop)?
   - MLflow serializes the path to `agent_model.py` and re-executes it inside the serving
     container, which has no access to your laptop, local `.env`, or any local process. If
     `agent_model.py` references a local database or a file that only exists on your machine, the
     container fails at startup with a connection error or `FileNotFoundError`. Self-containment
     means all credentials come from environment variables, all local packages ship via
     `code_paths`, and all external services are managed Databricks resources the container can
     reach with those credentials. (This turned out to matter for Bonus C too — the standalone MCP
     app needed the same treatment: a self-contained copy of `mcp_server.py` inside
     `deployment/mcp_app/`, since only that directory is synced to the app.)
2. Your model calls a managed Vector Search index at inference time rather than embedding documents into the container image. What are the tradeoffs (freshness, cold-start size, latency, failure modes) of querying an external index vs. baking the corpus into the model artifact?
   - External index: the corpus is always fresh — ingest new documents and the next query gets
     them without a model redeploy. The container image is small so cold-start is fast. The
     downside is a network call per query adds latency, and if the index is syncing or down,
     retrieval fails. Baking the corpus in: no network dependency, consistent retrieval latency, but
     the image is huge, cold-start is slow, and updating the corpus requires a full redeploy. For
     this assignment, the external index is the right call — the annual report doesn't change
     constantly, managed Vector Search is reliable, and the model already makes multiple LLM calls
     so one more retrieval hop isn't the bottleneck.

### Task 2.3 — Serving Endpoint
1. Why must you pass `DATABRICKS_TOKEN` as an environment variable to the endpoint, even though it's already authenticated to serve models?
   - The endpoint has system-level auth to receive traffic, but the code running inside the
     container makes outbound calls to Vector Search and the LLM endpoint at inference time. Those
     calls need an explicit bearer token with the right scopes — the serving identity doesn't
     automatically inject a usable token into the runtime. Passing it as a secret reference keeps
     the token out of logs and out of the model artifact.
2. What happens to in-flight requests when you deploy a new model version to the same endpoint? How does Databricks handle the transition?
   - Databricks does a blue-green swap: the existing version keeps serving traffic while the new
     version loads and reaches `READY`. Once healthy, traffic shifts to the new version. In-flight
     requests on the old version complete normally. If the new version fails to reach `READY`, the
     endpoint rolls back to the previous version automatically and keeps serving without downtime.
     (This is exactly the failure mode hit with the Bonus B v17 deploy — it never reached `READY`
     because of the missing `environment_vars`, so the endpoint kept the old version serving until
     v18 was redeployed correctly.)

### Task 3.2 — Client
1. Why is exponential backoff better than fixed-interval retries for a model serving endpoint?
   - If every client retries at the same fixed interval after a 429, they all hit the endpoint
     simultaneously again and keep it overwhelmed. Exponential backoff spreads retries out — 1s,
     2s, 4s — giving the endpoint time to recover and reducing the thundering-herd effect. Standard
     practice for any rate-limited service.
2. Your client has a `max_retries` parameter. What is the danger of setting it too high in a production system with many concurrent users?
   - High `max_retries` means each request holds an open connection much longer before failing. If
     the endpoint is genuinely down, 1000 concurrent users each retrying 10 times will exhaust
     client-side resources and block new requests from starting — the system appears frozen instead
     of failing fast. In production, fail fast with a small `max_retries`, surface the error
     clearly, and let the user or an alert system decide what to do next.
3. When would you choose `ask_streaming()` over `ask()`? Give a concrete UX example.
   - Use streaming when the response is long and the user needs to see progress. Example:
     "Summarize all key metrics from the 2023 annual report and flag anything unusual." With
     `ask()` the user stares at a spinner for 10-15 seconds then gets a wall of text. With
     `ask_streaming()` the first tokens appear in 1-2 seconds and they start reading while the rest
     generates — exactly how ChatGPT and Claude interfaces work. Wall-clock time is the same but
     perceived latency drops a lot.

### Bonus A — CI/CD (if attempted)
1. Why should the deploy step only run on `main` and not on feature branches?
   - Feature branches are work in progress. Deploying from one would overwrite the production
     endpoint with unreviewed code. Gating deploy on `main` means all changes go through a pull
     request, pass lint and tests, and get reviewed before hitting production — `main` is always
     deployable, branches are not.
2. What would you add to this pipeline to prevent deploying a model that performs worse than the current version? Describe the gate.
   - Add an evaluation gate between the test step and deploy. After registering the new model
     version, run a benchmark query set and compare accuracy, latency p95, and retrieval hit rate
     against the currently deployed version stored as a baseline in the MLflow experiment. If the
     new version scores below threshold on any metric, fail the gate and don't update the endpoint.
     `mlflow.evaluate()` makes this straightforward to log and compare across runs.

### Bonus B — `databricks-agents` SDK (if attempted)
1. Compare the `agents.deploy()` approach with the manual MLflow + CLI approach from Part 2. What control do you gain or lose with each?
   - `agents.deploy()` handles endpoint creation, secret injection (once you remember to pass
     `environment_vars` — see the bug in Deployment above), and spins up a Review App in one call.
     You lose fine-grained control: no custom scaling policy, and less visibility into exactly what
     endpoint config gets created. The manual approach (`deploy.py`) is more verbose but gives full
     control over every parameter, including recreate-on-type-mismatch handling that `agents.deploy()`
     doesn't expose. For rapid prototyping and feedback collection, `agents.deploy()` is faster; for
     production with specific SLA requirements, the manual approach is better.
2. The Review App enables human feedback collection. How would you use this feedback to improve the agent over time? Describe a concrete feedback loop.
   - Collect thumbs-up/thumbs-down ratings and free-text corrections; these land in the MLflow
     experiment as evaluation traces. Export low-rated responses periodically and look for patterns
     — in this submission, one failure pattern was already visible in the 3 test queries: the
     planner sometimes routes a self-contained numeric question through an unnecessary retrieval
     step, and a separate real arithmetic error surfaced in a compound-growth calculation. Both are
     concrete, reproducible cases to add to an offline benchmark so the CI gate catches regressions,
     then fix the relevant prompt (`agent/prompts.py`) or tool logic and redeploy. Deploy, collect
     feedback, analyze, fix, redeploy — a closed loop.

### Bonus C — Standalone MCP server (if attempted)
1. You moved the MCP server out of the model container. What did you gain (scaling, deployment, security, observability) and what new failure modes did you introduce (network, auth, latency, availability)?
   - Gains: the MCP server and model scale independently; tool code updates don't require a model
     redeploy; multiple agent versions could share one tool server; logs and metrics for tools are
     separate from model serving, so observability is cleaner. New failure modes: the agent now has
     a network dependency on the MCP app — verified directly in this submission by stopping
     `cs4603-mcp-tools` and getting a `503 Service Unavailable` on the next tool call. There's also
     one extra HTTPS round-trip per tool call adding latency, and auth has to be managed explicitly
     — which turned out to be more involved than expected, since Databricks Apps reject a plain
     personal access token and require a proper OAuth token (see the auth finding in Deployment
     above).
2. The remote MCP server now needs its own authentication. How would you secure it so that only your serving endpoint — not the public internet — can call the tools?
   - Two layers. First, require a valid OAuth bearer token in the `Authorization` header — as
     observed, Databricks Apps already enforce this by default (a bare PAT gets `401`), and only
     principals granted `CAN USE` on the app can obtain a token that's accepted. In production, that
     means creating a service principal for the serving endpoint, granting it `CAN USE` on the app,
     and having the endpoint mint M2M OAuth tokens from that service principal's client
     ID/secret rather than reusing its PAT. Second, restrict network access at the infrastructure
     level so the app URL is only routable from within the workspace, not the public internet. An
     attacker who gets the URL still can't call the tools without a valid token, and a leaked PAT
     alone is useless against the app since it isn't the accepted credential type.
3. When is bundling the tools in the container (Part 1) the *better* choice, and when is a separately deployed tool service (Bonus C) worth the extra moving parts?
   - Bundle when the tools are simple, stable, and only used by this one agent — if
     `tools/mcp_server.py` never changes and only this model calls it, the extra complexity of a
     separate app (plus the auth wrinkle documented above) isn't justified. A separate service is
     worth it when multiple agents or teams need the same tools, tool logic updates frequently and
     independently of the model, or you need to monitor and scale tool execution separately. In a
     real organization with multiple LLM products, a shared calculation service defined once and
     versioned is the right pattern — you define it once and everybody uses it, which is exactly
     the problem MCP was born to solve.
