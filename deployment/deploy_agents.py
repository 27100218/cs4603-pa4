"""Deploy via databricks-agents SDK (Bonus B).

Uses agents.deploy() instead of the manual WorkspaceClient approach in deploy.py.
This is the v2 path: same models-from-code definition + code_paths, but a single
call handles endpoint creation, secret injection, and spins up a Review App for
human feedback.

Run:  uv run python deployment/deploy_agents.py
"""

from __future__ import annotations

import os
import re
import tempfile

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
catalog = os.environ.get("UC_CATALOG", "cs4603")
schema = os.environ.get("UC_SCHEMA", "default")
secret_scope = os.environ.get("SECRET_SCOPE", "cs4603-deploy")
uc_model_name = f"{catalog}.{schema}.yahya_document_analyst"


def _patch_local_artifact(local_model_path: str) -> None:
    """Replace Windows absolute paths with the bare filename in every text file."""
    win_re = re.compile(r"[A-Za-z]:[/\\][^\n\r\t]+?agent_model\.py")
    for dirpath, _, filenames in os.walk(local_model_path):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                patched = win_re.sub("agent_model.py", content)
                if patched != content:
                    with open(fpath, "w", encoding="utf-8") as fh:
                        fh.write(patched)
                    print(f"Patched {os.path.relpath(fpath, local_model_path)}")
            except OSError:
                pass


def log_and_register() -> str:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    w = WorkspaceClient()
    username = w.current_user.me().user_name
    mlflow.set_experiment(f"/Users/{username}/pa4-document-analyst-v2")

    deploy_dir = os.path.dirname(os.path.abspath(__file__))
    orig_dir = os.getcwd()
    os.chdir(deploy_dir)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_model_path = os.path.join(tmp, "agent")

            mlflow.pyfunc.save_model(
                python_model="agent_model.py",
                path=local_model_path,
                code_paths=[
                    os.path.join(ROOT, "agent"),
                    os.path.join(ROOT, "rag"),
                    os.path.join(ROOT, "tools"),
                    os.path.join(ROOT, "config.py"),
                ],
                pip_requirements=[
                    "langgraph>=0.2.0",
                    "langchain>=0.3.0",
                    "langchain-core>=0.3.0",
                    "langchain-openai>=0.2.0",
                    "databricks-langchain>=0.1.0",
                    "databricks-vectorsearch>=0.40",
                    "databricks-ai-search",
                    "langchain-mcp-adapters>=0.0.5",
                    "mcp>=1.0.0",
                    "mlflow>=2.16.0",
                    "openai>=1.40.0",
                    "python-dotenv>=1.0.0",
                ],
                input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
            )

            _patch_local_artifact(local_model_path)

            client = mlflow.tracking.MlflowClient()
            with mlflow.start_run(run_name="document-analyst-agents-sdk") as run:
                client.log_artifacts(run.info.run_id, local_model_path, artifact_path="agent")
                print(f"Run ID: {run.info.run_id}")
            run_id = run.info.run_id
    finally:
        os.chdir(orig_dir)

    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/agent",
        name=uc_model_name,
    )
    print(f"Registered: {uc_model_name} version {registered.version}")
    return str(registered.version)


def deploy(version: str) -> None:
    w = WorkspaceClient()

    # agents.deploy() does a blue-green update which temporarily needs 2x capacity.
    # On the free tier this hits the provisioned-concurrency quota.  Delete the
    # existing endpoint first so the new deploy starts from zero.
    agents_endpoint = f"agents_{uc_model_name.replace('.', '-')}"
    try:
        w.serving_endpoints.get(agents_endpoint)
        print(f"Deleting existing endpoint '{agents_endpoint}' to free quota...")
        w.serving_endpoints.delete(agents_endpoint)
        import time as _time
        _time.sleep(10)
        print("Deleted.")
    except Exception:
        pass  # endpoint didn't exist yet

    deployment = agents.deploy(
        model_name=uc_model_name,
        model_version=version,
        scale_to_zero=True,
        environment_vars={
            "DATABRICKS_HOST": f"{{{{secrets/{secret_scope}/DATABRICKS_HOST}}}}",
            "DATABRICKS_TOKEN": f"{{{{secrets/{secret_scope}/DATABRICKS_TOKEN}}}}",
            "DATABRICKS_MODEL": f"{{{{secrets/{secret_scope}/DATABRICKS_MODEL}}}}",
            "VECTOR_SEARCH_ENDPOINT": os.environ.get("VECTOR_SEARCH_ENDPOINT", ""),
            "VECTOR_SEARCH_INDEX": os.environ.get("VECTOR_SEARCH_INDEX", ""),
            "EMBEDDINGS_ENDPOINT": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
        },
    )
    print(f"Endpoint: {deployment.endpoint_name}")
    print(f"Review App: {deployment.review_app_url}")


if __name__ == "__main__":
    ver = log_and_register()
    deploy(ver)
