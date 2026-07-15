"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""

from __future__ import annotations

import os
import re
import tempfile

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
catalog = os.environ.get("UC_CATALOG", "cs4603")
schema = os.environ.get("UC_SCHEMA", "default")
secret_scope = os.environ.get("SECRET_SCOPE", "cs4603-deploy")
endpoint_name = os.environ.get("SERVING_ENDPOINT_NAME", "yahya-document-analyst")


def _patch_local_artifact(local_model_path: str) -> None:
    """Replace Windows absolute paths with the bare filename in every text file.

    mlflow.langchain.save_model on Windows resolves lc_model to an absolute path
    before writing it into the MLmodel YAML.  Patching locally before upload means
    Databricks never sees the bad path.
    """
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


def log_and_register():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    w = WorkspaceClient()
    username = w.current_user.me().user_name
    mlflow.set_experiment(f"/Users/{username}/pa4-document-analyst")

    uc_model_name = f"{catalog}.{schema}.yahya_document_analyst"

    deploy_dir = os.path.dirname(os.path.abspath(__file__))
    orig_dir = os.getcwd()
    os.chdir(deploy_dir)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_model_path = os.path.join(tmp, "agent")

            # Save to a local temp directory so we can patch before upload.
            # Use mlflow.pyfunc (ChatModel) instead of mlflow.langchain so that
            # Databricks returns a plain JSON object, not a batch-list wrapper.
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

            # Fix Windows absolute paths in the locally-saved artifact.
            _patch_local_artifact(local_model_path)

            # Upload the clean artifact to a new MLflow run.
            client = mlflow.tracking.MlflowClient()
            with mlflow.start_run(run_name="document-analyst") as run:
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
    return uc_model_name, str(registered.version)


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    w = WorkspaceClient()
    host = os.environ["DATABRICKS_HOST"]

    config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(
                entity_name=uc_name,
                entity_version=version,
                workload_size="Small",
                scale_to_zero_enabled=True,
                environment_vars={
                    "DATABRICKS_HOST": f"{{{{secrets/{secret_scope}/DATABRICKS_HOST}}}}",
                    "DATABRICKS_TOKEN": f"{{{{secrets/{secret_scope}/DATABRICKS_TOKEN}}}}",
                    "DATABRICKS_MODEL": f"{{{{secrets/{secret_scope}/DATABRICKS_MODEL}}}}",
                    "VECTOR_SEARCH_ENDPOINT": os.environ.get("VECTOR_SEARCH_ENDPOINT", ""),
                    "VECTOR_SEARCH_INDEX": os.environ.get("VECTOR_SEARCH_INDEX", ""),
                    "EMBEDDINGS_ENDPOINT": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
                },
            )
        ],
    )

    try:
        w.serving_endpoints.get(endpoint_name)
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=config.served_entities,
        )
        print(f"Updated endpoint '{endpoint_name}'")
    except Exception as _e:
        err = str(_e).lower()
        if "does not exist" in err or "not found" in err:
            w.serving_endpoints.create(name=endpoint_name, config=config)
            print(f"Created endpoint '{endpoint_name}'")
        elif "non-agent" in err or "agent endpoint" in err:
            # pyfunc.ChatModel and mlflow.langchain are different endpoint types;
            # Databricks does not allow in-place conversion — delete and recreate.
            print(f"Endpoint type mismatch, deleting and recreating '{endpoint_name}'...")
            w.serving_endpoints.delete(endpoint_name)
            import time as _time
            _time.sleep(5)
            w.serving_endpoints.create(name=endpoint_name, config=config)
            print(f"Recreated endpoint '{endpoint_name}'")
        else:
            raise

    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Model version: {version}")
    print(f"URL: {url}")
    print("Check Serving UI for READY status (3-8 min).")
    return url


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)
