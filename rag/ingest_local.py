"""Run locally: uv run python rag/ingest_local.py"""
from __future__ import annotations

import os
import sys
import uuid
import time

from dotenv import load_dotenv

load_dotenv()


def _wait_for_statement(w, stmt_id: str, timeout: int = 120):
    from databricks.sdk.service.sql import StatementState

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = w.statement_execution.get_statement(stmt_id)
        state = resp.status.state
        if state in (StatementState.SUCCEEDED,):
            return resp
        if state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            raise RuntimeError(f"Statement {stmt_id} {state}: {resp.status.error}")
        time.sleep(3)
    raise TimeoutError(f"Statement {stmt_id} did not complete in {timeout}s")


def _exec(w, warehouse_id: str, sql: str):
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="50s",
    )
    from databricks.sdk.service.sql import StatementState
    if resp.status.state == StatementState.PENDING:
        resp = _wait_for_statement(w, resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {resp.status.error}\nSQL: {sql[:200]}")
    return resp


def main():
    try:
        import fitz
    except ImportError:
        print("Installing pymupdf...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf", "-q"])
        import fitz

    from databricks.sdk import WorkspaceClient
    from databricks.vector_search.client import VectorSearchClient

    catalog = os.environ.get("UC_CATALOG", "cs4603")
    schema = os.environ.get("UC_SCHEMA", "default")
    source_table = os.environ.get("SOURCE_TABLE", f"{catalog}.{schema}.yahya_analyst_chunks")
    vs_endpoint = os.environ.get("VECTOR_SEARCH_ENDPOINT", "cs4603_rag_endpoint")
    vs_index = os.environ.get("VECTOR_SEARCH_INDEX", f"{catalog}.{schema}.yahya_analyst_index")
    embeddings_endpoint = os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en")

    pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "annual_report.pdf")

    print(f"Parsing {pdf_path} ...")
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
        for para in paragraphs:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "chunk_to_retrieve": para,
                "chunk_to_embed": para,
                "source": "annual_report.pdf",
                "page": page_num + 1,
            })
    print(f"  {len(chunks)} chunks from {len(doc)} pages")

    w = WorkspaceClient()
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses found in workspace")
    warehouse_id = warehouses[0].id
    print(f"Using warehouse: {warehouses[0].name} ({warehouse_id})")

    print(f"Creating table {source_table} ...")
    _exec(w, warehouse_id, f"""
        CREATE OR REPLACE TABLE {source_table} (
            chunk_id STRING NOT NULL,
            chunk_to_retrieve STRING,
            chunk_to_embed STRING,
            source STRING,
            page INT
        )
    """)

    batch_size = 40
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        rows = []
        for c in batch:
            def esc(s):
                return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", "")
            rows.append(
                f"('{c['chunk_id']}', '{esc(c['chunk_to_retrieve'])}', "
                f"'{esc(c['chunk_to_embed'])}', '{c['source']}', {c['page']})"
            )
        values = ",\n".join(rows)
        _exec(w, warehouse_id, f"INSERT INTO {source_table} VALUES {values}")
        done = min(i + batch_size, total)
        print(f"  inserted {done}/{total} chunks")

    resp = _exec(w, warehouse_id, f"SELECT COUNT(*) FROM {source_table}")
    count = resp.result.data_array[0][0]
    print(f"Table ready: {count} rows in {source_table}")

    print(f"Creating vector index {vs_index} ...")
    vsc = VectorSearchClient()
    try:
        vsc.create_delta_sync_index(
            endpoint_name=vs_endpoint,
            index_name=vs_index,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=embeddings_endpoint,
        )
        print("Index created — wait for READY in Vector Search UI (usually 2-5 min)")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("Index already exists, triggering sync ...")
            vsc.get_index(vs_endpoint, vs_index).sync()
            print("Sync triggered")
        else:
            raise


if __name__ == "__main__":
    main()
