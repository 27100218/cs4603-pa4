"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirror PA2 Part 1:

TODO:
  - `build_chunks_table(spark, volume_path, chunks_table)`: parse the PDF with
    ai_parse_document, chunk with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enable Change Data
    Feed on the table.
  - `create_index()`: create a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
"""

from __future__ import annotations

import os


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    parse_table = chunks_table.rsplit(".", 1)[0] + ".yahya_analyst_parsed"

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {parse_table} (
            path STRING,
            parsed VARIANT
        ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    spark.sql(f"""
        INSERT OVERWRITE {parse_table}
        SELECT
            path,
            ai_parse_document(content) AS parsed
        FROM READ_FILES('{volume_path}', format => 'binaryFile')
        WHERE path LIKE '%.pdf'
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {chunks_table} (
            chunk_id STRING,
            chunk_to_retrieve STRING,
            chunk_to_embed STRING,
            source STRING,
            page STRING
        ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    spark.sql(f"""
        INSERT OVERWRITE {chunks_table}
        SELECT
            chunk.value:chunk_id::STRING AS chunk_id,
            chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
            chunk.value:chunk_to_embed::STRING AS chunk_to_embed,
            path AS source,
            chunk.value:page_number::STRING AS page
        FROM (
            SELECT path, ai_prep_search(parsed) AS result FROM {parse_table}
        ) prepped,
        LATERAL variant_explode(result:document.contents) AS chunk
    """)

    count = spark.table(chunks_table).count()
    print(f"Created {count} chunks in {chunks_table}")


def create_index() -> None:
    from databricks.vector_search.client import VectorSearchClient

    catalog = os.environ.get("UC_CATALOG", "cs4603")
    schema = os.environ.get("UC_SCHEMA", "default")
    vs_endpoint = os.environ.get("VECTOR_SEARCH_ENDPOINT", "cs4603_rag_endpoint")
    vs_index = os.environ.get("VECTOR_SEARCH_INDEX", f"{catalog}.{schema}.yahya_analyst_index")
    chunks_table = os.environ.get("SOURCE_TABLE", f"{catalog}.{schema}.yahya_analyst_chunks")
    embeddings_endpoint = os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en")

    vsc = VectorSearchClient()
    try:
        vsc.create_delta_sync_index(
            endpoint_name=vs_endpoint,
            index_name=vs_index,
            source_table_name=chunks_table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=embeddings_endpoint,
        )
        print(f"Created index {vs_index}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Index {vs_index} already exists, syncing...")
            vsc.get_index(vs_endpoint, vs_index).sync()
        else:
            raise
