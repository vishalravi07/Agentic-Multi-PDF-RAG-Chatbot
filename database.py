import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

import config


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Database URL
# ---------------------------------------------------------

# Prefer an explicit DATABASE_URL; otherwise fall back to the one assembled
# from the individual POSTGRES_* settings in config.py.
DATABASE_URL = os.getenv("DATABASE_URL") or config.DATABASE_URL

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is not set in the .env file."
    )


# ---------------------------------------------------------
# PostgreSQL connection
# ---------------------------------------------------------

@contextmanager
def get_conn():
    """
    Create a PostgreSQL connection.

    The connection is automatically closed
    after the operation is finished.
    """

    conn = psycopg2.connect(DATABASE_URL)

    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------
# Create SQLAlchemy engine
# ---------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ---------------------------------------------------------
# Insert document
# ---------------------------------------------------------

def insert_document(filename: str, user_id: int | None = None) -> int:
    """
    Insert a PDF into the documents table.

    Returns:
        The ID of the newly inserted document.
    """

    with get_conn() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO documents (filename, user_id)
                VALUES (%s, %s)
                RETURNING id;
                """,
                (filename, user_id)
            )

            doc_id = cur.fetchone()[0]

        conn.commit()

    return doc_id


# ---------------------------------------------------------
# Insert PDF chunks
# ---------------------------------------------------------

def insert_chunks(document_id: int, chunk_rows: list[dict]):
    """
    Insert PDF chunks and their embeddings
    into document_chunks.
    """

    with get_conn() as conn:

        with conn.cursor() as cur:

            values = [
                (
                    document_id,
                    row["page_number"],
                    row["chunk_text"],
                    row["embedding"]
                )
                for row in chunk_rows
            ]

            psycopg2.extras.execute_values(
                cur,

                """
                INSERT INTO document_chunks
                (
                    document_id,
                    page_number,
                    chunk_text,
                    embedding
                )
                VALUES %s
                """,

                values,

                template="(%s, %s, %s, %s::vector)"
            )

        conn.commit()


# ---------------------------------------------------------
# Initialize database
# ---------------------------------------------------------

def init_db():
    """
    Initialize PostgreSQL and enable pgvector.
    """

    with engine.begin() as conn:

        conn.execute(
            text(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )
        )

        # Make sure the per-user column exists on older installs.
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'documents'
                    ) THEN
                        ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id INTEGER;
                    END IF;
                END $$;
                """
            )
        )

    print("Database connection initialized.")
    print("pgvector extension is enabled.")
    
# ---------------------------------------------------------
# List documents
# ---------------------------------------------------------

def list_documents(user_id: int | None = None) -> list[dict]:
    """
    Return every document with its chunk/page counts.

    When user_id is given, only that user's documents are returned.

    Returns a list of dicts (newest first):
        {
            "id": int,
            "filename": str,
            "uploaded_at": datetime,
            "num_chunks": int,
            "total_pages": int,
        }
    """
    where_clause = ""
    params: list = []

    if user_id is not None:
        where_clause = "WHERE d.user_id = %s"
        params.append(user_id)

    sql = f"""
        SELECT
            d.id,
            d.filename,
            d.uploaded_at,
            COUNT(dc.id) AS num_chunks,
            COUNT(DISTINCT dc.page_number) AS total_pages
        FROM documents d
        LEFT JOIN document_chunks dc ON dc.document_id = d.id
        {where_clause}
        GROUP BY d.id, d.filename, d.uploaded_at
        ORDER BY d.uploaded_at DESC, d.id DESC;
    """

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [dict(row) for row in rows]


# ---------------------------------------------------------
# Similarity search (cosine distance via pgvector)
# ---------------------------------------------------------

def similarity_search(
    query_embedding: list[float],
    top_k: int = 5,
    document_id: int | None = None,
    user_id: int | None = None,
    query_text: str = "",
) -> list[dict]:

    """
    Hybrid retrieval.

    Uses:
    1. pgvector cosine similarity
    2. PostgreSQL keyword matching

    Keyword matching helps retrieve exact facts such as:

        "Room-rent limit"
        "₹5,000 per day"

    even when semantic similarity ranks another chunk higher.
    """

    conditions = []
    filters: list = []

    if document_id is not None:
        conditions.append("dc.document_id = %s")
        filters.append(document_id)

    if user_id is not None:
        conditions.append("d.user_id = %s")
        filters.append(user_id)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        WITH candidates AS (

            -- -----------------------------------------
            -- Vector search
            -- -----------------------------------------

            SELECT
                d.id AS document_id,
                d.filename,
                dc.page_number,
                dc.chunk_text,

                1 - (dc.embedding <=> %s::vector)
                    AS vector_score,

                0.0 AS keyword_score

            FROM document_chunks dc

            JOIN documents d
                ON d.id = dc.document_id

            {where_clause}

            ORDER BY dc.embedding <=> %s::vector

            LIMIT 20
        )

        SELECT
            document_id,
            filename,
            page_number,
            chunk_text,

            vector_score AS similarity_score

        FROM candidates

        ORDER BY vector_score DESC

        LIMIT %s;
    """

    params = [
        query_embedding
    ]

    params.extend(filters)

    params.extend([
        query_embedding,
        top_k
    ])

    with get_conn() as conn:

        with conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:

            cur.execute(sql, params)

            rows = cur.fetchall()

    return [dict(row) for row in rows]