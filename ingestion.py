from typing import List

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import text

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL
import database


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

# all-MiniLM-L6-v2 produces 384-dimensional embeddings
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ---------------------------------------------------------
# Text splitter
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# ---------------------------------------------------------
# Create database tables
# ---------------------------------------------------------

def create_ingestion_tables():
    """
    Creates the tables required for the RAG pipeline.
    """

    actual_dim = embedding_model.get_embedding_dimension()
    if actual_dim != EMBEDDING_DIM:
        raise ValueError(
            f"EMBEDDING_DIM is {EMBEDDING_DIM} but '{EMBEDDING_MODEL}' produces "
            f"{actual_dim}-dimensional vectors. Update EMBEDDING_DIM in .env to match."
        )

    with database.engine.begin() as conn:

        # Enable pgvector
        conn.execute(
            text(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )
        )

        # Documents table
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    filename VARCHAR(500) NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )

        # Existing installs: add the user_id column if it is missing.
        conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id INTEGER"
            )
        )

        # Chunks table
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,

                    document_id INTEGER
                        REFERENCES documents(id)
                        ON DELETE CASCADE,

                    page_number INTEGER,

                    chunk_text TEXT NOT NULL,

                    embedding VECTOR({EMBEDDING_DIM})
                )
                """
            )
        )


# ---------------------------------------------------------
# Extract PDF pages
# ---------------------------------------------------------

def extract_pdf_pages(pdf_path: str):
    """
    Extract text from every page of a PDF.

    Returns:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text_content = page.extract_text()

        if not text_content:
            continue

        text_content = text_content.strip()

        if not text_content:
            continue

        pages.append(
            {
                "page_number": page_number,
                "text": text_content
            }
        )

    return pages


# ---------------------------------------------------------
# Create chunks
# ---------------------------------------------------------

def create_chunks(pages):
    """
    Split each PDF page into smaller chunks.

    Page number is preserved so that we can show
    the source page to the user later.
    """

    chunks = []

    for page in pages:

        page_chunks = text_splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunk = chunk.strip()

            if not chunk:
                continue

            chunks.append(
                {
                    "page_number": page["page_number"],
                    "text": chunk
                }
            )

    return chunks


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

def generate_embeddings(texts: List[str]):
    """
    Generate embeddings using Sentence Transformers.
    """

    if not texts:
        raise ValueError("Cannot generate embeddings for an empty list of texts.")

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()


# ---------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------

def ingest_pdf(pdf_path: str, filename: str, user_id: int | None = None):

    # Make sure required tables exist
    create_ingestion_tables()

    # ---------------------------------------------
    # 1. Extract PDF
    # ---------------------------------------------

    pages = extract_pdf_pages(pdf_path)

    if not pages:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    # ---------------------------------------------
    # 2. Create chunks
    # ---------------------------------------------

    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError(
            "No text chunks could be created from the PDF."
        )

    # ---------------------------------------------
    # 3. Generate embeddings
    # ---------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    # ---------------------------------------------
    # 4. Store in PostgreSQL
    # ---------------------------------------------

    with database.engine.begin() as conn:

        # Insert document
        result = conn.execute(
            text(
                """
                INSERT INTO documents (filename, user_id)
                VALUES (:filename, :user_id)
                RETURNING id
                """
            ),
            {
                "filename": filename,
                "user_id": user_id,
            }
        )

        document_id = result.scalar_one()

        # Insert chunks
        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            conn.execute(
                text(
                    """
                    INSERT INTO document_chunks
                    (
                        document_id,
                        page_number,
                        chunk_text,
                        embedding
                    )
                    VALUES
                    (
                        :document_id,
                        :page_number,
                        :chunk_text,
                        CAST(:embedding AS vector)
                    )
                    """
                ),
                {
                    "document_id": document_id,
                    "page_number": chunk["page_number"],
                    "chunk_text": chunk["text"],
                    "embedding": str(embedding)
                }
            )

    # ---------------------------------------------
    # 5. Return summary
    # ---------------------------------------------

    return {
        "success": True,
        "document_id": document_id,
        "filename": filename,
        "total_pages": len(pages),
        "num_chunks": len(chunks)
    }
