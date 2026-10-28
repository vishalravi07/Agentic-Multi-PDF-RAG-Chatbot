# 🤖 Agentic Multi-PDF RAG Chatbot

An AI-powered chatbot that allows users to upload multiple PDF documents and ask questions about their contents. Built with Python and FastAPI, the project uses Retrieval-Augmented Generation (RAG), semantic search, and an agentic workflow to generate context-aware answers grounded in document content.

## ✨ Features

- **Multi-PDF Upload:** Upload multiple PDF documents for question answering.
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant document chunks to provide context to the language model.
- **Semantic Search:** Uses embeddings and vector similarity to find relevant information.
- **Agentic Workflow:** Supports an agent-oriented answering workflow alongside a simpler RAG mode.
- **Guardrails:** Includes checks intended to reduce irrelevant or unsupported responses.
- **Source Context:** Retrieves document passages that can help users understand the basis of an answer.
- **Web Interface:** Interact with the chatbot through a browser-based interface.
- **User Authentication:** Includes signup and login functionality.
- **Answer Caching:** Uses caching to reduce repeated processing for eligible queries.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| FastAPI | Backend API and web server |
| LangChain / text splitters | Document chunking and retrieval workflow |
| PostgreSQL + pgvector | Vector storage and similarity search |
| Sentence Transformers | Text embeddings |
| Ollama or Groq | Language model integration, depending on configuration |
| PyPDF | PDF text extraction |
| HTML, CSS, JavaScript | Frontend interface |
| Bcrypt | Password hashing |
| SQLite | Local user data storage, as configured |

## 🧠 How It Works

The chatbot follows a Retrieval-Augmented Generation pipeline:

1. **Upload:** The user uploads PDF documents.
2. **Extract:** Text is extracted from each PDF.
3. **Chunk:** The extracted text is split into smaller, manageable sections.
4. **Embed:** An embedding model converts the text chunks into numerical vectors.
5. **Store:** The vectors and associated document information are stored for retrieval.
6. **Retrieve:** When a user asks a question, the system searches for relevant chunks.
7. **Generate:** The retrieved context is passed to the language model to help generate an answer.
8. **Guardrails:** Application checks help reduce unsupported or irrelevant responses.

### Architecture

```text
             User
              |
              v
       Web Interface
              |
              v
         FastAPI
              |
       PDF Upload / Query
              |
        +-----+------+
        |            |
        v            v
   PDF Processing   User Auth
        |
        v
 Text Extraction
        |
        v
     Chunking
        |
        v
    Embeddings
        |
        v
 PostgreSQL + pgvector
        |
        v
   Semantic Retrieval
        |
        v
 Agentic / Simple RAG
        |
        v
   Guardrail Checks
        |
        v
   Answer to User
```

## 📁 Project Structure

```text
agentic-multipdf-rag/
├── main.py          # FastAPI application entry point
├── agent.py         # Agentic answering workflow
├── rag.py           # Retrieval and RAG logic
├── ingestion.py     # PDF ingestion and processing
├── database.py      # Database operations
├── Auth.py          # Authentication functionality
├── guardrail.py     # Guardrail checks
├── config.py        # Application configuration
├── index.html       # Web interface
├── tests/           # Test files
├── .env.example     # Example environment configuration
├── .gitignore       # Files excluded from Git
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
```

## ⚙️ Prerequisites

Install or configure the following before running the project:

- Python 3.10 or a compatible version for your dependencies
- PostgreSQL
- The pgvector PostgreSQL extension
- Ollama with a compatible local model, or a Groq API key
- Git

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vishalravi07/Agentic-Multi-PDF-RAG-Chatbot.git
cd Agentic-Multi-PDF-RAG-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root. Add the configuration required by your application, using the variable names expected by `config.py` and the database code.

For example:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
APP_API_KEY=your_application_api_key
```

If using Ollama, configure the relevant model and local endpoint settings required by your project.

**Security:** Never commit your actual `.env` file, API keys, database passwords, or personal database files to GitHub.

### 5. Configure PostgreSQL

Create the database configured for your application and enable pgvector.

For example, inside PostgreSQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Make sure the database connection settings in your `.env` file match your local PostgreSQL configuration.

### 6. Start the application

```bash
python -m uvicorn main:app --reload
```

Open the local address in your browser:

```text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation is usually available at:

```text
http://127.0.0.1:8000/docs
```

## 💡 Example Use Cases

- Ask questions across multiple PDF documents.
- Retrieve relevant passages from technical documentation.
- Explore insurance policy documents and compare their stated terms.
- Search lengthy reports without manually reading every page.
- Build a foundation for document research assistants.

For comparisons, the chatbot should retrieve the relevant conditions, exclusions, limits, and prices from the documents. Always verify important decisions against the original documents.

## 🔐 Security Considerations

- Store secrets in environment variables.
- Keep `.env`, virtual environments, uploaded private PDFs, and local databases out of version control.
- Hash passwords rather than storing plaintext passwords.
- Validate uploaded files and enforce file-size limits.
- Apply authentication and authorization checks to document upload and retrieval endpoints.
- Treat retrieved document content as untrusted input.
- Validate model output and test guardrails against prompt injection and unsupported claims.

## 🧪 Testing

Run the project's tests, if configured:

```bash
python -m pytest
```

Review the test results and add coverage for PDF ingestion, retrieval accuracy, authentication, access control, and guardrail behavior.

## 🔮 Future Improvements

- Add automated evaluation of retrieval relevance and answer faithfulness.
- Improve source citations with document names and page numbers.
- Add document-level access controls.
- Support more file formats.
- Add streaming responses and conversation history.
- Improve prompt-injection defenses and error handling.
- Add Docker-based deployment and automated tests.

## 👨‍💻 Author

**Vishal Ravichander**

GitHub: [@vishalravi07](https://github.com/vishalravi07)

---

*Built as a learning project to explore Retrieval-Augmented Generation, semantic search, LLM integration, and agentic AI workflows.*
