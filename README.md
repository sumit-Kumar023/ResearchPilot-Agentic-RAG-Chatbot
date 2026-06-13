# ResearchPilot — Agentic RAG Research Assistant

A conversational AI assistant that helps students and researchers upload, explore, verify, and interact with academic papers through natural language conversations.

## 🚀 Live Demo

**Try ResearchPilot Online:**

https://researchpilot-agentic-rag-chatbot-production-4bce.up.railway.app/

---

## Overview

ResearchPilot is an Agentic Retrieval-Augmented Generation (RAG) application built with LangGraph, LangChain, Streamlit, and Qdrant.

Users can upload research papers from PDFs, web URLs, or ArXiv, then ask questions about them in a conversational interface. The system intelligently decides whether to:

* Answer directly from uploaded documents
* Retrieve relevant research paper chunks
* Search the web for recent developments
* Verify whether a claim has been superseded by newer research

The goal is to help users understand complex papers faster, validate research findings, and stay updated with the latest advancements.

---

## ✨ Key Highlights

* Agentic RAG workflow powered by LangGraph
* Multi-source document ingestion
* Session-isolated vector databases
* Claim verification against current research
* Web-assisted knowledge discovery
* Streaming responses
* Automated evaluation using DeepEval
* Multi-session support
* Dockerized deployment

---

## 🏗️ Architecture

```text
User
 │
 ▼
Streamlit UI
 │
 ▼
LangGraph Agent
 │
 ├── Query Router
 │
 ├── RAG Pipeline
 │     ├── Qdrant Vector Store
 │     ├── OpenAI Embeddings
 │     ├── Retrieval
 │     └── Reranking
 │
 ├── Tavily Search
 │
 ├── Claim Verification
 │     ├── Web Search
 │     └── ArXiv Discovery
 │
 └── Response Generation
```

---

## 🛠️ Tech Stack

### AI & Agent Frameworks

* LangGraph
* LangChain
* OpenAI

### Retrieval & Storage

* Qdrant
* SQLite
* CacheBackedEmbeddings

### Search

* Tavily Search API

### Frontend

* Streamlit

### Evaluation

* DeepEval

### Deployment

* Docker
* Railway

### Language

* Python

---

## 🎯 Target Users

### Students

Understand dense academic papers through conversational Q&A.

### Researchers

Cross-reference findings and explore related research.

### Literature Reviewers

Verify whether claims and methods are still valid today.

### Knowledge Workers

Interact with documents using natural language instead of manual searching.

---

## 🚀 Features

| Feature                | Description                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| Paper Q&A              | Ask questions about uploaded papers and receive grounded answers based on retrieved context |
| Claim Verification     | Verify whether research findings remain valid or have been superseded                       |
| Research Discovery     | Find newer papers and developments related to existing research                             |
| Web Search             | Incorporates live web results when current information is required                          |
| Direct Answers         | Handles general knowledge questions without retrieval                                       |
| `/btw` Command         | Side-channel for off-topic questions outside session context                                |
| Multi-session Support  | Multiple independent paper workspaces                                                       |
| Auto Session Naming    | Session names generated automatically from the first message                                |
| Multi-Source Ingestion | PDFs, TXT, Markdown, URLs, and ArXiv papers                                                 |
| Graph State Inspector  | Debug LangGraph state through an expandable JSON view                                       |
| Streaming Responses    | Token-by-token streaming output                                                             |

---

## 📚 Supported Document Sources

### File Uploads

* PDF
* TXT
* Markdown

### Online Sources

* Web URLs
* ArXiv IDs
* ArXiv Title Search

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
cd rag-paper-project
```

### Install Dependencies

```bash
uv sync
```

### Run Application

```bash
uv run streamlit run app.py
```

### Run Backend Module

```bash
uv run python -m backend.<module_name>
```

---

## 🐳 Docker Setup

### Build Image

```bash
docker build -t research-pilot .
```

### Run Container

```bash
docker run -p 8501:8501 --env-file .env research-pilot
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=
TAVILY_API_KEY=

QDRANT_URL=
QDRANT_API_KEY=
```

---

## 🔄 How ResearchPilot Works

### 1. Document Ingestion

Users upload documents from:

* PDFs
* URLs
* ArXiv

Documents are parsed, chunked, embedded, and stored in Qdrant.

### 2. Intelligent Query Routing

The agent analyzes user intent and decides whether the query requires:

* Retrieval
* Web search
* Claim verification
* Direct answering

### 3. Retrieval & Reranking

Relevant chunks are retrieved and reranked before generation.

### 4. Claim Verification

Research claims are checked against current web sources and recent academic publications.

### 5. Response Generation

The final answer is generated using retrieved evidence and conversational history.

---

## ⚡ Production Optimizations

| Optimization                | Details                                                                 |
| --------------------------- | ----------------------------------------------------------------------- |
| Embedding Cache             | Prevents re-embedding identical content, reducing API calls and latency |
| Session Isolation           | Dedicated Qdrant collections per session                                |
| Graph Caching               | LangGraph instance cached using Streamlit resource caching              |
| Streaming Responses         | Responses appear token-by-token                                         |
| Session Persistence         | Conversations survive application restarts                              |
| Temporary File Cleanup      | Uploaded files are automatically removed after processing               |
| Async Evaluation            | Controlled concurrency to avoid rate-limit issues                       |
| Reliable ArXiv Verification | Uses targeted searches instead of unstable library dependencies         |

---

## 📏 Design Constraints

| Constraint                     | Reason                                                                |
| ------------------------------ | --------------------------------------------------------------------- |
| Max 3 Query Rewrites           | Prevents infinite retry loops                                         |
| Chunk Size 1000                | Balances retrieval quality and context preservation                   |
| Chunk Overlap 200              | Preserves information across chunk boundaries                         |
| Retrieval Top-K = 4            | Optimizes quality versus context length                               |
| Session-Scoped Collections     | Prevents cross-session data leakage                                   |
| Separate Verification Searches | Improves coverage of both web and academic sources                    |
| `/btw` Not Stored              | Prevents unrelated context from polluting paper-focused conversations |

---

## 📊 Evaluation

ResearchPilot includes an automated RAG evaluation pipeline powered by DeepEval.

### Metrics

| Metric               | What It Measures        |
| -------------------- | ----------------------- |
| Contextual Precision | Retrieval accuracy      |
| Contextual Recall    | Retrieval coverage      |
| Contextual Relevancy | Context relevance       |
| Answer Relevancy     | Answer quality          |
| Faithfulness         | Hallucination detection |

### Run Evaluation

```bash
uv run python evaluate.py
```

### Evaluation Workflow

* Generates synthetic golden datasets
* Stores generated goldens in `goldens.json`
* Reuses cached goldens on future runs
* Produces detailed evaluation reports
* Stores results in `eval_results.json`

---

## 📈 Future Improvements

* Hybrid Retrieval (BM25 + Vector Search)
* Multi-Agent Research Workflows
* Citation Generation
* PDF Annotation Support
* Research Paper Comparison Mode
* User Authentication
* Team Collaboration Features
* Knowledge Graph Integration

---

## 📂 Project Structure

```text
ResearchPilot/
│
├── app.py
├── backend/
├── evaluate.py
├── documents/
├── embedding_cache/
├── sessions.json
├── eval_results.json
├── goldens.json
├── Dockerfile
├── README.md
```

---

## 👨‍💻 Author

**Sumit Kumar**

AI/ML Engineer | Generative AI Developer | Full-Stack Developer

If you found this project useful, consider giving it a ⭐ on GitHub.
