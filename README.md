# RAG Reference Implementation

A framework-agnostic, domain-agnostic reference implementation of a Retrieval Augmented Generation (RAG) pipeline. 

The goal of this project is to demonstrate clean abstraction seams so the pipeline can be adapted to different domains and infrastructure backends without re-engineering the core. It evolves from a **Naïve RAG** prototype into a true enterprise-grade **Advanced RAG** architecture.

## Core Philosophy

- **Explicit over Implicit:** Adapters are assembled in explicit Python code (`compose.py`) rather than through config-driven factories.
- **Async by Default:** All 11 pipeline stage interfaces are `async` to support high concurrency, streaming, and future agentic RAG patterns.
- **Strict Boundaries:** The Orchestration Layer routes data; Backend Adapters talk to infrastructure. They never mix.
- **Advanced RAG as First-Class Citizens:** Context Enrichment, Query Rewriting, Reranking, and Evaluation are explicit pipeline stages, complete with `NoOp` / `Identity` fallbacks for simpler deployments.

## Architecture

### Data Flow (The Contracts)

**1. Ingest Pipeline:**
`DocumentLoader` → `Chunker` → `ContextEnricher` → `VectorStore.index()`

**2. Query Pipeline:**
`PromptStore` & `ConversationStore` → `QueryRewriter` → `VectorStore.search()` → `Reranker` → `Generator` → `ConversationStore.append_message()`

**3. Offline Evaluation:**
`Evaluator` runs against a test set to measure Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

## Navigability Contract

Knowing **where** to make a change is the most important feature of this codebase.

| What you want to change | Where you go |
|---|---|
| **Orchestration framework** (e.g., LangGraph) | `rag/pipeline/` |
| **Infrastructure backend** (e.g., pgvector, Chroma) | `rag/adapters/<backend>/` |
| **Domain** (system prompt, document schema) | `config/` |
| **UI** (e.g., replace Streamlit with React) | `ui/` |
| **Stage contract or Shared Types** | `rag/stages/` & `rag/types.py` |
| **Error handling / retry logic** | `rag/errors.py` & `rag/pipeline/` |
| **Credentials and config** | `rag/settings.py` & `.env` |
| **Retrieval quality** (enrichment, reranking) | `rag/adapters/local/`, `rag/adapters/generic/`, `rag/adapters/cohere/` |
| **RAG quality measurement** | `rag/adapters/ragas/` & `rag/pipeline/evaluate.py` |
| **Resource lifecycle** (connections, sessions) | `AsyncExitStack` in `compose.py` |

## Infrastructure Implementations

This project ships with two complete reference backend pairs to prove the abstraction boundaries:

1. **Commodity/Open Stack:** OpenAI/Anthropic (LLM/Embeddings), Chroma/pgvector (VectorStore), SQLite/Postgres (Stores), Cohere (Reranker).
2. **Snowflake Native Stack:** Snowflake Cortex (LLM, Embeddings, Vector Search) and Snowflake Tables (Stores), maintaining strict zero-dependency isolation from OpenAI.

## Getting Started

1. **Environment Setup:** Copy `.env.example` to `.env` and fill in your keys.
2. **Choose your Stack:** - `python compose.py` (Simple Local/OpenAI stack)
   - `python compose_cortex.py` (Snowflake Cortex native stack)
   - `python compose_advanced.py` (Full Advanced RAG with Cohere & LLM enrichment)
3. **Run the UI:**
```bash
   streamlit run ui/app.py
```

## Key Architectural Decisions (ADRs)

* **ADR-0001:** Python implementation with Snowflake Cortex & OpenAI as reference pairs.
* **ADR-0002:** Explicit Python code wiring (`compose.py`) over config assembly.
* **ADR-0003:** Streamlit UI isolated to `ui/`.
* **ADR-0004:** All 11 stage interfaces are async.
* **ADR-0005:** `VectorStore` owns embedding. Orchestration only passes chunks.
* **ADR-0006:** Advanced RAG components (`ContextEnricher`, `QueryRewriter`, `Reranker`, `Evaluator`) are explicit interfaces with `NoOp` passthroughs.
* **ADR-0007:** Resource lifecycle managed via `AsyncExitStack` in composition roots.

