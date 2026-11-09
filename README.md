# GroundGate Lite

**Hybrid RAG that says "I don't know" instead of guessing.** Open source, plain Python, no
framework lock-in.

Most RAG demos answer every question, even when nothing in your documents is relevant. GroundGate
Lite checks relevance *before* it calls the LLM. If the best chunk isn't relevant enough, it
returns "I don't know based on the provided documents." and the model never gets a chance to
invent an answer.

```
question ─▶ vector top-50 ┐
            BM25 top-50   ┴▶ Reciprocal Rank Fusion ─▶ rerank ─▶ relevance gate ─▶ cited answer
                                                                   └─▶ "I don't know"
```

## Quick start (no API key needed)

```bash
pip install -e ".[dev]"
ggl ingest sample_data/docs
ggl ask "What does error LMR-4031 mean?"      # cited answer
ggl ask "Who is the CEO of Lumora?"            # I don't know based on the provided documents.
ggl eval evals/golden.jsonl                    # hit rate, MRR, refusal rates
```

The sample corpus is a fictional SaaS company with error codes, tables and an unanswerable-question
set. Offline mode uses a hashing embedder and an extractive answerer, so you can see the pipeline
work before spending a cent.

## Use real models

```bash
cp .env.example .env
# OpenAI
GG_LLM_PROVIDER=openai GG_EMBED_PROVIDER=openai GG_OPENAI_API_KEY=sk-...
# or fully local and free with Ollama
GG_LLM_PROVIDER=ollama GG_EMBED_PROVIDER=ollama   # ollama pull llama3.1:8b && ollama pull nomic-embed-text
```

Delete `.groundgate-lite/` and re-ingest after changing the embedding model.

## What's inside

| Piece | Why it matters |
|---|---|
| Structure-aware chunking (~512 tokens, heading breadcrumbs, tables kept whole) | Chunks know which product/section they belong to |
| Vector + BM25 hybrid search | Exact identifiers (error codes, SKUs, versions) that embeddings blur |
| Reciprocal Rank Fusion (k=60) | Merges cosine and BM25 rankings without score normalisation |
| Lexical reranker with pool-IDF weighting | A term in every chunk (your brand name) proves nothing; identifiers count double |
| Relevance gate | Refuses before generation instead of answering from the best of a bad lot |
| Dedup, per-document cap, MMR, best-at-edges ordering | No five copies of one paragraph; strongest chunks where models read best |
| Idempotent ingest | Re-runs are free; changed docs replace their old chunks atomically |
| Postgres + pgvector store | HNSW index, OR-semantics full-text search (`websearch_to_tsquery` alone ANDs every word and silently returns nothing for most questions) |
| Retrieval eval | hit rate@5, MRR@10, false-refusal and correct-refusal rates on a golden set |
| Providers | OpenAI (and any OpenAI-compatible API), Ollama, offline |

## Library use

```python
from groundgate_lite import Lite, Document

rag = Lite()
rag.ingest([Document(doc_id="handbook.md", text=open("handbook.md").read(), title="Handbook")])
print(rag.ask("How many vacation days do new hires get?"))
```

## Need it production-ready?

**[GroundGate](LINK_TO_GUMROAD)** is the full kit this was cut from. It adds:

- **Claim-by-claim grounding**: every sentence of the answer is checked against its cited source;
  invented numbers are stripped before users see them
- **Prompt-injection defence**: hidden-HTML stripping, quarantine of poisoned documents, canary
  leak detection
- **Multi-tenant isolation** (tenant from API key, pre-filtering, Postgres row-level security)
- **Full evaluation + CI quality gate**: faithfulness, correctness, citation precision, injection
  probes, latency and cost; `gg gate` fails your pull request on regressions; threshold
  calibration; synthetic test-set generation
- Cross-encoder / Cohere / LLM rerankers, Anthropic Claude, contextual retrieval, query rewriting
- FastAPI API + web console, answer caching, per-request traces with token cost, Docker Compose
- A 34-page guide and a 280+ source research log behind every default

## License

MIT
