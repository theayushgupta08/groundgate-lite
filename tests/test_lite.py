from groundgate_lite import Document, Lite, Settings
from groundgate_lite.evals import evaluate


def make():
    rag = Lite(Settings(_env_file=None))
    rag.ingest_dir("sample_data/docs")
    return rag


def test_answers_with_citation():
    out = make().ask("What does error LMR-4031 mean?")
    assert not out["abstained"] and "[1]" in out["answer"] and out["sources"][0]["doc_id"] == "error-codes.md"


def test_says_i_dont_know():
    out = make().ask("Who is the CEO of Lumora?")
    assert out["abstained"] and out["sources"] == []


def test_idempotent_ingest():
    rag = make()
    assert rag.ingest_dir("sample_data/docs")["added"] == []


def test_update_replaces_chunks():
    rag = Lite(Settings(_env_file=None))
    rag.ingest([Document(doc_id="p.md", text="# P\n\nThe limit is 50 requests per minute for the plan.")])
    rag.ingest([Document(doc_id="p.md", text="# P\n\nThe limit is 75 requests per minute for the plan.")])
    chunks = rag.store.all_chunks("default")
    assert len(chunks) == 1 and "75" in chunks[0].text


def test_retrieval_eval():
    m = evaluate(make(), "evals/golden.jsonl")
    assert m["hit_rate@5"] >= 0.9 and m["correct_refusal_rate"] >= 0.8
