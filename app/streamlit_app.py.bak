"""Streamlit chat UI.

    streamlit run app/streamlit_app.py

Requires the artifacts from Kaggle and one run of scripts/build_index.py.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.rag import MultimodalRAG  # noqa: E402

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Retrieval-Augmented Multimodal Reasoning")
st.caption("Ask questions over text, tables, charts and images. Every answer cites its sources.")


@st.cache_resource(show_spinner="Loading index and models ...")
def load(backend: str, rerank: bool):
    return MultimodalRAG.from_config(backend=backend, use_reranker=rerank)


with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("LLM backend", ["ollama", "transformers", "extractive"],
                           help="ollama = fastest on CPU (run `ollama pull qwen2.5vl:3b` first). "
                                "transformers = Qwen2.5-1.5B via Hugging Face (slow on CPU). "
                                "extractive = no LLM, shows best evidence.")
    rerank = st.checkbox("Cross-encoder reranking", value=True)
    mode = st.radio("Mode", ["multimodal", "text_only", "no_rag"],
                    help="Compare our system with text-only RAG and with the LLM alone.")
    try:
        rag = load(backend, rerank)
    except Exception as e:
        st.error(f"Could not load the system: {e}")
        st.stop()
    docs = rag.store.list_docs()
    selected = st.multiselect("Search in documents (empty = all)", docs)
    st.caption(f"{len(docs)} documents · {rag.manifest.get('n_chunks', '?')} chunks · "
               f"{rag.manifest.get('n_images', '?')} images indexed")

question = st.chat_input("Ask a question about the documents ...")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning ..."):
            try:
                out = rag.answer(question, doc_ids=selected or None, mode=mode)
            except Exception as e:
                st.error(str(e))
                st.stop()
        st.markdown(out["answer"])
        if out["weak_evidence"] and mode != "no_rag":
            st.warning("Retrieved evidence was weak, so the model was told to abstain if unsure.")
        st.caption(f"retrieval {out['latency']['retrieval_s']}s · generation {out['latency']['generation_s']}s")

    if out["contexts"]:
        st.subheader("Evidence")
        cited = {c["n"] for c in out["citations"]}
        for i, it in enumerate(out["contexts"], 1):
            mark = "✅ cited" if i in cited else ""
            with st.expander(f"[{i}] {it['doc_id']} · page {it['page_num']} · {it['modality']} {mark}", expanded=i in cited):
                if it.get("image_path"):
                    img = rag.artifacts_dir / it["image_path"]
                    if img.exists():
                        st.image(str(img), width=420)
                st.text(it["content"][:1500])
                scores = {k: round(it[k], 3) for k in ("text_score", "clip_score", "rerank") if k in it}
                st.caption(str(scores))
