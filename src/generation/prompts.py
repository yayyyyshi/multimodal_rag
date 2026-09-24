"""Prompts that restrict the model to the given context, require [n] citations and allow abstaining."""

from __future__ import annotations

ABSTAIN = "Not answerable"

SYSTEM_RAG = f"""You are a careful assistant that answers questions about documents.
Rules:
1. Use ONLY the numbered context (text passages, tables, figure descriptions and images). Do not use outside knowledge.
2. Cite the source number after each fact in square brackets, e.g. [2]. Cite only numbers that exist.
3. If the context does not contain the answer, reply exactly: "{ABSTAIN}".
4. Be concise: first the answer (a number, a name, a short phrase or a short list), then at most two sentences of justification."""

SYSTEM_NO_RAG = f"""You are a helpful assistant. Answer the question concisely.
If you do not know the answer, reply exactly: "{ABSTAIN}"."""

WEAK_EVIDENCE_NOTE = ("NOTE: the retrieved context is only weakly related to the question. "
                      f"If it does not clearly contain the answer, reply \"{ABSTAIN}\".")

MODALITY_LABEL = {"text": "text", "table": "table", "image": "figure"}


def format_context(items: list[dict], max_chars: int = 1200) -> str:
    blocks = []
    for i, it in enumerate(items, 1):
        label = MODALITY_LABEL.get(it.get("modality"), it.get("modality"))
        head = f"[{i}] ({it.get('doc_id', '?')}, page {it.get('page_num', '?')}, {label})"
        body = (it.get("content") or "").strip()
        if len(body) > max_chars:
            body = body[:max_chars] + " ..."
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks)


def build_rag_messages(question: str, items: list[dict], weak: bool = False, image_refs: dict | None = None) -> list[dict]:
    """image_refs maps context numbers to the images sent to a vision-language model."""
    user = f"Context:\n{format_context(items)}\n\n"
    if image_refs:
        user += f"The images attached are sources {', '.join(f'[{n}]' for n in image_refs)} (in that order).\n\n"
    if weak:
        user += WEAK_EVIDENCE_NOTE + "\n\n"
    user += f"Question: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}]


def build_no_rag_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_NO_RAG}, {"role": "user", "content": f"Question: {question}\nAnswer:"}]
