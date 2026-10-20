"""Prompt templates for retrieval-augmented generation."""

ABSTAIN = "Not answerable"

# Kept deliberately short: a 1.5B model follows a few concrete rules much
# better than a long rule list, and every prompt token costs CPU time.
SYSTEM_RAG = f"""You answer questions using only the numbered passages in the user message.

- Use only facts stated in the passages. Never add outside knowledge.
- The passages come from OCR and may be garbled. Ignore unreadable fragments, page headers, dates and author names unless the question asks about them.
- Keep times, numbers and names exactly as written.
- Write 1 to 4 short sentences of plain text: no lists, no bold, no headings.
- Do not repeat a fact. Do not write phrases like "as mentioned in the context". Stop once the question is answered.
- End each sentence with the number of the passage it comes from, like [2].
- If the passages do not contain the answer, reply exactly: {ABSTAIN}

Example
Context:
[1] (page 2, text passage)
The reading room opens at nine and closes at five.

[2] (page 3, text passage)
Membership is free for students.

Question: When does the reading room open, and who gets free membership?
Answer: The reading room opens at nine. [1] Students get free membership. [2]
"""

SYSTEM_NO_RAG = """You are a helpful assistant.
Answer the user's question clearly and concisely using your general knowledge.
Do not claim that you consulted documents or retrieved sources.
"""


def format_context(items: list[dict], max_chars: int = 1200) -> str:
    """Format retrieved items as numbered evidence passages."""
    sections = []

    for i, item in enumerate(items, start=1):
        doc_id = item.get("doc_id", "unknown")
        page_num = item.get("page_num", "?")
        modality = item.get("modality", "text")

        label = {
            "text": "text passage",
            "table": "table",
            "image": "figure description",
        }.get(modality, modality)

        head = f"[{i}] ({doc_id}, page {page_num}, {label})"

        body = (
            item.get("content")
            or item.get("caption")
            or ""
        ).strip()

        if not body:
            body = "[No text content available for this context.]"

        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "..."

        sections.append(f"{head}\n{body}")

    return "\n\n".join(sections)


def build_rag_messages(
    question: str,
    items: list[dict],
    weak: bool = False,
    image_refs: dict[int, object] | None = None,
) -> list[dict]:
    """Build messages for a RAG question.

    The "Context:" and "Question:" labels and the phrase "weakly related"
    are relied on by ExtractiveGenerator; keep them.
    """
    context = format_context(items)

    user_content = f"Context:\n{context}\n\n"

    if image_refs:
        user_content += (
            "Attached images are additional evidence; cite the numbered "
            "passage that describes them.\n\n"
        )

    if weak:
        user_content += (
            "Note: the passages are only weakly related to the question. "
            f'If they do not support a reliable answer, reply exactly "{ABSTAIN}".\n\n'
        )

    user_content += f"Question: {question}\nAnswer:"

    return [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": user_content},
    ]


def build_no_rag_messages(question: str) -> list[dict]:
    """Build messages when retrieval is disabled."""
    return [
        {"role": "system", "content": SYSTEM_NO_RAG},
        {"role": "user", "content": question},
    ]