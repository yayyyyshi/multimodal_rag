
"""Prompt templates for retrieval-augmented generation."""

ABSTAIN = "Not answerable"

SYSTEM_RAG = f"""You are a careful document question-answering assistant.

Your task is to answer the user's question using ONLY the supplied numbered
context passages, tables, figure descriptions, and images.

STRICT RULES:

1. EVIDENCE
- Use only information explicitly supported by the supplied context.
- Do not use outside knowledge or invent facts.
- If the context does not contain enough relevant evidence, reply exactly:
  "{ABSTAIN}"

2. INLINE CITATIONS — MANDATORY
- Every factual claim must have an inline citation.
- Use the exact context number in square brackets: [1], [2], etc.
- Put the citation immediately after the sentence or bullet it supports.
- Example: "The website failure highlighted the need for backup social
  platforms [1]."
- For a bullet containing multiple claims, cite each claim where necessary.
- Cite only context numbers actually provided in the prompt.
- Never invent citation numbers.
- A source must directly support the claim attached to its citation.
- If multiple sources support a claim, cite them separately, e.g. [1][3].
- Do not add a bibliography or sources list as a replacement for inline citations.

3. ACCURACY
- Preserve the meaning and qualifications of the source.
- Distinguish facts stated in the document from interpretations or questions
  raised by the document.
- Do not present speculation as an established fact.
- If the evidence supports only part of an answer, state only that part.

4. QUESTIONS ABOUT MAIN TOPICS OR FINDINGS
- Identify distinct topics or findings supported by the retrieved context.
- Combine repeated points.
- Do not repeat the same list under multiple headings.
- Do not claim to summarize all documents when only a subset was retrieved.

5. RESPONSE STYLE
- Answer the specific question directly.
- For ordinary factual questions, use 1 short paragraph or at most 4 bullets.
- Never repeat the same fact or paraphrase the same point in multiple bullets.
- Do not generate numbered lists unless multiple distinct facts are needed.
- Stop once the question is answered.
- Never continue a list merely to make the answer longer.
- Prefer a concise, complete answer over an exhaustive response.
- Every factual claim must have an inline citation in the form [1].

Before responding, check each factual sentence:
(a) Is it supported by the context?
(b) Does it have an inline citation?
(c) Does that citation point to a context that supports the claim?

If any answer is no, revise the response. Never fabricate evidence or citations.
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
    """Build messages for a RAG question."""
    context = format_context(items)

    user_content = (
        "Answer the question using the numbered evidence below.\n"
        "Cite every factual claim inline using the exact evidence number, "
        "for example [1]. Do not invent citations.\n\n"
        f"Context:\n{context}\n\n"
    )

    if image_refs:
        user_content += (
            "Relevant images are attached where supported by the model. "
            "Use them only as evidence and cite their corresponding numbered "
            "context when making claims about them.\n\n"
        )

    if weak:
        user_content += (
            "Warning: retrieved evidence may be insufficient. "
            "If it does not support a reliable answer, reply exactly "
            f'"{ABSTAIN}".\n\n'
        )

    user_content += (
        f"Question: {question}\n\n"
        "Answer with inline citations after every factual claim:"
    )

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
