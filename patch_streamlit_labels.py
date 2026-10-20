"""One-off patch: stop the Streamlit app presenting model-written [n] markers as verified citations.

Run from the project root:   python patch_streamlit_labels.py
Creates app/streamlit_app.py.bak, then makes three exact-match edits.
If any expected text is not found exactly once, nothing is written.
"""
import sys
from pathlib import Path

PATH = Path("app/streamlit_app.py")

EDITS = [
    (   # 1) note when the generation limit cut the answer
        '        st.markdown(out["answer"])\n'
        '        if out["weak_evidence"] and mode != "no_rag":\n',
        '        st.markdown(out["answer"])\n'
        '        if out.get("truncated"):\n'
        '            st.caption("The answer reached the generation length limit; an unfinished "\n'
        '                       "final sentence may have been removed.")\n'
        '        if out["weak_evidence"] and mode != "no_rag":\n',
    ),
    (   # 2) honest caption + separate model-referenced vs lexical-suggestion sets
        '        cited = {c["n"] for c in out["citations"]}\n',
        '        st.caption("Retrieved evidence, verification: none. A tag means only that the model "\n'
        '                   "wrote that [n] in its answer; it does not check that the source supports the claim.")\n'
        '        cited = {c["n"] for c in out["citations"] if c.get("status") != "suggested_lexical"}\n'
        '        suggested = {c["n"] for c in out["citations"] if c.get("status") == "suggested_lexical"}\n',
    ),
    (   # 3) replace the green check "cited" mark
        '            mark = "\u2705 cited" if i in cited else ""\n',
        '            mark = ("\u00b7 referenced by model (unverified)" if i in cited\n'
        '                    else "\u00b7 lexical suggestion (not verification)" if i in suggested else "")\n',
    ),
]


def main() -> int:
    if not PATH.exists():
        print(f"{PATH} not found; run this from the project root.")
        return 1

    raw = PATH.read_bytes()
    text = raw.decode("utf-8")
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")

    if "referenced by model (unverified)" in text:
        print("Already patched; nothing to do.")
        return 0

    for number, (old, _) in enumerate(EDITS, start=1):
        count = text.count(old)
        if count != 1:
            print(f"Edit {number}: expected text found {count} times (need exactly 1). Nothing written.")
            print("Expected:\n" + old)
            return 1

    for old, new in EDITS:
        text = text.replace(old, new)

    PATH.with_suffix(".py.bak").write_bytes(raw)
    PATH.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("Patched app/streamlit_app.py (backup: app/streamlit_app.py.bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
