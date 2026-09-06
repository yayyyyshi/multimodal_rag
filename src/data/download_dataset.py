"""
Download a small subset of the DocVQA dataset from Hugging Face for use
as this project's sample multimodal corpus (real scanned document images:
letters, memos, forms, reports, tables, advertisements).

DocVQA: https://huggingface.co/datasets/lmms-lab/DocVQA
License: Apache-2.0
Citation: Mathew et al., "DocVQA: A Dataset for VQA on Document Images" (2021)

The full validation split is ~12GB, so this script STREAMS the dataset and
only saves images for a limited number of unique documents (one image per
docId — DocVQA has multiple questions per document image, we only need the
image once). QA pairs are still saved for every question, for later use in
evaluation (Week 11).

Usage:
    python src/data/download_dataset.py --num-docs 100
    python src/data/download_dataset.py --num-docs 100 --out-dir data/raw/docvqa
"""

import argparse
import json
import os

from datasets import load_dataset


def download_subset(num_docs: int, out_dir: str, split: str = "validation") -> None:
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"[INFO] Streaming lmms-lab/DocVQA ({split} split) — no full download...")
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split, streaming=True)

    seen_doc_ids = set()
    qa_records = []
    saved_images = 0

    for example in ds:
        doc_id = example["docId"]

        # Save the image only the first time we see this docId
        if doc_id not in seen_doc_ids:
            if len(seen_doc_ids) >= num_docs:
                # We've collected enough unique documents; stop scanning
                # once we've also grabbed any trailing QA pairs for docs
                # already saved. Since rows are grouped by doc in this
                # dataset, it's safe to break here.
                break

            image_path = os.path.join(images_dir, f"{doc_id}.jpg")
            example["image"].convert("RGB").save(image_path, "JPEG", quality=90)
            seen_doc_ids.add(doc_id)
            saved_images += 1

            if saved_images % 20 == 0:
                print(f"[INFO] Saved {saved_images}/{num_docs} document images...")

        qa_records.append({
            "doc_id": doc_id,
            "question_id": example["questionId"],
            "question": example["question"],
            "question_types": example["question_types"],
            "answers": example["answers"],
            "image_path": os.path.join("images", f"{doc_id}.jpg"),
        })

    qa_path = os.path.join(out_dir, "qa_metadata.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_records, f, indent=2)

    print(f"[DONE] Saved {saved_images} document images -> {images_dir}")
    print(f"[DONE] Saved {len(qa_records)} QA records -> {qa_path}")


def main():
    parser = argparse.ArgumentParser(description="Download a DocVQA subset for the multimodal RAG project")
    parser.add_argument("--num-docs", type=int, default=100, help="Number of unique document images to download")
    parser.add_argument("--out-dir", default="data/raw/docvqa", help="Output directory")
    parser.add_argument("--split", default="validation", choices=["validation", "test"], help="Dataset split to use")
    args = parser.parse_args()

    download_subset(args.num_docs, args.out_dir, args.split)


if __name__ == "__main__":
    main()
