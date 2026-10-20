"""Measure retrieval quality for several image_weight values (no LLM, no reranker).

Run from the project root:
    python scripts/tune_retrieval.py --limit 120

For each weight this writes a temporary copy of the config (auto-detected, or --config), runs
scripts/evaluate.py --retrieval-only on it, parses the "## Overall" table it
prints, and then compares. It never modifies config.yaml.

Notes
- text_only retrieval ignores image hits, so its numbers must be identical
  for every weight. That is used as a sanity check that the setting really
  was applied and that runs are deterministic.
- The recommendation is deliberately conservative: it needs a clear gain and
  no loss of top-5 hit rate. Small differences are noise at this sample size.
- Weights are chosen on the same questions that are reported, so confirm any
  change with a larger run (omit --limit) before trusting the gain.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path

QUALITY = ("Hit@5", "Recall@5", "MRR")
IMAGE_WEIGHT_RE = re.compile(r"^(\s*image_weight:\s*)([0-9.]+)", re.MULTILINE)
MIN_GAIN = 0.03        # mean of QUALITY metrics must improve by at least this
MAX_HIT5_LOSS = 0.02   # and Hit@5 must not drop by more than this
NOISE_NOTE = 0.05


SKIP_DIRS = {"venv", ".venv", "env", ".git", "node_modules", "site-packages",
             "__pycache__", "results", "artifacts", "data"}


def find_config(explicit: str | None, root: Path = Path(".")) -> Path:
    """Locate the base config: --config, else common places, else a unique config.yaml."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--config {explicit} does not exist.")
        return path

    for relative in ("config.yaml", "configs/config.yaml", "config/config.yaml"):
        if (root / relative).is_file():
            return root / relative

    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]   # never descend into venv etc.
        if "config.yaml" in files:
            found.append(Path(folder) / "config.yaml")

    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError("No config.yaml found. Pass it with --config PATH.")
    listing = "\n  ".join(str(p) for p in found)
    raise LookupError(f"Several config.yaml files found; pass one with --config PATH:\n  {listing}")


def read_image_weight(config_text: str) -> float | None:
    match = IMAGE_WEIGHT_RE.search(config_text)
    return float(match.group(2)) if match else None


def set_image_weight(config_text: str, weight: float) -> str:
    new_text, count = IMAGE_WEIGHT_RE.subn(
        lambda m: f"{m.group(1)}{weight:g}", config_text, count=1
    )
    if count != 1:
        raise ValueError("No 'image_weight:' line found in the config file.")
    return new_text


def parse_overall(output: str) -> dict[str, dict[str, float]]:
    """Parse the '## Overall' markdown table printed by evaluate.py.

    Returns {metric: {mode: value}}, e.g. {"Hit@5": {"text_only": 0.721, ...}}.
    """
    lines = output.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "## Overall")
    except StopIteration:
        raise ValueError("No '## Overall' table found in evaluate.py output.")

    header: list[str] | None = None
    table: dict[str, dict[str, float]] = {}

    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            if table:
                break
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Metric":
            header = cells[1:]
            continue
        if header is None or set(cells[0]) <= set("-: "):
            continue

        values = {}
        for mode, cell in zip(header, cells[1:]):
            try:
                values[mode] = float(cell)
            except ValueError:
                values[mode] = math.nan
        table[cells[0]] = values

    if not table:
        raise ValueError("The '## Overall' table was empty.")
    return table


def quality(table: dict, mode: str) -> float:
    values = [table[m][mode] for m in QUALITY if m in table and mode in table[m]]
    return sum(values) / len(values) if values else math.nan


def recommend(results: dict[float, dict], baseline: float, mode: str = "multimodal"):
    """Return (best_weight, reason). Conservative: needs a clear gain and no top-5 loss."""
    base = results[baseline]
    base_q, base_hit5 = quality(base, mode), base["Hit@5"][mode]

    best_weight, best_q = baseline, base_q
    for weight, table in results.items():
        q = quality(table, mode)
        if q > best_q and table["Hit@5"][mode] >= base_hit5 - MAX_HIT5_LOSS:
            best_weight, best_q = weight, q

    if best_weight == baseline or best_q - base_q < MIN_GAIN:
        return baseline, (
            f"keep image_weight: {baseline:g}. No setting improved the mean of "
            f"{'/'.join(QUALITY)} by at least {MIN_GAIN:.2f} without losing top-5 hits."
        )
    return best_weight, (
        f"image_weight: {best_weight:g} improves the mean of {'/'.join(QUALITY)} "
        f"by {best_q - base_q:+.3f} over {baseline:g}. Confirm with a larger run "
        f"(omit --limit) before adopting it."
    )


def run_one(weight, base_text, args) -> dict:
    tag = f"{weight:g}".replace(".", "p")
    tmp_config = Path(f".tune_cfg_iw{tag}.yaml")
    tmp_config.write_bytes(set_image_weight(base_text, weight).encode("utf-8"))  # no BOM

    cmd = [
        sys.executable, args.evaluate_script,
        "--retrieval-only", "--modes", "text_only", "multimodal",
        "--no-rerank", "--limit", str(args.limit),
        "--config", str(tmp_config), "--run-name", f"tune_iw{tag}",
    ]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-8:])
            raise RuntimeError(f"evaluate.py failed (exit {proc.returncode}):\n{tail}")
        return parse_overall(proc.stdout)
    finally:
        if not args.keep:
            tmp_config.unlink(missing_ok=True)


def print_report(results: dict[float, dict], baseline: float) -> None:
    print("\n=== multimodal retrieval by image_weight (no reranker) ===")
    print(f"{'image_weight':<14}{'Hit@1':>8}{'Hit@5':>8}{'Recall@5':>10}{'MRR':>8}{'mean(H5,R5,MRR)':>18}")
    for weight in sorted(results):
        t = results[weight]
        tag = " (current)" if weight == baseline else ""
        print(f"{weight:<14g}{t['Hit@1']['multimodal']:>8.3f}{t['Hit@5']['multimodal']:>8.3f}"
              f"{t['Recall@5']['multimodal']:>10.3f}{t['MRR']['multimodal']:>8.3f}"
              f"{quality(t, 'multimodal'):>18.3f}{tag}")

    text_only = {w: tuple(round(results[w][m]["text_only"], 4) for m in ("Hit@1", "Hit@5", "Recall@5", "MRR"))
                 for w in results}
    if len(set(text_only.values())) == 1:
        print("\nSanity check OK: text_only numbers are identical for every weight, as they should be.")
    else:
        print("\nWARNING: text_only numbers differ between runs. Either the runs are not "
              "deterministic or something other than image_weight changed. Do not trust the comparison.")
        for w, v in sorted(text_only.items()):
            print(f"  image_weight={w:g}: text_only (Hit@1, Hit@5, Recall@5, MRR) = {v}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Search image_weight with retrieval-only evaluation.")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--limit", type=int, default=120, help="Questions per run")
    parser.add_argument("--config", default=None,
                        help="Base config (never modified); auto-detected if omitted")
    parser.add_argument("--evaluate-script", default="scripts/evaluate.py")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary config files")
    args = parser.parse_args()

    try:
        config_path = find_config(args.config)
    except (FileNotFoundError, LookupError) as error:
        print(error)
        return 1

    print(f"Using base config: {config_path}")
    base_text = config_path.read_text(encoding="utf-8")
    baseline = read_image_weight(base_text)
    if baseline is None:
        print(f"No 'image_weight:' line in {config_path}.")
        return 1

    weights = sorted(set(args.weights) | {baseline})
    results: dict[float, dict] = {}

    for index, weight in enumerate(weights, start=1):
        print(f"[{index}/{len(weights)}] image_weight={weight:g} ...", flush=True)
        try:
            results[weight] = run_one(weight, base_text, args)
        except Exception as error:  # report and continue with the other weights
            print(f"    skipped: {error}")

    if baseline not in results:
        print("The baseline run failed, so there is nothing to compare against.")
        return 1

    print_report(results, baseline)
    best, reason = recommend(results, baseline)
    print(f"\nRecommendation: {reason}")
    print(f"(Differences under about {NOISE_NOTE:.2f} are noise at this sample size.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())