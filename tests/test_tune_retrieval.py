"""Tests for scripts/tune_retrieval.py. Run: python -m pytest tests/test_tune_retrieval.py -q"""

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tune_retrieval", ROOT / "scripts" / "tune_retrieval.py")
tune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune)

# Real "## Overall" tables printed by evaluate.py on the user's machine.
REAL_NORERANK = """\
## Overall

| Metric | text_only | multimodal |
|---|---|---|
| Hit@1 | 0.512 | 0.372 |
| Hit@5 | 0.721 | 0.767 |
| Recall@5 | 0.621 | 0.698 |
| MRR | 0.582 | 0.538 |

## By evidence type: text_only

| Evidence | n | Acc. | Hit@5 | Recall@5 | MRR |
|---|---|---|---|---|---|
| ? | 2 | - | 1.000 | 1.000 | 1.000 |
"""

CONFIG = "retrieval:\n  top_k_text: 12\n  image_weight: 0.7\n  final_k: 6\n"


def table(hit1, hit5, rec5, mrr, text=(0.512, 0.721, 0.621, 0.582)):
    keys = ("Hit@1", "Hit@5", "Recall@5", "MRR")
    mm = dict(zip(keys, (hit1, hit5, rec5, mrr)))
    to = dict(zip(keys, text))
    return {k: {"text_only": to[k], "multimodal": mm[k]} for k in keys}


def test_parse_real_output():
    t = tune.parse_overall("noise before\n" + REAL_NORERANK + "[eval] results -> x")
    assert t["Hit@5"] == {"text_only": 0.721, "multimodal": 0.767}
    assert t["MRR"]["multimodal"] == 0.538 and set(t) == {"Hit@1", "Hit@5", "Recall@5", "MRR"}


def test_parse_errors():
    with pytest.raises(ValueError):
        tune.parse_overall("no table here")


def test_image_weight_edit_only_touches_that_line():
    assert tune.read_image_weight(CONFIG) == 0.7
    new = tune.set_image_weight(CONFIG, 0.3)
    assert "  image_weight: 0.3\n" in new and "top_k_text: 12" in new and "final_k: 6" in new
    assert tune.set_image_weight(CONFIG, 0.0).count("image_weight: 0") == 1
    with pytest.raises(ValueError):
        tune.set_image_weight("retrieval:\n  final_k: 6\n", 0.3)


def test_recommend_requires_clear_gain_without_top5_loss():
    base = table(0.372, 0.767, 0.698, 0.538)
    clear = table(0.60, 0.767, 0.700, 0.65)           # big ordering gain, Hit@5 intact
    lossy = table(0.60, 0.70, 0.700, 0.70)            # gain but Hit@5 drops 0.067
    tiny = table(0.38, 0.77, 0.70, 0.54)              # inside noise
    assert tune.recommend({0.7: base, 0.5: clear}, 0.7)[0] == 0.5
    assert tune.recommend({0.7: base, 0.5: lossy}, 0.7)[0] == 0.7
    assert tune.recommend({0.7: base, 0.5: tiny}, 0.7)[0] == 0.7
    assert tune.recommend({0.7: base}, 0.7)[0] == 0.7


FAKE_EVAL = textwrap.dedent('''
    import re, sys
    args = sys.argv
    cfg = open(args[args.index("--config") + 1], encoding="utf-8").read()
    w = float(re.search(r"image_weight:\\s*([0-9.]+)", cfg).group(1))
    if "FAIL_AT" in cfg and w == float(re.search(r"FAIL_AT:\\s*([0-9.]+)", cfg).group(1)):
        sys.stderr.write("boom"); sys.exit(3)
    mm = {0.0: (0.40, 0.70, 0.65, 0.50), 0.3: (0.55, 0.76, 0.69, 0.60), 0.5: (0.62, 0.77, 0.70, 0.66),
          0.7: (0.37, 0.767, 0.698, 0.538), 1.0: (0.30, 0.74, 0.66, 0.45)}[w]
    wobble = 0.01 if "NONDET" in cfg and w == 0.3 else 0.0
    print("## Overall\\n\\n| Metric | text_only | multimodal |\\n|---|---|---|")
    for name, m, t in zip(("Hit@1", "Hit@5", "Recall@5", "MRR"), mm, (0.512, 0.721, 0.621, 0.582)):
        print(f"| {name} | {t + wobble:.3f} | {m:.3f} |")
''')


def run_main(tmp_path, monkeypatch, capsys, extra_config="", extra_args=()):
    (tmp_path / "config.yaml").write_text(CONFIG + extra_config, encoding="utf-8")
    (tmp_path / "fake_eval.py").write_text(FAKE_EVAL, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tune_retrieval.py", "--evaluate-script", "fake_eval.py", *extra_args])
    code = tune.main()
    return code, capsys.readouterr().out


def test_main_end_to_end_picks_peak_and_cleans_up(tmp_path, monkeypatch, capsys):
    before = CONFIG
    code, out = run_main(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "Sanity check OK" in out
    assert "image_weight: 0.5 improves" in out
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == before      # never modified
    assert not list(tmp_path.glob(".tune_cfg_*.yaml"))                            # temp files removed


def test_main_warns_when_text_only_differs(tmp_path, monkeypatch, capsys):
    code, out = run_main(tmp_path, monkeypatch, capsys, extra_config="# NONDET\n")
    assert code == 0 and "WARNING: text_only numbers differ" in out


def test_main_skips_a_failing_weight_but_fails_if_baseline_fails(tmp_path, monkeypatch, capsys):
    code, out = run_main(tmp_path, monkeypatch, capsys, extra_config="FAIL_AT: 0.3\n")
    assert code == 0 and "skipped" in out and "boom" in out
    code, out = run_main(tmp_path, monkeypatch, capsys, extra_config="FAIL_AT: 0.7\n")
    assert code == 1 and "baseline run failed" in out


def test_find_config_locations(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text(CONFIG, encoding="utf-8")
    assert tune.find_config(None, tmp_path) == tmp_path / "configs" / "config.yaml"      # the reported failure
    (tmp_path / "config.yaml").write_text(CONFIG, encoding="utf-8")
    assert tune.find_config(None, tmp_path) == tmp_path / "config.yaml"                  # root wins
    explicit = tmp_path / "other.yaml"
    explicit.write_text(CONFIG, encoding="utf-8")
    assert tune.find_config(str(explicit), tmp_path) == explicit


def test_find_config_unique_nested_ambiguous_and_missing(tmp_path):
    (tmp_path / "venv" / "lib").mkdir(parents=True)
    (tmp_path / "venv" / "lib" / "config.yaml").write_text("x: 1", encoding="utf-8")    # must be ignored
    with pytest.raises(FileNotFoundError):
        tune.find_config(None, tmp_path)
    (tmp_path / "settings").mkdir()
    (tmp_path / "settings" / "config.yaml").write_text(CONFIG, encoding="utf-8")
    assert tune.find_config(None, tmp_path) == tmp_path / "settings" / "config.yaml"
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "config.yaml").write_text(CONFIG, encoding="utf-8")
    with pytest.raises(LookupError):
        tune.find_config(None, tmp_path)
    with pytest.raises(FileNotFoundError):
        tune.find_config("nope.yaml", tmp_path)


def test_main_uses_configs_folder(tmp_path, monkeypatch, capsys):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "fake_eval.py").write_text(FAKE_EVAL, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tune_retrieval.py", "--evaluate-script", "fake_eval.py"])
    code = tune.main()
    out = capsys.readouterr().out
    assert code == 0 and "Using base config:" in out and "configs" in out and "image_weight: 0.5 improves" in out