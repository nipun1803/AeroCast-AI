"""LaTeX export helpers."""
import pandas as pd

from aerocast import report


def test_booktabs_structure_and_escaping():
    df = pd.DataFrame({"model": ["LogReg (Platt, rule-selected)", "A_b 50%"],
                       "pr_auc": [0.39369, float("nan")], "thr": [float("inf"), 0.5]})
    tex = report.to_booktabs(df, decimals=3)
    lines = tex.strip().splitlines()
    assert lines[0] == r"\begin{tabular}{lrr}"
    assert r"\toprule" in tex and r"\midrule" in tex and tex.rstrip().endswith(r"\end{tabular}")
    assert r"A\_b 50\%" in tex                       # special characters escaped
    assert "0.394" in tex and "--" in tex and r"$\infty$" in tex
    assert tex.count(r"\\") == 3                     # header + 2 body rows


def test_symbols_are_latex_safe():
    assert report.tex_escape("P≥0.4") == r"P$\geq$0.4"
