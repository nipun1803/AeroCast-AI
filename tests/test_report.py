"""LaTeX export helpers, and agreement between report/main.tex and what the export produces."""
import re

import pandas as pd

from aerocast import config, report

TEX = (report.REPORT_DIR / "main.tex").read_text()


def test_report_uses_exactly_the_exported_tables():
    used = set(re.findall(r"\\input\{tables/([\w-]+)\.tex\}", TEX))
    assert used == set(report.report_tables())


def test_report_uses_exactly_the_exported_figures():
    used = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([\w-]+)\.png\}", TEX))
    assert used == set(report.REPORT_FIGURES)
    assert all((config.FIGURES_DIR / f"{f}.png").exists() for f in report.REPORT_FIGURES)


def test_bibliography_matches_citations():
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", TEX) for k in group.split(",")}
    bib = set(re.findall(r"^@\w+\{([^,]+),", (report.REPORT_DIR / "references.bib").read_text(), re.M))
    assert cited == bib


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
