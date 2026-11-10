"""Agreement between report/main.tex, the exported assets and the bibliography."""
import re

from aerocast_dl import config, report

TEX = (report.REPORT_DIR / "main.tex").read_text()


def test_no_unfilled_placeholders():
    assert not re.findall(r"%%[A-Z]+%%", TEX)


def test_report_uses_exactly_the_exported_tables():
    used = set(re.findall(r"\\input\{tables/([\w-]+)\.tex\}", TEX))
    assert used == set(report.report_tables())


def test_report_uses_exactly_the_exported_figures():
    used = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{(?:figures/)?([\w-]+)\.png\}", TEX))
    assert used == set(report.REPORT_FIGURES)
    assert all((config.FIGURES_DIR / f"{f}.png").exists() for f in report.REPORT_FIGURES)


def test_bibliography_matches_citations():
    cited = {k.strip() for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", TEX) for k in group.split(",")}
    bib = set(re.findall(r"^@\w+\{([^,]+),", (report.REPORT_DIR / "references.bib").read_text(), re.M))
    assert cited == bib
