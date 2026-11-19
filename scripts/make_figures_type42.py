"""Regenerate the report tables and figures with TrueType (Type 42) fonts in the PDF figures, as used in the paper.

Runs the unmodified syscalib.make_report. The only difference from `python -m syscalib.make_report` is a
runtime Matplotlib setting (pdf.fonttype/ps.fonttype 3 -> 42) applied after the module's own rcParams.update,
which changes how fonts are embedded in the PDF figures. Data, layout and plotting code are unchanged.

Usage (from the repository root):
    python scripts/make_figures_type42.py --results results/rerun_seeds1000-1029 --outdir results_repro/report --figdir results_repro/figs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import syscalib.make_report as mr

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
mr.main()   # parses the same --results/--outdir/--figdir arguments as syscalib.make_report
