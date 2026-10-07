"""Paths and experiment settings shared by every notebook and module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "cfb_box_scores_2000_2010.csv"
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results"

SEED = 42
FIRST_STATS_SEASON = 2004   # box-score stats are empty before 2004 (see notebook 01 audit)
MIN_GAMES = 8               # min regular-season games with stats for a team to be usable
PRIMARY_MODEL = "Differential Ridge"   # fixed BEFORE looking at results


def ensure_dirs():
    for d in (DATA_PROC, FIG_DIR, RES_DIR):
        d.mkdir(parents=True, exist_ok=True)
