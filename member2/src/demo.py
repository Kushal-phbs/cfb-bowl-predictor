"""Live-demo helpers.  (Notebook 03)  Reads the CSVs written by notebooks 01 and 02."""
import numpy as np
import pandas as pd

from .config import DATA_PROC, RES_DIR, PRIMARY_MODEL
from .features import FEATS
from .models import MODELS, KIND

_cache = {}


def _data():
    if not _cache:
        _cache["bowls"] = pd.read_csv(DATA_PROC / "bowl_matchups.csv", parse_dates=["date"])
        _cache["team_feats"] = pd.read_csv(DATA_PROC / "team_season_features.csv").set_index(["season", "team"])
        _cache["cv"] = pd.read_csv(RES_DIR / "cv_predictions.csv")
    return _cache["bowls"], _cache["team_feats"], _cache["cv"]


def _row(team_feats, team_a, team_b, season):
    fa, fb = team_feats.loc[(season, team_a)], team_feats.loc[(season, team_b)]
    r = {f"a_{c}": fa[c] for c in FEATS} | {f"b_{c}": fb[c] for c in FEATS} | {f"d_{c}": fa[c] - fb[c] for c in FEATS}
    return pd.DataFrame([r])


def predict_matchup(team_a, team_b, season, model_name=PRIMARY_MODEL, verbose=True):
    """Predict one bowl game. The chosen season is excluded from training (fair out-of-sample prediction)."""
    bowls, team_feats, cv = _data()
    model = MODELS[model_name]().fit(bowls[bowls.season != season])
    sig = float(model.predict(_row(team_feats, team_a, team_b, season))[0])
    winner = team_a if sig > 0 else team_b
    hist = cv[(cv.model == model_name) & (cv.conf >= abs(sig))]
    hist_acc = hist.correct.mean() if len(hist) >= 10 else np.nan
    if verbose:
        unit = "pts" if KIND[model_name] == "margin" else "(signal)"
        print(f"{season}: {team_a} vs {team_b}")
        print(f"  Predicted winner : {winner}")
        print(f"  Predicted margin : {abs(sig):.1f} {unit}")
        if not np.isnan(hist_acc):
            print(f"  Historical accuracy at this confidence or higher: {hist_acc:.0%} ({len(hist)} CV games)")
    return winner, sig


def pickem_sheet(season, model_name=PRIMARY_MODEL):
    """Full Bowl Pick'em sheet for a season (that season is held out of training)."""
    bowls, _, _ = _data()
    te = bowls[bowls.season == season].copy()
    model = MODELS[model_name]().fit(bowls[bowls.season != season])
    te["signal"] = model.predict(te)
    te["pick"] = np.where(te.signal > 0, te.team_a, te.team_b)
    te["winner"] = np.where(te.margin > 0, te.team_a, te.team_b)
    te["confidence_pts"] = te.signal.abs().rank(method="first").astype(int)
    te["points_won"] = np.where(te.pick == te.winner, te.confidence_pts, 0)
    sheet = te.sort_values("confidence_pts", ascending=False)[
        ["team_a", "team_b", "pick", "confidence_pts", "winner", "points_won"]]
    n = len(te)
    print(f"Season {season}: {int(te.points_won.sum())} / {n * (n + 1) // 2} possible points "
          f"({te.points_won.sum() / (n * (n + 1) / 2):.0%}); correct picks {int((te.pick == te.winner).sum())}/{n}")
    return sheet
