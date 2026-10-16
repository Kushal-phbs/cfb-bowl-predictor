"""Models, leave-one-season-out CV and metrics.  (Notebook 02)"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from .config import SEED
from .features import D_COLS


def ols_pvalues(X, y):
    """Two-sided p-values for no-intercept OLS coefficients (numpy only)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return 2 * stats.t.sf(np.abs(beta / se), dof)


def scaled(est):
    # scale only (no centring) so that the model stays odd: f(-x) = -f(x)
    return make_pipeline(StandardScaler(with_mean=False), est)


class RuleModel:
    kind = "rule"
    def __init__(self, col=None): self.col = col
    def fit(self, tr): return self
    def predict(self, te):
        s = np.ones(len(te)) if self.col is None else te[self.col].to_numpy().astype(float)
        return np.where(s == 0, 1e-9, s)              # ties -> pick team A


class DiffModel:
    """Symmetric model on feature differences d = f(A) - f(B)."""
    def __init__(self, make_est, kind="margin", select_p=None, augment=False):
        self.make_est, self.kind, self.select_p, self.augment = make_est, kind, select_p, augment

    def fit(self, tr):
        cols = list(D_COLS)
        if self.select_p is not None:                  # selection uses ONLY the training fold
            p = ols_pvalues(tr[cols].to_numpy(), tr["margin"].to_numpy())
            keep = [c for c, pv in zip(cols, p) if pv < self.select_p]
            cols = keep if len(keep) >= 3 else [cols[i] for i in np.argsort(p)[:3]]
        self.cols_ = cols
        X = tr[cols].to_numpy()
        y = tr["margin"].to_numpy() if self.kind == "margin" else tr["a_won"].to_numpy()
        if self.augment:                               # mirror every game (A<->B) for non-linear models
            X = np.vstack([X, -X])
            y = np.r_[y, -y] if self.kind == "margin" else np.r_[y, 1 - y]
        self.est_ = self.make_est().fit(X, y)
        return self

    def predict(self, te):
        X = te[self.cols_].to_numpy()
        if self.kind == "margin":
            return self.est_.predict(X)
        if self.kind == "decision":
            return self.est_.decision_function(X)
        return self.est_.predict_proba(X)[:, 1] - 0.5


OFF = ["ppg", "ypg", "pass_ypg", "rush_ypg", "ypc", "ypa", "comp_pct", "first_downs_pg", "third_pct", "giveaways_pg"]
DEF = ["papg", "ypg_allowed", "pass_ypg_allowed", "rush_ypg_allowed", "ypc_allowed", "ypa_allowed",
       "comp_pct_allowed", "first_downs_allowed_pg", "third_pct_allowed", "takeaways_pg"]


class ScoreRegression:
    """Predict each team's points from (its offence, opponent's defence); margin = predicted A - predicted B."""
    kind = "margin"
    def _X(self, te, me, opp):
        return np.hstack([te[[f"{me}_{c}" for c in OFF]].to_numpy(), te[[f"{opp}_{c}" for c in DEF]].to_numpy()])

    def fit(self, tr):
        X = np.vstack([self._X(tr, "a", "b"), self._X(tr, "b", "a")])
        y = np.r_[tr["score_home"].to_numpy(), tr["score_away"].to_numpy()]
        self.est_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(X, y)
        return self

    def predict(self, te):
        return self.est_.predict(self._X(te, "a", "b")) - self.est_.predict(self._X(te, "b", "a"))


ALPHAS = np.logspace(-2, 4, 25)
MODELS = {
    "Baseline: listed home team":       lambda: RuleModel(),
    "Baseline: higher win %":           lambda: RuleModel("d_win_pct"),
    "Baseline: higher avg margin":      lambda: RuleModel("d_margin_avg"),
    "Score Regression":                 lambda: ScoreRegression(),
    "Differential OLS (all)":           lambda: DiffModel(lambda: scaled(LinearRegression(fit_intercept=False))),
    "Differential Ridge":               lambda: DiffModel(lambda: scaled(RidgeCV(alphas=ALPHAS, fit_intercept=False))),
    "Differential OLS + p-value sel.":  lambda: DiffModel(lambda: scaled(LinearRegression(fit_intercept=False)), select_p=0.10),
    "Logistic Regression":              lambda: DiffModel(lambda: scaled(LogisticRegressionCV(Cs=10, cv=5, fit_intercept=False, max_iter=2000)), kind="proba"),
    "SVM (RBF)":                        lambda: DiffModel(lambda: scaled(SVC(kernel="rbf", C=1.0)), kind="decision", augment=True),
    "Random Forest":                    lambda: DiffModel(lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1), kind="proba", augment=True),
}
KIND = {k: v().kind for k, v in MODELS.items()}


def loso_cv(models, data):
    """Leave-one-season-out CV. Every data-dependent step happens inside the training fold."""
    out = []
    for season in sorted(data.season.unique()):
        tr, te = data[data.season != season], data[data.season == season]
        for name, make in models.items():
            m = make().fit(tr)
            out.append(pd.DataFrame({"model": name, "season": season, "game": te.index.values,
                                     "signal": m.predict(te), "margin": te["margin"].values}))
    res = pd.concat(out, ignore_index=True)
    res["correct"] = (res.signal > 0) == (res.margin > 0)
    res["conf"] = res.signal.abs()
    return res


def pickem_score(g):
    """Share of the maximum Pick'em points earned (least confident game = 1 point, most confident = N)."""
    order = g.conf.rank(method="first").astype(int)
    return (order * g.correct).sum() / (len(g) * (len(g) + 1) / 2)


def evaluate(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in cv.groupby("model", sort=False):
        k, n = int(g.correct.sum()), len(g)
        ci = stats.binomtest(k, n, 0.5).proportion_ci(method="wilson")
        per_season = g.groupby("season").correct.mean()
        rows.append({
            "model": name,
            "mean season acc": per_season.mean(),
            "pooled acc": k / n,
            "95% CI low": ci.low, "95% CI high": ci.high,
            "p-value vs 50%": stats.binomtest(k, n, 0.5).pvalue,
            "margin MAE": (g.signal - g.margin).abs().mean() if KIND[name] == "margin" else np.nan,
            "pick'em score": g.groupby("season").apply(pickem_score).mean(),
        })
    return pd.DataFrame(rows).set_index("model")
