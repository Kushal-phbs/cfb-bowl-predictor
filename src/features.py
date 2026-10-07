"""Cleaning, team-season feature engineering and bowl-matchup dataset.  (Notebook 01)"""
import numpy as np
import pandas as pd

from .config import DATA_RAW, FIRST_STATS_SEASON, MIN_GAMES

STAT_COLS = ["first_downs", "third_down_comp", "third_down_att", "pass_comp", "pass_att", "pass_yards",
             "rush_att", "rush_yards", "total_yards", "fum", "int", "pen_num", "pen_yards"]

FEATS = ["win_pct", "ppg", "papg", "margin_avg", "pts_std", "margin_std", "ypg", "ypg_allowed",
         "pass_ypg", "pass_ypg_allowed", "rush_ypg", "rush_ypg_allowed", "yards_std", "ypc", "ypc_allowed",
         "ypa", "ypa_allowed", "comp_pct", "comp_pct_allowed", "first_downs_pg", "first_downs_allowed_pg",
         "third_pct", "third_pct_allowed", "giveaways_pg", "takeaways_pg", "to_margin_pg", "pen_yds_pg", "sos"]
D_COLS = ["d_" + c for c in FEATS]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW, parse_dates=["date"])


def clean(raw: pd.DataFrame, verbose=True):
    """Keep seasons with stats, blank stats of games with extreme yardage inconsistencies.
    Returns (regular_season_games, bowl_games)."""
    df = raw[raw.season >= FIRST_STATS_SEASON].copy()
    for c in ("away", "home"):
        df[c] = df[c].astype(str).str.strip()

    # pass + rush yards exceed total yards by ~13 yds on average (likely sack yardage): systematic, not corruption.
    # We only flag extreme gaps (> 100 yds) as suspicious and blank their stats.
    bad = pd.Series(False, index=df.index)
    for side in ("away", "home"):
        diff = (df[f"pass_yards_{side}"] + df[f"rush_yards_{side}"] - df[f"total_yards_{side}"]).abs()
        bad |= diff.fillna(0) > 100
    for s in STAT_COLS:
        for side in ("away", "home"):
            df.loc[bad, f"{s}_{side}"] = np.nan      # keep the score, drop untrustworthy stats

    reg = df[df.game_type == "regular"].copy()
    post = df[df.game_type == "post"].copy()
    if verbose:
        print(f"Games with an extreme yardage inconsistency: {int(bad.sum())}")
        print("Duplicate games:", int(df.duplicated(["date", "away", "home"]).sum()))
        print("Tied games:", int((df.score_away == df.score_home).sum()))
        print(f"Regular-season games: {len(reg)} | Bowl (post) games: {len(post)}")
    return reg, post


def to_team_games(g: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with own stats, opponent stats ('_opp') and score info."""
    parts = []
    for me, opp, is_home in (("away", "home", 0), ("home", "away", 1)):
        t = pd.DataFrame({"game_id": g.index, "season": g.season.values, "team": g[me].values,
                          "opp": g[opp].values, "is_home": is_home,
                          "pts": g[f"score_{me}"].values, "pts_allowed": g[f"score_{opp}"].values})
        for s in STAT_COLS:
            t[s] = g[f"{s}_{me}"].values
            t[s + "_opp"] = g[f"{s}_{opp}"].values
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out["win"] = (out.pts > out.pts_allowed).astype(int)
    out["margin"] = out.pts - out.pts_allowed
    return out


def build_team_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team-games into one row per (season, team). Rates are ratios of sums (not means of ratios)."""
    grp = tg.groupby(["season", "team"])
    s = grp[[c for c in tg.columns if c in STAT_COLS or c.endswith("_opp")] + ["pts", "pts_allowed", "win"]].sum()
    n = grp.size()
    ns = grp["total_yards"].count()          # games that have stats
    f = pd.DataFrame(index=s.index)
    f["games"], f["games_stats"] = n, ns
    # results / scoring
    f["win_pct"] = s.win / n
    f["ppg"] = s.pts / n
    f["papg"] = s.pts_allowed / n
    f["margin_avg"] = f.ppg - f.papg
    f["pts_std"] = grp["pts"].std()                         # consistency (variance-style feature)
    f["margin_std"] = grp["margin"].std()
    # yardage
    f["ypg"], f["ypg_allowed"] = s.total_yards / ns, s.total_yards_opp / ns
    f["pass_ypg"], f["pass_ypg_allowed"] = s.pass_yards / ns, s.pass_yards_opp / ns
    f["rush_ypg"], f["rush_ypg_allowed"] = s.rush_yards / ns, s.rush_yards_opp / ns
    f["yards_std"] = grp["total_yards"].std()
    # efficiency
    f["ypc"], f["ypc_allowed"] = s.rush_yards / s.rush_att, s.rush_yards_opp / s.rush_att_opp
    f["ypa"], f["ypa_allowed"] = s.pass_yards / s.pass_att, s.pass_yards_opp / s.pass_att_opp
    f["comp_pct"], f["comp_pct_allowed"] = s.pass_comp / s.pass_att, s.pass_comp_opp / s.pass_att_opp
    f["first_downs_pg"], f["first_downs_allowed_pg"] = s.first_downs / ns, s.first_downs_opp / ns
    f["third_pct"] = s.third_down_comp / s.third_down_att
    f["third_pct_allowed"] = s.third_down_comp_opp / s.third_down_att_opp
    # ball security (fum = fumbles, int = interceptions thrown; used as a proxy for turnovers)
    f["giveaways_pg"] = (s.fum + s.int) / ns
    f["takeaways_pg"] = (s.fum_opp + s.int_opp) / ns
    f["to_margin_pg"] = f.takeaways_pg - f.giveaways_pg
    f["pen_yds_pg"] = s.pen_yards / ns
    # strength of schedule = mean win % of opponents faced
    wp = f["win_pct"].rename("opp_win_pct").reset_index().rename(columns={"team": "opp"})
    tmp = tg[["season", "team", "opp"]].merge(wp, on=["season", "opp"], how="left")
    f["sos"] = tmp.groupby(["season", "team"])["opp_win_pct"].mean()
    return f


def lookup(team_feats: pd.DataFrame, teams, seasons) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays([seasons.values, teams.values], names=["season", "team"])
    return team_feats.reindex(idx)


def build_bowl_matchups(post: pd.DataFrame, team_feats: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """One row per bowl game: team A = listed home team, B = the other. Differences d_* = f(A) - f(B)."""
    bowl = post.copy()
    bowl["team_a"], bowl["team_b"] = bowl["home"], bowl["away"]
    bowl["margin"] = bowl["score_home"] - bowl["score_away"]
    bowl["a_won"] = (bowl["margin"] > 0).astype(int)

    FA, FB = lookup(team_feats, bowl.team_a, bowl.season), lookup(team_feats, bowl.team_b, bowl.season)
    valid = (FA["games_stats"].values >= MIN_GAMES) & (FB["games_stats"].values >= MIN_GAMES)
    if verbose:
        print(f"Bowl games: {len(bowl)} | dropped (a team has < {MIN_GAMES} regular games with stats): {int((~valid).sum())}")

    wide = pd.concat([
        FA[FEATS].add_prefix("a_").set_axis(bowl.index),
        FB[FEATS].add_prefix("b_").set_axis(bowl.index),
        pd.DataFrame(FA[FEATS].values - FB[FEATS].values, columns=D_COLS, index=bowl.index),
    ], axis=1)
    bowls = pd.concat([bowl[["season", "date", "team_a", "team_b", "score_home", "score_away", "margin", "a_won"]], wide], axis=1)
    bowls = bowls[valid].dropna(subset=D_COLS).sort_values("date").reset_index(drop=True)
    if verbose:
        print("Final modelling set:", bowls.shape, "| games per season:")
        print(bowls.groupby("season").size().to_dict())
        print(f"Listed-home-team win rate: {bowls.a_won.mean():.3f}")
    return bowls
