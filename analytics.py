"""analytics.py — pure pandas statistics over the cached tables.

No Streamlit / Plotly imports: every function takes DataFrame(s) plus simple
filter arguments and returns a DataFrame, so the stat logic stays testable in
isolation. Definitional rules follow CLAUDE.md exactly.

So far this module implements the shared helpers and **season batting** stats.
More stat groups (bowling, fielding, partnerships, ...) come in later phases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

# Dismissal kinds that are NOT a batting dismissal (the batter remains "not
# out" for average / not-out purposes).
NOT_OUT_KINDS = {"retired hurt", "retired not out"}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def load_deliveries(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the cached deliveries table with stable dtypes."""
    return pd.read_csv(data_dir / "deliveries.csv")


def load_matches(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the cached matches table."""
    return pd.read_csv(data_dir / "matches.csv")


def load_reviews(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the cached DRS reviews side table."""
    return pd.read_csv(data_dir / "reviews.csv")


def load_replacements(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the cached substitutions side table."""
    return pd.read_csv(data_dir / "replacements.csv")


def _genuine_out_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that are a real dismissal (a wicket), excluding retirements."""
    return (
        df["player_out"].notna()
        & (df["player_out"] != "")
        & (~df["wicket_kind"].isin(NOT_OUT_KINDS))
    )


def core_innings(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Main two innings only — drops super-over balls (innings 3/4) so they
    don't distort batting/bowling aggregates (CLAUDE.md edge-case rule)."""
    return deliveries[deliveries["innings"] <= 2]


def filter_phase(deliveries: pd.DataFrame, phase: str | None) -> pd.DataFrame:
    """Restrict to a single phase ('Powerplay'/'Middle'/'Death'); ``None``
    returns the frame unchanged."""
    if phase is None:
        return deliveries
    return deliveries[deliveries["phase"] == phase]


def _rate(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    """Safe element-wise rate: 0/NaN denominators yield NaN, never inf.

    Rates are always reported beside their volume column, so a NaN here simply
    means "no sample" — never silently dropped (CLAUDE.md: no thresholds)."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom * scale


# --------------------------------------------------------------------------- #
# Batting
# --------------------------------------------------------------------------- #
def _batting_innings_count(df: pd.DataFrame) -> pd.Series:
    """Distinct (match, innings) a player was at the crease (as striker OR
    non-striker), so innings/not-outs count even a non-striker run-out."""
    striker = df[["match_id", "innings", "batter"]].rename(columns={"batter": "player"})
    non_striker = df[["match_id", "innings", "non_striker"]].rename(
        columns={"non_striker": "player"}
    )
    at_crease = pd.concat([striker, non_striker], ignore_index=True)
    at_crease = at_crease[at_crease["player"].notna() & (at_crease["player"] != "")]
    return at_crease.groupby("player")[["match_id", "innings"]].apply(
        lambda g: g.drop_duplicates().shape[0]
    )


def _batting_dismissals(df: pd.DataFrame) -> pd.Series:
    """Times each player was genuinely out (excludes retired hurt/not out)."""
    out = df[(df["player_out"].notna()) & (df["player_out"] != "")]
    out = out[~out["wicket_kind"].isin(NOT_OUT_KINDS)]
    return out.groupby("player_out").size()


def _highest_score(df: pd.DataFrame) -> pd.Series:
    """Best single-innings runs off the bat for each batter."""
    per_innings = df.groupby(["batter", "match_id", "innings"])["runs_batter"].sum()
    return per_innings.groupby("batter").max()


def _innings_totals(df: pd.DataFrame) -> pd.Series:
    """Runs off the bat per (batter, match, innings) — basis for milestones."""
    return df.groupby(["batter", "match_id", "innings"])["runs_batter"].sum()


def _milestones(df: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping innings buckets: thirties (30-49), fifties (50-99),
    hundreds (100+) — matching scorecard convention."""
    totals = _innings_totals(df)
    buckets = pd.DataFrame({
        "thirties": ((totals >= 30) & (totals < 50)),
        "fifties": ((totals >= 50) & (totals < 100)),
        "hundreds": (totals >= 100),
    })
    return buckets.groupby(level="batter").sum()


def _fastest_fifty_balls(df: pd.DataFrame) -> pd.Series:
    """Fewest legal balls a batter took to reach 50 in any innings (NaN if
    they never made a fifty)."""
    legal = df[df["legal_ball"]].sort_values(
        ["batter", "match_id", "innings", "over", "ball_in_over"]
    )
    grp = legal.groupby(["batter", "match_id", "innings"])
    cum_runs = grp["runs_batter"].cumsum()
    ball_no = grp.cumcount() + 1
    reached = ball_no[cum_runs >= 50]
    if reached.empty:
        return pd.Series(dtype="float64")
    keys = legal.loc[reached.index, ["batter", "match_id", "innings"]]
    per_innings = (
        keys.assign(ball_no=reached.values)
        .groupby(["batter", "match_id", "innings"])["ball_no"].min()
    )
    return per_innings.groupby("batter").min()


def _scoring_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Count of legal balls a batter played for 0/1/2/3 off the bat. (4s and
    6s are reported separately as the headline boundary columns.)"""
    legal = df[df["legal_ball"]]
    counts = (
        legal.assign(n=1)
        .pivot_table(index="batter", columns="runs_batter", values="n",
                     aggfunc="sum", fill_value=0)
    )
    out = pd.DataFrame(index=counts.index)
    for runs, name in [(0, "zeros"), (1, "ones"), (2, "twos"), (3, "threes")]:
        out[name] = counts[runs] if runs in counts.columns else 0
    return out.astype(int)


def _team_run_share(core: pd.DataFrame) -> pd.Series:
    """Each batter's runs off the bat as a % of their team's total runs
    scored (runs_total, extras included) across the season."""
    team_runs = core.groupby("batting_team")["runs_total"].sum()
    player_team = core.groupby("batter")["batting_team"].agg(
        lambda s: s.value_counts().index[0]
    )
    player_runs = core.groupby("batter")["runs_batter"].sum()
    share = player_runs / player_team.map(team_runs) * 100
    return share.round(2)


def batting_season(deliveries: pd.DataFrame, phase: str | None = None) -> pd.DataFrame:
    """Per-player season batting stats.

    Rules (CLAUDE.md): balls faced count **legal balls only**; average uses
    genuine dismissals; every rate ships beside its volume column and no
    low-sample players are filtered out.

    When ``phase`` is given, all counting stats are restricted to that phase,
    but innings/not-outs/highest-score remain whole-innings concepts and are
    only reported in the unfiltered (season) view.
    """
    df = filter_phase(core_innings(deliveries), phase)

    grouped = df.groupby("batter")
    runs = grouped["runs_batter"].sum()
    balls_faced = grouped["legal_ball"].sum()
    fours = grouped["is_four"].sum()
    sixes = grouped["is_six"].sum()
    dots = grouped["is_dot"].sum()

    stats = pd.DataFrame({
        "runs": runs,
        "balls_faced": balls_faced,
        "fours": fours,
        "sixes": sixes,
        "_dots": dots,
    })

    # Phase-level counting stats (split with the rest of the table by phase).
    stats = stats.join(_scoring_breakdown(df))
    for col in ["zeros", "ones", "twos", "threes"]:
        stats[col] = stats[col].fillna(0).astype(int)

    # Whole-innings / season concepts are computed on the unfiltered core frame.
    core = core_innings(deliveries)
    innings = _batting_innings_count(core).rename("innings")
    outs = _batting_dismissals(core).rename("_outs")
    high = _highest_score(core).rename("highest_score")

    stats = stats.join(innings).join(outs).join(high)
    stats = stats.join(_milestones(core))
    stats = stats.join(_fastest_fifty_balls(core).rename("fastest_fifty_balls"))
    stats = stats.join(_team_run_share(core).rename("team_run_share"))
    stats["innings"] = stats["innings"].fillna(0).astype(int)
    stats["_outs"] = stats["_outs"].fillna(0).astype(int)
    stats["not_outs"] = (stats["innings"] - stats["_outs"]).clip(lower=0)
    for col in ["thirties", "fifties", "hundreds"]:
        stats[col] = stats[col].fillna(0).astype(int)

    # Average must use dismissals scoped to the SAME frame as runs: phase-local
    # dismissals when a phase filter is active (season dismissals when not).
    # Mirrors batting_by_position's dismissal counting; _rate yields NaN — not a
    # divide-by-zero hybrid — when a player has no dismissals in the scope.
    avg_outs = _batting_dismissals(df).rename("_avg_outs")
    stats = stats.join(avg_outs)
    stats["_avg_outs"] = stats["_avg_outs"].fillna(0).astype(int)

    stats["strike_rate"] = _rate(stats["runs"], stats["balls_faced"], 100).round(2)
    stats["average"] = _rate(stats["runs"], stats["_avg_outs"]).round(2)
    stats["boundary_pct"] = _rate(
        stats["fours"] + stats["sixes"], stats["balls_faced"], 100
    ).round(2)
    stats["dot_pct"] = _rate(stats["_dots"], stats["balls_faced"], 100).round(2)
    stats["balls_per_boundary"] = _rate(
        stats["balls_faced"], stats["fours"] + stats["sixes"]
    ).round(2)

    stats = stats.reset_index().rename(columns={"batter": "player"})
    # Volume columns sit beside the rates they explain (no thresholds applied).
    columns = [
        "player", "innings", "not_outs", "runs", "balls_faced",
        "strike_rate", "average", "highest_score",
        "fours", "sixes", "boundary_pct", "dot_pct", "balls_per_boundary",
        "zeros", "ones", "twos", "threes",
        "thirties", "fifties", "hundreds",
        "fastest_fifty_balls", "team_run_share",
    ]
    stats["highest_score"] = stats["highest_score"].fillna(0).astype(int)
    return stats[columns].sort_values("runs", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Bowling
# --------------------------------------------------------------------------- #
# Credited dismissal kinds (must mirror parse.py's bowler_credited rule).
CREDITED_KINDS = [
    "bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket",
]


def _bowler_runs(df: pd.DataFrame) -> pd.Series:
    """Runs charged to the bowler per ball: runs off the bat + wides +
    no-balls. Byes, legbyes and penalty runs are NOT charged (CLAUDE.md)."""
    return df["runs_batter"] + df["wide"] + df["noball"]


def _overs_notation(balls: pd.Series) -> pd.Series:
    """Balls -> cricket overs notation as a float (e.g. 26 balls -> 4.2)."""
    return (balls // 6) + (balls % 6) / 10


def _bowler_innings_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """Per (bowler, match, innings): wickets credited, runs conceded, balls.

    Basis for best figures and 3/4/5-wicket hauls."""
    work = df.assign(
        _conceded=_bowler_runs(df),
        _wkt=(df["bowler_credited"] == df["bowler"]) & (df["bowler_credited"] != ""),
    )
    g = work.groupby(["bowler", "match_id", "innings"])
    return pd.DataFrame({
        "wkts": g["_wkt"].sum().astype(int),
        "runs": g["_conceded"].sum().astype(int),
        "balls": g["legal_ball"].sum().astype(int),
    }).reset_index()


def _best_figures(per_innings: pd.DataFrame) -> pd.Series:
    """Best bowling in an innings per bowler: most wickets, fewest runs as
    tie-break, formatted 'W/R'."""
    ranked = per_innings.sort_values(
        ["bowler", "wkts", "runs"], ascending=[True, False, True]
    )
    best = ranked.groupby("bowler").first()
    return (best["wkts"].astype(str) + "/" + best["runs"].astype(str)).rename(
        "best_figures"
    )


def _hauls(per_innings: pd.DataFrame) -> pd.DataFrame:
    """Counts of innings with 3+/4+/5+ wickets per bowler."""
    g = per_innings.groupby("bowler")["wkts"]
    return pd.DataFrame({
        "hauls_3w": g.apply(lambda s: (s >= 3).sum()),
        "hauls_4w": g.apply(lambda s: (s >= 4).sum()),
        "hauls_5w": g.apply(lambda s: (s >= 5).sum()),
    })


def _maidens(df: pd.DataFrame) -> pd.Series:
    """Maiden overs per bowler: a full over (6 legal balls) conceding 0 runs
    including extras — any extra (a wide, a bye, ...) breaks it."""
    g = df.groupby(["bowler", "match_id", "innings", "over"])
    over_summary = pd.DataFrame({
        "runs_total": g["runs_total"].sum(),
        "legal": g["legal_ball"].sum(),
    })
    is_maiden = (over_summary["runs_total"] == 0) & (over_summary["legal"] == 6)
    return is_maiden.groupby(level="bowler").sum().astype(int)


def _wickets_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bowler credited wickets split by dismissal kind."""
    credited = df[(df["bowler_credited"] == df["bowler"]) & (df["bowler_credited"] != "")]
    table = (
        credited.assign(n=1)
        .pivot_table(index="bowler", columns="wicket_kind", values="n",
                     aggfunc="sum", fill_value=0)
    )
    out = pd.DataFrame(index=table.index)
    for kind in CREDITED_KINDS:
        col = "w_" + kind.replace(" ", "_")
        out[col] = table[kind] if kind in table.columns else 0
    return out.astype(int)


def bowling_season(deliveries: pd.DataFrame, phase: str | None = None) -> pd.DataFrame:
    """Per-player season bowling stats.

    Rules (CLAUDE.md): overs/balls bowled count **legal balls only**; economy
    charges wides + no-balls but **excludes byes + legbyes**; wickets are
    bowler-credited only (run-outs / retirements excluded). No thresholds —
    every rate ships beside its volume column.

    When ``phase`` is given, counting stats and their rates are phase-scoped
    (balls, runs conceded, wickets, economy, average, strike rate, dot %,
    boundaries). Whole-innings figures — innings bowled, maidens, best figures
    and 3/4/5-wicket hauls — stay season-wide and do not change with phase.
    """
    core = core_innings(deliveries)
    core = core.assign(
        _conceded=_bowler_runs(core),
        _wkt=((core["bowler_credited"] == core["bowler"]) & (core["bowler_credited"] != "")),
    )
    df = filter_phase(core, phase)  # phase-scoped counting frame

    grouped = df.groupby("bowler")
    balls = grouped["legal_ball"].sum()
    runs_conceded = grouped["_conceded"].sum()
    wickets = grouped["_wkt"].sum().astype(int)
    fours = grouped["is_four"].sum()
    sixes = grouped["is_six"].sum()
    dots = grouped["is_dot"].sum()
    wides = grouped["wide"].sum()
    noballs = grouped["noball"].sum()
    # Whole-innings volume (does NOT change with phase), like batting_season.
    innings_bowled = (
        core[["bowler", "match_id", "innings"]]
        .drop_duplicates()
        .groupby("bowler")
        .size()
    )

    # Index is the phase-active bowlers (from df); whole-innings series are
    # left-joined onto it so season-only bowlers don't leak into a phase view.
    stats = pd.DataFrame({
        "balls_bowled": balls,
        "runs_conceded": runs_conceded,
        "wickets": wickets,
        "fours_conceded": fours,
        "sixes_conceded": sixes,
        "_dots": dots,
        "wides": wides,
        "noballs": noballs,
    })
    stats = stats.join(innings_bowled.rename("innings_bowled"))
    stats["innings_bowled"] = stats["innings_bowled"].fillna(0).astype(int)
    stats["overs"] = _overs_notation(stats["balls_bowled"]).round(1)
    stats["boundaries_conceded"] = stats["fours_conceded"] + stats["sixes_conceded"]
    stats["extras_conceded"] = stats["wides"] + stats["noballs"]

    # Whole-innings derived (do NOT change with phase): maidens, best figures,
    # hauls — computed on the full core frame, not the phase slice.
    per_innings = _bowler_innings_grouped(core)
    stats = stats.join(_maidens(core).rename("maidens"))
    stats = stats.join(_best_figures(per_innings))
    stats = stats.join(_hauls(per_innings))
    # Wickets-by-type stays phase-scoped (sums to the phase 'wickets' column).
    stats = stats.join(_wickets_by_type(df))
    stats["maidens"] = stats["maidens"].fillna(0).astype(int)
    for col in ["hauls_3w", "hauls_4w", "hauls_5w"]:
        stats[col] = stats[col].fillna(0).astype(int)
    for kind in CREDITED_KINDS:
        col = "w_" + kind.replace(" ", "_")
        stats[col] = stats[col].fillna(0).astype(int)
    stats["best_figures"] = stats["best_figures"].fillna("0/0")

    overs_decimal = stats["balls_bowled"] / 6
    stats["economy"] = _rate(stats["runs_conceded"], overs_decimal).round(2)
    stats["average"] = _rate(stats["runs_conceded"], stats["wickets"]).round(2)
    stats["strike_rate"] = _rate(stats["balls_bowled"], stats["wickets"]).round(2)
    stats["dot_pct"] = _rate(stats["_dots"], stats["balls_bowled"], 100).round(2)

    stats = stats.reset_index().rename(columns={"bowler": "player"})
    columns = [
        "player", "innings_bowled", "overs", "balls_bowled", "maidens",
        "runs_conceded", "wickets", "economy", "average", "strike_rate",
        "dot_pct", "best_figures", "hauls_3w", "hauls_4w", "hauls_5w",
        "fours_conceded", "sixes_conceded", "boundaries_conceded",
        "wides", "noballs", "extras_conceded",
    ] + ["w_" + k.replace(" ", "_") for k in CREDITED_KINDS]
    # Drop the synthetic empty-name bowler row if present.
    stats = stats[stats["player"] != ""]
    return stats[columns].sort_values(
        ["wickets", "economy"], ascending=[False, True]
    ).reset_index(drop=True)


def batting_by_position(deliveries: pd.DataFrame, phase: str | None = None) -> pd.DataFrame:
    """League-wide batting output grouped by batting position (1-11)."""
    df = filter_phase(core_innings(deliveries), phase)
    grouped = df.groupby("batting_position")
    runs = grouped["runs_batter"].sum()
    balls = grouped["legal_ball"].sum()
    out = pd.DataFrame({
        "runs": runs,
        "balls_faced": balls,
        "fours": grouped["is_four"].sum(),
        "sixes": grouped["is_six"].sum(),
    })
    dismissals = _batting_dismissals_by_position(df)
    out = out.join(dismissals.rename("dismissals"))
    out["dismissals"] = out["dismissals"].fillna(0).astype(int)
    out["strike_rate"] = _rate(out["runs"], out["balls_faced"], 100).round(2)
    out["average"] = _rate(out["runs"], out["dismissals"]).round(2)
    return out.reset_index().rename(columns={"batting_position": "position"})


def _batting_dismissals_by_position(df: pd.DataFrame) -> pd.Series:
    """Dismissals attributed to the dismissed batter's position."""
    out = df[(df["player_out"].notna()) & (df["player_out"] != "")]
    out = out[~out["wicket_kind"].isin(NOT_OUT_KINDS)]
    # position of the player who got out (batter on strike == player_out for
    # most kinds; for non-striker run-outs we map via the player's position)
    pos = df.groupby("batter")["batting_position"].first()
    return out.assign(pos=out["player_out"].map(pos)).groupby("pos").size()


def batting_phase_splits(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Per-player runs / balls / strike-rate in each phase, side by side."""
    core = core_innings(deliveries)
    frames = []
    for phase in ["Powerplay", "Middle", "Death"]:
        d = core[core["phase"] == phase].groupby("batter")
        part = pd.DataFrame({
            f"{phase.lower()}_runs": d["runs_batter"].sum(),
            f"{phase.lower()}_balls": d["legal_ball"].sum(),
        })
        part[f"{phase.lower()}_sr"] = _rate(
            part[f"{phase.lower()}_runs"], part[f"{phase.lower()}_balls"], 100
        ).round(2)
        frames.append(part)
    out = pd.concat(frames, axis=1)
    for col in out.columns:
        if col.endswith(("_runs", "_balls")):
            out[col] = out[col].fillna(0).astype(int)
    out = out.reset_index().rename(columns={"batter": "player"})
    return out.sort_values("powerplay_runs", ascending=False).reset_index(drop=True)


def bowling_phase_splits(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Per-player runs / balls / wickets / economy in each phase, side by side."""
    core = core_innings(deliveries)
    core = core.assign(
        _conceded=_bowler_runs(core),
        _wkt=((core["bowler_credited"] == core["bowler"]) & (core["bowler_credited"] != "")),
    )
    frames = []
    for phase in ["Powerplay", "Middle", "Death"]:
        d = core[core["phase"] == phase].groupby("bowler")
        p = phase.lower()
        part = pd.DataFrame({
            f"{p}_balls": d["legal_ball"].sum(),
            f"{p}_runs": d["_conceded"].sum(),
            f"{p}_wkts": d["_wkt"].sum(),
        })
        part[f"{p}_econ"] = _rate(part[f"{p}_runs"], part[f"{p}_balls"] / 6).round(2)
        frames.append(part)
    out = pd.concat(frames, axis=1)
    for col in out.columns:
        if col.endswith(("_runs", "_balls", "_wkts")):
            out[col] = out[col].fillna(0).astype(int)
    out = out.reset_index().rename(columns={"bowler": "player"})
    out = out[out["player"] != ""]
    return out.sort_values("death_wkts", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Fielding
# --------------------------------------------------------------------------- #
def fielding_season(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Per-fielder catches, caught-&-bowled, stumpings and run-outs effected.

    Built from the ``fielders`` array + dismissal ``kind``. A run-out can list
    multiple fielders; each is credited (run-outs *effected/involved*)."""
    core = core_innings(deliveries)
    w = core[core["wicket_kind"].notna() & (core["wicket_kind"] != "")].copy()
    w["fielder"] = w["fielders"].fillna("").str.split(";")
    w = w.explode("fielder")
    w["fielder"] = w["fielder"].str.strip()
    w = w[w["fielder"] != ""]

    def _by_kind(kind: str) -> pd.Series:
        return w[w["wicket_kind"] == kind].groupby("fielder").size()

    out = pd.DataFrame({
        "catches": _by_kind("caught"),
        "caught_and_bowled": _by_kind("caught and bowled"),
        "stumpings": _by_kind("stumped"),
        "run_outs": _by_kind("run out"),
    }).fillna(0).astype(int)
    out["total_dismissals"] = out.sum(axis=1)
    out = out.reset_index().rename(columns={"fielder": "player"})
    return out.sort_values(
        ["total_dismissals", "catches"], ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Partnerships
# --------------------------------------------------------------------------- #
def partnerships(deliveries: pd.DataFrame) -> pd.DataFrame:
    """One row per partnership (a maximal run of balls with the same pair at
    the crease). ``wicket_number`` is the partnership's order in the innings."""
    core = core_innings(deliveries).sort_values(
        ["match_id", "innings", "over", "ball_in_over"]
    ).copy()
    pair = pd.Series(
        list(zip(
            core["batter"].astype(str).where(core["batter"] <= core["non_striker"],
                                              core["non_striker"].astype(str)),
            core["non_striker"].astype(str).where(core["batter"] <= core["non_striker"],
                                                  core["batter"].astype(str)),
        )),
        index=core.index,
    )
    core["_pair"] = pair
    seq = core.groupby(["match_id", "innings"])["_pair"].transform(
        lambda s: s.ne(s.shift()).cumsum()
    )
    core["wicket_number"] = seq

    g = core.groupby(["match_id", "innings", "wicket_number"])
    out = g.agg(
        batting_team=("batting_team", "first"),
        runs=("runs_total", "sum"),
        balls=("legal_ball", "sum"),
        batter1=("_pair", lambda s: s.iloc[0][0]),
        batter2=("_pair", lambda s: s.iloc[0][1]),
    ).reset_index()
    out["run_rate"] = _rate(out["runs"], out["balls"] / 6).round(2)
    return out.sort_values("runs", ascending=False).reset_index(drop=True)


def best_partnership_by_wicket(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Highest stand recorded for each wicket number (1st, 2nd, ...)."""
    p = partnerships(deliveries)
    idx = p.groupby("wicket_number")["runs"].idxmax()
    return p.loc[idx].sort_values("wicket_number").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Team & innings
# --------------------------------------------------------------------------- #
def innings_totals(deliveries: pd.DataFrame) -> pd.DataFrame:
    """One row per team innings: total, wickets, overs, run rate, boundaries,
    extras and dot %."""
    core = core_innings(deliveries).copy()
    core["_out"] = _genuine_out_mask(core)
    g = core.groupby(["match_id", "innings", "batting_team", "bowling_team"])
    out = g.agg(
        runs=("runs_total", "sum"),
        balls=("legal_ball", "sum"),
        wickets=("_out", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        extras=("runs_extras", "sum"),
        dots=("is_dot", "sum"),
    ).reset_index()
    out["overs"] = _overs_notation(out["balls"]).round(1)
    out["run_rate"] = _rate(out["runs"], out["balls"] / 6).round(2)
    out["dot_pct"] = _rate(out["dots"], out["balls"], 100).round(2)
    return out


def team_summary(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Per-team batting aggregates, extras conceded, and bat-first vs chase wins."""
    it = innings_totals(deliveries)
    bat = it.groupby("batting_team").agg(
        innings=("runs", "size"),
        runs=("runs", "sum"),
        balls=("balls", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        dots=("dots", "sum"),
        highest_total=("runs", "max"),
        lowest_total=("runs", "min"),
    )
    bat["run_rate"] = _rate(bat["runs"], bat["balls"] / 6).round(2)
    bat["dot_pct"] = _rate(bat["dots"], bat["balls"], 100).round(2)
    bat = bat.join(it.groupby("bowling_team")["extras"].sum().rename("extras_conceded"))

    wins = matches[matches["result"] == "win"]
    wbf = wins[wins["winner"] == wins["bat_first_team"]].groupby("winner").size()
    wch = wins[wins["winner"] == wins["chasing_team"]].groupby("winner").size()
    bat = bat.join(wbf.rename("wins_batting_first")).join(wch.rename("wins_chasing"))
    for col in ["wins_batting_first", "wins_chasing", "extras_conceded"]:
        bat[col] = bat[col].fillna(0).astype(int)

    bat = bat.reset_index().rename(columns={"batting_team": "team"})
    cols = [
        "team", "innings", "runs", "run_rate", "fours", "sixes",
        "highest_total", "lowest_total", "dot_pct", "extras_conceded",
        "wins_batting_first", "wins_chasing",
    ]
    return bat[cols].sort_values("runs", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Season standings (points table)
# --------------------------------------------------------------------------- #
def standings(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """League points table: played, won, lost, tied, no-result, points, NRR.

    Only **league-stage** matches count (playoffs are excluded, identified by a
    present ``match_number``). Points: 2 per win, 1 each for a tie or no-result.
    NRR per CLAUDE.md — (runs scored / overs faced) - (runs conceded / overs
    bowled). A side bowled out counts as having faced the **full allotted
    quota** for that innings; that quota is read from the match data (revised
    ``target_overs`` for a reduced/DLS match, else the scheduled ``info.overs``)
    — never a hardcoded 20. Super-over innings (3/4) are already dropped by
    ``core_innings``; no-result matches are excluded from NRR entirely.

    D/L matches use the par-score rule: the side batting first is credited with
    (revised target - 1) runs over the revised target overs, the chasing side
    keeps its actual runs/overs.
    """
    matches = matches[matches["match_number"].notna()]
    league_ids = set(matches["match_id"])
    it = innings_totals(deliveries).copy()
    it = it[it["match_id"].isin(league_ids)]
    it = it.merge(
        matches[["match_id", "method", "target_runs", "target_overs",
                 "scheduled_overs"]],
        on="match_id", how="left",
    )
    it["method"] = it["method"].fillna("")
    is_dl = it["method"] == "D/L"

    # Per-innings allotted overs (the quota a bowled-out side is charged):
    # the revised target overs when the match was reduced/DLS, otherwise the
    # scheduled overs. Both innings of a reduced match share the revised
    # allotment. No literal 20 anywhere — the figure comes from the data.
    quota = it["target_overs"].fillna(it["scheduled_overs"]).astype(float)
    # Effective overs: full quota if bowled out, else actual (capped at quota).
    it["eff_overs"] = np.where(
        it["wickets"] >= 10, quota, np.minimum(it["balls"] / 6, quota)
    )
    # Runs used for NRR — actual, except the D/L par override below.
    it["runs_eff"] = it["runs"].astype(float)
    # D/L par rule: first innings credited with (target - 1) over target overs.
    dl_first = is_dl & (it["innings"] == 1)
    it.loc[dl_first, "runs_eff"] = it.loc[dl_first, "target_runs"] - 1
    it.loc[dl_first, "eff_overs"] = it.loc[dl_first, "target_overs"]

    no_result_ids = set(matches.loc[matches["result"] == "no result", "match_id"])
    it_nrr = it[~it["match_id"].isin(no_result_ids)]

    scored = it_nrr.groupby("batting_team").agg(
        runs_for=("runs_eff", "sum"), overs_for=("eff_overs", "sum")
    )
    conceded = it_nrr.groupby("bowling_team").agg(
        runs_against=("runs_eff", "sum"), overs_against=("eff_overs", "sum")
    )
    nrr = (
        scored["runs_for"] / scored["overs_for"]
        - conceded["runs_against"] / conceded["overs_against"]
    ).rename("nrr")

    teams = sorted(set(matches["team1"]) | set(matches["team2"]))
    rec = {t: dict(played=0, won=0, lost=0, tied=0, no_result=0) for t in teams}
    for _, m in matches.iterrows():
        t1, t2 = m["team1"], m["team2"]
        rec[t1]["played"] += 1
        rec[t2]["played"] += 1
        if m["result"] == "win":
            winner = m["winner"]
            loser = t2 if winner == t1 else t1
            rec[winner]["won"] += 1
            rec[loser]["lost"] += 1
        elif m["result"] == "tie":
            rec[t1]["tied"] += 1
            rec[t2]["tied"] += 1
        elif m["result"] == "no result":
            rec[t1]["no_result"] += 1
            rec[t2]["no_result"] += 1

    table = pd.DataFrame.from_dict(rec, orient="index")
    table["points"] = 2 * table["won"] + table["tied"] + table["no_result"]
    table = table.join(nrr)
    table["nrr"] = table["nrr"].round(3)
    table = table.reset_index().rename(columns={"index": "team"})
    return table.sort_values(
        ["points", "nrr"], ascending=[False, False]
    ).reset_index(drop=True)


def toss_match_win(matches: pd.DataFrame) -> pd.DataFrame:
    """Toss-win to match-win correlation, overall and split by toss decision.
    Only decided matches (a winner) are counted."""
    decided = matches[matches["result"] == "win"].copy()
    decided["toss_won_match"] = decided["toss_winner"] == decided["winner"]

    def row(label: str, df: pd.DataFrame) -> dict:
        n = len(df)
        w = int(df["toss_won_match"].sum())
        return {"segment": label, "matches": n, "toss_winner_won": w,
                "win_pct": round(w / n * 100, 1) if n else float("nan")}

    return pd.DataFrame([
        row("overall", decided),
        row("toss winner chose to bat", decided[decided["toss_decision"] == "bat"]),
        row("toss winner chose to field", decided[decided["toss_decision"] == "field"]),
    ])


def bat_chase_win_rates(matches: pd.DataFrame) -> pd.DataFrame:
    """How often the side batting first vs chasing won (decided matches only).

    Super-over wins are excluded: the main match was tied, so attributing the
    win to batting first or chasing would be meaningless.
    """
    decided = matches[(matches["result"] == "win")
                      & (~matches["super_over"].fillna(False).astype(bool))]
    n = len(decided)
    bf = int((decided["winner"] == decided["bat_first_team"]).sum())
    ch = int((decided["winner"] == decided["chasing_team"]).sum())
    return pd.DataFrame([
        {"outcome": "won batting first", "wins": bf,
         "win_pct": round(bf / n * 100, 1) if n else float("nan")},
        {"outcome": "won chasing", "wins": ch,
         "win_pct": round(ch / n * 100, 1) if n else float("nan")},
    ])


def margin_distribution(matches: pd.DataFrame) -> pd.DataFrame:
    """Distribution of victory margins, bucketed by runs and by wickets."""
    decided = matches[matches["result"] == "win"]
    runs = decided["win_by_runs"].dropna()
    wkts = decided["win_by_wickets"].dropna()

    run_labels = ["1-10", "11-25", "26-50", "51+"]
    run_counts = pd.cut(
        runs, bins=[0, 10, 25, 50, 10_000], labels=run_labels
    ).value_counts().reindex(run_labels, fill_value=0)
    wkt_labels = ["1-2", "3-4", "5-6", "7-8", "9-10"]
    wkt_counts = pd.cut(
        wkts, bins=[0, 2, 4, 6, 8, 10], labels=wkt_labels
    ).value_counts().reindex(wkt_labels, fill_value=0)

    rows = [{"margin_type": "by runs", "bucket": lab, "matches": int(c)}
            for lab, c in run_counts.items()]
    rows += [{"margin_type": "by wickets", "bucket": lab, "matches": int(c)}
             for lab, c in wkt_counts.items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Venue & toss
# --------------------------------------------------------------------------- #
def venue_summary(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Per venue: matches, average first-innings total, and defend vs chase
    win counts/rates."""
    it = innings_totals(deliveries)
    first = it[it["innings"] == 1].merge(
        matches[["match_id", "venue"]], on="match_id", how="left"
    )
    avg_first = first.groupby("venue")["runs"].mean().round(1).rename("avg_first_innings")

    decided = matches[matches["result"] == "win"]
    defend = decided[decided["winner"] == decided["bat_first_team"]].groupby("venue").size()
    chase = decided[decided["winner"] == decided["chasing_team"]].groupby("venue").size()
    played = matches.groupby("venue").size().rename("matches")

    out = pd.concat([
        played, avg_first,
        defend.rename("defend_wins"), chase.rename("chase_wins"),
    ], axis=1)
    out["defend_wins"] = out["defend_wins"].fillna(0).astype(int)
    out["chase_wins"] = out["chase_wins"].fillna(0).astype(int)
    decided_n = out["defend_wins"] + out["chase_wins"]
    out["chase_win_pct"] = _rate(out["chase_wins"], decided_n, 100).round(1)
    out = out.reset_index().rename(columns={"index": "venue"})
    return out.sort_values("matches", ascending=False).reset_index(drop=True)


def toss_decision_trends(matches: pd.DataFrame) -> pd.DataFrame:
    """How often the toss winner chose to bat vs field, per venue (+ overall)."""
    g = (
        matches.groupby(["venue", "toss_decision"]).size()
        .unstack(fill_value=0)
    )
    for col in ["bat", "field"]:
        if col not in g.columns:
            g[col] = 0
    g["matches"] = g["bat"] + g["field"]
    g["field_pct"] = (g["field"] / g["matches"] * 100).round(1)
    g = g.reset_index()
    overall = pd.DataFrame([{
        "venue": "ALL VENUES",
        "bat": int(g["bat"].sum()), "field": int(g["field"].sum()),
        "matches": int(g["matches"].sum()),
        "field_pct": round(g["field"].sum() / g["matches"].sum() * 100, 1),
    }])
    return pd.concat([overall, g.sort_values("matches", ascending=False)],
                     ignore_index=True)


# --------------------------------------------------------------------------- #
# Over-level & momentum
# --------------------------------------------------------------------------- #
def _over_table(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Per (match, innings, over): bowler, runs scored, wickets, legal balls."""
    core = core_innings(deliveries).copy()
    core["_out"] = _genuine_out_mask(core)
    g = core.groupby(["match_id", "innings", "over"])
    return g.agg(
        batting_team=("batting_team", "first"),
        bowling_team=("bowling_team", "first"),
        bowler=("bowler", "first"),
        runs=("runs_total", "sum"),
        wickets=("_out", "sum"),
        balls=("legal_ball", "sum"),
    ).reset_index()


def over_progression(deliveries: pd.DataFrame, match_id) -> pd.DataFrame:
    """Per-over runs, wickets and cumulative runs for both innings of a match —
    the data behind Manhattan (runs/over) and worm (cumulative) charts."""
    ot = _over_table(deliveries)
    ot = ot[ot["match_id"] == match_id].sort_values(["innings", "over"]).copy()
    ot["cumulative_runs"] = ot.groupby("innings")["runs"].cumsum()
    return ot[["innings", "batting_team", "over", "runs", "wickets",
               "cumulative_runs"]].reset_index(drop=True)


def wickets_by_over(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Season totals per over number (1-20): runs, wickets, run rate."""
    ot = _over_table(deliveries)
    g = ot.groupby("over").agg(
        runs=("runs", "sum"), wickets=("wickets", "sum"), balls=("balls", "sum")
    ).reset_index()
    g["over_number"] = g["over"] + 1  # human-friendly 1-20
    g["run_rate"] = _rate(g["runs"], g["balls"] / 6).round(2)
    return g[["over_number", "runs", "wickets", "balls", "run_rate"]]


def over_extremes(deliveries: pd.DataFrame, n: int = 10) -> dict[str, pd.DataFrame]:
    """Most expensive and most economical single overs (full 6-ball overs)."""
    ot = _over_table(deliveries)
    full = ot[ot["balls"] == 6].copy()
    full["over_number"] = full["over"] + 1
    cols = ["match_id", "innings", "over_number", "bowler", "bowling_team",
            "runs", "wickets"]
    return {
        "most_expensive": full.sort_values("runs", ascending=False).head(n)[cols]
            .reset_index(drop=True),
        "most_economical": full.sort_values(["runs", "wickets"],
                                            ascending=[True, False]).head(n)[cols]
            .reset_index(drop=True),
    }


# --------------------------------------------------------------------------- #
# Extras analysis
# --------------------------------------------------------------------------- #
EXTRA_TYPES = ["wide", "noball", "bye", "legbye", "penalty"]


def extras_summary(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Season totals for each extra type and their share of all runs."""
    core = core_innings(deliveries)
    total_runs = int(core["runs_total"].sum())
    rows = []
    for col in EXTRA_TYPES:
        s = int(core[col].sum())
        rows.append({"extra_type": col, "runs": s,
                     "pct_of_all_runs": round(s / total_runs * 100, 2)})
    total = sum(r["runs"] for r in rows)
    rows.append({"extra_type": "ALL EXTRAS", "runs": total,
                 "pct_of_all_runs": round(total / total_runs * 100, 2)})
    return pd.DataFrame(rows)


def extras_by_team(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Extras conceded by each team (while bowling), by type."""
    core = core_innings(deliveries)
    g = core.groupby("bowling_team")[EXTRA_TYPES].sum().astype(int)
    g["total_extras"] = g.sum(axis=1)
    g = g.reset_index().rename(columns={"bowling_team": "team"})
    return g.sort_values("total_extras", ascending=False).reset_index(drop=True)


def extras_by_bowler(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Bowler-attributable extras (wides + no-balls) leaked, per bowler."""
    core = core_innings(deliveries)
    g = core.groupby("bowler").agg(
        wides=("wide", "sum"), noballs=("noball", "sum")
    )
    g["wides_noballs"] = g["wides"] + g["noballs"]
    g = g.reset_index().rename(columns={"bowler": "player"})
    g = g[g["player"] != ""]
    return g.sort_values("wides_noballs", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Matchups (batter vs bowler)
# --------------------------------------------------------------------------- #
def head_to_head(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Per (batter, bowler): balls, runs, dismissals, SR, boundaries, dots."""
    core = core_innings(deliveries)
    g = core.groupby(["batter", "bowler"])
    out = pd.DataFrame({
        "balls": g["legal_ball"].sum(),
        "runs": g["runs_batter"].sum(),
        "fours": g["is_four"].sum(),
        "sixes": g["is_six"].sum(),
        "dots": g["is_dot"].sum(),
    })
    dmask = (
        (core["bowler_credited"] == core["bowler"])
        & (core["bowler_credited"] != "")
        & (core["player_out"] == core["batter"])
    )
    dismissals = core[dmask].groupby(["batter", "bowler"]).size().rename("dismissals")
    out = out.join(dismissals)
    out["dismissals"] = out["dismissals"].fillna(0).astype(int)
    out["strike_rate"] = _rate(out["runs"], out["balls"], 100).round(2)
    return out.reset_index().sort_values("balls", ascending=False).reset_index(drop=True)


def matchup(deliveries: pd.DataFrame, batter: str | None = None,
            bowler: str | None = None, bowling_team: str | None = None) -> pd.DataFrame:
    """One-row head-to-head summary for a filtered subset — supports batter vs a
    single bowler, or batter vs a team's whole attack (pass ``bowling_team``)."""
    core = core_innings(deliveries)
    if batter is not None:
        core = core[core["batter"] == batter]
    if bowler is not None:
        core = core[core["bowler"] == bowler]
    if bowling_team is not None:
        core = core[core["bowling_team"] == bowling_team]
    balls = int(core["legal_ball"].sum())
    runs = int(core["runs_batter"].sum())
    dmask = (
        (core["bowler_credited"] == core["bowler"])
        & (core["bowler_credited"] != "")
        & (core["player_out"] == core["batter"])
    )
    dismissals = int(dmask.sum())
    return pd.DataFrame([{
        "batter": batter or "(any)",
        "bowler": bowler or "(any)",
        "bowling_team": bowling_team or "(any)",
        "balls": balls, "runs": runs, "dismissals": dismissals,
        "fours": int(core["is_four"].sum()), "sixes": int(core["is_six"].sum()),
        "strike_rate": round(runs / balls * 100, 2) if balls else float("nan"),
    }])


# --------------------------------------------------------------------------- #
# Niche: DRS reviews, impact players, player-of-the-match
# --------------------------------------------------------------------------- #
def drs_summary(reviews: pd.DataFrame) -> pd.DataFrame:
    """Per team: DRS reviews taken and success rate (+ an overall row)."""
    g = reviews.groupby("reviewing_team").agg(
        reviews=("successful", "size"), successful=("successful", "sum")
    )
    g["successful"] = g["successful"].astype(int)
    g["success_pct"] = (g["successful"] / g["reviews"] * 100).round(1)
    g = g.reset_index().rename(columns={"reviewing_team": "team"})
    overall = pd.DataFrame([{
        "team": "OVERALL", "reviews": int(g["reviews"].sum()),
        "successful": int(g["successful"].sum()),
        "success_pct": round(g["successful"].sum() / g["reviews"].sum() * 100, 1),
    }])
    return pd.concat([overall, g.sort_values("reviews", ascending=False)],
                     ignore_index=True)


def impact_player_usage(replacements: pd.DataFrame) -> pd.DataFrame:
    """Impact-player substitutions per team, and the players most brought in."""
    imp = replacements[replacements["reason"] == "impact_player"]
    by_team = imp.groupby("team").size().rename("impact_subs")
    return by_team.reset_index().sort_values("impact_subs", ascending=False).reset_index(drop=True)


def pom_tally(matches: pd.DataFrame) -> pd.DataFrame:
    """Player-of-the-match awards across the season."""
    poms = matches["player_of_match"].dropna()
    poms = poms[poms != ""]
    exploded = poms.str.split(";").explode().str.strip()
    exploded = exploded[exploded != ""]
    return (
        exploded.value_counts().rename_axis("player").reset_index(name="awards")
    )


if __name__ == "__main__":
    deliveries = load_deliveries()
    matches = load_matches()
    reviews = load_reviews()
    replacements = load_replacements()
    core = core_innings(deliveries)

    def check(label, ok):
        print(f"  [{'OK ' if ok else 'XX '}] {label}")

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("=== Venue summary (top 5 by matches) ===\n")
        vs = venue_summary(matches, deliveries)
        print(vs.head(5).to_string(index=False))
        check("venue 'matches' sums to 74", int(vs["matches"].sum()) == len(matches))

        print("\n=== Toss-decision trends (overall + top venues) ===\n")
        print(toss_decision_trends(matches).head(4).to_string(index=False))

        print("\n=== Wickets by over (season) ===\n")
        wbo = wickets_by_over(deliveries)
        print(wbo.to_string(index=False))
        tot_w = int(core.pipe(lambda d: _genuine_out_mask(d).sum()))
        check(f"wickets-by-over sum ({int(wbo['wickets'].sum())}) == total wickets ({tot_w})",
              int(wbo["wickets"].sum()) == tot_w)

        print("\n=== Most expensive overs (top 5) ===\n")
        print(over_extremes(deliveries, 5)["most_expensive"].to_string(index=False))

        print("\n=== Extras summary (season) ===\n")
        es = extras_summary(deliveries)
        print(es.to_string(index=False))
        ext_total = int(es[es["extra_type"] == "ALL EXTRAS"]["runs"].iloc[0])
        check(f"extras total ({ext_total}) == sum of runs_extras ({int(core['runs_extras'].sum())})",
              ext_total == int(core["runs_extras"].sum()))

        print("\n=== Matchup example: V Kohli vs all bowlers (top 3 by balls) ===\n")
        h2h = head_to_head(deliveries)
        print(h2h[h2h["batter"] == "V Kohli"].head(3).to_string(index=False))
        check(f"head-to-head total balls ({int(h2h['balls'].sum())}) == legal balls ({int(core['legal_ball'].sum())})",
              int(h2h["balls"].sum()) == int(core["legal_ball"].sum()))
        check(f"head-to-head total runs ({int(h2h['runs'].sum())}) == batter runs ({int(core['runs_batter'].sum())})",
              int(h2h["runs"].sum()) == int(core["runs_batter"].sum()))

        print("\n=== DRS summary (overall + top 3 teams) ===\n")
        drs = drs_summary(reviews)
        print(drs.head(4).to_string(index=False))
        check(f"DRS reviews ({int(drs[drs['team']=='OVERALL']['reviews'].iloc[0])}) == rows in reviews.csv ({len(reviews)})",
              int(drs[drs["team"] == "OVERALL"]["reviews"].iloc[0]) == len(reviews))

        print("\n=== Impact-player usage (top 5) ===\n")
        print(impact_player_usage(replacements).head(5).to_string(index=False))

        print("\n=== Player-of-the-match tally (top 5) ===\n")
        pom = pom_tally(matches)
        print(pom.head(5).to_string(index=False))
        n_pom = int((matches["player_of_match"].fillna("") != "").sum())
        check(f"POM awards sum ({int(pom['awards'].sum())}) == matches with a POM ({n_pom})",
              int(pom["awards"].sum()) == n_pom)
