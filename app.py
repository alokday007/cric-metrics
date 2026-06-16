"""app.py — Cric Metrics dashboard (Streamlit + Plotly).

Reads the cached CSVs produced by parse.py and renders descriptive IPL 2026
statistics. This module owns all UI/Plotly concerns; the stat logic lives in
analytics.py (kept import-free of Streamlit/Plotly so it stays testable).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A

DATA_DIR = Path("data")
PLOTLY_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(0,0,0,0)"

st.set_page_config(page_title="Cric Metrics", page_icon="🏏", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading (cached — the app never re-parses JSON)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data():
    deliveries = A.load_deliveries(DATA_DIR)
    matches = A.load_matches(DATA_DIR)
    reviews = A.load_reviews(DATA_DIR)
    replacements = A.load_replacements(DATA_DIR)
    return deliveries, matches, reviews, replacements


def _require_data() -> None:
    if not (DATA_DIR / "deliveries.csv").exists():
        st.error(
            "Cached data not found. Run **`python parse.py`** once to build "
            "`data/deliveries.csv` and `data/matches.csv`, then reload."
        )
        st.stop()


def style_fig(fig):
    """Common dark styling for every Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def legend_inside(fig):
    """Anchor the legend inside the plot's top-right so it never collides with
    a left-aligned title (the standard fix for stacked/multi-series charts)."""
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=0.99,
                    xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.35)"),
    )
    return fig


# --------------------------------------------------------------------------- #
# Batting tab
# --------------------------------------------------------------------------- #
def render_batting(deliveries: pd.DataFrame) -> None:
    st.subheader("Batting")
    st.caption(
        "Every player is shown — no qualification thresholds. Each rate sits "
        "beside its volume column so you can judge sample size yourself."
    )

    teams = sorted(deliveries["batting_team"].dropna().unique())
    c1, c2, c3 = st.columns([3, 1.2, 1])
    picked = c1.multiselect("Teams", teams, default=[],
                            help="Leave empty to show all teams.",
                            key="batting_teams")
    phase = c2.selectbox("Phase", ["All", "Powerplay", "Middle", "Death"],
                         key="batting_phase")
    top_n = c3.slider("Chart: top N", 5, 25, 10, key="batting_topn")

    sel = deliveries if not picked else deliveries[deliveries["batting_team"].isin(picked)]
    phase_arg = None if phase == "All" else phase
    bat = A.batting_season(sel, phase=phase_arg)

    if phase_arg:
        st.caption(
            f"Showing **{phase}** counting stats. Innings, not-outs, highest "
            "score, milestones, fastest-fifty and team-run-share remain "
            "whole-innings season figures."
        )

    if bat.empty:
        st.info("No deliveries match the current filters.")
        return

    top = bat.head(top_n)
    g1, g2 = st.columns(2)

    fig_runs = px.bar(
        top.sort_values("runs"), x="runs", y="player", orientation="h",
        title=f"Top {len(top)} run-scorers", text="runs",
        color="strike_rate", color_continuous_scale="Tealgrn",
        labels={"strike_rate": "SR"},
    )
    fig_runs.update_traces(textposition="outside", cliponaxis=False)
    # Headroom so the longest bar's value label (e.g. 776) isn't clipped.
    fig_runs.update_xaxes(range=[0, float(top["runs"].max()) * 1.15])
    g1.plotly_chart(style_fig(fig_runs), width="stretch")

    fig_sr = px.scatter(
        bat, x="balls_faced", y="strike_rate", size="runs", color="average",
        hover_name="player", title="Strike rate vs balls faced (size = runs)",
        color_continuous_scale="Tealgrn",
        labels={"balls_faced": "Balls faced", "strike_rate": "Strike rate"},
    )
    g2.plotly_chart(style_fig(fig_sr), width="stretch")

    fig_bnd = px.bar(
        top.sort_values("runs"), y="player", x=["fours", "sixes"],
        orientation="h", title=f"Boundaries — top {len(top)} run-scorers",
        labels={"value": "count", "variable": ""},
        color_discrete_map={"fours": "#38bdf8", "sixes": "#22c55e"},
    )
    fig_bnd = style_fig(fig_bnd)
    # Anchor the legend inside the top-right so it clears the (left-aligned) title.
    fig_bnd.update_layout(
        legend=dict(orientation="h", yanchor="top", y=0.99,
                    xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.35)"),
    )
    st.plotly_chart(fig_bnd, width="stretch")

    st.markdown("##### Season batting table")
    st.dataframe(bat, width="stretch", hide_index=True)

    with st.expander("Batting by position (selected teams & phase)"):
        st.caption("Reflects the current Teams and Phase filters selected above.")
        st.dataframe(A.batting_by_position(sel, phase=phase_arg),
                     width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Bowling tab
# --------------------------------------------------------------------------- #
def render_bowling(deliveries: pd.DataFrame) -> None:
    st.subheader("Bowling")
    st.caption(
        "Every bowler is shown — no qualification thresholds. Each rate sits "
        "beside its volume column so you can judge sample size yourself."
    )

    teams = sorted(deliveries["bowling_team"].dropna().unique())
    c1, c2, c3 = st.columns([3, 1.2, 1])
    picked = c1.multiselect("Teams", teams, default=[],
                            help="Leave empty to show all teams.",
                            key="bowling_teams")
    phase = c2.selectbox("Phase", ["All", "Powerplay", "Middle", "Death"],
                         key="bowling_phase")
    top_n = c3.slider("Chart: top N", 5, 25, 10, key="bowling_topn")

    sel = deliveries if not picked else deliveries[deliveries["bowling_team"].isin(picked)]
    phase_arg = None if phase == "All" else phase
    bowl = A.bowling_season(sel, phase=phase_arg)

    if phase_arg:
        st.caption(
            f"Showing **{phase}** counting stats (wickets, overs/balls, runs "
            "conceded, economy, average, strike rate, dot %, boundaries "
            "conceded). Best figures, maidens, 3/4/5-wicket hauls and innings "
            "bowled remain whole-innings season figures."
        )

    if bowl.empty:
        st.info("No deliveries match the current filters.")
        return

    top = bowl.head(top_n)  # already sorted by wickets desc, economy asc
    g1, g2 = st.columns(2)

    fig_wkts = px.bar(
        top.sort_values("wickets"), x="wickets", y="player", orientation="h",
        title=f"Top {len(top)} wicket-takers", text="wickets",
        color="economy", color_continuous_scale="Tealgrn",
        labels={"economy": "Econ"},
    )
    fig_wkts.update_traces(textposition="outside", cliponaxis=False)
    fig_wkts.update_xaxes(range=[0, float(top["wickets"].max()) * 1.15])
    g1.plotly_chart(style_fig(fig_wkts), width="stretch")

    # Size = wickets, but floor the bubble so 0-wicket bowlers still render;
    # hover shows the true wicket count. Sample size is the overs (x) axis.
    scat = bowl.assign(bubble=bowl["wickets"].clip(lower=1))
    fig_econ = px.scatter(
        scat, x="overs", y="economy", size="bubble", color="average",
        hover_name="player",
        hover_data={"bubble": False, "wickets": True, "average": True},
        title="Economy vs overs bowled (size = wickets)",
        color_continuous_scale="Tealgrn",
        labels={"overs": "Overs bowled", "economy": "Economy"},
    )
    g2.plotly_chart(style_fig(fig_econ), width="stretch")

    fig_bnd = px.bar(
        top.sort_values("wickets"), y="player",
        x=["fours_conceded", "sixes_conceded"], orientation="h",
        title=f"Boundaries conceded — top {len(top)} wicket-takers",
        labels={"value": "count", "variable": ""},
        color_discrete_map={"fours_conceded": "#38bdf8", "sixes_conceded": "#22c55e"},
    )
    fig_bnd = style_fig(fig_bnd)
    fig_bnd.update_layout(
        legend=dict(orientation="h", yanchor="top", y=0.99,
                    xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.35)"),
    )
    st.plotly_chart(fig_bnd, width="stretch")

    st.markdown("##### Season bowling table")
    st.dataframe(bowl, width="stretch", hide_index=True)

    with st.expander("Bowling phase splits (selected teams)"):
        st.caption(
            "Reflects the Teams filter above. Shows all three phases side by "
            "side regardless of the Phase selector (it is itself a phase "
            "breakdown)."
        )
        st.dataframe(A.bowling_phase_splits(sel),
                     width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Teams & Standings tab
# --------------------------------------------------------------------------- #
def render_standings(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    st.subheader("Teams & Standings")
    st.caption(
        "League-stage points table — always all teams, phase-agnostic. "
        "Points: 2 per win, 1 per tie/no-result, 0 per loss. NRR charges a "
        "bowled-out side its full allotted overs (read from the match data, not "
        "assumed 20); super-over and no-result matches are excluded."
    )

    table = A.standings(matches, deliveries).copy()
    table.insert(0, "pos", range(1, len(table) + 1))

    highlight = st.selectbox("Highlight a team", ["(none)"] + list(table["team"]),
                             key="standings_highlight")

    disp = table.rename(columns={
        "pos": "#", "team": "Team", "played": "P", "won": "W", "lost": "L",
        "tied": "T", "no_result": "NR", "points": "Pts", "nrr": "NRR",
    })

    def _hl_row(row):
        on = highlight != "(none)" and row["Team"] == highlight
        return ["background-color: #14532d" if on else "" for _ in row]

    styler = disp.style.apply(_hl_row, axis=1).format({"NRR": "{:+.3f}"})
    st.dataframe(styler, width="stretch", hide_index=True)

    # NRR-by-team bar — signed (green positive, red negative), no legend so the
    # title can't collide; the highlighted team is recoloured.
    bar = table.sort_values("nrr")
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in bar["nrr"]]
    if highlight != "(none)":
        colors = [("#eab308" if t == highlight else c)
                  for t, c in zip(bar["team"], colors)]
    fig = go.Figure(go.Bar(
        x=bar["nrr"], y=bar["team"], orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in bar["nrr"]], textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(title="Net run rate by team (green + / red −)")
    span = float(max(abs(bar["nrr"].min()), abs(bar["nrr"].max()))) * 1.25
    fig.update_xaxes(range=[-span, span], zeroline=True, zerolinecolor="#888")
    st.plotly_chart(style_fig(fig), width="stretch")

    if highlight != "(none)":
        with st.expander(f"Match-by-match — {highlight} (league stage)"):
            league = matches[matches["match_number"].notna()].copy()
            mt = league[(league["team1"] == highlight) | (league["team2"] == highlight)].copy()

            def _opp(r):
                return r["team2"] if r["team1"] == highlight else r["team1"]

            def _res(r):
                if r["result"] == "tie":
                    return "Tie"
                if r["result"] == "no result":
                    return "No result"
                return "Won" if r["winner"] == highlight else "Lost"

            def _margin(r):
                if pd.notna(r["win_by_runs"]):
                    return f"by {int(r['win_by_runs'])} runs"
                if pd.notna(r["win_by_wickets"]):
                    return f"by {int(r['win_by_wickets'])} wkts"
                return ""

            mt["Opponent"] = mt.apply(_opp, axis=1)
            mt["Result"] = mt.apply(_res, axis=1)
            mt["Margin"] = mt.apply(_margin, axis=1)
            show = mt.sort_values("date")[["date", "Opponent", "Result", "Margin", "venue"]]
            show = show.rename(columns={"date": "Date", "venue": "Venue"})
            st.dataframe(show, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Fielding tab
# --------------------------------------------------------------------------- #
def render_fielding(deliveries: pd.DataFrame) -> None:
    st.subheader("Fielding")
    st.caption(
        "Catches, caught-&-bowled, stumpings and run-outs effected (from the "
        "fielders array). Every fielder shown — no thresholds. A run-out can "
        "credit more than one fielder."
    )

    teams = sorted(deliveries["bowling_team"].dropna().unique())
    c1, c2 = st.columns([3, 1])
    picked = c1.multiselect("Teams (fielding side)", teams, default=[],
                            help="Leave empty to show all teams.",
                            key="fielding_teams")
    top_n = c2.slider("Chart: top N", 5, 25, 10, key="fielding_topn")

    sel = deliveries if not picked else deliveries[deliveries["bowling_team"].isin(picked)]
    fld = A.fielding_season(sel)
    if fld.empty:
        st.info("No dismissals match the current filters.")
        return

    top = fld.head(top_n)
    fig = px.bar(
        top.sort_values("total_dismissals"), y="player",
        x=["catches", "caught_and_bowled", "stumpings", "run_outs"],
        orientation="h", title=f"Top {len(top)} fielders — dismissals by type",
        labels={"value": "dismissals", "variable": ""},
    )
    st.plotly_chart(legend_inside(style_fig(fig)), width="stretch")

    st.markdown("##### Season fielding table")
    st.dataframe(fld, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Phases & Momentum tab
# --------------------------------------------------------------------------- #
def render_phases(deliveries: pd.DataFrame, matches: pd.DataFrame) -> None:
    st.subheader("Phases & Momentum")
    st.caption(
        "Match momentum (Manhattan & worm) for any game, plus league-wide "
        "over-by-over trends and phase splits. The over-level charts and phase "
        "tables are league-wide."
    )

    opts = {}
    for _, r in matches.sort_values("date").iterrows():
        opts[f"{r['date']} — {r['team1']} vs {r['team2']}"] = r["match_id"]
    pick = st.selectbox("Match (for Manhattan & worm)", list(opts),
                        key="phases_match")
    prog = A.over_progression(deliveries, opts[pick]).copy()
    prog["over_number"] = prog["over"] + 1

    g1, g2 = st.columns(2)
    fig_man = px.bar(
        prog, x="over_number", y="runs", color="batting_team", barmode="group",
        title="Manhattan — runs per over",
        labels={"over_number": "Over", "runs": "Runs", "batting_team": ""},
    )
    g1.plotly_chart(legend_inside(style_fig(fig_man)), width="stretch")
    fig_worm = px.line(
        prog, x="over_number", y="cumulative_runs", color="batting_team",
        markers=True, title="Worm — cumulative runs",
        labels={"over_number": "Over", "cumulative_runs": "Runs", "batting_team": ""},
    )
    g2.plotly_chart(legend_inside(style_fig(fig_worm)), width="stretch")

    wbo = A.wickets_by_over(deliveries)
    h1, h2 = st.columns(2)
    fig_rr = px.bar(
        wbo, x="over_number", y="run_rate", title="Run rate by over (league)",
        labels={"over_number": "Over", "run_rate": "Run rate"},
    )
    h1.plotly_chart(style_fig(fig_rr), width="stretch")
    fig_wk = px.bar(
        wbo, x="over_number", y="wickets", title="Wickets by over (league)",
        color="wickets", color_continuous_scale="Tealgrn",
        labels={"over_number": "Over", "wickets": "Wickets"},
    )
    h2.plotly_chart(style_fig(fig_wk), width="stretch")

    ext = A.over_extremes(deliveries, 10)
    e1, e2 = st.columns(2)
    e1.markdown("##### Most expensive overs")
    e1.dataframe(ext["most_expensive"], width="stretch", hide_index=True)
    e2.markdown("##### Most economical overs (full 6-ball)")
    e2.dataframe(ext["most_economical"], width="stretch", hide_index=True)

    with st.expander("Batting phase splits (league-wide)"):
        st.dataframe(A.batting_phase_splits(deliveries), width="stretch", hide_index=True)
    with st.expander("Bowling phase splits (league-wide)"):
        st.dataframe(A.bowling_phase_splits(deliveries), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Matchups tab
# --------------------------------------------------------------------------- #
def render_matchups(deliveries: pd.DataFrame) -> None:
    st.subheader("Matchups")
    st.caption(
        "Head-to-head batter vs bowler, or a batter vs a team's whole attack. "
        "Legal balls faced; dismissals credited to the bowler. No thresholds."
    )

    batters = sorted(deliveries["batter"].dropna().unique())
    c1, c2, c3 = st.columns(3)
    batter = c1.selectbox("Batter", batters, key="matchups_batter")

    h2h = A.head_to_head(deliveries)
    faced = h2h[h2h["batter"] == batter]
    bowler = c2.selectbox("Bowler", ["(all bowlers)"] + sorted(faced["bowler"].unique()),
                          key="matchups_bowler")
    teams = ["(all teams)"] + sorted(deliveries["bowling_team"].dropna().unique())
    team = c3.selectbox("vs Team attack", teams, key="matchups_team")

    summ = A.matchup(
        deliveries, batter=batter,
        bowler=None if bowler == "(all bowlers)" else bowler,
        bowling_team=None if team == "(all teams)" else team,
    ).iloc[0]

    m = st.columns(5)
    m[0].metric("Balls", int(summ["balls"]))
    m[1].metric("Runs", int(summ["runs"]))
    m[2].metric("Dismissals", int(summ["dismissals"]))
    m[3].metric("Strike rate",
                "—" if pd.isna(summ["strike_rate"]) else f"{summ['strike_rate']:.1f}")
    m[4].metric("4s / 6s", f"{int(summ['fours'])} / {int(summ['sixes'])}")

    topf = faced.sort_values("balls", ascending=False).head(12)
    if not topf.empty:
        fig = px.bar(
            topf.sort_values("balls"), y="bowler", x="runs", orientation="h",
            title=f"{batter} — runs vs most-faced bowlers", text="runs",
            color="strike_rate", color_continuous_scale="Tealgrn",
            labels={"strike_rate": "SR"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, float(topf["runs"].max()) * 1.15])
        st.plotly_chart(style_fig(fig), width="stretch")

    st.markdown(f"##### {batter} vs every bowler faced")
    st.dataframe(faced.drop(columns=["batter"]).reset_index(drop=True),
                 width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Extras & Niche tab
# --------------------------------------------------------------------------- #
def render_extras(deliveries: pd.DataFrame, reviews: pd.DataFrame,
                  replacements: pd.DataFrame, matches: pd.DataFrame) -> None:
    st.subheader("Extras & Niche")
    st.caption(
        "Extras leaked (typed: wides/no-balls are charged to the bowler, "
        "byes/legbyes are not), plus DRS reviews, impact-player usage and "
        "Player-of-the-Match."
    )

    es = A.extras_summary(deliveries)
    e1, e2 = st.columns([1, 1.4])
    e1.markdown("##### Extras (season)")
    e1.dataframe(es, width="stretch", hide_index=True)
    ebt = A.extras_by_team(deliveries)
    fig = px.bar(
        ebt.sort_values("total_extras"), y="team",
        x=["wide", "noball", "bye", "legbye", "penalty"], orientation="h",
        title="Extras conceded by team", labels={"value": "runs", "variable": ""},
    )
    e2.plotly_chart(legend_inside(style_fig(fig)), width="stretch")

    with st.expander("Extras by bowler (wides + no-balls leaked)"):
        st.dataframe(A.extras_by_bowler(deliveries), width="stretch", hide_index=True)

    st.divider()
    d1, d2 = st.columns(2)
    d1.markdown("##### DRS reviews & success rate")
    d1.dataframe(A.drs_summary(reviews), width="stretch", hide_index=True)
    d2.markdown("##### Impact-player usage")
    d2.dataframe(A.impact_player_usage(replacements), width="stretch", hide_index=True)

    st.markdown("##### Player-of-the-Match")
    pom = A.pom_tally(matches)
    p1, p2 = st.columns([1.3, 1])
    figp = px.bar(
        pom.head(12).sort_values("awards"), y="player", x="awards",
        orientation="h", title="Most POM awards", text="awards",
        color="awards", color_continuous_scale="Tealgrn",
    )
    figp.update_traces(textposition="outside", cliponaxis=False)
    if not pom.empty:
        figp.update_xaxes(range=[0, float(pom["awards"].max()) * 1.3])
    p1.plotly_chart(style_fig(figp), width="stretch")
    p2.dataframe(pom, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #
def main() -> None:
    _require_data()
    deliveries, matches, reviews, replacements = load_data()
    season = matches["season"].iloc[0] if len(matches) else ""

    left, right = st.columns([4, 1])
    left.title("🏏 Cric Metrics")
    left.caption("A descriptive ball-by-ball statistics dashboard.")
    right.metric("Season", f"IPL {season}")

    tabs = st.tabs([
        "Batting", "Bowling", "Fielding", "Teams & Standings",
        "Phases & Momentum", "Matchups", "Extras & Niche",
    ])
    with tabs[0]:
        render_batting(deliveries)
    with tabs[1]:
        render_bowling(deliveries)
    with tabs[2]:
        render_fielding(deliveries)
    with tabs[3]:
        render_standings(matches, deliveries)
    with tabs[4]:
        render_phases(deliveries, matches)
    with tabs[5]:
        render_matchups(deliveries)
    with tabs[6]:
        render_extras(deliveries, reviews, replacements, matches)

    st.divider()
    st.caption(
        "Data source: [Cricsheet](https://cricsheet.org), licensed under "
        "ODC-BY. Cric Metrics is an independent stats project and is not "
        "affiliated with the BCCI or any team."
    )


if __name__ == "__main__":
    main()
