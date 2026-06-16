"""parse.py — Cricsheet JSON -> two tidy tables.

Reads every ``JSON Source Files/*.json`` and writes:

  * ``data/deliveries.csv`` — one row per ball
  * ``data/matches.csv``    — one row per match

Plus two small supplementary tables for the niche views whose source fields are
not part of the two core schemas:

  * ``data/reviews.csv``      — one row per DRS review (review.decision etc.)
  * ``data/replacements.csv`` — one row per substitution (impact players etc.)

Run this once (and again only when the source files change). The Streamlit app
reads the cached CSVs and never re-parses JSON on every run.

Standard library only — no third-party parser. See CLAUDE.md for the exact
column lists and the definitional rules implemented here.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path("JSON Source Files")
OUT_DIR = Path("data")

# Dismissal kinds credited to the bowler (run-outs / retirements excluded).
BOWLER_CREDITED_KINDS = {
    "bowled",
    "caught",
    "lbw",
    "stumped",
    "caught and bowled",
    "hit wicket",
}

# Recognised extras keys (typed). Anything else is reported in the summary.
KNOWN_EXTRAS = {"wides", "noballs", "byes", "legbyes", "penalty"}

# Deliveries with a phase computed from the innings' powerplays field. The
# powerplay's last over comes from the data; Middle/Death use the 15/16 split.
DEATH_FIRST_OVER = 15  # 0-indexed: 16th over onwards
MIDDLE_LAST_OVER = 14  # 0-indexed: through the 15th over


def _powerplay_last_over(innings: dict) -> int:
    """0-indexed last over of the (mandatory) powerplay for this innings.

    Cricsheet ``from``/``to`` use ``over.ball`` notation with a 0-indexed over,
    so ``to: 5.6`` means the powerplay ends with over index 5.
    """
    pps = innings.get("powerplays") or []
    last = 5  # sensible default if a shortened/super-over innings omits it
    for pp in pps:
        if pp.get("type", "mandatory") == "mandatory" and "to" in pp:
            last = int(float(pp["to"]))
    return last


def _phase(over_idx: int, pp_last: int) -> str:
    if over_idx <= pp_last:
        return "Powerplay"
    if over_idx <= MIDDLE_LAST_OVER:
        return "Middle"
    return "Death"


def _extras_amounts(delivery: dict, unknown_keys: Counter) -> dict:
    """Return typed extra amounts; record any unexpected keys."""
    extras = delivery.get("extras") or {}
    for k in extras:
        if k not in KNOWN_EXTRAS:
            unknown_keys[k] += 1
    return {
        "wide": extras.get("wides", 0),
        "noball": extras.get("noballs", 0),
        "bye": extras.get("byes", 0),
        "legbye": extras.get("legbyes", 0),
        "penalty": extras.get("penalty", 0),
    }


def _wicket_fields(delivery: dict, bowler: str, unknown_kinds: Counter) -> dict:
    """Flatten the (usually single) wicket on a delivery into flat columns."""
    wickets = delivery.get("wickets") or []
    if not wickets:
        return {
            "wicket_kind": "",
            "player_out": "",
            "fielders": "",
            "bowler_credited": "",
        }

    kinds, outs, fielders, credited = [], [], [], []
    for w in wickets:
        kind = w.get("kind", "")
        if kind and kind not in BOWLER_CREDITED_KINDS and kind not in {
            "run out",
            "retired hurt",
            "retired out",
            "retired not out",
            "obstructing the field",
            "handled the ball",
            "hit the ball twice",
            "timed out",
        }:
            unknown_kinds[kind] += 1
        kinds.append(kind)
        outs.append(w.get("player_out", ""))
        names = [f.get("name", "") for f in (w.get("fielders") or [])]
        fielders.extend(n for n in names if n)
        if kind in BOWLER_CREDITED_KINDS:
            credited.append(bowler)

    return {
        "wicket_kind": ";".join(k for k in kinds if k),
        "player_out": ";".join(o for o in outs if o),
        "fielders": ";".join(fielders),
        "bowler_credited": ";".join(credited),
    }


def parse_match(path: Path, unknown_extras: Counter, unknown_kinds: Counter):
    """Return ``(match_row, [delivery_rows], [review_rows], [replacement_rows])``."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    info = data["info"]
    match_id = path.stem
    season = info.get("season", "")
    event = info.get("event", {}) or {}
    # League games carry event.match_number (1..N); playoffs carry event.stage
    # (e.g. "Qualifier 1", "Final") and no match_number.
    match_number = event.get("match_number")
    stage = event.get("stage", "")
    date = (info.get("dates") or [""])[0]
    venue = info.get("venue", "")
    city = info.get("city", "")
    teams = info.get("teams", [])
    innings_list = data.get("innings", [])

    # ---- match row -------------------------------------------------------
    outcome = info.get("outcome", {}) or {}
    by = outcome.get("by", {}) or {}
    # A tie decided by a super over: Cricsheet names the winner under
    # `eliminator` and sets result "tie". IPL awards this as a WIN to the
    # super-over winner (2 pts), not a shared 1-point tie — so score it as a win
    # and flag it. A genuine shared tie has no eliminator.
    eliminator = outcome.get("eliminator", "")
    super_over = bool(eliminator)
    if eliminator:
        result = "win"
        winner = eliminator
    elif outcome.get("result") in {"tie", "no result", "draw"}:
        result = outcome["result"]
        winner = outcome.get("winner", "")
    elif outcome.get("winner"):
        result = "win"
        winner = outcome["winner"]
    else:
        result = ""
        winner = outcome.get("winner", "")

    bat_first_team = innings_list[0]["team"] if innings_list else ""
    chasing_team = innings_list[1]["team"] if len(innings_list) > 1 else ""
    target_block = (
        innings_list[1].get("target", {}) if len(innings_list) > 1 else {}
    )
    target = target_block.get("runs")
    target_overs = target_block.get("overs")
    # Scheduled overs per innings (info.overs). For a match reduced before the
    # toss this is the reduced figure; if only the chase is revised by rain the
    # revised allotment lives in target_overs instead. NRR reads its quota from
    # these fields rather than assuming 20.
    scheduled_overs = info.get("overs")
    # method is set when the result was decided by Duckworth-Lewis (D/L).
    method = outcome.get("method", "")
    pom = info.get("player_of_match") or []

    match_row = {
        "match_id": match_id,
        "season": season,
        "match_number": match_number,
        "stage": stage,
        "date": date,
        "venue": venue,
        "city": city,
        "team1": teams[0] if len(teams) > 0 else "",
        "team2": teams[1] if len(teams) > 1 else "",
        "toss_winner": info.get("toss", {}).get("winner", ""),
        "toss_decision": info.get("toss", {}).get("decision", ""),
        "bat_first_team": bat_first_team,
        "chasing_team": chasing_team,
        "winner": winner,
        "result": result,
        "super_over": super_over,
        "method": method,
        "win_by_runs": by.get("runs"),
        "win_by_wickets": by.get("wickets"),
        "target_runs": target,
        "target_overs": target_overs,
        "scheduled_overs": scheduled_overs,
        "player_of_match": ";".join(pom),
    }

    # ---- delivery rows (+ reviews, replacements) -------------------------
    delivery_rows = []
    review_rows = []
    replacement_rows = []
    for innings_no, innings in enumerate(innings_list, start=1):
        batting_team = innings["team"]
        bowling_team = next((t for t in teams if t != batting_team), "")
        pp_last = _powerplay_last_over(innings)
        positions: dict[str, int] = {}
        next_pos = 1

        for over in innings.get("overs", []):
            over_idx = over["over"]
            for ball_seq, delivery in enumerate(over.get("deliveries", []), start=1):
                batter = delivery.get("batter", "")
                non_striker = delivery.get("non_striker", "")
                bowler = delivery.get("bowler", "")

                # batting position = order a player first reaches the crease
                for name in (batter, non_striker):
                    if name and name not in positions:
                        positions[name] = next_pos
                        next_pos += 1

                runs = delivery.get("runs", {}) or {}
                runs_batter = runs.get("batter", 0)
                extras = _extras_amounts(delivery, unknown_extras)
                legal_ball = extras["wide"] == 0 and extras["noball"] == 0

                row = {
                    "match_id": match_id,
                    "season": season,
                    "date": date,
                    "venue": venue,
                    "city": city,
                    "innings": innings_no,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over_idx,
                    "ball_in_over": ball_seq,
                    "legal_ball": legal_ball,
                    "phase": _phase(over_idx, pp_last),
                    "batter": batter,
                    "non_striker": non_striker,
                    "bowler": bowler,
                    "batting_position": positions.get(batter),
                    "runs_batter": runs_batter,
                    "runs_extras": runs.get("extras", 0),
                    "runs_total": runs.get("total", 0),
                    **extras,
                    "is_four": runs_batter == 4,
                    "is_six": runs_batter == 6,
                    "is_dot": legal_ball and runs_batter == 0,
                    **_wicket_fields(delivery, bowler, unknown_kinds),
                }
                delivery_rows.append(row)

                review = delivery.get("review")
                if review:
                    decision = review.get("decision", "")
                    review_rows.append({
                        "match_id": match_id,
                        "innings": innings_no,
                        "over": over_idx,
                        "ball_in_over": ball_seq,
                        "reviewing_team": review.get("by", ""),
                        "batter": review.get("batter", ""),
                        "umpire": review.get("umpire", ""),
                        "type": review.get("type", ""),
                        "decision": decision,
                        # Cricsheet: "upheld" = the review succeeded for the team
                        "successful": decision == "upheld",
                    })

                for rep in (delivery.get("replacements", {}) or {}).get("match", []):
                    replacement_rows.append({
                        "match_id": match_id,
                        "innings": innings_no,
                        "over": over_idx,
                        "ball_in_over": ball_seq,
                        "team": rep.get("team", ""),
                        "player_in": rep.get("in", ""),
                        "player_out": rep.get("out", ""),
                        "reason": rep.get("reason", ""),
                    })

    return match_row, delivery_rows, review_rows, replacement_rows


def main() -> None:
    files = sorted(SOURCE_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {SOURCE_DIR}/")

    unknown_extras: Counter = Counter()
    unknown_kinds: Counter = Counter()
    match_rows, delivery_rows, review_rows, replacement_rows = [], [], [], []

    for path in files:
        m, ds, rv, rp = parse_match(path, unknown_extras, unknown_kinds)
        match_rows.append(m)
        delivery_rows.extend(ds)
        review_rows.extend(rv)
        replacement_rows.extend(rp)

    OUT_DIR.mkdir(exist_ok=True)
    deliveries = pd.DataFrame(delivery_rows)
    matches = pd.DataFrame(match_rows)
    reviews = pd.DataFrame(review_rows)
    replacements = pd.DataFrame(replacement_rows)
    deliveries.to_csv(OUT_DIR / "deliveries.csv", index=False)
    matches.to_csv(OUT_DIR / "matches.csv", index=False)
    reviews.to_csv(OUT_DIR / "reviews.csv", index=False)
    replacements.to_csv(OUT_DIR / "replacements.csv", index=False)

    # ---- summary ---------------------------------------------------------
    super_over_balls = int((deliveries["innings"] > 2).sum())
    print(f"Parsed {len(files)} files")
    print(f"  matches.csv      : {len(matches):>6} rows")
    print(f"  deliveries.csv   : {len(deliveries):>6} rows")
    print(f"  reviews.csv      : {len(reviews):>6} rows")
    print(f"  replacements.csv : {len(replacements):>6} rows")
    print(f"  legal balls    : {int(deliveries['legal_ball'].sum()):>6}")
    print(f"  super-over balls (innings > 2): {super_over_balls}")
    print(f"  phases         : {dict(deliveries['phase'].value_counts())}")
    if unknown_extras:
        print(f"  ! unexpected extras keys : {dict(unknown_extras)}")
    if unknown_kinds:
        print(f"  ! uncredited/unknown wicket kinds : {dict(unknown_kinds)}")
    if not unknown_extras and not unknown_kinds:
        print("  no unexpected extras keys or wicket kinds")


if __name__ == "__main__":
    main()
