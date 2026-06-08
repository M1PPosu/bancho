"""
score_risk_engine.py

A multi-factor risk scoring engine for osu! private server score submissions.
Replaces the simple dynamic PP cap with a layered confidence model that avoids
false-positive bans on legitimately skilled players.

Risk score: 0–100
  0–30   → Accept (clean)
  31–55  → Accept + log (monitor)
  56–75  → Hide/freeze pending review
  76–90  → Temporary restriction + review queue
  91–100 → Auto-restrict

Each component contributes a weighted sub-score. Trust adjustments can lower
the final value for known-good players.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any
import asyncio
from app.discord import Embed, Webhook
import app.settings

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class RiskAction(IntEnum):
    ACCEPT          = 0   # 0–30
    ACCEPT_LOG      = 1   # 31–55
    FREEZE_REVIEW   = 2   # 56–75
    TEMP_RESTRICT   = 3   # 76–90
    AUTO_RESTRICT   = 4   # 91–100


@dataclass
class RiskResult:
    final_score: float
    action: RiskAction
    components: dict[str, float]
    trust_adjustment: float
    reasons: list[str]

    @property
    def should_restrict(self) -> bool:
        return self.action == RiskAction.AUTO_RESTRICT

    @property
    def should_temp_restrict(self) -> bool:
        return self.action == RiskAction.TEMP_RESTRICT

    @property
    def should_freeze(self) -> bool:
        return self.action == RiskAction.FREEZE_REVIEW

    def __str__(self) -> str:
        return (
            f"RiskResult(score={self.final_score:.1f}, action={self.action.name}, "
            f"trust_adj={self.trust_adjustment:+.1f}, reasons={self.reasons})"
        )


# ---------------------------------------------------------------------------
# Input context – filled from DB / in-memory state before calling engine
# ---------------------------------------------------------------------------

@dataclass
class PlayerContext:
    """All player-level data needed for risk evaluation."""

    user_id: int
    playcount: int
    total_playtime_seconds: int
    account_age_days: int
    overall_pp: float

    # Top plays (pp values, sorted descending)
    top_plays_pp: list[float]       # full top list available
    recent_plays_pp: list[float]    # last ~20 submitted scores' pp values

    # Mode-specific data (for the mode being submitted)
    mode: int                        # 0=std,1=taiko,2=catch,3=mania; +4=rx,+8=ap
    mode_playcount: int
    mode_pp: float
    mode_top_plays_pp: list[float]  # top plays for this specific mode

    # Mod distribution: {mod_bitmask: count_of_scores}
    mod_distribution: dict[int, int] = field(default_factory=dict)

    # Account trust signals
    is_whitelisted: bool = False
    has_previous_restriction: bool = False
    is_supporter: bool = False
    previous_risk_flags: int = 0    # count of prior logged/frozen events

    # Session info
    scores_this_session: int = 0    # scores submitted in the last hour
    session_start_ts: float = 0.0   # unix timestamp of session start


@dataclass
class ScoreContext:
    """Data about the specific score being submitted."""

    pp: float
    acc: float
    mods: int
    max_combo: int
    miss_count: int
    mode: int

    # Beatmap info
    map_star_rating: float
    map_max_combo: int
    map_ranked_status: int          # 2=ranked, 3=approved, 5=loved

    # Performance on this map historically
    player_prev_best_pp: float | None   # their old PB on this map, if any

    # Replay signals (pass None if replay wasn't analysed)
    replay_ur: float | None = None          # unstable rate
    replay_mean_error: float | None = None  # mean hit error ms
    replay_has_suspicous_inputs: bool = False

    # Timestamp
    submitted_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Individual risk components (each returns 0.0–100.0)
# ---------------------------------------------------------------------------

def _map_pp_outlier_risk(score: ScoreContext, player: PlayerContext) -> tuple[float, list[str]]:
    """
    How anomalous is this score's PP relative to the map and the player's history?
    Weight: 25 %
    """
    reasons: list[str] = []
    risk = 0.0

    if not player.top_plays_pp:
        return 0.0, reasons   # bootstrap phase – no data

    top1  = player.top_plays_pp[0]
    top10 = _weighted_average(player.top_plays_pp[:10])
    top50 = _weighted_average(player.top_plays_pp[:50])

    # --- How far does this score exceed the player's best ever? ---
    if top1 > 0:
        ratio_vs_top1 = score.pp / top1
        if ratio_vs_top1 > 2.0:
            risk += 60.0
            reasons.append(f"score pp is {ratio_vs_top1:.1f}× their all-time top play")
        elif ratio_vs_top1 > 1.5:
            risk += 35.0
            reasons.append(f"score pp is {ratio_vs_top1:.1f}× their all-time top play")
        elif ratio_vs_top1 > 1.2:
            risk += 15.0
            reasons.append(f"score pp is {ratio_vs_top1:.1f}× their all-time top play")
        elif ratio_vs_top1 > 1.05:
            risk += 5.0

    # --- How far above their top-10 average? ---
    if top10 > 0:
        ratio_vs_top10 = score.pp / top10
        if ratio_vs_top10 > 3.0:
            risk += 20.0
            reasons.append(f"score pp is {ratio_vs_top10:.1f}× their top-10 avg")
        elif ratio_vs_top10 > 2.0:
            risk += 10.0

    # --- Improvement over their previous PB on this map ---
    prev_best_pp = score.player_prev_best_pp
    if prev_best_pp:
        if prev_best_pp > 0:
            pb_ratio = score.pp / prev_best_pp
            if pb_ratio > 3.0:
                risk += 25.0
                reasons.append(f"score is {pb_ratio:.1f}× their previous PB on this map")
            elif pb_ratio > 2.0:
                risk += 10.0

    # --- Absolute floor: tiny PP scores can't be suspicious ---
    if score.pp < 150:
        risk = min(risk, 10.0)

    return min(risk, 100.0), reasons


def _player_progression_risk(score: ScoreContext, player: PlayerContext) -> tuple[float, list[str]]:
    """
    Is this score consistent with plausible human skill progression?
    Weight: 25 %
    """
    reasons: list[str] = []
    risk = 0.0

    # --- Playcount gate ---
    # Very low playcount relative to claimed skill is a red flag
    expected_plays_for_pp = _expected_plays_for_pp(score.pp)
    if player.playcount < expected_plays_for_pp * 0.05:
        risk += 50.0
        reasons.append(
            f"only {player.playcount} plays for {score.pp:.0f}pp "
            f"(expected ≥ {int(expected_plays_for_pp * 0.05)})"
        )
    elif player.playcount < expected_plays_for_pp * 0.15:
        risk += 25.0
        reasons.append(f"low playcount ({player.playcount}) relative to claimed skill")

    # --- Playtime plausibility ---
    # Rough heuristic: skilled play requires many hours
    hours_played = player.total_playtime_seconds / 3600
    expected_hours = _expected_hours_for_pp(score.pp)
    if hours_played < expected_hours * 0.05:
        risk += 30.0
        reasons.append(
            f"only {hours_played:.0f}h played for {score.pp:.0f}pp "
            f"(expected ≥ {expected_hours * 0.05:.0f}h)"
        )
    elif hours_played < expected_hours * 0.15:
        risk += 15.0

    # --- Account age ---
    if player.account_age_days < 7 and score.pp > 400:
        risk += 20.0
        reasons.append(f"new account ({player.account_age_days}d) with high pp score")
    elif player.account_age_days < 30 and score.pp > 600:
        risk += 10.0

    # --- Recent improvement speed ---
    if len(player.recent_plays_pp) >= 5:
        recent_avg = _weighted_average(player.recent_plays_pp[-5:])
        if recent_avg > 0:
            improvement = score.pp / recent_avg
            if improvement > 4.0:
                risk += 25.0
                reasons.append(
                    f"score is {improvement:.1f}× recent avg "
                    f"({recent_avg:.0f}pp) — sudden spike"
                )
            elif improvement > 2.5:
                risk += 12.0
                reasons.append(f"rapid skill spike ({improvement:.1f}× recent avg)")

    return min(risk, 100.0), reasons


def _replay_behavior_risk(score: ScoreContext, player: PlayerContext) -> tuple[float, list[str]]:
    """
    Signals from replay data (if available).
    Weight: 20 %
    """
    reasons: list[str] = []
    risk = 0.0

    if score.replay_has_suspicous_inputs:
        risk += 50.0
        reasons.append("replay contains suspicious input patterns")

    if score.replay_ur is not None:
        ur = score.replay_ur
        # Superhuman consistency on hard maps
        if ur < 40 and score.map_star_rating >= 6.0:
            risk += 35.0
            reasons.append(f"suspiciously low UR ({ur:.1f}) on {score.map_star_rating:.1f}★ map")
        elif ur < 55 and score.map_star_rating >= 7.0:
            risk += 20.0
            reasons.append(f"low UR ({ur:.1f}) on {score.map_star_rating:.1f}★ map")

    if score.replay_mean_error is not None:
        # Perfectly centred on 0ms is unusual for a human
        if abs(score.replay_mean_error) < 1.0 and score.acc > 98.0:
            risk += 15.0
            reasons.append(
                f"near-perfect mean error ({score.replay_mean_error:.2f}ms) with {score.acc:.2f}% acc"
            )

    # FC on extreme maps with very low miss is fine; but check acc consistency
    if score.miss_count == 0 and score.acc < 92.0 and score.map_star_rating >= 8.0:
        # FC with low acc on an insane map: possible relax/aimbot
        risk += 10.0
        reasons.append(f"FC with only {score.acc:.1f}% acc on {score.map_star_rating:.1f}★ map")

    return min(risk, 100.0), reasons


def _account_context_risk(score: ScoreContext, player: PlayerContext) -> tuple[float, list[str]]:
    """
    Account-level red flags unrelated to the score itself.
    Weight: 15 %
    """
    reasons: list[str] = []
    risk = 0.0

    if player.has_previous_restriction:
        risk += 20.0
        reasons.append("account has a prior restriction on record")

    if player.previous_risk_flags > 0:
        flag_risk = min(player.previous_risk_flags * 5, 25.0)
        risk += flag_risk
        reasons.append(f"{player.previous_risk_flags} prior risk flag(s) on record")

    # Mode mismatch: submitted in a mode they have almost no plays in
    if player.mode_playcount < 10 and score.pp > 300:
        risk += 15.0
        reasons.append(
            f"only {player.mode_playcount} plays in this mode but scoring {score.pp:.0f}pp"
        )

    # Mod skills that don't match their profile
    if score.mods != 0 and player.mod_distribution:
        total_with_mods = sum(
            v for k, v in player.mod_distribution.items() if k != 0
        )
        if total_with_mods == 0 and score.pp > 500:
            risk += 10.0
            reasons.append("high pp score with mods but no prior mod plays in profile")

    return min(risk, 100.0), reasons


def _session_pattern_risk(score: ScoreContext, player: PlayerContext) -> tuple[float, list[str]]:
    """
    Suspicious session behaviour (burst of high scores, odd timing patterns).
    Weight: 15 %
    """
    reasons: list[str] = []
    risk = 0.0

    # Too many scores in a short window
    if player.scores_this_session > 60:
        risk += 20.0
        reasons.append(f"{player.scores_this_session} scores submitted in one session")
    elif player.scores_this_session > 30:
        risk += 8.0

    # High scores at implausible speed
    if player.scores_this_session > 0 and player.session_start_ts > 0:
        elapsed_minutes = (score.submitted_at - player.session_start_ts) / 60
        if elapsed_minutes > 0:
            rate = player.scores_this_session / elapsed_minutes
            if rate > 3.0:   # more than 3 maps per minute
                risk += 15.0
                reasons.append(f"submitting at {rate:.1f} scores/min — possible tool abuse")

    return min(risk, 100.0), reasons


# ---------------------------------------------------------------------------
# Trust adjustments (negative = lowers risk score)
# ---------------------------------------------------------------------------

def _compute_trust_adjustment(score: ScoreContext, player: PlayerContext) -> tuple[float, list[str]]:
    """
    Returns a value (typically negative) applied after weighted sum.
    Trusted signals reduce risk; suspicious history can increase it.
    """
    adjustment = 0.0
    reasons: list[str] = []

    if player.is_whitelisted:
        adjustment -= 30.0
        reasons.append("player is whitelisted (-30)")

    if player.is_supporter:
        adjustment -= 5.0
        reasons.append("supporter account (-5)")

    # Long-standing account with lots of plays is less likely to be a cheater
    if player.account_age_days > 365 and player.playcount > 5000:
        adjustment -= 10.0
        reasons.append("veteran account with high playcount (-10)")

    # New PB improvement (not a massive spike) is expected behaviour
    if score.player_prev_best_pp is not None:
        ratio = score.pp / score.player_prev_best_pp if score.player_prev_best_pp > 0 else 1.0
        if 1.0 < ratio <= 1.15:
            adjustment -= 5.0
            reasons.append("small, plausible PB improvement (-5)")

    # Prior restriction = trust penalty
    if player.has_previous_restriction:
        adjustment += 10.0
        reasons.append("previous restriction on record (+10)")

    return adjustment, reasons


# ---------------------------------------------------------------------------
# Component weights
# ---------------------------------------------------------------------------

COMPONENT_WEIGHTS: dict[str, float] = {
    "map_pp_outlier":       0.25,
    "player_progression":   0.25,
    "replay_behavior":      0.20,
    "account_context":      0.15,
    "session_pattern":      0.15,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_score_risk(
    score: ScoreContext,
    player: PlayerContext,
) -> RiskResult:
    """
    Evaluate the risk of a score submission and return a RiskResult.

    Usage in the submission handler::

        from app.anticheat.score_risk_engine import (
            ScoreContext, PlayerContext, evaluate_score_risk, RiskAction
        )

        risk = evaluate_score_risk(score_ctx, player_ctx)
        log(f"[risk] {player} → {risk}", Ansi.LCYAN)

        if risk.action == RiskAction.AUTO_RESTRICT:
            await player.restrict(admin=bot, reason=str(risk))
            ...
        elif risk.action == RiskAction.TEMP_RESTRICT:
            # temporary restriction logic
            ...
        elif risk.action == RiskAction.FREEZE_REVIEW:
            # hide score, add to review queue
            ...
        elif risk.action == RiskAction.ACCEPT_LOG:
            # log for monitoring but let through
            ...
        # else: clean, proceed normally
    """
    # Bootstrap: skip engine when there's not enough history
    BOOTSTRAP_PLAYS = 10
    if player.playcount < BOOTSTRAP_PLAYS:
        return RiskResult(
            final_score=0.0,
            action=RiskAction.ACCEPT,
            components={},
            trust_adjustment=0.0,
            reasons=[f"bootstrap phase ({player.playcount} plays < {BOOTSTRAP_PLAYS})"],
        )

    # --- Evaluate each component ---
    raw: dict[str, float] = {}
    all_reasons: list[str] = []

    score1, r1 = _map_pp_outlier_risk(score, player)
    raw["map_pp_outlier"] = score1
    all_reasons.extend(r1)

    score2, r2 = _player_progression_risk(score, player)
    raw["player_progression"] = score2
    all_reasons.extend(r2)

    score3, r3 = _replay_behavior_risk(score, player)
    raw["replay_behavior"] = score3
    all_reasons.extend(r3)

    score4, r4 = _account_context_risk(score, player)
    raw["account_context"] = score4
    all_reasons.extend(r4)

    score5, r5 = _session_pattern_risk(score, player)
    raw["session_pattern"] = score5
    all_reasons.extend(r5)

    # --- Weighted sum ---
    weighted = sum(raw[k] * COMPONENT_WEIGHTS[k] for k in raw)

    # --- Trust adjustment ---
    trust_adj, trust_reasons = _compute_trust_adjustment(score, player)
    all_reasons.extend(trust_reasons)

    final = max(0.0, min(100.0, weighted + trust_adj))

    # --- Action thresholds ---
    if final <= 30:
        action = RiskAction.ACCEPT
    elif final <= 55:
        action = RiskAction.ACCEPT_LOG
    elif final <= 75:
        action = RiskAction.FREEZE_REVIEW
    elif final <= 90:
        action = RiskAction.TEMP_RESTRICT
    else:
        action = RiskAction.AUTO_RESTRICT
    
    if action in (RiskAction.FREEZE_REVIEW, RiskAction.TEMP_RESTRICT, RiskAction.AUTO_RESTRICT):
        colour = {
            RiskAction.FREEZE_REVIEW:  0xFFA500,  # orange
            RiskAction.TEMP_RESTRICT:  0xFF4500,  # red-orange
            RiskAction.AUTO_RESTRICT:  0xFF0000,  # red
        }[action]

        embed = Embed(
            title=f"[{action.name}] Risk flag — {player.user_id}",
            description="\n".join(f"• {r}" for r in all_reasons) or "No specific reasons.",
            color=colour,
            timestamp=datetime.utcnow().isoformat(),
        )
        embed.add_field(name="Risk Score", value=f"{final:.1f} / 100", inline=True)
        embed.add_field(name="Action", value=action.name, inline=True)
        embed.add_field(name="Score PP", value=f"{score.pp:.2f}pp", inline=True)
        embed.add_field(name="Map SR", value=f"{score.map_star_rating:.2f}★", inline=True)
        embed.add_field(name="Playcount", value=str(player.playcount), inline=True)
        embed.add_field(name="Account Age", value=f"{player.account_age_days}d", inline=True)
        embed.add_field(
            name="Components",
            value="\n".join(f"{k}: {v:.1f}" for k, v in raw.items()),
            inline=False,
        )

        webhook = Webhook(url=app.settings.ANTICHEAT_WEBHOOK)
        webhook.add_embed(embed)
        asyncio.get_event_loop().create_task(webhook.post())
    
    return RiskResult(
        final_score=final,
        action=action,
        components=raw,
        trust_adjustment=trust_adj,
        reasons=all_reasons,
    )


# ---------------------------------------------------------------------------
# Helper: build contexts from the data already available in the handlers
# ---------------------------------------------------------------------------

async def build_player_context(
    player: "Player",             # app.objects.player.Player
    mode: int,
    database: Any,
) -> PlayerContext:
    """
    Convenience builder that queries the DB for history needed by the engine.
    Call this inside osuSubmitModularSelector / osuSubmitModular.

    Example::

        player_ctx = await build_player_context(score.player, score.mode, db)
        score_ctx  = build_score_context(score, bmap)
        risk = evaluate_score_risk(score_ctx, player_ctx)
    """
    stats = player.stats[mode]

    # Top 50 pp scores for this mode
    top_rows = await database.fetch_all(
        "SELECT pp FROM scores "
        "WHERE userid = :uid AND mode = :mode AND status = 2 "
        "ORDER BY pp DESC LIMIT 50",
        {"uid": player.id, "mode": mode},
    )
    top_plays_pp = [float(r["pp"]) for r in top_rows]

    # Recent 20 pp scores (any status) for progression analysis
    recent_rows = await database.fetch_all(
        "SELECT pp FROM scores "
        "WHERE userid = :uid AND mode = :mode "
        "ORDER BY play_time DESC LIMIT 20",
        {"uid": player.id, "mode": mode},
    )
    recent_plays_pp = [float(r["pp"]) for r in recent_rows]

    # Mod distribution
    mod_rows = await database.fetch_all(
        "SELECT mods, COUNT(*) AS cnt FROM scores "
        "WHERE userid = :uid AND mode = :mode AND status = 2 "
        "GROUP BY mods",
        {"uid": player.id, "mode": mode},
    )
    mod_dist = {r["mods"]: r["cnt"] for r in mod_rows}

    # Scores submitted in the last hour (session estimate)
    one_hour_ago = int(time.time()) - 3600
    session_row = await database.fetch_one(
        "SELECT COUNT(*) AS cnt, MIN(UNIX_TIMESTAMP(play_time)) AS first_ts "
        "FROM scores "
        "WHERE userid = :uid AND UNIX_TIMESTAMP(play_time) > :cutoff",
        {"uid": player.id, "cutoff": one_hour_ago},
    )
    scores_this_session = session_row["cnt"] if session_row else 0
    session_start_ts    = float(session_row["first_ts"] or 0) if session_row else 0.0

    # Account age (days since registration)
    user_row = await database.fetch_one(
        "SELECT creation_time FROM users WHERE id = :uid",
        {"uid": player.id},
    )
    if user_row and user_row["creation_time"]:
        account_age_days = (time.time() - float(user_row["creation_time"])) / 86400
    else:
        account_age_days = 0

    # Prior restriction history
    restrict_row = await database.fetch_one(
        "SELECT COUNT(*) AS cnt FROM logs "
        "WHERE `to` = :uid AND `action` = 'restrict'",
        {"uid": player.id},
    )
    has_prev_restriction = bool(restrict_row and restrict_row["cnt"] > 0)

    # Prior risk flag count
    flag_row = await database.fetch_one(
        "SELECT COUNT(*) AS cnt FROM anticheat_flags WHERE userid = :uid",
        {"uid": player.id},
    )
    prev_flags = flag_row["cnt"] if flag_row else 0

    return PlayerContext(
        user_id=player.id,
        playcount=stats.plays,
        total_playtime_seconds=stats.playtime,
        account_age_days=int(account_age_days),
        overall_pp=stats.pp,
        top_plays_pp=top_plays_pp,
        recent_plays_pp=recent_plays_pp,
        mode=mode,
        mode_playcount=stats.plays,
        mode_pp=stats.pp,
        mode_top_plays_pp=top_plays_pp,
        mod_distribution=mod_dist,
        is_whitelisted=bool(player.priv & 0x4),   # adjust to your Privileges.WHITELISTED
        has_previous_restriction=has_prev_restriction,
        is_supporter=bool(player.priv & 0x8),     # adjust to your Privileges.SUPPORTER
        previous_risk_flags=prev_flags,
        scores_this_session=scores_this_session,
        session_start_ts=session_start_ts,
    )


def build_score_context(
    score: "Score",        # app.objects.score.Score
    bmap:  "Beatmap",      # app.objects.beatmap.Beatmap
) -> ScoreContext:
    """Build a ScoreContext from an already-parsed score + beatmap."""
    return ScoreContext(
        pp=score.pp,
        acc=score.acc,
        mods=int(score.mods),
        max_combo=score.max_combo,
        miss_count=score.nmiss,
        mode=int(score.mode),
        map_star_rating=bmap.diff,      # adjust field name if different
        map_max_combo=bmap.max_combo,   # adjust field name if different
        map_ranked_status=int(bmap.status),
        player_prev_best_pp=(
            float(score.prev_best.pp) if score.prev_best else None
        ),
        # replay fields – populate from your replay analyser if available
        replay_ur=None,
        replay_mean_error=None,
        replay_has_suspicous_inputs=False,
    )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_average(values: list[float]) -> float:
    if not values:
        return 0.0
    total_weight = sum(0.95**i for i in range(len(values)))
    weighted_sum = sum(v * 0.95**i for i, v in enumerate(values))
    return weighted_sum / total_weight


def _expected_plays_for_pp(pp: float) -> float:
    """
    Very rough heuristic: how many plays should a player of this skill have?
    Based on typical osu! progression patterns.
    """
    if pp < 200:
        return 500
    elif pp < 400:
        return 2_000
    elif pp < 600:
        return 5_000
    elif pp < 800:
        return 10_000
    elif pp < 1000:
        return 20_000
    elif pp < 1500:
        return 40_000
    else:
        return 80_000


def _expected_hours_for_pp(pp: float) -> float:
    """Rough heuristic: hours of playtime expected to reach this skill level."""
    if pp < 200:
        return 50
    elif pp < 400:
        return 150
    elif pp < 600:
        return 400
    elif pp < 800:
        return 800
    elif pp < 1000:
        return 1_500
    elif pp < 1500:
        return 3_000
    else:
        return 6_000

