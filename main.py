"""Launch-day Twitch CCU forecaster for gta6-twitch-ccu-forecast-01 (Python 3.13, stdlib only).

Model: a launch's release-day shape, hour by hour, is borrowed from the past
launches in the case's `comparables`. For every comparable with a complete
release day, take the log-ratio of each hour's viewers to its own hour-1
viewers. For the target, weight each comparable by a Gaussian kernel on the
distance between its log hour-1 viewers and the target's: launches of a
similar scale share a shape (a rolling-midnight console launch that is
already large in hour 1 decays and re-peaks; a launch that unlocks late in
the UTC day explodes from a near-zero hour 1). The inner quantiles of the
weighted log-ratios are read off the weighted empirical distribution; the
two tail quantiles come from the weighted normal fit, where the local pool
is too thin to read tails directly. The five log-quantile curves are
smoothed across neighbouring hours, then scaled by the target's hour-1
print. If no comparable has a complete release day, the forecast falls back
to persistence with a widening spread (the house baseline's shape).

The grader runs `python3 -P -s /bot/main.py` once per case with only that
case's file in the input directory; the program reads every *.in file in
$MAKEFUN_INPUT_DIR (else /input) and writes one <id>.out per case to
$MAKEFUN_OUTPUT_DIR (else /output). It is fully deterministic: no clocks,
no randomness, pool order fixed by comparable name.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Gaussian kernel bandwidth on the log hour-1 distance between launches.
KERNEL_BW = 1.0
# The house baseline's spread, used only when no comparable is usable.
FALLBACK_SIGMA = 0.15

_NORM = NormalDist()


def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_key(moment):
    """The `from` key of the hour that starts at `moment` (UTC)."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def round2(x):
    """x rounded to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def weighted_quantile(pairs, tau):
    """Quantile of (value, weight) pairs by linear interpolation on weight.

    `pairs` need not be sorted. With total weight W the quantile sits where
    the cumulative weight first reaches tau * W, interpolated between the
    two neighbouring points.
    """
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[0][0]
    goal = tau * total
    acc = 0.0
    prev_v, prev_acc = ordered[0][0], 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= goal:
            if acc == prev_acc:
                return value
            frac = (goal - prev_acc) / (acc - prev_acc)
            return prev_v + frac * (value - prev_v)
        prev_v, prev_acc = value, acc
    return ordered[-1][0]


def comparable_day(comp):
    """A comparable's release-day viewers, hours 1..24, or None if incomplete."""
    try:
        start = parse_ts(comp["day_start"])
        points = {}
        for row in comp.get("hourly") or []:
            viewers = row.get("viewers")
            if viewers is None:
                continue
            viewers = float(viewers)
            if math.isfinite(viewers):
                points[row["from"]] = max(1.0, viewers)
        day = []
        for k in range(24):
            value = points.get(hour_key(start + timedelta(hours=k)))
            if value is None:
                return None
            day.append(value)
        return day
    except (KeyError, TypeError, ValueError):
        return None


def build_pool(case):
    """Usable comparables as (log hour-1, [log-ratio for hours 2..24]), name order."""
    pool = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: str(c.get("name", ""))):
        day = comparable_day(comp)
        if day is None:
            continue
        base = day[0]
        ratios = [math.log(day[h - 1] / base) for h in range(2, 25)]
        pool.append((math.log(base), ratios))
    return pool


def hour1_print(case):
    """The target's hour-1 print: at least 1, from `known` if present."""
    known = case.get("known") or []
    row = next((k for k in known if k.get("hour") == 1), None)
    if row is None and known:
        row = min(known, key=lambda k: k.get("hour", 99))
    if row is not None and row.get("viewers") is not None:
        return max(1.0, float(row["viewers"]))
    history = case.get("history") or []
    if history and history[-1].get("viewers") is not None:
        return max(1.0, float(history[-1]["viewers"]))
    return 1.0


def hour_log_quantiles(pairs, total_weight):
    """One hour's five log-ratio quantiles from weighted (log-ratio, weight) pairs.

    The three inner quantiles are read off the weighted empirical
    distribution. The two tails come from the weighted normal fit (mean and
    standard deviation of the same pairs): where the kernel has thinned the
    pool, a handful of launches cannot pin a 5% tail, but their centre and
    spread still can. With no spread (a single usable launch) every level
    falls back to the empirical read.
    """
    mean = sum(v * w for v, w in pairs) / total_weight
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total_weight
    sd = math.sqrt(var) if var > 0 else 0.0
    qs = []
    for tau in LEVELS:
        if sd > 0 and tau in (LEVELS[0], LEVELS[-1]):
            qs.append(mean + _NORM.inv_cdf(tau) * sd)
        else:
            qs.append(weighted_quantile(pairs, tau))
    return qs


def log_quantiles(pool, p1, hours):
    """Blended log-ratio quantiles per hour: {hour: [lr at each of LEVELS]}."""
    log_p1 = math.log(p1)
    weights = [math.exp(-0.5 * ((lp - log_p1) / KERNEL_BW) ** 2) for lp, _ in pool]
    if not pool or sum(weights) <= 0:
        weights = [1.0] * len(pool)
    total = sum(weights)
    raw = {}
    for h in hours:
        idx = h - 2
        pairs = [(ratios[idx], w) for (_, ratios), w in zip(pool, weights)]
        raw[h] = hour_log_quantiles(pairs, total)
    # Smooth each quantile curve across neighbouring hours.
    smoothed = {}
    for h in hours:
        acc = list(raw[h])
        count = 1
        for other in (h - 1, h + 1):
            if other in raw:
                for i in range(len(LEVELS)):
                    acc[i] += raw[other][i]
                count += 1
        smoothed[h] = [a / count for a in acc]
    return smoothed


def baseline_log_quantiles(hours):
    """Persistence fallback: log-ratio 0 with a spread widening like sqrt(h-1)."""
    return {
        h: [_NORM.inv_cdf(tau) * FALLBACK_SIGMA * math.sqrt(h - 1) for tau in LEVELS]
        for h in hours
    }


def forecast_case(case):
    hours = [row["hour"] for row in case.get("forecast") or []]
    if not hours:
        hours = list(range(2, 25))
    p1 = hour1_print(case)
    pool = build_pool(case)
    curves = log_quantiles(pool, p1, hours) if pool else baseline_log_quantiles(hours)
    rows = []
    for h in hours:
        qs = []
        prev = 0.0
        for lr in curves[h]:
            value = p1 * math.exp(lr)
            value = min(max(value, 1e-6), 1e9)
            value = max(value, prev)  # quantiles never decrease
            qs.append(round2(value))
            prev = float(qs[-1])
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
