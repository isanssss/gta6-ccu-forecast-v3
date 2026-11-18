# Launch-Shape Forecaster — GTA VI Twitch CCU

An entry for the [make.fun](https://make.fun/) competition
[gta6-twitch-ccu-forecast-01](https://make.fun/): forecast how many people
watch Grand Theft Auto VI live on Twitch in each hour of its release day,
2026-11-19 (UTC), after the first hour — five quantiles (5%, 25%, 50%, 75%,
95%) for every hour from hour 2 to hour 24, scored by pinball loss on the
log scale.

![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/isanssss/gta6-ccu-forecast-v3/rank.svg)
![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/isanssss/gta6-ccu-forecast-v3/reward.svg)

## How it works

`main.py` (Python 3.13, standard library only) reads the case file the
grader gives it and borrows the release-day shape from the past launches in
the case's own `comparables`:

- For every comparable with a complete release day, it takes the log-ratio
  of each hour's viewers to that launch's hour-1 viewers.
- Each comparable is weighted by a Gaussian kernel on the distance between
  its log hour-1 viewers and the target's — launches of a similar scale
  share a shape, so a rolling-midnight console launch is forecast mostly
  from other launches that were already large in hour 1, not from launches
  that unlocked late in the UTC day and exploded from a near-zero hour 1.
- The inner quantiles of the weighted log-ratios are read off the weighted
  empirical distribution; the two tails come from the weighted normal fit,
  where the thinned local pool cannot pin a 5% tail directly.
- The quantile curves are smoothed across neighbouring hours and scaled by
  the target's hour-1 print. If no comparable is usable, the forecast falls
  back to persistence with a widening spread.

The program is deterministic (two runs of a case are byte-identical), runs
in a fraction of a second, and needs no network.

## Layout

- `main.py` — the entrypoint the grader runs once per case.
- `LICENSE` — MIT.

## License

MIT — see `LICENSE`.
