"""Day-ahead net load forecasting at the distribution transformer.

Neighbourhood flexibility needs a number before it needs an algorithm: how much
load, and how much rooftop generation, is this transformer going to see
tomorrow. Demand response has to be requested in advance, and a community
battery has to know at breakfast what it is saving its charge for.

The target is *net* load, load minus rooftop PV, because that is the quantity
the transformer and the feeder actually experience. It is also much harder to
forecast than gross load: PV makes the series bimodal, drives it negative at
midday under high penetration, and adds weather variance that no calendar
feature can see.

Aggregation is done over the *learned* topology. Forecasting a transformer
using the utility's ledger would pull in consumers that belong to a different
transformer and leave out ones that belong to this one, and with a third of
records wrong that is not a small effect.

Method
------
One gradient boosted model per transformer, predicting any horizon up to 24
hours from features that are all known at forecast time: calendar position,
the same interval yesterday and a week ago, and recent daily aggregates. No
recursion, so a 24-hour-ahead prediction does not compound its own errors.

Every model is scored against three baselines a utility could deploy for free,
and the honest comparison is reported rather than the raw error alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Lags available at forecast time for any horizon inside a day.
DAY = 96
WEEK = 7 * DAY
TRAIN_FRACTION = 0.7


@dataclass
class ForecastResult:
    """Forecasts and skill scores for one transformer."""

    dt_id: str
    timestamps: pd.DatetimeIndex
    actual: np.ndarray
    predicted: np.ndarray
    baselines: dict[str, np.ndarray]
    metrics: dict[str, float]

    @property
    def skill(self) -> float:
        """Fractional MAE reduction against the best baseline."""
        return self.metrics.get("skill_vs_best_baseline", 0.0)


def aggregate_to_transformer(net_p_kw: np.ndarray, meter_ids: np.ndarray,
                             assignments: pd.DataFrame) -> pd.DataFrame:
    """Sum consumer net power onto each transformer using learned connectivity."""
    index = {str(m): i for i, m in enumerate(meter_ids)}
    columns: dict[str, np.ndarray] = {}
    for dt_id, block in assignments.groupby("inferred_dt_id"):
        cols = [index[m] for m in block["meter_id"] if m in index]
        if cols:
            columns[str(dt_id)] = net_p_kw[:, cols].sum(axis=1)
    return pd.DataFrame(columns)


def aggregate_to_phase(net_p_kw: np.ndarray, meter_ids: np.ndarray,
                       assignments: pd.DataFrame) -> pd.DataFrame:
    """Sum consumer net power onto each (transformer, phase) pair.

    Forecasting per phase rather than per transformer is what makes the rest of
    the flexibility stack possible. A transformer is rarely in trouble as a
    whole; one phase is overloaded while another sits idle, which is the normal
    condition of an unbalanced LV feeder. Aggregating the three together hides
    exactly the problem worth acting on, and a demand response signal that does
    not know the phase cannot fix it.

    Columns are named ``DT07|1`` so the phase survives into every downstream
    table without a second lookup.
    """
    index = {str(m): i for i, m in enumerate(meter_ids)}
    columns: dict[str, np.ndarray] = {}
    for (dt_id, phase), block in assignments.groupby(["inferred_dt_id",
                                                      "inferred_phase"]):
        cols = [index[m] for m in block["meter_id"] if m in index]
        if cols:
            columns[f"{dt_id}|{int(phase)}"] = net_p_kw[:, cols].sum(axis=1)
    return pd.DataFrame(columns)


def split_key(key: str) -> tuple[str, int]:
    """Split a ``DT07|1`` column name back into transformer and phase."""
    dt_id, phase = key.rsplit("|", 1)
    return dt_id, int(phase)


def pv_day_ahead_forecast(solar_kw: np.ndarray, steps_per_day: int,
                          seed: int = 0, daily_bias: float = 0.18,
                          interval_noise: float = 0.07) -> np.ndarray:
    """A realistic day-ahead PV forecast for one transformer.

    A DISCOM buys irradiance forecasts; it does not get tomorrow's generation
    for free. So this deliberately degrades the truth rather than using it.

    The dominant error in a day-ahead solar forecast is getting the day's cloud
    cover wrong, which biases the whole daylight period in one direction, so the
    error here is a per-day multiplicative bias with a much smaller
    interval-level component on top. Published day-ahead GHI forecasts land
    around 15 to 20% normalised error, which is what ``daily_bias`` reproduces.

    Feeding the model the true generation instead would inflate every score in
    this module and none of it would survive contact with a real deployment.
    """
    rng = np.random.default_rng(seed)
    n_days = int(np.ceil(len(solar_kw) / steps_per_day))
    bias = rng.lognormal(0.0, daily_bias, size=n_days)
    bias = np.repeat(bias, steps_per_day)[:len(solar_kw)]
    jitter = rng.normal(1.0, interval_noise, size=len(solar_kw))
    return np.clip(solar_kw * bias * jitter, 0.0, None)


def _features(series: np.ndarray, timestamps: pd.DatetimeIndex,
              pv_forecast: np.ndarray | None = None) -> pd.DataFrame:
    """Calendar and lag features, all knowable a day in advance."""
    s = pd.Series(series)
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    extra = {}
    if pv_forecast is not None:
        pv = pd.Series(pv_forecast)
        extra = {
            "pv_forecast": pv_forecast,
            "pv_forecast_day": pv.rolling(DAY, min_periods=1).mean().to_numpy(),
        }
    return pd.DataFrame({
        **extra,
        # Cyclical encoding so 23:45 and 00:00 are neighbours, not opposites.
        "sin_day": np.sin(2 * np.pi * hour / 24.0),
        "cos_day": np.cos(2 * np.pi * hour / 24.0),
        "dayofweek": timestamps.dayofweek.to_numpy(),
        "is_weekend": (timestamps.dayofweek.to_numpy() >= 5).astype(int),
        "lag_day": s.shift(DAY).to_numpy(),
        "lag_2day": s.shift(2 * DAY).to_numpy(),
        "lag_week": s.shift(WEEK).to_numpy(),
        # Yesterday's shape around the same time, which carries weather.
        "lag_day_mean": s.shift(DAY).rolling(DAY).mean().to_numpy(),
        "lag_day_max": s.shift(DAY).rolling(DAY).max().to_numpy(),
        "lag_day_min": s.shift(DAY).rolling(DAY).min().to_numpy(),
    })


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = np.abs(actual - predicted)
    mae = float(np.nanmean(error))
    # Normalised by the range rather than by the value: net load crosses zero
    # under PV, and a percentage error against a near-zero denominator is
    # meaningless.
    span = float(np.nanmax(actual) - np.nanmin(actual))
    return mae, float(mae / span * 100.0) if span > 0 else float("nan")


def forecast_transformer(series: np.ndarray, timestamps: pd.DatetimeIndex,
                         dt_id: str,
                         pv_forecast: np.ndarray | None = None) -> ForecastResult | None:
    """Fit and evaluate a day-ahead forecaster for one transformer."""
    features = _features(series, timestamps, pv_forecast)
    usable = features.notna().all(axis=1).to_numpy() & np.isfinite(series)
    if usable.sum() < 4 * DAY:
        return None

    idx = np.where(usable)[0]
    split = int(len(idx) * TRAIN_FRACTION)
    train, test = idx[:split], idx[split:]
    if len(test) < DAY:
        return None

    model = HistGradientBoostingRegressor(
        max_iter=220, learning_rate=0.06, max_depth=6,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0)
    model.fit(features.iloc[train], series[train])
    predicted = model.predict(features.iloc[test])
    actual = series[test]

    # Baselines a utility already has, for free.
    baselines = {
        "yesterday": features["lag_day"].to_numpy()[test],
        "last week": features["lag_week"].to_numpy()[test],
        "hour-of-day mean": pd.Series(series[train]).groupby(
            timestamps[train].hour).transform("mean").reindex(
            range(len(train))).to_numpy(),
    }
    # Hour-of-day climatology, evaluated properly on the test index.
    hourly = pd.Series(series[train]).groupby(timestamps[train].hour).mean()
    baselines["hour-of-day mean"] = hourly.reindex(
        timestamps[test].hour).to_numpy()

    mae, nmae = _metrics(actual, predicted)
    metrics = {"mae_kw": mae, "nmae_pct": nmae, "n_test": int(len(test))}
    for name, values in baselines.items():
        b_mae, _ = _metrics(actual, values)
        metrics[f"mae_{name.replace(' ', '_').replace('-', '_')}"] = b_mae

    best_baseline = min(_metrics(actual, v)[0] for v in baselines.values())
    metrics["mae_best_baseline"] = best_baseline
    metrics["skill_vs_best_baseline"] = float(
        (best_baseline - mae) / best_baseline) if best_baseline > 0 else 0.0

    return ForecastResult(dt_id=dt_id, timestamps=timestamps[test], actual=actual,
                          predicted=predicted, baselines=baselines, metrics=metrics)


def forecast_all(net_p_kw: np.ndarray, meter_ids: np.ndarray,
                 assignments: pd.DataFrame, timestamps: pd.DatetimeIndex,
                 solar_kw: np.ndarray | None = None,
                 steps_per_day: int = DAY) -> dict[str, ForecastResult]:
    """Fit a forecaster for every transformer in the learned topology.

    ``solar_kw`` is the per-meter rooftop generation. It is not passed to the
    model directly; a degraded day-ahead forecast is derived from it first, so
    the model sees what a utility would actually have bought.
    """
    frame = aggregate_to_phase(net_p_kw, meter_ids, assignments)
    pv_frame = (aggregate_to_phase(solar_kw, meter_ids, assignments)
                if solar_kw is not None else None)

    out: dict[str, ForecastResult] = {}
    for seed, dt_id in enumerate(frame.columns):
        pv = None
        if pv_frame is not None and dt_id in pv_frame:
            pv = pv_day_ahead_forecast(pv_frame[dt_id].to_numpy(),
                                       steps_per_day, seed=seed)
        result = forecast_transformer(frame[dt_id].to_numpy(), timestamps,
                                      dt_id, pv_forecast=pv)
        if result is not None:
            out[dt_id] = result
    return out


def summary(results: dict[str, ForecastResult]) -> pd.DataFrame:
    """One row per transformer of forecast error and skill."""
    return pd.DataFrame([
        {"dt_id": r.dt_id, **r.metrics} for r in results.values()
    ]).sort_values("skill_vs_best_baseline", ascending=False).reset_index(drop=True)
