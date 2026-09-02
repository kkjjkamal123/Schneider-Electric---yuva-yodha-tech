"""Single entry point for reading the generated dataset.

Everything downstream - the pipeline, the API, the tests - loads through here
rather than touching ``.npz`` files directly. Timestamps in particular are
stored as int64 nanoseconds since the epoch, which pandas will happily
misinterpret if the unit is left implicit, so the conversion lives in exactly
one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from entitygrid.config import RAW_DIR

LOCAL_TZ = "Asia/Kolkata"


@dataclass
class Dataset:
    """Everything a DISCOM would actually hand over, plus truth for scoring."""

    timestamps: pd.DatetimeIndex
    meter_ids: np.ndarray
    voltage: np.ndarray            # (n_steps, n_meters)
    current: np.ndarray
    net_p_kw: np.ndarray
    solar_kw: np.ndarray

    dt_ids: np.ndarray
    dt_voltage: np.ndarray         # (n_dts, n_steps, 3)
    dt_current: np.ndarray
    dt_neutral: np.ndarray

    last_gasp: pd.DataFrame
    recorded_connectivity: pd.DataFrame

    truth_meters: pd.DataFrame
    truth_transformers: pd.DataFrame
    truth_events: dict

    @property
    def interval_minutes(self) -> int:
        return int(self.truth_events.get("interval_minutes", 15))

    @property
    def steps_per_day(self) -> int:
        return (24 * 60) // self.interval_minutes

    @property
    def n_steps(self) -> int:
        return self.voltage.shape[0]

    def step_time(self, step: int) -> pd.Timestamp:
        """Wall-clock time of a simulation step index."""
        return self.timestamps[min(max(step, 0), len(self.timestamps) - 1)]


def parse_epoch_ns(values: np.ndarray) -> pd.DatetimeIndex:
    """Convert stored int64 nanoseconds to a timezone-aware index.

    The explicit ``unit`` is not optional: without it pandas infers a unit from
    magnitude and silently lands three decades away from the real date.
    """
    index = pd.to_datetime(np.asarray(values, dtype="int64"), unit="ns", utc=True)
    return index.tz_convert(LOCAL_TZ)


def load_dataset(raw_dir: Path | None = None) -> Dataset:
    """Load the full generated dataset from ``data/raw``."""
    raw = Path(raw_dir or RAW_DIR)
    if not (raw / "ami.npz").exists():
        raise FileNotFoundError(
            f"no dataset at {raw}; run `python -m entitygrid.sim.generate` first")

    import json

    ami = np.load(raw / "ami.npz", allow_pickle=False)
    tel = np.load(raw / "dt_telemetry.npz", allow_pickle=False)

    last_gasp = pd.read_csv(raw / "last_gasp.csv")
    if len(last_gasp):
        last_gasp["received_at"] = pd.to_datetime(
            last_gasp["received_at"], utc=True, format="mixed").dt.tz_convert(LOCAL_TZ)

    return Dataset(
        timestamps=parse_epoch_ns(ami["timestamps"]),
        meter_ids=ami["meter_ids"],
        voltage=ami["voltage"],
        current=ami["current"],
        net_p_kw=ami["net_p_kw"],
        solar_kw=ami["solar_kw"],
        dt_ids=tel["dt_ids"],
        dt_voltage=tel["voltage"],
        dt_current=tel["current"],
        dt_neutral=tel["neutral_current"],
        last_gasp=last_gasp,
        recorded_connectivity=pd.read_csv(raw / "recorded_connectivity.csv"),
        truth_meters=pd.read_csv(raw / "truth_meters.csv"),
        truth_transformers=pd.read_csv(raw / "truth_transformers.csv"),
        truth_events=json.loads((raw / "truth_events.json").read_text()),
    )
