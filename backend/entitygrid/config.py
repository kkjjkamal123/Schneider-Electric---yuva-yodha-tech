"""Central configuration for the ENTITY GRID reference deployment.

Every module reads its constants from here so that a demo, a unit test and a
pilot deployment differ only by the ``SimConfig`` / ``GridConfig`` instance
they are handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class GridConfig:
    """Electrical parameters of the modelled LV network.

    Defaults follow a typical Indian urban/peri-urban 11 kV / 433 V
    distribution transformer with a four-wire (3-phase + neutral) LV feeder.
    """

    # --- source / transformer -------------------------------------------------
    nominal_ll_voltage: float = 433.0          # volts, line-to-line
    dt_ratings_kva: tuple[int, ...] = (63, 100, 160, 250, 400)
    # Transformer series impedance referred to LV, as %Z on the DT rating.
    dt_impedance_pct: float = 4.5
    dt_x_over_r: float = 2.0

    # --- conductors (ohm / km) ------------------------------------------------
    # 3x95 mm^2 + 70 mm^2 N aerial bundled cable - LV backbone.
    main_r_per_km: float = 0.320
    main_x_per_km: float = 0.075
    main_neutral_r_per_km: float = 0.443
    main_neutral_x_per_km: float = 0.075
    # 2x16 mm^2 service drop to an individual consumer.
    service_r_per_km: float = 1.910
    service_x_per_km: float = 0.098

    # --- geometry -------------------------------------------------------------
    section_length_m: tuple[float, float] = (25.0, 60.0)
    service_length_m: tuple[float, float] = (8.0, 35.0)

    @property
    def nominal_ln_voltage(self) -> float:
        """Nominal line-to-neutral (phase) voltage in volts."""
        return self.nominal_ll_voltage / (3 ** 0.5)


@dataclass(frozen=True)
class SimConfig:
    """Shape and duration of the synthetic dataset."""

    seed: int = 20260404

    n_transformers: int = 12
    meters_per_dt: tuple[int, int] = (28, 55)

    days: int = 30
    interval_minutes: int = 15

    # Share of consumers carrying rooftop PV, and its kWp range.
    solar_penetration: float = 0.22
    solar_kwp: tuple[float, float] = (1.5, 6.0)

    # Smart-meter instrumentation limits (this is what makes the problem hard).
    voltage_resolution_v: float = 0.1     # meters report 0.1 V steps
    voltage_noise_v: float = 0.25         # class 1 metrology + ADC noise
    missing_read_rate: float = 0.015      # comms gaps in the AMI head-end

    # Ground-truth impairments ENTITY GRID has to cope with / detect.
    n_degrading_dts: int = 3              # DTs walked toward failure
    n_outage_events: int = 6              # LV faults producing last-gasp bursts
    connectivity_record_error_rate: float = 0.31  # utility's *recorded* mapping

    grid: GridConfig = field(default_factory=GridConfig)

    @property
    def steps_per_day(self) -> int:
        return (24 * 60) // self.interval_minutes

    @property
    def n_steps(self) -> int:
        return self.days * self.steps_per_day


DEFAULT_SIM = SimConfig()
