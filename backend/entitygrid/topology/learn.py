"""Self-learning LV topology: which meter sits on which transformer and phase.

This is the module the rest of ENTITY GRID depends on. Health scoring needs to know
which meters aggregate to which DT; fault localisation needs the connectivity
graph; volt-var needs to know which phase is rising. A utility's own ledger is
wrong for roughly a third of consumers, so none of it can be trusted as input.

The method, in three steps:

1. **Group.** Meters are clustered on the correlation of their common-mode-
   removed voltage increments. The natural grouping that falls out is not the
   transformer but the *(transformer, phase)* pair, because meters sharing a
   phase conductor share its voltage drop almost perfectly, while meters on
   different phases of the same transformer are coupled only weakly, and with
   the opposite sign, through the neutral.

2. **Label.** Each group is matched against per-phase DT busbar telemetry. The
   best-correlating reference names both the transformer and the phase, so the
   clusters come out labelled with real asset identifiers rather than
   arbitrary numbers.

3. **Score.** Every meter is given a confidence from how cleanly its group
   matched, and how well the meter itself tracks its group. Low-confidence
   meters are what a crew should actually be sent to verify - the output is a
   ranked work list, not an unqualified answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from entitygrid.sim.network import PHASE_NAMES
from entitygrid.topology.features import (
    correlation_distance,
    masked_correlation,
    reference_signatures,
    residual_delta_voltage,
    usable_intervals,
)

# Below this margin between the best and runner-up transformer match, the
# assignment is not trusted without a field check.
LOW_CONFIDENCE_MARGIN = 0.15


@dataclass
class TopologyResult:
    """Inferred connectivity plus the evidence behind it."""

    assignments: pd.DataFrame   # meter_id, inferred_dt_id, inferred_phase, confidence...
    cluster_labels: np.ndarray  # (n_meters,) raw cluster index
    correlation: np.ndarray     # (n_meters, n_meters) meter-to-meter correlation

    @property
    def needs_verification(self) -> pd.DataFrame:
        """Meters a crew should physically confirm, worst first."""
        flagged = self.assignments[self.assignments["needs_verification"]]
        return flagged.sort_values("confidence")


def learn_topology(voltage: np.ndarray, meter_ids: np.ndarray,
                   dt_voltage: np.ndarray, dt_ids: np.ndarray,
                   *,
                   common_mode: bool = True,
                   difference: bool = True,
                   own_load: np.ndarray | None = None,
                   hours: np.ndarray | None = None,
                   exclude_hours: tuple[int, int] | None = None,
                   ) -> TopologyResult:
    """Infer meter-to-transformer-to-phase connectivity from voltage alone.

    Parameters
    ----------
    voltage:
        ``(n_steps, n_meters)`` observed meter voltages, NaN for missing reads.
    meter_ids:
        ``(n_meters,)`` meter identifiers, aligned to ``voltage`` columns.
    dt_voltage:
        ``(n_dts, n_steps, 3)`` per-phase DT busbar voltages.
    dt_ids:
        ``(n_dts,)`` transformer identifiers, aligned to ``dt_voltage``.

    Notes
    -----
    No ground truth and no utility ledger is consulted. The only structural
    assumption is that the asset register knows how many transformers exist,
    which every DISCOM does.
    """
    residual = residual_delta_voltage(
        voltage, common_mode=common_mode, difference=difference)

    # Interval selection runs on the differenced grid, so drop the first sample
    # to keep the mask aligned with the residual it filters.
    mask = usable_intervals(
        None if own_load is None else own_load[1:] if difference else own_load,
        None if hours is None else hours[1:] if difference else hours,
        exclude_hours)

    corr = masked_correlation(residual, mask)
    distance = correlation_distance(corr)

    n_groups = len(dt_ids) * 3
    n_groups = min(n_groups, len(meter_ids))
    labels = AgglomerativeClustering(
        n_clusters=n_groups, metric="precomputed", linkage="average",
    ).fit_predict(distance)

    references = reference_signatures(dt_voltage)
    # The references are always differenced. When the meter side is not, the
    # two grids differ by one sample, so align on the common tail.
    span = min(residual.shape[0], references.shape[1])
    sig_source, references = residual[-span:], references[:, -span:]

    rows = []
    for cluster in range(n_groups):
        members = np.where(labels == cluster)[0]
        if len(members) == 0:
            continue

        signature = sig_source[:, members].mean(axis=1)
        signature = (signature - signature.mean()) / (signature.std() + 1e-9)
        scores = references @ signature / len(signature)

        order = np.argsort(scores)[::-1]
        best, runner_up = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        margin = float(scores[best] - scores[runner_up])

        dt_id = str(dt_ids[best // 3])
        phase = int(best % 3)

        # How tightly does each member track its own group?
        member_fit = corr[np.ix_(members, members)].mean(axis=1)

        for meter_index, fit in zip(members, member_fit):
            confidence = float(np.clip(scores[best], 0.0, 1.0) * np.clip(fit, 0.0, 1.0))
            rows.append({
                "meter_id": str(meter_ids[meter_index]),
                "inferred_dt_id": dt_id,
                "inferred_phase": phase,
                "inferred_phase_name": PHASE_NAMES[phase],
                "cluster": cluster,
                "cluster_size": len(members),
                "match_score": float(scores[best]),
                "match_margin": margin,
                "group_fit": float(fit),
                "confidence": confidence,
                "needs_verification": bool(margin < LOW_CONFIDENCE_MARGIN or fit < 0.35),
            })

    assignments = pd.DataFrame(rows)
    # Restore the caller's meter ordering.
    assignments = (assignments.set_index("meter_id")
                   .reindex([str(m) for m in meter_ids])
                   .reset_index())
    return TopologyResult(assignments=assignments, cluster_labels=labels, correlation=corr)
