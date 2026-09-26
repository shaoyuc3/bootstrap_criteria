"""Participant-bootstrap tests for within- and between-participant correlations.

The bootstrap unit is the participant/interview, not the flattened segment row.
For each bootstrap draw, participants are sampled with replacement. Within-
participant correlations are recomputed from participant-centered sufficient
statistics, and between-participant correlations are recomputed from participant
means. A point is marked significant when its 95% participant-bootstrap CI
excludes zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapInference:
    """Bootstrap result for one feature."""

    feature: str
    within_r: float
    between_r: float
    within_ci_low: float
    within_ci_high: float
    within_p: float
    within_significant: bool
    between_ci_low: float
    between_ci_high: float
    between_p: float
    between_significant: bool
    n_boot: int
    n_participants: int

    @property
    def both_significant(self) -> bool:
        return self.within_significant and self.between_significant


def _safe_corr_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlations between X columns and y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y[:, None])
    x_mean = np.divide(
        np.where(valid, x, 0.0).sum(axis=0),
        valid.sum(axis=0),
        out=np.full(x.shape[1], np.nan, dtype=np.float64),
        where=valid.sum(axis=0) > 0,
    )
    y_mean = np.divide(
        np.where(valid, y[:, None], 0.0).sum(axis=0),
        valid.sum(axis=0),
        out=np.full(x.shape[1], np.nan, dtype=np.float64),
        where=valid.sum(axis=0) > 0,
    )
    x_centered = np.where(valid, x - x_mean, 0.0)
    y_centered = np.where(valid, y[:, None] - y_mean, 0.0)
    denom = np.sqrt((x_centered * x_centered).sum(axis=0) * (y_centered * y_centered).sum(axis=0))
    return np.divide(
        (x_centered * y_centered).sum(axis=0),
        denom,
        out=np.full(x.shape[1], np.nan, dtype=np.float64),
        where=denom > 0,
    )


def compute_au_associations(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    participant_col: str = "pid",
    score_col: str = "score",
) -> pd.DataFrame:
    """Compute observed within- and between-participant correlations.

    Parameters
    ----------
    df:
        Long segment-level dataframe. Each row is one scored segment.
    feature_cols:
        Feature columns to correlate with ``score_col``.
    participant_col:
        Column identifying participants/interviews.
    score_col:
        Segment-level target score column.
    """
    feature_cols = list(feature_cols)
    rows = []
    within_x = []
    within_y = []
    between_x = []
    between_y = []

    for _, group in df.groupby(participant_col, sort=False):
        x = group[feature_cols].to_numpy(dtype=np.float64)
        y = group[score_col].to_numpy(dtype=np.float64)
        within_x.append(x - np.nanmean(x, axis=0, keepdims=True))
        within_y.append(y - np.nanmean(y))
        between_x.append(np.nanmean(x, axis=0))
        between_y.append(np.nanmean(y))

    within_r = _safe_corr_columns(np.vstack(within_x), np.concatenate(within_y))
    between_r = _safe_corr_columns(np.vstack(between_x), np.asarray(between_y, dtype=np.float64))
    for idx, feature in enumerate(feature_cols):
        rows.append(
            {
                "feature": feature,
                "within_r": float(within_r[idx]),
                "between_r": float(between_r[idx]),
            }
        )
    return pd.DataFrame(rows)


def participant_bootstrap_inference(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    participant_col: str = "pid",
    score_col: str = "score",
    n_boot: int = 10000,
    seed: int = 20260828,
) -> pd.DataFrame:
    """Compute observed correlations, bootstrap CIs, p-values, and flags.

    The returned significance flags use the same criterion as the paper figure:
    ``ci_low > 0`` or ``ci_high < 0`` for the corresponding axis.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    feature_cols = list(feature_cols)
    observed = compute_au_associations(
        df,
        feature_cols,
        participant_col=participant_col,
        score_col=score_col,
    )
    grouped = list(df.groupby(participant_col, sort=False))
    n_participants = len(grouped)
    n_features = len(feature_cols)

    within_sxy = np.zeros((n_participants, n_features), dtype=np.float64)
    within_sx2 = np.zeros((n_participants, n_features), dtype=np.float64)
    within_sy2 = np.zeros(n_participants, dtype=np.float64)
    between_x = np.zeros((n_participants, n_features), dtype=np.float64)
    between_y = np.zeros(n_participants, dtype=np.float64)

    for idx, (_, group) in enumerate(grouped):
        x = group[feature_cols].to_numpy(dtype=np.float64)
        y = group[score_col].to_numpy(dtype=np.float64)
        x_centered = x - np.nanmean(x, axis=0, keepdims=True)
        y_centered = y - np.nanmean(y)
        valid = np.isfinite(x_centered) & np.isfinite(y_centered[:, None])
        within_sxy[idx] = np.nansum(np.where(valid, x_centered * y_centered[:, None], 0.0), axis=0)
        within_sx2[idx] = np.nansum(np.where(valid, x_centered * x_centered, 0.0), axis=0)
        within_sy2[idx] = np.nansum(np.where(np.isfinite(y_centered), y_centered * y_centered, 0.0))
        between_x[idx] = np.nanmean(x, axis=0)
        between_y[idx] = np.nanmean(y)

    rng = np.random.default_rng(seed)
    within_boot = np.empty((n_boot, n_features), dtype=np.float64)
    between_boot = np.empty((n_boot, n_features), dtype=np.float64)

    chunk_size = 200
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        sample_idx = rng.integers(0, n_participants, size=(stop - start, n_participants))

        sxy = within_sxy[sample_idx].sum(axis=1)
        sx2 = within_sx2[sample_idx].sum(axis=1)
        sy2 = within_sy2[sample_idx].sum(axis=1)
        denom = np.sqrt(sx2 * sy2[:, None])
        within_boot[start:stop] = np.divide(
            sxy,
            denom,
            out=np.full_like(sxy, np.nan, dtype=np.float64),
            where=denom > 0,
        )

        sampled_x = between_x[sample_idx]
        sampled_y = between_y[sample_idx]
        x_centered = sampled_x - sampled_x.mean(axis=1, keepdims=True)
        y_centered = sampled_y - sampled_y.mean(axis=1, keepdims=True)
        denom = np.sqrt(
            np.sum(x_centered * x_centered, axis=1)
            * np.sum(y_centered * y_centered, axis=1)[:, None]
        )
        between_boot[start:stop] = np.divide(
            np.sum(x_centered * y_centered[:, :, None], axis=1),
            denom,
            out=np.full((stop - start, n_features), np.nan, dtype=np.float64),
            where=denom > 0,
        )

    rows: list[BootstrapInference] = []
    for feature_idx, feature in enumerate(feature_cols):
        observed_row = observed.loc[observed["feature"] == feature].iloc[0]
        values = {}
        for prefix, boot_values in (
            ("within", within_boot[:, feature_idx]),
            ("between", between_boot[:, feature_idx]),
        ):
            finite = boot_values[np.isfinite(boot_values)]
            if finite.size == 0:
                values[f"{prefix}_ci_low"] = float("nan")
                values[f"{prefix}_ci_high"] = float("nan")
                values[f"{prefix}_p"] = float("nan")
                values[f"{prefix}_significant"] = False
                continue
            ci_low, ci_high = np.percentile(finite, [2.5, 97.5])
            lower_tail = np.sum(finite <= 0.0)
            upper_tail = np.sum(finite >= 0.0)
            p_value = min(1.0, 2.0 * (min(lower_tail, upper_tail) + 1.0) / (finite.size + 1.0))
            values[f"{prefix}_ci_low"] = float(ci_low)
            values[f"{prefix}_ci_high"] = float(ci_high)
            values[f"{prefix}_p"] = float(p_value)
            values[f"{prefix}_significant"] = bool(ci_low > 0.0 or ci_high < 0.0)

        rows.append(
            BootstrapInference(
                feature=feature,
                within_r=float(observed_row["within_r"]),
                between_r=float(observed_row["between_r"]),
                within_ci_low=values["within_ci_low"],
                within_ci_high=values["within_ci_high"],
                within_p=values["within_p"],
                within_significant=values["within_significant"],
                between_ci_low=values["between_ci_low"],
                between_ci_high=values["between_ci_high"],
                between_p=values["between_p"],
                between_significant=values["between_significant"],
                n_boot=n_boot,
                n_participants=n_participants,
            )
        )

    return pd.DataFrame([row.__dict__ | {"both_significant": row.both_significant} for row in rows])
