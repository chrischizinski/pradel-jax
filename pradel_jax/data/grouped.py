"""Exact frequency-weighted grouping for Pradel encounter histories.

This is not a new statistical approximation.  Rows are combined only when
their complete capture history and every covariate trajectory are identical,
so multiplying a group's log likelihood by its frequency reproduces the
individual-history likelihood exactly (up to floating-point summation order).
"""

from dataclasses import replace
from typing import List

import jax.numpy as jnp
import numpy as np

from .adapters import DataContext


def _grouping_key(data: DataContext) -> np.ndarray:
    """Return one numeric key row per individual without losing trajectories."""
    parts: List[np.ndarray] = [np.asarray(data.capture_matrix, dtype=np.float64)]
    for name in sorted(data.covariates):
        values = np.asarray(data.covariates[name], dtype=np.float64)
        # Older adapters retain a few scalar descriptive flags in covariates.
        # They do not vary by person and cannot affect a likelihood contribution.
        if values.ndim == 0:
            continue
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[0] != data.n_individuals:
            raise ValueError(f"Covariate '{name}' is not indexed by individual.")
        parts.append(values.reshape(data.n_individuals, -1))

    key = np.concatenate(parts, axis=1)
    # NaNs do not compare equal, so use a sentinel solely for row grouping.
    return np.nan_to_num(key, nan=-9.87654321e37)


def group_data_context(data: DataContext) -> DataContext:
    """Collapse duplicate likelihood contributions into a weighted context.

    The returned context has one row per unique complete record and a
    ``frequency`` vector.  It preserves the original context metadata and adds
    transparent compression counts for reporting.
    """
    if data.frequency is not None:
        raise ValueError("DataContext is already frequency-weighted.")

    _, first_index, inverse, counts = np.unique(
        _grouping_key(data),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    order = np.argsort(first_index)
    first_index = first_index[order]
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    counts = np.bincount(remap[inverse], minlength=len(order))

    metadata = dict(data.metadata or {})
    metadata["grouped"] = {
        "n_original_individuals": int(data.n_individuals),
        "n_unique_records": int(len(first_index)),
        "compression_ratio": float(data.n_individuals / len(first_index)),
    }
    return replace(
        data,
        capture_matrix=jnp.asarray(np.asarray(data.capture_matrix)[first_index]),
        covariates={
            name: (
                jnp.asarray(np.asarray(values)[first_index])
                if np.asarray(values).ndim > 0
                else values
            )
            for name, values in data.covariates.items()
        },
        n_individuals=int(len(first_index)),
        individual_ids=(
            None
            if data.individual_ids is None
            else [data.individual_ids[i] for i in first_index]
        ),
        metadata=metadata,
        frequency=jnp.asarray(counts, dtype=jnp.float64),
    )
