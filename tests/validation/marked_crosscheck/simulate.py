"""Write the capture histories for the marked cross-check.

Deterministic: rerunning reproduces ``histories.csv`` exactly. The histories
come from the reduced model in ``tests/integration/test_marked_crosscheck.py``
(multistate CJS, three live states, constant parameters). They are synthetic --
no hunter data is involved.

    python tests/validation/marked_crosscheck/simulate.py
"""

from pathlib import Path

import numpy as np

from pradel_jax.models.multistate import (
    OBS_NONE,
    OBS_TIER1,
    OBS_TIER2,
    STATE_TIER1,
    STATE_TIER2,
)
from tests.integration.test_marked_crosscheck import (
    N_OCCASIONS,
    TRUTH,
    reduced_matrices,
)

N_INDIVIDUALS = 5000
LETTERS = {OBS_NONE: "0", OBS_TIER1: "A", OBS_TIER2: "B"}


def simulate(rng):
    n = N_INDIVIDUALS
    matrices = np.asarray(reduced_matrices(TRUTH, n))
    first = rng.integers(0, N_OCCASIONS - 1, size=n)
    state = np.where(rng.random(n) < 0.3, STATE_TIER2, STATE_TIER1)
    history = np.zeros((n, N_OCCASIONS), dtype=np.int32)
    for t in range(N_OCCASIONS):
        if t > 0:
            cumulative = np.cumsum(matrices[np.arange(n), t - 1, state], axis=1)
            moved = (rng.random(n)[:, None] > cumulative).sum(axis=1)
            state = np.where(first < t, moved, state)
        observed = np.where(
            state == STATE_TIER1,
            OBS_TIER1,
            np.where(state == STATE_TIER2, OBS_TIER2, OBS_NONE),
        )
        history[:, t] = np.where(first <= t, observed, OBS_NONE)
    return history


if __name__ == "__main__":
    history = simulate(np.random.default_rng(5))
    lines = ["".join(LETTERS[c] for c in row) for row in history]
    out = Path(__file__).with_name("histories.csv")
    out.write_text("ch\n" + "\n".join(lines) + "\n")
    print(f"wrote {len(lines)} histories to {out}")
