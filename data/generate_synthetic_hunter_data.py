#!/usr/bin/env python3
"""
Generate a fully synthetic capture-recapture dataset with the same column
schema as the real Nebraska/South Dakota HIP hunter-tier exports (see
scripts/analysis/run_mr_nebraska_sd.R), so the pipeline and the pradel_jax
package can be demonstrated/tested without any real hunter data.

Every value is simulated from a known Pradel (1996) process (entry,
survival, detection) with true parameters set below — nothing here is
derived from or resembles real individuals. IDs use a "SYNTH_" prefix
specifically so they can never be mistaken for real or fuzzy-matched IDs.

Usage:
    python data/generate_synthetic_hunter_data.py
"""

import numpy as np
import pandas as pd

SEED = 42
N_INDIVIDUALS = 2000
YEARS = list(range(2016, 2025))  # 9 occasions, matches the real exports
N_OCCASIONS = len(YEARS)

# True data-generating parameters (illustrative, not fitted to any real
# data). phi = apparent survival (per-occasion, logit link), f-like entry
# process controls when individuals join the study, p = detection
# probability given alive and in the study.
TRUE_PARAMS = dict(
    p_detect=0.6,
    phi_base=0.70,       # survival for a baseline (female, non-tier2) individual
    phi_gender_boost=0.12,   # male survival boost
    phi_tier2_boost=0.15,    # "ever tier2" individuals survive better
    tier2_given_capture=0.25,  # P(tier2 | captured), for tier2-type individuals
    entry_weights=[0.55, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03],
    gender_p_male=0.55,
    gender_p_unknown=0.04,
    tier2_type_rate=0.15,
    age_min=12,
    age_max=70,
)


def simulate(n=N_INDIVIDUALS, seed=SEED, params=TRUE_PARAMS):
    rng = np.random.default_rng(seed)

    entry_occasion = rng.choice(N_OCCASIONS, size=n, p=params["entry_weights"])

    gender = np.full(n, "F", dtype=object)
    is_male = rng.random(n) < params["gender_p_male"]
    gender[is_male] = "M"
    is_unknown = rng.random(n) < params["gender_p_unknown"]
    gender[is_unknown] = "UNKNOWN"

    tier2_type = rng.random(n) < params["tier2_type_rate"]
    age_2016 = rng.integers(params["age_min"], params["age_max"] + 1, size=n)

    phi = np.full(n, params["phi_base"])
    phi = np.where(is_male, phi + params["phi_gender_boost"], phi)
    phi = np.where(tier2_type, phi + params["phi_tier2_boost"], phi)
    phi = np.clip(phi, 0.01, 0.99)

    tier = np.zeros((n, N_OCCASIONS), dtype=int)
    alive = np.zeros(n, dtype=bool)

    for t in range(N_OCCASIONS):
        just_entered = entry_occasion == t
        alive = np.where(just_entered, True, alive)

        already_in_study = entry_occasion < t
        survives = rng.random(n) < phi
        alive = np.where(already_in_study, alive & survives, alive)

        detected = alive & (rng.random(n) < params["p_detect"])
        upgraded = detected & tier2_type & (rng.random(n) < params["tier2_given_capture"])
        tier[:, t] = np.where(detected, np.where(upgraded, 2, 1), 0)

    age_cols = {
        f"age_{year}": age_2016 + i for i, year in enumerate(YEARS)
    }
    tier_cols = {f"tier_{year}": tier[:, i] for i, year in enumerate(YEARS)}

    ch = np.array(["".join("1" if v > 0 else "0" for v in row) for row in tier])

    df = pd.DataFrame({
        "customer_id": [f"SYNTH_{i:05d}" for i in range(n)],
        **tier_cols,
        "gender": gender,
        **age_cols,
        "th": ch,
        "ch": ch,
    })
    return df


if __name__ == "__main__":
    df = simulate()
    out_path = "data/synthetic_capture_recapture_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic individuals to {out_path}")
    print(f"Gender counts:\n{df['gender'].value_counts()}")
    n_ever_captured = (df.filter(like="tier_").to_numpy() > 0).any(axis=1).sum()
    print(f"Ever captured: {n_ever_captured} ({n_ever_captured / len(df):.1%})")
