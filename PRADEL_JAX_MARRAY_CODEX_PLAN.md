# Pradel-JAX m-Array Refactor and Two-Tier Hunter Modeling Plan

## Purpose

This document is an implementation brief for Codex to extend the
existing `pradel-jax` workflow used for Nebraska and South Dakota
duck-hunter mark--recapture analyses.

The immediate goal is **not to rewrite the package**. The goal is to
preserve the existing Pradel-JAX API, optimization, inference, and
validation machinery while adding a faster grouped/m-array likelihood
backend and correcting the architecture needed for genuinely
time-varying parameters.

The scientific motivation is to remove computational/modeling
restrictions that previously forced simplified Pradel models and
ultimately allow richer analysis of the two-tier waterfowl regulation
experiment, especially:

-   annual Tier I/Tier II status;
-   Tier × year effects;
-   demographic heterogeneity in Tier effects;
-   time-varying detection;
-   recruitment and retention differences by Tier;
-   eventual extension to explicit Tier I ↔ Tier II transitions and
    reactivation.

The implementation should proceed incrementally and be validated at
every stage against the current individual-history likelihood and, where
feasible, MARK/RMark.

------------------------------------------------------------------------

## 1. Existing analysis and constraints

### 1.1 Reproduction workflow

The supplied `pradel_mr_reproduction` package reproduces the
constant-(p) Nebraska and South Dakota analysis.

Current data preparation:

-   annual fields `tier_2016` through `tier_2024`;
-   `0` = not captured/registered;
-   `1` or `2` = annual tier level;
-   binary capture histories are rebuilt from the tier fields;
-   `gender` is coded M/F;
-   `first_age` is based on `age_2016`;
-   `tier2_dummy` indicates whether the hunter was ever Tier II during
    the study period.

The reproduction README reports approximately:

-   Nebraska: \~118,000 hunters;
-   South Dakota: \~71,000 hunters;
-   12 candidate models per state;
-   full run in under a minute using JAX.

### 1.2 Current candidate models

Detection probability is fixed at:

``` text
p ~ 1
```

The candidate models vary (`\phi`{=tex}) and (f), primarily using:

``` text
gender
first_age
tier2_dummy
```

with additive effects only.

Examples include:

``` text
phi ~ gender + first_age + tier2_dummy
p   ~ 1
f   ~ gender
```

### 1.3 Why the model was restricted

These restrictions should **not** be interpreted as the desired
biological model.

The original MARK/RMark workflow had computational/convergence
difficulties with richer models. `pradel-jax` was subsequently developed
to improve computation, but the final reproducible analysis still holds
(p) constant and uses a restricted additive model set to obtain
reliable, identifiable fits.

The purpose of this refactor is to determine whether grouped/m-array
likelihoods can remove enough computational burden to fit the richer
models that were originally scientifically desirable.

------------------------------------------------------------------------

## 2. Critical prerequisite: verify and fix time-varying parameter architecture

Before implementing an m-array backend, audit the existing handling of
time-varying covariates and parameters.

### 2.1 Required parameter structure

A genuine time-varying Pradel model needs interval/occasion-specific
parameters:

\[ `\phi`{=tex}*{it},`\qquad `{=tex}p*{it},`\qquad `{=tex}f\_{it}. \]

The architecture should support arrays conceptually equivalent to:

``` python
phi.shape == (n_individuals, n_intervals)
p.shape   == (n_individuals, n_occasions)
f.shape   == (n_individuals, n_intervals)
```

with exact indexing determined by the Pradel likelihood convention.

Do **not** assume that detecting columns such as `tier_2021`,
`tier_2022`, etc. is sufficient. Trace the complete path from formula
parsing → design matrices → linear predictors → transformed parameters →
likelihood and verify that annual covariates actually produce annual
parameter values.

### 2.2 Required tests

Create tests demonstrating that changing a covariate at one occasion
changes only the appropriate parameter(s)/interval contribution.

Test at minimum:

``` text
phi ~ year
p   ~ 1
f   ~ 1
```

``` text
phi ~ tier
p   ~ year
f   ~ 1
```

``` text
phi ~ tier * year
p   ~ year
f   ~ tier * year
```

### 2.3 Tier must not be generically imputed

Annual Tier is a state-like categorical variable.

Do not use generic numeric time-varying-covariate imputation for Tier.

The values have substantive meaning:

``` text
0 = inactive/not registered
1 = Tier I
2 = Tier II
```

Missing values, inactivity, and unavailable pre-implementation Tier
status must be handled explicitly.

------------------------------------------------------------------------

## 3. Preserve the existing individual-history backend

Do not remove or substantially rewrite the current individual Pradel
likelihood.

It is needed as:

1.  a reference implementation;
2.  a regression-testing target;
3.  a fallback for models that cannot be represented by an m-array;
4.  a tool for verifying the new grouped/m-array likelihood on small
    datasets.

The desired architecture is:

``` text
                         ┌──────────────────────┐
                         │ Individual likelihood│
                         │ existing/reference   │
                         └──────────┬───────────┘
                                    │
Data + Formula ── backend selector ─┤
                                    │
                         ┌──────────▼───────────┐
                         │ Grouped/m-array      │
                         │ new/fast backend     │
                         └──────────────────────┘
```

A user-facing API could eventually support something similar to:

``` python
fit_model(
    model=model,
    formula=formula_spec,
    data=data_context,
    backend="individual"
)
```

or:

``` python
fit_model(
    model=model,
    formula=formula_spec,
    data=data_context,
    backend="marray"
)
```

Avoid breaking existing callers.

------------------------------------------------------------------------

## 4. Stage 1: frequency-weighted grouped encounter histories

Before implementing a formal m-array likelihood, implement an exact
grouped-history backend.

This provides a low-risk demonstration of how much computation can be
eliminated by collapsing duplicate likelihood contributions.

### 4.1 Mathematical idea

The current individual likelihood is:

\[ `\ell`{=tex}(`\theta`{=tex}) = `\sum`{=tex}\_{i=1}\^{N}
`\log `{=tex}P(h_i `\mid `{=tex}`\theta`{=tex}). \]

If hunters share identical encounter histories and identical relevant
covariate patterns, collapse them into group (g) with frequency (n_g):

\[ `\ell`{=tex}(`\theta`{=tex}) = `\sum`{=tex}\_{g=1}\^{G} n_g
`\log `{=tex}P(h_g `\mid `{=tex}`\theta`{=tex}). \]

For valid grouping keys, this is **exactly the same likelihood**, not an
approximation.

### 4.2 Grouping rules

Only collapse records when they have identical information required by
the fitted likelihood.

For a simple static model, grouping might include:

``` text
capture_history
gender
first_age class/value as appropriate
tier2_dummy
state if states are pooled
```

For time-varying models, the entire relevant covariate trajectory may
need to be part of the grouping key.

Never collapse across records whose likelihood contributions differ.

### 4.3 Suggested implementation

Add a data structure such as:

``` text
pradel_jax/data/grouped.py
```

with something conceptually like:

``` python
@dataclass
class GroupedDataContext:
    capture_matrix: ...
    covariates: ...
    frequency: ...
    n_unique_records: int
    n_original_individuals: int
```

Add a grouped likelihood that evaluates each unique record once and
multiplies its log-likelihood by its frequency.

### 4.4 Required exact-equivalence tests

For identical starting values, verify:

\[ `\ell`{=tex}*{`\text{individual}`{=tex}}(`\theta`{=tex}) =
`\ell`{=tex}*{`\text{grouped}`{=tex}}(`\theta`{=tex}) \]

within strict numerical tolerance.

After optimization, verify:

-   parameter estimates;
-   log-likelihood;
-   AIC;
-   gradient near optimum;
-   Hessian/SEs where estimable.

Recommended tolerance for log-likelihood comparisons:

``` text
absolute difference < 1e-6
```

or a documented floating-point tolerance justified by JAX precision
settings.

------------------------------------------------------------------------

## 5. Benchmark Stage 1 before proceeding

Benchmark the current individual backend versus the grouped backend.

Use:

1.  simulated small data;
2.  simulated large data;
3.  Nebraska data;
4.  South Dakota data.

Record:

``` text
N original individuals
G unique grouped records
compression ratio N/G
data preparation time
JIT compilation time
optimization time
number of likelihood evaluations
peak memory if practical
log-likelihood
AIC
parameter estimates
convergence status
```

Benchmark at least:

``` text
phi ~ 1
p   ~ 1
f   ~ 1
```

``` text
phi ~ gender
p   ~ 1
f   ~ 1
```

``` text
phi ~ gender + first_age + tier2_dummy
p   ~ 1
f   ~ gender
```

Do not proceed under the assumption that m-arrays will necessarily
outperform JAX for every model. Quantify the gain.

------------------------------------------------------------------------

## 6. Stage 2: formal Pradel m-array backend

After grouped-history equivalence is established, implement a genuine
m-array likelihood.

### 6.1 General structure

A standard forward m-array records, for individuals released/encountered
at occasion (i), the occasion (j) at which they are next encountered,
plus a terminal cell for never encountered again.

Each release row contributes a multinomial likelihood:

\[ `\mathbf{m}`{=tex}\_i `\sim`{=tex} `\operatorname{Multinomial}`{=tex}
(R_i,`\boldsymbol{\pi}`{=tex}\_i). \]

The cell probabilities are functions of apparent survival and encounter
probability.

For (j\>i), the forward probability has the general form:

\[ `\pi`{=tex}\^{F}\_{ij} = `\phi`{=tex}\_i `\left[
\prod_{k=i+1}^{j-1}
\phi_k(1-p_k)
\right]`{=tex}p_j, \]

subject to exact indexing/conditioning conventions verified against the
published Pradel/MARK likelihood.

The terminal probability represents no subsequent encounter.

### 6.2 Reverse-time component

Pradel temporal symmetry obtains recruitment information from the
reverse-time analogue of the survival process.

Using the recruitment parameterization in the current project:

\[ `\lambda`{=tex}\_t=`\phi`{=tex}\_t+f_t \]

and seniority is:

\[ `\gamma`{=tex}\_{t+1} = `\frac{\phi_t}{\lambda_t}`{=tex} =
`\frac{\phi_t}{\phi_t+f_t}`{=tex}. \]

The reverse-time m-array should use (`\gamma`{=tex}) and the appropriate
reverse encounter probabilities.

### 6.3 Important warning

Do **not** simply translate the current individual JAX code into m-array
form and declare it correct.

Independently verify:

-   conditioning;
-   first-capture treatment;
-   first-recapture treatment;
-   tail/never-seen-again probabilities;
-   forward/reverse indexing;
-   relationship among (`\phi`{=tex}), (f), (`\lambda`{=tex}), and
    (`\gamma`{=tex}).

Use the published Pradel temporal-symmetry likelihood and MARK/RMark
behavior as external references.

This is especially important because the project previously encountered
confusion over the recruitment parameterization.

------------------------------------------------------------------------

## 7. Correct Pradel parameterization

The target `Pradrec`-style parameterization for this project is:

\[ `\phi`{=tex}*t =
`\operatorname{logit}`{=tex}\^{-1}(X*{`\phi`{=tex},t}`\beta`{=tex}\_`\phi`{=tex})
\]

\[ p_t = `\operatorname{logit}`{=tex}\^{-1}(X\_{p,t}`\beta`{=tex}\_p) \]

\[ f_t = `\exp`{=tex}(X\_{f,t}`\beta`{=tex}\_f) \]

where (f) is a **per-capita recruitment rate**, not a probability.

Population growth is:

\[ `\boxed{\lambda_t=\phi_t+f_t}`{=tex} \]

and:

\[ `\boxed{\gamma_{t+1}=\frac{\phi_t}{\phi_t+f_t}}`{=tex}. \]

Add explicit tests for these transformations and document them in the
package.

Do not use:

\[ `\lambda`{=tex}=`\frac{\phi}{1-f}`{=tex} \]

for this parameterization.

------------------------------------------------------------------------

## 8. Proposed file organization

Prefer additive modules over major changes to working code.

Suggested structure:

``` text
pradel_jax/
├── data/
│   ├── grouped.py
│   └── marray.py
│
├── models/
│   ├── pradel.py
│   ├── pradel_grouped.py        # optional if separation helps
│   └── pradel_marray.py
│
├── formulas/
│   ├── design_matrix.py
│   └── time_varying.py
│
└── tests/
    ├── test_time_varying_parameters.py
    ├── test_grouped_equivalence.py
    ├── test_marray_construction.py
    ├── test_marray_likelihood.py
    ├── test_pradel_parameterization.py
    ├── test_mark_equivalence.py
    └── test_large_n_benchmark.py
```

Adapt this to the actual repository layout rather than forcing these
exact paths if equivalent modules already exist.

------------------------------------------------------------------------

## 9. Validation ladder

Do not move directly to the full Tier II model.

Validate progressively.

### Level 1: intercept-only

``` text
phi ~ 1
p   ~ 1
f   ~ 1
```

Require agreement among:

``` text
MARK/RMark
individual JAX
grouped JAX
m-array JAX
```

where possible.

### Level 2: simple static covariates

``` text
phi ~ gender
p   ~ 1
f   ~ 1
```

then:

``` text
phi ~ gender + first_age
p   ~ 1
f   ~ gender
```

### Level 3: existing reproduction model

Reproduce the best-supported models from the current Nebraska/South
Dakota reproduction workflow.

Require essentially identical:

-   MLEs;
-   log-likelihood;
-   AIC;
-   derived (`\phi`{=tex}), (f), and (`\lambda`{=tex});
-   standard errors where numerically stable.

### Level 4: time variation

Test:

``` text
phi ~ year
p   ~ 1
f   ~ year
```

then allow:

``` text
p ~ year
```

### Level 5: interactions

Test:

``` text
phi ~ tier2_dummy * year + gender + age
p   ~ year
f   ~ tier2_dummy * year + gender + age
```

### Level 6: annual Tier status

Only after all previous levels pass should annual Tier I/Tier II status
enter the model.

------------------------------------------------------------------------

## 10. Simulation-based validation

Create a simulation framework with known truth.

At minimum simulate scenarios with:

1.  constant (`\phi`{=tex},p,f);
2.  strongly varying (p_t);
3.  varying (`\phi`{=tex}\_t);
4.  varying (f_t);
5.  Tier × year effect on (`\phi`{=tex});
6.  Tier × year effect on (f);
7.  age × Tier effect;
8.  rare Tier II participation;
9.  high churn;
10. low detection;
11. very large (N).

Example truth:

\[ `\phi`{=tex}\_{T1}=0.70 \]

\[ `\phi`{=tex}\_{T2}=0.55 \]

with a known positive Tier-II temporal slope.

Check:

-   bias;
-   RMSE;
-   confidence interval coverage;
-   convergence;
-   Hessian behavior;
-   runtime;
-   scaling with (N).

Distinguish **statistical identifiability** from **computational
failure**. A faster optimizer cannot solve a fundamentally
non-identifiable model.

------------------------------------------------------------------------

## 11. Immediate scientific target: richer two-tier Pradel model

Once the m-array backend is validated, fit models that were previously
difficult or impractical.

A useful initial target is:

\[ `\operatorname{logit}`{=tex}(`\phi`{=tex}\_{it}) = `\beta`{=tex}\_0+
`\beta`{=tex}*1 Tier*{it}+ `\beta`{=tex}\_2 Year_t+
`\beta`{=tex}*3(Tier*{it}`\times `{=tex}Year_t)+ `\beta`{=tex}*4
Age*{it}+ `\beta`{=tex}\_5 Sex_i+ `\beta`{=tex}\_6 State_i. \]

For encounter/detection probability:

\[ `\operatorname{logit}`{=tex}(p\_{it}) = `\alpha`{=tex}\_0+
`\alpha`{=tex}\_1 Year_t+ `\alpha`{=tex}\_2 State_i \]

with additional Tier effects tested if identifiable and scientifically
meaningful.

For recruitment:

\[ `\log`{=tex}(f\_{it}) = `\delta`{=tex}\_0+ `\delta`{=tex}*1
Tier*{it}+ `\delta`{=tex}\_2 Year_t+
`\delta`{=tex}*3(Tier*{it}`\times `{=tex}Year_t)+ `\delta`{=tex}*4
Age*{it}+ `\delta`{=tex}\_5 Sex_i+ `\delta`{=tex}\_6 State_i. \]

Do not automatically fit every possible interaction.

Prioritize biologically/management-motivated terms:

``` text
Tier × Year
Tier × Age
Tier × Sex
Tier × State
```

and consider higher-order interactions only when supported by sample
size, hypotheses, and identifiability.

------------------------------------------------------------------------

## 12. Two definitions of Tier exposure

The project should distinguish two analyses.

### 12.1 Ever Tier II

Define:

``` text
ever_tier2 = 1 if hunter used Tier II at least once
             0 otherwise
```

Advantages:

-   fixed individual-level grouping variable;
-   straightforward grouped/m-array stratification;
-   useful first application and computational benchmark.

Limitation:

`ever_tier2` uses future information to classify earlier years. A hunter
who first becomes Tier II in 2024 is classified as "ever Tier II" during
2021--2023.

Therefore this variable should primarily support a **descriptive
trajectory comparison**, not a causal claim that Tier II caused
subsequent retention.

### 12.2 Annual Tier status

The scientifically preferred exposure is:

\[ Tier\_{it}. \]

For retention, Tier in year (t) should predict persistence/participation
from (t) to (t+1).

This supports questions such as:

> Does Tier II status in year (t) predict subsequent retention after
> accounting for age, sex, state, and year?

and:

> Did the relationship between Tier II participation and subsequent
> retention change from 2021 through 2024?

------------------------------------------------------------------------

## 13. Stage 3: explicit Tier-state extension

Do **not** force the full `0/Tier I/Tier II` history into a standard
binary Pradel m-array if that discards state-transition information.

The available annual histories are:

``` text
0 = inactive
1 = Tier I
2 = Tier II
```

The existing R3 analysis already uses these histories descriptively to
calculate:

-   recruitment;
-   retention;
-   reactivation;
-   churn;
-   Tier I → Tier II transitions;
-   Tier II → Tier I transitions.

The long-term goal is to integrate those processes into an inferential
model.

### 13.1 Desired state-transition quantities

Among active hunters, estimate:

\[ `\psi`{=tex}\^{T1`\rightarrow `{=tex}T1}\_t \]

\[ `\psi`{=tex}\^{T1`\rightarrow `{=tex}T2}\_t \]

\[ `\psi`{=tex}\^{T2`\rightarrow `{=tex}T1}\_t \]

\[ `\psi`{=tex}\^{T2`\rightarrow `{=tex}T2}\_t. \]

Potentially model transition probabilities as functions of:

``` text
year
age
sex
state
prior participation
```

### 13.2 Reactivation

The existing project defines reactivation as return following at least
three years of inactivity.

A future multistate/state-memory extension should investigate whether
returning hunters enter through:

``` text
inactive/lapsed → Tier I
inactive/lapsed → Tier II
```

and whether Tier II disproportionately serves as a re-entry pathway.

### 13.3 Keep this separate from Stage 1/2

Implement this only after the ordinary Pradel grouped/m-array backend is
correct and validated.

A separate model class is preferable, e.g.:

``` text
MultistatePradelModel
```

or another clearly named state-transition model.

Do not silently change the interpretation of the standard `PradelModel`.

------------------------------------------------------------------------

## 14. Questions the final model should be able to address

The improved analysis should ultimately support questions such as:

### Retention

-   Is apparent retention different for Tier I and Tier II hunters?
-   Does that difference change by year?
-   Does the Tier effect differ by age?
-   Does it differ by sex?
-   Does it differ between Nebraska and South Dakota?

### Recruitment

-   Is Tier II associated with higher entry/recruitment?
-   Has Tier-II-associated recruitment changed over the experimental
    period?
-   Which demographic groups are most likely to enter through Tier II?

### Detection/annual participation

-   Is the assumption of constant (p) supported?
-   Does annual participation/encounter probability vary by year?
-   Does it differ by Tier or state?

### Tier transitions

-   What is the probability that a Tier I hunter subsequently chooses
    Tier II?
-   What is the probability that a Tier II hunter subsequently chooses
    Tier I?
-   Which hunters persist in Tier II?
-   Which hunters appear to use Tier II as an entry pathway before
    transitioning to Tier I?

### Reactivation

-   Are lapsed hunters more likely to return through Tier II than Tier
    I?
-   Does Tier II improve subsequent persistence among reactivated
    hunters?

------------------------------------------------------------------------

## 15. Performance targets

Do not set an arbitrary required speedup before benchmarking.

Report performance transparently.

For each backend/model report:

``` text
data-prep time
compile time
optimization time
total time
memory if available
N individuals
G grouped histories / m-array dimensions
number of parameters
number of optimizer evaluations
convergence
log-likelihood
```

The main computational question is:

> How does runtime scale with the number of hunters once the sufficient
> encounter information is compressed?

A particularly useful benchmark would hold the number of occasions and
covariate strata constant while increasing:

``` text
N = 10,000
N = 100,000
N = 1,000,000
```

The grouped/m-array likelihood should become increasingly advantageous
if the number of unique sufficient records grows much more slowly than
(N).

------------------------------------------------------------------------

## 16. Reproduction workflow integration

The supplied reproduction bundle currently uses:

``` text
run_mr_nebraska_sd.R
    ↓ reticulate
pradel_mr_bridge.py
    ↓
pradel-jax
```

Preserve this workflow while adding backend selection.

For example, extend the bridge function from conceptually:

``` python
fit_pradel_model(
    csv_path,
    phi_formula,
    p_formula,
    f_formula
)
```

to something backward-compatible such as:

``` python
fit_pradel_model(
    csv_path,
    phi_formula,
    p_formula,
    f_formula,
    backend="individual"
)
```

Then allow the R reproduction script to benchmark:

``` r
backend = "individual"
backend = "grouped"
backend = "marray"
```

without duplicating the full analysis pipeline.

Do not change the existing output schema unless necessary. Existing
downstream report-processing scripts depend on the model comparison and
confidence-interval CSV formats.

------------------------------------------------------------------------

## 17. Data protection requirements

The Nebraska and South Dakota HIP files contain pseudonymous
hunter-level data.

Maintain the current safeguards:

-   never commit raw hunter-level CSV files;
-   keep `data/*.csv` ignored by git;
-   tests should use simulated or synthetic data;
-   benchmark summaries may report counts and timings but should not
    expose identifiers;
-   no test fixture should contain real hunter IDs or raw records.

------------------------------------------------------------------------

## 18. Documentation requirements

Add documentation explaining:

### Pradel parameters

\[ `\phi `{=tex}= `\text{apparent persistence/retention}`{=tex} \]

\[ p =
`\text{encounter/annual participation probability conditional on being in the population}`{=tex}
\]

\[ f = `\text{per-capita recruitment rate}`{=tex} \]

\[ `\lambda`{=tex}=`\phi`{=tex}+f \]

For hunters, avoid biological language implying literal survival/death.

### Backend choices

Document:

``` text
individual
grouped
marray
```

including when each is mathematically valid.

Explicitly state that arbitrary individual/time-varying covariates
cannot necessarily be collapsed into a conventional m-array without
preserving the required covariate/state trajectory.

### Tier interpretation

Document the distinction between:

``` text
ever_tier2
```

and:

``` text
annual Tier status
```

and warn against interpreting `ever_tier2` as a causal exposure.

------------------------------------------------------------------------

## 19. Do not do these things

Codex should **not**:

1.  Rewrite `pradel-jax` from scratch.
2.  Remove the individual-history likelihood.
3.  Assume the existing time-varying machinery is correct without tests.
4.  Treat Tier as an ordinary numeric covariate and impute it.
5.  Collapse hunters whose likelihood contributions differ.
6.  Assume faster computation solves identifiability.
7.  Change the Pradel parameterization.
8.  Use (`\lambda`{=tex}=`\phi`{=tex}/(1-f)) for the current `Pradrec`
    recruitment parameterization.
9.  Jump directly to the multistate Tier model.
10. Break the existing R/reticulate reproduction workflow.
11. Commit or expose real hunter-level data.
12. Trust the new m-array likelihood solely because it agrees with the
    current JAX implementation; use independent
    MARK/RMark/published-likelihood checks where possible.

------------------------------------------------------------------------

## 20. Recommended implementation sequence

Execute the work in this order.

### Milestone 1 --- Audit

-   Trace formula → design matrix → parameter → likelihood indexing.
-   Verify whether current annual covariates produce annual parameters.
-   Document any mismatch with tests before changing behavior.

### Milestone 2 --- Interval-specific parameters

-   Implement correct (`\phi`{=tex}*{it}), (p*{it}), (f\_{it}) support.
-   Preserve constant/static models.
-   Add regression tests.

### Milestone 3 --- Grouped histories

-   Add exact frequency-weighted grouping.
-   Validate against individual likelihood.
-   Benchmark Nebraska and South Dakota.

### Milestone 4 --- m-array construction

-   Implement forward/reverse array builders.
-   Unit-test array cells against hand-calculated toy histories.

### Milestone 5 --- m-array likelihood

-   Implement multinomial likelihood.
-   Verify conditioning/indexing independently.
-   Compare with individual JAX and MARK/RMark.

### Milestone 6 --- Existing analysis reproduction

-   Refit current 12-model constant-(p) analysis.
-   Confirm estimates/AIC/results.
-   Document runtime improvement.

### Milestone 7 --- Free (p) and add year

Fit progressively:

``` text
phi ~ year
p   ~ year
f   ~ year
```

and simpler nested variants.

Evaluate identifiability and convergence.

### Milestone 8 --- Tier interactions

Fit scientifically motivated models including:

``` text
Tier × Year
Tier × Age
Tier × Sex
Tier × State
```

beginning with `ever_tier2` where necessary for validation.

### Milestone 9 --- Annual Tier

Implement annual Tier as an interval-specific predictor where
mathematically appropriate.

### Milestone 10 --- Multistate extension

Only after all preceding milestones pass, design the explicit Tier
I/Tier II transition model.

------------------------------------------------------------------------

## 21. Definition of done for the first development cycle

The first development cycle is complete when all of the following are
true:

-   [ ] Existing constant models still run unchanged.
-   [ ] Time-varying parameter indexing is tested and correct.
-   [ ] Grouped-history likelihood matches individual-history
    likelihood.
-   [ ] Formal m-arrays can be constructed from toy encounter histories.
-   [ ] m-array likelihood matches known/reference results.
-   [ ] Current Nebraska/South Dakota constant-(p) analysis is
    reproduced.
-   [ ] Runtime benchmarks are documented.
-   [ ] At least one model with year-varying parameters fits
    successfully.
-   [ ] At least one Tier × year model fits successfully.
-   [ ] Tests use only synthetic data.
-   [ ] Existing R/reticulate workflow remains functional.
-   [ ] Documentation explains backend validity and Pradel
    parameterization.

The multistate Tier I/Tier II model is **not** required for the first
development cycle.

------------------------------------------------------------------------

## 22. Desired Codex workflow

Before editing code:

1.  Inspect the repository structure.
2.  Read the current Pradel likelihood in full.
3.  Read formula/design-matrix/time-varying code in full.
4.  Read existing tests and MARK/RMark validation code.
5.  Run the existing test suite.
6.  Run the supplied reproduction workflow if local data are available;
    otherwise use synthetic data.
7.  Produce a short implementation plan tied to actual files/functions.
8.  Make changes milestone-by-milestone.
9.  Run tests after every milestone.
10. Do not proceed past a failed equivalence test.

For each milestone, report:

``` text
files changed
behavior changed
tests added
tests passed/failed
numerical equivalence results
benchmark results
remaining concerns
```

Prefer small reviewable commits/changes over one large refactor.

------------------------------------------------------------------------

## 23. Central scientific objective

The purpose of the computational work is not merely to make an existing
model faster.

The goal is to make a more appropriate model of the two-tier experiment
computationally feasible.

The existing analysis separates:

``` text
binary Pradel histories
        ↓
phi, p, f, lambda
```

from:

``` text
0 / Tier I / Tier II histories
        ↓
descriptive recruitment
retention
reactivation
churn
tier transitions
```

The long-term objective is to reduce that separation.

The desired analysis should eventually explain **how hunters enter,
persist, lapse, reactivate, and move between Tier I and Tier II**, how
those processes change over time, and how they differ among demographic
groups and states.

The grouped/m-array Pradel implementation is the first computational
step toward that objective.
