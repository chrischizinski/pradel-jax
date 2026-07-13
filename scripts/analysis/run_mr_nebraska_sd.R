# =========================================================================
# Pradel Mark-Recapture Analysis — Nebraska & South Dakota
# =========================================================================
#
# Standalone R script. tidyverse handles data prep and result wrangling;
# reticulate drives the Python pradel-jax package (JAX autodiff/Hessian SEs
# only exist on the Python side, so the actual model fitting always happens
# there — this script is an orchestration/reporting layer around it).
#
# Reproduces the candidate model set and data-prep rules used in the
# corrected-likelihood re-fit (see hip-matches-data-lineage /
# pradel-correction-and-report-impact project notes):
#   - capture history (ch) is REBUILT from the tier_2016..2024 columns for
#     BOTH states, ignoring the file's own `ch` column (Nebraska's `ch` is
#     corrupted — leading zeros stripped on ~25% of rows).
#   - gender coded M=1/F=0; UNKNOWN/NA gender dropped.
#   - first_age = age_2016, z-standardized (raw age overflows the logit).
#   - tier2_dummy = ever tiered into tier 2 (2016-2024).
#   - detection probability p held constant (Constant-p approach) to keep
#     the model identifiable.
#
# Output: two CSVs per run, in the format `process_pradel_results.R`
# (two-tier-book repo) expects:
#   {state}_constant_p_constant_p_analysis_{timestamp}.csv   (model comparison)
#   confidence_interval_summary_{timestamp}.csv               (best-model CIs)
# Point `pradel_dir` in that script at OUTPUT_DIR (below) to pick these up.
#
# =========================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(reticulate)
})

# -------------------------------------------------------------------------
# Configuration — edit these paths if your checkout layout differs
# -------------------------------------------------------------------------

REPO_ROOT <- "/Users/cchizinski2/gitlab/student_work/ava_britton/pradel-jax"

PYTHON_VENV <- file.path(REPO_ROOT, "pradel_env", "bin", "python")
BRIDGE_PY   <- file.path(REPO_ROOT, "scripts", "analysis", "pradel_mr_bridge.py")
OUTPUT_DIR  <- file.path(REPO_ROOT, "results", "mr_analysis")

DATA_FILES <- list(
  Nebraska     = "/Users/cchizinski2/gitlab/student_work/ava_britton/hip_matches/data/final_output/20250904_ne_hip_tier_data.csv",
  `South Dakota` = "/Users/cchizinski2/gitlab/student_work/ava_britton/hip_matches/data/final_output/20250903_sd_hip_tier_data.csv"
)

STATE_FILE_TAG <- c("Nebraska" = "nebraska", "South Dakota" = "south_dakota")

# Candidate model set (matches the reported constant-p model comparison).
MODELS <- tribble(
  ~model,                       ~phi,                                    ~p,   ~f,
  "Intercept-only",             "~1",                                    "~1", "~1",
  "Gender on phi",               "~1 + gender",                          "~1", "~1",
  "Age on phi",                  "~1 + first_age",                       "~1", "~1",
  "Tier2 on phi",                "~1 + tier2_dummy",                     "~1", "~1",
  "Gender + Age on phi",         "~1 + gender + first_age",              "~1", "~1",
  "Gender + Tier2 on phi",       "~1 + gender + tier2_dummy",            "~1", "~1",
  "Age + Tier2 on phi",          "~1 + first_age + tier2_dummy",         "~1", "~1",
  "All covariates on phi",       "~1 + gender + first_age + tier2_dummy","~1", "~1",
  "Gender on f",                 "~1",                                    "~1", "~1 + gender",
  "Age on f",                    "~1",                                    "~1", "~1 + first_age",
  "Gender on phi + f",           "~1 + gender",                          "~1", "~1 + gender",
  "Full phi + Gender on f",      "~1 + gender + first_age + tier2_dummy","~1", "~1 + gender",
)

# -------------------------------------------------------------------------
# Data prep (tidyverse) — mirrors the validated prep from the corrected
# likelihood re-fit. Do not "fix" first_age filtering/order: raw age must
# be filtered to > 0 BEFORE standardizing, or the mean/sd get skewed by
# bad rows.
# -------------------------------------------------------------------------

prepare_state_data <- function(path) {
  df <- read_csv(path, col_types = cols(.default = "c"))

  tier_cols <- names(df) %>% keep(~ str_starts(.x, "tier_2"))

  tier_num <- df %>%
    select(all_of(tier_cols)) %>%
    mutate(across(everything(), ~ suppressWarnings(as.numeric(.x)))) %>%
    mutate(across(everything(), ~ replace_na(.x, 0)))

  ch <- tier_num %>%
    mutate(across(everything(), ~ if_else(.x > 0, "1", "0"))) %>%
    reduce(paste0)

  tier2_dummy <- as.integer(rowSums(tier_num == 2, na.rm = TRUE) > 0)

  n0 <- nrow(df)

  out <- tibble(
    ch = ch,
    gender = recode(df$gender, "M" = 1, "F" = 0, .default = NA_real_),
    first_age = suppressWarnings(as.numeric(df$age_2016)),
    tier2_dummy = tier2_dummy
  ) %>%
    filter(!is.na(gender), !is.na(first_age), first_age > 0) %>%
    mutate(
      gender = as.integer(gender),
      # z-standardize age: raw age (~11-80) as a logit predictor overflows
      # eta and breaks the fit. Standardizing only rescales the age
      # coefficient — it does not change the model, log-likelihood, or AIC.
      first_age = as.numeric(scale(first_age))
    )

  list(data = out, n_dropped = n0 - nrow(out))
}

# -------------------------------------------------------------------------
# Python bridge setup
# -------------------------------------------------------------------------

if (!file.exists(PYTHON_VENV)) {
  stop("pradel-jax virtualenv python not found at: ", PYTHON_VENV,
       "\nRun ./quickstart.sh in the pradel-jax repo first.")
}
use_python(PYTHON_VENV, required = TRUE)
source_python(BRIDGE_PY)  # provides fit_pradel_model(csv_path, phi_formula, p_formula, f_formula)

# -------------------------------------------------------------------------
# Fit one state: all candidate models, then best-model parameter table
# -------------------------------------------------------------------------

fit_state <- function(path, state_name) {
  message(glue::glue("\n===== {state_name}: preparing data ====="))
  prep <- prepare_state_data(path)
  sub  <- prep$data
  n    <- nrow(sub)
  message(glue::glue("  n = {n} (dropped {prep$n_dropped} for UNKNOWN gender / missing age)"))

  tmp_csv <- tempfile(fileext = ".csv")
  sub %>%
    mutate(individual_id = row_number(), .before = 1) %>%
    write_csv(tmp_csv)
  on.exit(unlink(tmp_csv), add = TRUE)

  message(glue::glue("  fitting {nrow(MODELS)} candidate models..."))
  fits <- MODELS %>%
    pmap(function(model, phi, p, f) {
      res <- fit_pradel_model(tmp_csv, phi, p, f)
      res$model <- model
      res$phi_formula <- phi
      res$p_formula <- p
      res$f_formula <- f
      res
    })
  names(fits) <- MODELS$model

  comparison <- fits %>%
    map_dfr(function(r) {
      tibble(
        state = state_name,
        model = r$model,
        phi_formula = r$phi_formula,
        p_formula = r$p_formula,
        f_formula = r$f_formula,
        log_likelihood = r$log_likelihood %||% NA_real_,
        aic = r$aic %||% NA_real_,
        n_parameters = r$n_parameters %||% NA_integer_,
        fit_time = r$fit_time %||% NA_real_,
        success = r$success,
        strategy_used = r$optimizer_used %||% NA_character_,
        parameters = if (is.null(r$parameters)) NA_character_
                     else paste0("[", paste(unlist(r$parameters), collapse = ", "), "]"),
        error = r$error %||% NA_character_
      )
    }) %>%
    mutate(bic = -2 * log_likelihood + n_parameters * log(n)) %>%
    relocate(bic, .after = aic) %>%
    arrange(aic)

  failed <- comparison %>% filter(!success)
  if (nrow(failed) > 0) {
    message(glue::glue("  [FAIL] {failed$model}: {failed$error}"))
  }

  ok <- comparison %>% filter(success, !is.na(aic))
  if (nrow(ok) == 0) stop(glue::glue("All candidate models failed for {state_name}"))

  best_row   <- ok %>% slice_min(aic, n = 1)
  best_model <- best_row$model[1]
  best_fit   <- fits[[best_model]]

  message(glue::glue("  BEST: {best_model}  (AIC = {round(best_row$aic[1], 1)})"))

  se <- best_fit$parameter_se
  ci <- tibble(
    state = state_name,
    model = best_model,
    # pradel-jax names intercepts "phi_(Intercept)"; normalize to
    # "phi_intercept" to match the convention process_pradel_results.R
    # (two-tier-book repo) filters on for the lambda table.
    parameter = str_replace(unlist(best_fit$parameter_names), "\\(Intercept\\)$", "intercept"),
    estimate = unlist(best_fit$parameters),
    std_error = if (is.null(se)) NA_real_ else unlist(se),
    aic = best_fit$aic,
    log_likelihood = best_fit$log_likelihood
  ) %>%
    mutate(
      ci_lower = estimate - 1.96 * std_error,
      ci_upper = estimate + 1.96 * std_error
    ) %>%
    relocate(ci_lower, ci_upper, .after = std_error) %>%
    relocate(aic, log_likelihood, .after = ci_upper)

  list(comparison = comparison, ci = ci, n = n, best_model = best_model)
}

# -------------------------------------------------------------------------
# Run both states
# -------------------------------------------------------------------------

results <- imap(DATA_FILES, fit_state)

all_comparison <- map_dfr(results, "comparison")
all_ci         <- map_dfr(results, "ci")

# -------------------------------------------------------------------------
# Write output CSVs (format expected by process_pradel_results.R)
# -------------------------------------------------------------------------

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
timestamp <- as.integer(Sys.time())

iwalk(results, function(res, state_name) {
  tag <- STATE_FILE_TAG[[state_name]]
  out_path <- file.path(OUTPUT_DIR, glue::glue("{tag}_constant_p_constant_p_analysis_{timestamp}.csv"))
  write_csv(res$comparison, out_path)
  message(glue::glue("Wrote {out_path}"))
})

ci_path <- file.path(OUTPUT_DIR, glue::glue("confidence_interval_summary_{timestamp}.csv"))
write_csv(all_ci, ci_path)
message(glue::glue("Wrote {ci_path}"))

# -------------------------------------------------------------------------
# Console summary: lambda = phi + f for each state's best model
# (phi on logit link, f on log link — Pradel 1996 / MARK "Pradrec")
# -------------------------------------------------------------------------

lambda_summary <- all_ci %>%
  filter(parameter %in% c("phi_intercept", "f_intercept")) %>%
  select(state, model, parameter, estimate, std_error) %>%
  pivot_wider(names_from = parameter, values_from = c(estimate, std_error)) %>%
  mutate(
    phi = plogis(estimate_phi_intercept),
    f = exp(estimate_f_intercept),
    lambda = phi + f
  ) %>%
  select(state, model, phi, f, lambda)

message("\n===== Lambda summary (reference individual, best model) =====")
print(lambda_summary, n = Inf)
