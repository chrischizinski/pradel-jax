# Fit the reduced model with marked and store what it finds.
#
#   Rscript tests/validation/marked_crosscheck/fit_marked.R
#
# Writes marked_results.json next to this script. The Python test compares
# pradel-jax against that file, so CI needs no R.
#
# Multistate CJS with strata A (Tier I), B (Tier II), C (sitting out). C is
# unobservable, A and B are seen with certainty -- registration is complete,
# the same assumption the five-state model makes. S is common to all strata.
# Psi is saturated: six free transitions, the same parameter space as the
# pradel-jax reduced model, so the two maxima must coincide.
#
# Needs marked (tested with 1.2.8) and TMB. marked's default MSCJS fitter
# needs ADMB; use.tmb = TRUE avoids it. nlminb is used because the default
# optim stopped with a gradient of ~0.07, too loose for this comparison.

suppressMessages({
  library(marked)
  library(jsonlite)
})

here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
ch <- read.csv(file.path(here, "histories.csv"), colClasses = "character")

proc <- process.data(ch, model = "MSCJS", strata.labels = c("A", "B", "C"))
ddl <- make.design.data(proc)
ddl$p$fix <- ifelse(ddl$p$stratum == "C", 0, 1)

fit <- crm(
  proc, ddl,
  model.parameters = list(
    S = list(formula = ~1),
    p = list(formula = ~1),
    Psi = list(formula = ~ -1 + stratum:tostratum)
  ),
  hessian = FALSE, use.tmb = TRUE, method = "nlminb",
  control = list(eval.max = 5000, iter.max = 5000), silent = TRUE
)
stopifnot(fit$results$convergence == 0)

# predict() fails on this fit in marked 1.2.8, so the reals are built from the
# betas: Psi is a multinomial logit per origin stratum, staying put is the
# reference cell.
beta <- fit$results$beta$Psi
psi <- list()
for (from in c("A", "B", "C")) {
  to <- setdiff(c("A", "B", "C"), from)
  odds <- exp(beta[paste0("stratum", from, ":tostratum", to)])
  probabilities <- c(odds, 1) / (1 + sum(odds))
  psi[[from]] <- as.list(setNames(probabilities, c(to, from)))
}

results <- list(
  marked_version = as.character(packageVersion("marked")),
  loglik = -fit$results$neg2lnl / 2,
  S = unname(plogis(fit$results$beta$S)),
  psi = psi
)
write(toJSON(results, digits = NA, auto_unbox = TRUE, pretty = TRUE),
      file.path(here, "marked_results.json"))
cat("wrote", file.path(here, "marked_results.json"), "\n")
