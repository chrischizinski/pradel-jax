# Load packages
library(RMark)
library(tidyverse)

# -------------------------------------------------------------------------
# Load data

# Filter out bad ages
wf.dat <- wf.dat %>% filter(age > 0)

# Create ID column
wf.dat$id <- 1:nrow(wf.dat)

# -------------------------------------------------------------------------
# Process data for Pradel recruitment model
wf.pro <- process.data(wf.dat,
                       model = "Pradrec",
                       begin.time = 2016,
                       groups = c("gender"))

# Create design data
wf.ddl <- make.design.data(wf.pro)
wf.ddl$Phi$id <- as.numeric(rownames(wf.ddl$Phi))
wf.ddl$p$id   <- as.numeric(rownames(wf.ddl$p))
wf.ddl$f$id   <- as.numeric(rownames(wf.ddl$f))

# Add 'tier' to design data
for (year in 2016:2024) {
  year_col <- paste0("tier", year)
  
  phi_rows <- wf.ddl$Phi$time == year
  p_rows   <- wf.ddl$p$time == year
  f_rows   <- wf.ddl$f$time == year
  
  wf.ddl$Phi$tier[phi_rows] <- wf.dat[[year_col]][wf.ddl$Phi$id[phi_rows]]
  wf.ddl$p$tier[p_rows]     <- wf.dat[[year_col]][wf.ddl$p$id[p_rows]]
  wf.ddl$f$tier[f_rows]     <- wf.dat[[year_col]][wf.ddl$f$id[f_rows]]
}

# -------------------------------------------------------------------------
# Define covariate models

# Survival (Phi)
Phi.1  <- list(formula = ~1)
Phi.2  <- list(formula = ~time)
Phi.3  <- list(formula = ~gender)
Phi.4  <- list(formula = ~age)
Phi.5  <- list(formula = ~tier)
Phi.6  <- list(formula = ~gender + age)
Phi.7  <- list(formula = ~gender + time)
Phi.8  <- list(formula = ~age + time)
Phi.9  <- list(formula = ~gender + tier)
Phi.10 <- list(formula = ~age + tier)
Phi.11 <- list(formula = ~time + tier)
Phi.12 <- list(formula = ~gender + age + time)
Phi.13 <- list(formula = ~gender + age + tier)
Phi.14 <- list(formula = ~gender + time + tier)
Phi.15 <- list(formula = ~age + time + tier)
Phi.16 <- list(formula = ~gender + age + time + tier)

# Detection (p)
p.1  <- list(formula = ~1)
p.2  <- list(formula = ~time)
p.3  <- list(formula = ~gender)
p.4  <- list(formula = ~age)
p.5  <- list(formula = ~tier)
p.6  <- list(formula = ~gender + age)
p.7  <- list(formula = ~gender + time)
p.8  <- list(formula = ~age + time)
p.9  <- list(formula = ~gender + tier)
p.10 <- list(formula = ~age + tier)
p.11 <- list(formula = ~time + tier)
p.12 <- list(formula = ~gender + age + time)
p.13 <- list(formula = ~gender + age + tier)
p.14 <- list(formula = ~gender + time + tier)
p.15 <- list(formula = ~age + time + tier)
p.16 <- list(formula = ~gender + age + time + tier)

# Recruitment (f)
f.1  <- list(formula = ~1)
f.2  <- list(formula = ~time)
f.3  <- list(formula = ~gender)
f.4  <- list(formula = ~age)
f.5  <- list(formula = ~tier)
f.6  <- list(formula = ~gender + age)
f.7  <- list(formula = ~gender + time)
f.8  <- list(formula = ~age + time)
f.9  <- list(formula = ~gender + tier)
f.10 <- list(formula = ~age + tier)
f.11 <- list(formula = ~time + tier)
f.12 <- list(formula = ~gender + age + time)
f.13 <- list(formula = ~gender + age + tier)
f.14 <- list(formula = ~gender + time + tier)
f.15 <- list(formula = ~age + time + tier)
f.16 <- list(formula = ~gender + age + time + tier)

# -------------------------------------------------------------------------
# Create all combinations
wf.model.list <- create.model.list("Pradrec")

# Run models (you may want to test first with fewer models)
wf.results <- mark.wrapper(wf.model.list,
                           data = wf.pro,
                           ddl = wf.ddl,
                           external = TRUE,
                           delete = TRUE,
                           accumulate = TRUE)
