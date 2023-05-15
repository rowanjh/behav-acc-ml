# ~~~~~~~~~~~~~~~ Script overview ---------------------------------------
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#'
#' https://github.com/rowanjh/behav-acc-ml
#' 
#' Purpose: 
#'      This script runs a grid of random forest models. All models can be run 
#'      at once, or the task can be broken up into smaller parts (see notes)
#'
#' Notes: 
#'      We recommend running from command line using Rscript for stability e.g.:
#'          Rscript scripts/rf-runCV.R              (runs all models)
#'          Rscript scripts/rf-runCV.R - lsio       (only runs LSIO models)
#' 
#'      By default this script runs all models. Accepted command-line args include:
#'          -lsio (only runs LSIO models)
#'          -timestrat (only runs timestrat models)
#'          -randstrat (only runs random-stratified models) 
#' 
#' Date Created: 2022-10-04
#'
#' Outputs: 
#'      Predictions and model metrics for each validation fold are exported to:
#'      ./outputs/rf-results/{datetime}/
#'      
# ~~~~~~~~~~~~~~~ 1. Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
# ---- Start logging ---
log_start_time <- Sys.time()
start_time_string <- format(log_start_time, "%Y-%m-%d_%H%M")
print(paste0("Starting Random Forests. Start time: ", start_time_string))

# ---- Packages ---
#' Package colino requires FCBF package installed from Bioconductor
#' Installation instructions:
#'   install.packages("BiocManager")
#'   BiocManager::install("FCBF")
#'   remotes::install_github("stevenpawley/colino")

message("--------SETUP--------")
suppressMessages({
    library(here)
    library(tidymodels)
    library(themis)
    library(FCBF)
    library(colino)
    library(doParallel)
    library(glue)
    library(data.table)
})
source(here("scripts", "r", "rf-helpers.R"))

# ---- Parse command line arguments (if any) ---
myargs <- commandArgs(trailingOnly=TRUE)
if("-randstrat" %in% tolower(myargs)){
    run_randstrat <- TRUE
    message(glue("Running randstrat models"))
} else run_randstrat <- FALSE
if ("-lsio" %in% tolower(myargs)){
    run_LSIO <- TRUE
    message(glue("Running LSIO models"))
} else run_LSIO <- FALSE
if("-timesplit" %in% tolower(myargs)){
    run_timesplit <- TRUE
    message(glue("Running timesplit models"))
} else run_timesplit <- FALSE

if(!run_randstrat & !run_LSIO & !run_timesplit){
    run_randstrat <- TRUE
    run_LSIO <- TRUE
    run_timesplit <- TRUE
    message(glue("Running all randstrat, LSIO, and timesplit models"))
}

# ---- Script inputs ---
path_windowed_data  <- here("data", "windowed", "windowed-data.csv") 
lsio_fold_spec        <- here("config", "lsio-folds.csv")
timesplit_fold_spec <- here("config", "timesplit-folds.csv")

# ---- Parameters ---
n_CPU_cores <- min(10, parallel::detectCores())

# ---- Load and prep dataset ---
dat <- fread(path_windowed_data, data.table = FALSE)

# Subset to required variables
dat <- dat %>% select(-matches("^beh_"))

# Create variable for rand-strat splitting
dat <- dat %>% unite(split_var, ruff_id, transition, 
                     majority_behaviour, remove = FALSE)

# Convert outcome variable to factor for tidymodels
dat <- dat %>% mutate(majority_behaviour = factor(majority_behaviour))

# CV folds for LSIO and timestrat were specified by the researchers, import spec
fold_spec_LSIO <- read.csv(lsio_fold_spec) |> 
    rename(LSIO_fold = fold)
fold_spec_time <- read.csv(timesplit_fold_spec) |> 
    rename(timesplit_fold = fold)

dat <- left_join(dat, fold_spec_LSIO, by = "ruff_id")
dat <- left_join(dat, fold_spec_time, by = c("recording_id", "window_id"))
out_dir <- here("outputs", "rf-results", start_time_string)
if(!dir.exists(out_dir)) 
    dir.create(out_dir, recursive = TRUE)

message(glue("Starting Analysis. \nSaving output to {out_dir}\n"))

# ~~~~~~~~~~~~~~~ 2. Prepare models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
message("--------PREPARE PREPROCESSORS & MODELS--------")

# ---- Split data ---
# note: "too little data to stratify" warning can be ignored
if(run_LSIO){
    LSIO_cvfolds <- custom_cv_split(dat, split_id_col = "LSIO_fold")
    # For models excluding all transitions from train & val folds
    LSIO_cvfolds_notrans <- custom_cv_split(dat |> filter(!transition), 
                                            split_id_col = "LSIO_fold")
}
if(run_timesplit){
    timesplit_cvfolds <- dat %>% 
        filter(timesplit_fold != "test") %>%
        custom_cv_split(split_id_col = "timesplit_fold")
    # For models excluding all transitions from train & val folds
    timesplit_cvfolds_notrans <- dat %>% 
        filter(timesplit_fold != "test",
               !transition) |>
        custom_cv_split(split_id_col = "timesplit_fold")
}

if(run_randstrat){
    # Create train-test split
    set.seed(123)
    strat_split <- initial_split(dat, strata = 'split_var', prop = 0.7)
    strat_train <- training(strat_split)
    
    # Create 10 cv folds from training dataset
    set.seed(2106)
    strat_cvfolds <- vfold_cv(strat_train, v = 10, strata = "split_var")
    
    # Save fold assignments for reproducibility
    # Get test set assignments
    strat_assignments <- strat_split$data |>
        slice(-strat_split$in_id) |>
        select(recording_id, window_id, window_start) %>%
        mutate(fold = "test")
    
    # Get cv fold assignments
    for (i in 10:1){
        strat_assignments <- strat_cvfolds$splits[[i]]$data |>
            slice(-strat_cvfolds$splits[[i]]$in_id) |>
            select(recording_id, window_id, window_start) |>
            mutate(fold = strat_cvfolds$id[i]) |>
            bind_rows(strat_assignments)
    }
    path <- file.path(out_dir, "randstrat_fold_assignments.csv")
    write.csv(strat_assignments, path, row.names = FALSE)
    
    # Get splits for randstrat with transitions excluded from train and val sets
    strat_cvfolds_notrans <- left_join(dat, strat_assignments) |>
        filter(!grepl("test", fold),
               !transition) |>
        custom_cv_split(split_id_col = "fold")
}


# # ######################################################################
# # Simulated dummy dataset for testing/development
# dat_orig <- dat
# behs <- unique(dat$majority_behaviour)
# folds <- unique(dat$LSIO_fold)
# nsamples <- length(behs) * length(folds) * 3
# 
# # Replace data with development dat
# dat <- dat_orig[1:nsamples,]
# dat$majority_behaviour <- factor(rep(behs, times = ceiling(nsamples/length(behs)))[1:nsamples])
# # dat[,20:400] <- NULL
# dat <- dat %>% select('recording_id':'morph', mixed, LSIO_fold)
# # unbalance the classes so SMOTE doesn't break
# for (i in 1:5){
#     idx <- which(dat$majority_behaviour == unique(dat$majority_behaviour)[1])
#     dat <- bind_rows(dat, dat[idx,])
# }
# nsamples <- nrow(dat)
# dat$LSIO_fold <- rep(folds, each = ceiling(nsamples/length(folds)))[1:nsamples]
# dat$transition <- sample(0:1, nsamples, TRUE, c(0.7, 0.3))
# dat$morph <- sample(LETTERS[1:3], nsamples, TRUE, c(0.33, 0.33, 0.33))
# 
# dat <- dat %>% unite(split_var, recording_id, transition,
#                      majority_behaviour, remove = FALSE)
# 
# # Create some features with perfect predictive info, then add noise
# 
# dat$accX.mean <- as.numeric(dat$majority_behaviour)
# dat$accY.mean <- as.numeric(dat$majority_behaviour)
# dat$accX.mean[dat$accX.mean[sample(1:nrow(dat), 50)]] <- 4
# dat$accY.mean[dat$accY.mean[sample(1:nrow(dat), 50)]] <- 4
# dat$mixed <- sample(0:1, nsamples, TRUE, c(0.7, 0.3))
# 
# strat_split <- initial_split(dat, strata = 'split_var', prop = 0.7)
# strat_train <- training(strat_split)
# strat_cvfolds <- vfold_cv(strat_train, v = 10, strata = "split_var")
# LSIO_cvfolds <- custom_cv_split(dat, split_id_col = "LSIO_fold")
# # ######################################################################

# ~~~~~~~~~~~~~~~ 3. Prepare model workflows ~~~~~~~~~~~~~~~~~~~~~~~----
#' Preprocessing recipes vary the following:
#'   - Transitions included (_Tin) or excluded (Tout) from validation fold
#'   - SMOTE upsampling levels (_sm)
   
# Transitions included in train and validation sets
rec_stem_Tin <- 
    recipe(majority_behaviour ~ ., data = head(dat)) %>%
    update_role(split_var, recording_id, ruff_id, segment_id, win_in_segment_id, 
                window_id, loop_j, window_start, window_end, transition, mixed, 
                LSIO_fold, timesplit_fold, morph, new_role = "ID") %>%
    step_select_fcbf(all_predictors(), threshold = 0.0035) %>%
    step_normalize(all_numeric_predictors())# %>%
    #step_dummy(all_nominal_predictors(), id = "dummy_morph")

# Transitions excluded from train set
rec_stem_Tout <- 
    recipe(majority_behaviour ~ ., data = head(dat)) %>%
    update_role(split_var, recording_id, ruff_id, segment_id, win_in_segment_id, 
                window_id, loop_j, window_start, window_end, transition, mixed, 
                LSIO_fold, timesplit_fold, morph, new_role = "ID") %>%
    step_filter(transition == 0) %>%
    step_select_fcbf(all_predictors(), threshold = 0.01) %>%
    step_normalize(all_numeric_predictors())# %>%
    #step_dummy(all_nominal_predictors(), id = "dummy_morph")

# Add upsampling levels from 0 (none) to 1 (equal sample size for all classes)
rec_sm0_Tin <- rec_stem_Tin
rec_sm07_Tin <- rec_stem_Tin %>%
    step_smote(majority_behaviour, over_ratio = 0.07, seed = 231)
rec_sm5_Tin <- rec_stem_Tin %>%
    step_smote(majority_behaviour, over_ratio = 0.5, seed = 231)
rec_sm10_Tin <- rec_stem_Tin %>%
    step_smote(majority_behaviour, over_ratio = 1.0, seed = 231)

rec_sm0_Tout <- rec_stem_Tout
rec_sm07_Tout <- rec_stem_Tout %>%
    step_smote(majority_behaviour, over_ratio = 0.07, seed = 231)
rec_sm5_Tout <- rec_stem_Tout %>%
    step_smote(majority_behaviour, over_ratio = 0.5, seed = 231)
rec_sm10_Tout <- rec_stem_Tout %>%
    step_smote(majority_behaviour, over_ratio = 1.0, seed = 231)

# ---- Specify model(s) ---
rf_spec_cv <-
    rand_forest(trees=1000, mtry=tune(), min_n=tune()) %>% # tune allows us to try and compare different values for parameters
    set_mode("classification") %>%
    set_engine("randomForest")

# ---- Prepare tidymodel workflows ---
wf_sm0_Tin <- workflow(rec_sm0_Tin, rf_spec_cv)
wf_sm07_Tin <- workflow(rec_sm07_Tin, rf_spec_cv)
wf_sm5_Tin <- workflow(rec_sm5_Tin, rf_spec_cv)
wf_sm10_Tin <- workflow(rec_sm10_Tin, rf_spec_cv)
wf_sm0_Tout <- workflow(rec_sm0_Tout, rf_spec_cv)
wf_sm07_Tout <- workflow(rec_sm07_Tout, rf_spec_cv)
wf_sm5_Tout <- workflow(rec_sm5_Tout, rf_spec_cv)
wf_sm10_Tout <- workflow(rec_sm10_Tout, rf_spec_cv)

# ---- Create hyperparameter tuning grid ---
rf_grid <- crossing(mtry = c(1,2,4,8),
                    min_n = c(1,2))

# ---- Create metric set ---
my_metrics <- metric_set(accuracy, roc_auc, f_meas, sens, spec, precision, mcc)
    
# ~~~~~~~~~~~~~~~ 4. Run cross-validation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
## Save analysis settings
# Text output: data path, rf model spec, grid spec.
outpath <- file.path(out_dir, "parameters.txt")

capture.output(file = outpath,
cat(glue('
====================================================
Log for cross-validation. More details in subfolders
====================================================
Analysis start time: {start_time_string}

====================================================
Analyses requested
====================================================
{paste("LSIO cv requested: ", run_LSIO, "\n")}
{paste("randstrat cv requested: ", run_randstrat, "\n")}
{paste("timestrat cv requested: ", run_timesplit, "\n")}

====================================================
Model Specification
====================================================\n
')),
rf_spec_cv,
cat("
====================================================
Tuning grid
====================================================\n"),
rf_grid)


message("--------START CV--------")
# Run stratified models
if(run_randstrat){
    message ("Running strat models")
    run_CV(wf_sm0_Tin, strat_cvfolds, rf_grid, my_metrics, out_dir, "strat_sm0_Tin")
    run_CV(wf_sm07_Tin, strat_cvfolds, rf_grid, my_metrics, out_dir, "strat_sm07_Tin")
    run_CV(wf_sm5_Tin, strat_cvfolds, rf_grid, my_metrics, out_dir, "strat_sm5_Tin")
    run_CV(wf_sm10_Tin, strat_cvfolds, rf_grid, my_metrics, out_dir, "strat_sm10_Tin")
    run_CV(wf_sm0_Tout, strat_cvfolds, rf_grid, my_metrics, out_dir, "strat_sm0_Tout")
    run_CV(wf_sm07_Tout, strat_cvfolds, rf_grid, my_metrics, out_dir, "strat_sm07_Tout")
    run_CV(wf_sm5_Tout, strat_cvfolds, rf_grid, my_metrics, out_dir, "strat_sm5_Tout")
    run_CV(wf_sm10_Tout, strat_cvfolds, rf_grid, my_metrics, out_dir, "strat_sm10_Tout")
    # Transitions are pre-filtered from dataset, so the Tout recipe is not needed
    run_CV(wf_sm0_Tin, strat_cvfolds_notrans, rf_grid, my_metrics, out_dir, "strat_sm0_ToutAll")
    run_CV(wf_sm07_Tin, strat_cvfolds_notrans, rf_grid, my_metrics, out_dir, "strat_sm07_ToutAll")
    run_CV(wf_sm5_Tin, strat_cvfolds_notrans, rf_grid, my_metrics, out_dir, "strat_sm5_ToutAll")
    run_CV(wf_sm10_Tin, strat_cvfolds_notrans, rf_grid, my_metrics, out_dir, "strat_sm10_ToutAll")
}

# Run LSIO models
if(run_LSIO){
    message ("Running LSIO models")
    run_CV(wf_sm0_Tin, LSIO_cvfolds, rf_grid, my_metrics, out_dir, "LSIO_sm0_Tin")
    run_CV(wf_sm07_Tin, LSIO_cvfolds, rf_grid, my_metrics, out_dir, "LSIO_sm07_Tin")
    run_CV(wf_sm5_Tin, LSIO_cvfolds, rf_grid, my_metrics, out_dir, "LSIO_sm5_Tin")
    run_CV(wf_sm10_Tin, LSIO_cvfolds, rf_grid, my_metrics, out_dir, "LSIO_sm10_Tin")
    run_CV(wf_sm0_Tout, LSIO_cvfolds, rf_grid, my_metrics, out_dir, "LSIO_sm0_Tout")
    run_CV(wf_sm07_Tout, LSIO_cvfolds, rf_grid, my_metrics, out_dir, "LSIO_sm07_Tout")
    run_CV(wf_sm5_Tout, LSIO_cvfolds, rf_grid, my_metrics, out_dir, "LSIO_sm5_Tout")
    run_CV(wf_sm10_Tout, LSIO_cvfolds, rf_grid, my_metrics, out_dir, "LSIO_sm10_Tout")
    # Transitions are pre-filtered from dataset, so the Tout recipe is not needed
    run_CV(wf_sm0_Tin, LSIO_cvfolds_notrans, rf_grid, my_metrics, out_dir, "LSIO_sm0_ToutAll")
    run_CV(wf_sm07_Tin, LSIO_cvfolds_notrans, rf_grid, my_metrics, out_dir, "LSIO_sm07_ToutAll")
    run_CV(wf_sm5_Tin, LSIO_cvfolds_notrans, rf_grid, my_metrics, out_dir, "LSIO_sm5_ToutAll")
    run_CV(wf_sm10_Tin, LSIO_cvfolds_notrans, rf_grid, my_metrics, out_dir, "LSIO_sm10_ToutAll")
}

if(run_timesplit){
    message ("Running timesplit models")
    run_CV(wf_sm0_Tin, timesplit_cvfolds, rf_grid, my_metrics, out_dir, "timesplit_sm0_Tin")
    run_CV(wf_sm07_Tin, timesplit_cvfolds, rf_grid, my_metrics, out_dir, "timesplit_sm07_Tin")
    run_CV(wf_sm5_Tin, timesplit_cvfolds, rf_grid, my_metrics, out_dir, "timesplit_sm5_Tin")
    run_CV(wf_sm10_Tin, timesplit_cvfolds, rf_grid, my_metrics, out_dir, "timesplit_sm10_Tin")
    run_CV(wf_sm0_Tout, timesplit_cvfolds, rf_grid, my_metrics, out_dir, "timesplit_sm0_Tout")
    run_CV(wf_sm07_Tout, timesplit_cvfolds, rf_grid, my_metrics, out_dir, "timesplit_sm07_Tout")
    run_CV(wf_sm5_Tout, timesplit_cvfolds, rf_grid, my_metrics, out_dir, "timesplit_sm5_Tout")
    run_CV(wf_sm10_Tout, timesplit_cvfolds, rf_grid, my_metrics, out_dir, "timesplit_sm10_Tout")
    # Transitions are pre-filtered from dataset, so the Tout recipe is not needed
    run_CV(wf_sm0_Tin, timesplit_cvfolds_notrans, rf_grid, my_metrics, out_dir, "timesplit_sm0_ToutAll")
    run_CV(wf_sm07_Tin, timesplit_cvfolds_notrans, rf_grid, my_metrics, out_dir, "timesplit_sm07_ToutAll")
    run_CV(wf_sm5_Tin, timesplit_cvfolds_notrans, rf_grid, my_metrics, out_dir, "timesplit_sm5_ToutAll")
    run_CV(wf_sm10_Tin, timesplit_cvfolds_notrans, rf_grid, my_metrics, out_dir, "timesplit_sm10_ToutAll")
}

message(glue("Analysis Finished. Output saved to {out_dir}\n",
             "Total runtime: {difftime(Sys.time(), log_start_time, units = 'mins')} mins"))
