# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' github.com/...
#'
#' Purpose: 
#'      This script implements FCBF feature selection for hidden markov models.
#'      FCBF is run on folds with variable thresholds so that 15 features are 
#'      selected. Having an equal number features per fold simplifies the HMM 
#'      implementation, but 
#' 
#' Notes:
#'      Room to improve the efficiency of this process, but it gets the job done
#' 
#' Date Created: 
#'      May 2, 2023
#'      
#' Outputs:
#'      ./config/hmm-fcbf-spec.csv

# ~~~~~~~~~~~~~~~ Load packages & Initialization ~~~~~~~~~~~~~~~~~~~~~~~~----
library(here)
library(recipes)
library(colino)
library(glue)
library(data.table, include.only = "fread")
library(purrr)
source(here("scripts", "r", "hmm-helpers.R"))

# ---- Script inputs ---
path_windowed_data    <- here("data", "windowed", "windowed-data.csv") 
timesplit_fold_spec   <- here("config", "timesplit-folds.csv")
LSIO_fold_spec        <- here("config", "lsio-folds_2022-11-16.csv")

# ---- Prep data ---
dat <- fread(path_windowed_data, data.table = FALSE)
fold_spec_timesplit <- read.csv(timesplit_fold_spec) |> 
    rename(timesplit_fold = fold) |>
    mutate(timesplit_fold = gsub("fold", "", timesplit_fold))
fold_spec_LSIO <- read.csv(LSIO_fold_spec) |> 
    rename(LSIO_fold = fold)

dat <- dat |> select(-matches("^beh_(?!event)", perl = TRUE))
dat <- left_join(dat, fold_spec_timesplit, by = c("recording_id", "window_id"))
dat <- left_join(dat, fold_spec_LSIO, by = "ruff_id")

# ~~~~~~~~~~~~~~~ Run FCBF on each fold ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Get 15 features for each fold by tweaking the threshold. A bit tedious, but 
#' not too bad. It would also be possible to adjust the hmm algorithm to 
#' tolerate variable numbers of features.

#' Run FCBF on a fold
#' 
#' @param fold a cross-validation fold (1-10) present in the dat object
#' @param threshold the threshold for FCBF to apply
#' @param fold_type 'LSIO_fold' or 'timesplit_fold', should be a column in dat
#' @return a character vector of selected features
get_features_per_fold <- function(fold, threshold, fold_type){
    # Get data for this training fold
    folddat <- dat %>% 
        filter(.data[[fold_type]] != fold & .data[[fold_type]] != "test") 
    
    # Exclude variables with unsavoury distributions
    folddat <- folddat %>% 
        select(-matches("roll_"))
    
    rec <- 
        recipe(majority_behaviour ~ ., data = head(folddat)) %>%
        update_role(recording_id, ruff_id, segment_id, win_in_segment_id, 
                    window_id, loop_j, window_start, window_end, transition, mixed, 
                    LSIO_fold, timesplit_fold, beh_event_id, new_role = "ID") %>%
        # Morph will be a covariate for sHMM, select different features with FCBF
        update_role(morph, new_role = "ID") %>% 
        step_select_fcbf(all_predictors(), threshold = threshold)
    
    prepped <- prep(rec, folddat)    
    selected_feats <- prepped$steps[[1]]$features_retained$variable
    print(glue("fold {fold} selected {length(selected_feats)} feats"))
    return(list(selected_feats))
}
# A bit slow
features_LSIO <- list(
    f1 = get_features_per_fold(1, 0.00197, "LSIO_fold"),
    f2 = get_features_per_fold(2, 0.005, "LSIO_fold"),
    f3 = get_features_per_fold(3, 0.0026, "LSIO_fold"),
    f4 = get_features_per_fold(4, 0.0025, "LSIO_fold"),
    f5 = get_features_per_fold(5, 0.003, "LSIO_fold"),
    f6 = get_features_per_fold(6, 0.002, "LSIO_fold"),
    f7 = get_features_per_fold(7, 0.0015, "LSIO_fold"),
    f8 = get_features_per_fold(8, 0.00265, "LSIO_fold"),
    f9 = get_features_per_fold(9, 0.003, "LSIO_fold"),
    f10 = get_features_per_fold(10, 0.0045, "LSIO_fold")
)

features_timesplit <- list(
    f1 = get_features_per_fold(1, 0.0035, "timesplit_fold"),
    f2 = get_features_per_fold(2, 0.00275, "timesplit_fold"),
    f3 = get_features_per_fold(3, 0.0020, "timesplit_fold"),
    f4 = get_features_per_fold(4, 0.004, "timesplit_fold"),
    f5 = get_features_per_fold(5, 0.0024, "timesplit_fold"),
    f6 = get_features_per_fold(6, 0.0026, "timesplit_fold"),
    f7 = get_features_per_fold(7, 0.0025, "timesplit_fold"),
    f8 = get_features_per_fold(8, 0.0028, "timesplit_fold"),
    f9 = get_features_per_fold(9, 0.0028, "timesplit_fold"),
    f10 = get_features_per_fold(10, 0.0030, "timesplit_fold")
)

LSIO_df <- map_df(features_LSIO, unlist) |> 
    pivot_longer(everything(), names_to = "fold", values_to = "feature") |>
    arrange(fold) |> 
    mutate(split = "LSIO")
timesplit_df <- map_df(features_timesplit, unlist) |> 
    pivot_longer(everything(), names_to = "fold", values_to = "feature") |>
    arrange(fold) |> 
    mutate(split = "timesplit")

allfeat_spec <- bind_rows(LSIO_df, timesplit_df) |>
    mutate(fold = gsub("f", "fold", fold))

write.csv(allfeat_spec, file = here("config", "hmm-fcbf-spec.csv"),
          row.names = FALSE)

