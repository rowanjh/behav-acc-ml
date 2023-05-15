# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'      This script implements FCBF feature selection for hidden markov models.
#'      FCBF is run on folds with variable thresholds so that 15 features are 
#'      selected. Having an equal number features per fold simplifies the HMM 
#'      implementation, but is more tedious for FCBF which doesn't currently 
#'      support a specification for the number of features to select, so
#'      different thresholds need to be attempted.
#' 
#' Notes:
#'      Room to improve the efficiency of this process, but it gets the job done
#' 
#' Date Created: 
#'      May 2, 2023
#'      
#' Outputs:
#'      the selected features for each fold are expored to:
#'      ./config/hmm-fcbf-spec.csv
#'
# ~~~~~~~~~~~~~~~ Load packages & Initialization ~~~~~~~~~~~~~~~~~~~~~~~~----
library(here)
library(recipes)
library(colino)
library(glue)
library(data.table, include.only = "fread")
library(purrr)
library(doParallel)
source(here("scripts", "r", "hmm-helpers.R"))

# ---- Script inputs ---
path_windowed_data    <- here("data", "windowed", "windowed-data.csv") 
timesplit_fold_spec   <- here("config", "timesplit-folds.csv")
LSIO_fold_spec        <- here("config", "lsio-folds.csv")

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

# Exclude some features with a distribution that was found to be untennable
dat <- dat |> select(-roll_min, -roll_max, -roll_mean, -roll_median)

# ~~~~~~~~~~~~~~~ Run FCBF on each fold ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Get 15 features for each fold by tweaking the threshold. A bit tedious, but 
#' not too bad. It would also be possible to adjust the hmm algorithm to 
#' tolerate variable numbers of features.

# Specify the threshold for each fold below
runs <- list(
    lsio_1 =  list(fold = 1, thresh = 0.00197, split = "LSIO"),
    lsio_2 =  list(fold = 2, thresh = 0.005,   split = "LSIO"),
    lsio_3 =  list(fold = 3, thresh = 0.0026,  split = "LSIO"),
    lsio_4 =  list(fold = 4, thresh = 0.0025,  split = "LSIO"),
    lsio_5 =  list(fold = 5, thresh = 0.003,   split = "LSIO"),
    lsio_6 =  list(fold = 6, thresh = 0.002,   split = "LSIO"),
    lsio_7 =  list(fold = 7, thresh = 0.0015,  split = "LSIO"),
    lsio_8 =  list(fold = 8, thresh = 0.00265, split = "LSIO"),
    lsio_9 =  list(fold = 9, thresh = 0.003,   split = "LSIO"),
    lsio_10 = list(fold = 10, thresh = 0.0045, split = "LSIO"),
    ts_1 =    list(fold = 1, thresh = 0.0035,  split = "timesplit"),
    ts_2 =    list(fold = 2, thresh = 0.00275, split = "timesplit"),
    ts_3 =    list(fold = 3, thresh = 0.0020,  split = "timesplit"),
    ts_4 =    list(fold = 4, thresh = 0.004,   split = "timesplit"),
    ts_5 =    list(fold = 5, thresh = 0.0024,  split = "timesplit"),
    ts_6 =    list(fold = 6, thresh = 0.0026,  split = "timesplit"),
    ts_7 =    list(fold = 7, thresh = 0.0025,  split = "timesplit"),
    ts_8 =    list(fold = 8, thresh = 0.0028,  split = "timesplit"),
    ts_9 =    list(fold = 9, thresh = 0.0028,  split = "timesplit"),
    ts_10 =   list(fold = 10, thresh = 0.0030, split = "timesplit"),
    ts_test = list(fold = "test", thresh = 0.0030, split = "timesplit")
)

# Run FCBF in parallel over all folds
cl <- makeCluster(min(parallel::detectCores(), 21))
registerDoParallel(cl)

outputs <- foreach(i = 1:length(runs), 
                   .packages = c('dplyr','recipes', 'colino','glue')) %dopar% {
    # init
    run <- runs[[i]]
    splitcol <- paste0(run$split, "_fold")
    
    # load training fold for this cross-validation split
    folddat <- dat %>% 
        filter(.data[[splitcol]] != run$fold) 
    
    # Exclude test set data (unless this is the test fold)
    if (run$fold != "test"){
        folddat <- folddat %>% 
            filter(.data[[splitcol]] != "test") 
    }
    # setup recipe to do FCBF
    rec <- 
        recipe(majority_behaviour ~ ., data = head(folddat)) %>%
        update_role(recording_id, ruff_id, segment_id, win_in_segment_id, 
                    window_id, loop_j, window_start, window_end, transition, mixed, 
                    LSIO_fold, timesplit_fold, beh_event_id, new_role = "ID") %>%
        update_role(morph, new_role = "ID") %>% 
        step_select_fcbf(all_predictors(), threshold = run$thresh)
    
    prepped <- prep(rec, folddat)    
    selected_feats <- prepped$steps[[1]]$features_retained$variable
    
    data.frame(feat = selected_feats,
               split = run$split,
               fold = as.character(run$fold))
}
stopCluster(cl)
registerDoSEQ()

fcbf_selections <- bind_rows(outputs)

# Check number of features in each fold
fcbf_selections |> 
    group_by(split, fold) |>
    count()

write.csv(fcbf_selections, file = here("config", "hmm-fcbf-spec.csv"),
          row.names = FALSE)

