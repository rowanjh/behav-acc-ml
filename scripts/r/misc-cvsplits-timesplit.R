# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'       This script creates train/test sets and cross-validation folds using 
#'       the time stratification method.
#' 
#' Notes:
#'      In time stratification, the data from a given bird is split between the
#'      train and test sets according to timestamp. Instead of randomly 
#'      allocating windows to each set, earlier behaviours are use for training,  
#'      and later behaviours are used for testing. This gives protection against
#'      overoptimistic performance estimates that could arise if behaviours 
#'      closer in time are more similar.
#'      
#'      We implemented time stratification as follows: For the train/test split, 
#'      each behaviour/bird combination is split independently. For a given 
#'      behaviour for a given bird, the earliest 70% of windows will be 
#'      attempted to be assigned to train, and the latest 30% to test. There is 
#'      a constraint that windows sharing the same behaviour event id must be 
#'      assigned to the same fold. This way consecutive windows of the same 
#'      behaviour (e.g. when the bird is resting for multiple windows) are 
#'      treated as not independent, and must be kept together.
#'      
#'      For cross-validation we split the training set into 10 folds each with a 
#'      roughly equal number of windows of each behaviour. Each behaviour for 
#'      each bird is split into 10 chunks based on datetime quantiles. The 
#'      timestamp or temporal order of chunks is not considered beyond this 
#'      point, and event id is ignored. The quantiled chunks are simply split 
#'      into folds to roughly balance the number of behaviours in each fold
#'             
#' Date Created: 
#'      May 2, 2023
#' 
#' Output:
#'      Specification of the time-stratification folds exported to:
#'      ./config/timesplit-folds.csv
#'      
# ~~~~~~~~~~~~~~~ Load packages & Initialization ~~~~~~~~~~~~~~~~~~~~~~~~----
library(here)
library(data.table, include.only = c("fread", "fwrite"))
library(dplyr)
library(tidyr)
library(purrr)
library(glue)
# library(tidymodels)
dat <- fread(here("data", "windowed", "windowed-data.csv"), data.table = FALSE)

# ---- Load and prep dataset ----
# Create column to easily split by ruff_id and behaviour (used later to join)
dat <- dat %>% 
    unite(bird_beh, ruff_id, majority_behaviour, remove = FALSE) |>
    arrange(ruff_id, window_start)

# Transition windows have 2+ IDs. Only only use the first
splitdat <- dat %>% 
    # Only keep the first event id, exclude cases where first event is NA
    mutate(event = gsub("_.*", "", beh_event_id), 
           event = if_else(event == "NA", gsub(".*_", "", beh_event_id), event))

# ---- Train-test split ----

splitdat <- splitdat %>%
    # Group by bird, behaviour, and event
    group_by(bird_beh, event) %>%
    # Number of windows for each event.
    summarise(n_windows = n(), time = min(window_start)) %>%
    arrange(bird_beh, time) %>% 
    group_by(bird_beh) %>%
    # Time at which 70% of events are earlier, and 30% later (by bird and beh)
    mutate(t70 = quantile(time, 0.7))

# # Some duplicate events got made - these are transitions, not a problem. Just 
# # need to ensure bird_beh AND event are used when joining back to the orig data.
# splitdat[duplicated(splitdat$event) | 
#              duplicated(splitdat$event, fromLast = TRUE),] |>
#     arrange(event)

# Create train and test folds
dat_train <- splitdat %>% filter(time < t70) %>% select(-t70)
dat_test <- splitdat %>% filter(time >= t70) %>% select(-t70)

# ---- Create 10 cross-validation folds ----
#' Break up the training set into 10 quantiles, for each bird and behaviour.
#' Disregard event id for now.
cv_split <- dat_train %>% 
    group_by(bird_beh) %>%
    arrange(bird_beh, time) %>%
    mutate(qntl = ntile(time, 10),
           count = n())
    
# Spread out the q values randomly across folds
assign_random_fold <- function(df){
    # Take a df with event ids and quantile assignments.
    # Assign each quantile to a random fold ID
    fold_ids <- paste0("fold", 1:10)
    df[['fold']] <- NA
    for (i in 1:length(unique(df$qntl))){
        fold <- sample(fold_ids, size = 1, replace = FALSE)
        fold_ids <- fold_ids[!fold_ids==fold]
        df[df$qntl == unique(df$qntl)[i], 'fold'] <- fold
    }
    return(df)
}

# I tried some different seeds until I got a split that looked reasonably 
# balanced
set.seed(321)
event_fold_assignments <- map(split(cv_split, ~bird_beh), assign_random_fold) %>% bind_rows()

# Check assignment
table(event_fold_assignments$qntl, event_fold_assignments$fold)

# rejoin fold assignment to original dataset
dat_with_fold <- dat %>%    
    mutate(event_truncate = gsub("_.*", "", beh_event_id),
           event_truncate = if_else(event_truncate == "NA", 
                                    gsub(".*_", "", beh_event_id), 
                                    event_truncate)) %>%
    unite(bird_beh, ruff_id, majority_behaviour, remove = FALSE) %>% 
    left_join(event_fold_assignments %>% select(event, fold), 
              by = c("event_truncate" = "event",
                     "bird_beh" = "bird_beh"))

x <- table(dat_with_fold$majority_behaviour, dat_with_fold$fold) %>% as.data.frame() %>%
    pivot_wider(values_from = Freq, names_from = Var2)

# A measure of variation between folds in sample size
x$pct_deviance <- apply(x[,-1], 1, max) / apply(x[,-1], 1, min) * 100
x
max(x$pct_deviance)

# Add test ids
dat_with_fold <- dat_with_fold %>%
    mutate(fold = if_else(is.na(fold), "test", fold))
# Save fold assignment
fwrite(dat_with_fold %>% select(recording_id, window_id, fold), 
          file = here("config", glue("timesplit-folds.csv")))
