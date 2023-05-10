# This script creates a stratified split of the data, but instead of randomly
# allocating windows to each fold, the split will be based on time. Nearby
# behaviours are not as independent, and are kept together, to reduce data leakage
library(here)
library(data.table, include.only = c("fread", "fwrite"))
library(dplyr)
library(tidyr)
library(purrr)
library(glue)
# library(tidymodels)
dat <- fread(here("data", "windowed", "windowed-data.csv"), data.table = FALSE)

# ---- Load and prep dataset ----
# Split by ruff_id and behaviour
dat <- dat %>% 
    unite(bird_beh, ruff_id, majority_behaviour, remove = FALSE) %>% 
    arrange(ruff_id, window_start)

# Stratification works as follows: For the train/test split, each behaviour/bird
# combination will be split independently. For a given behaviour for a given bird,
# 70% of windows will be attempted to be allocated to train, and 30% to test. 
# The allocation will be based on time: the 30% of points with the latest timestamps
# will go into the test set. For CV, an equal number of windows will be attempted
# to be put in each fold. They will be split into 10 chunks based on datetime
# quantiles, and then each of those chunks will be put into one of the 10 folds.
# It doesn't matter which fold per se, different assignments may be attempted to 
# balance the overall number of behaviours in each fold across all birds.

# Window ID is not necessary in order with respect to datetime. Within a segment
# yes, but not across segments.
# 
# unique(dat$bird_beh)
# 
# quantiles <- dat %>% group_by(bird_beh) %>%
#     summarise(n = n(),
#               q1 = quantile(window_start, 0.1),
#               q2 = quantile(window_start, 0.2),
#               q3 = quantile(window_start, 0.3),
#               q4 = quantile(window_start, 0.4),
#               q5 = quantile(window_start, 0.5),
#               q6 = quantile(window_start, 0.6),
#               q7 = quantile(window_start, 0.7),
#               q8 = quantile(window_start, 0.8),
#               q9 = quantile(window_start, 0.9)
#     )
# 
# test <- dat %>% group_by(bird_beh) %>%
#     mutate(q = ntile(window_start, 10),
#            count = n()) %>%
#     select(ruff_id, bird_beh, window_start, q, count, beh_event_id) %>%
#     arrange(ruff_id, bird_beh, window_start)
# test %>%
#     group_by(bird_beh, q) %>%
#     summarise(n = n()) %>%
#     pivot_wider(names_from = q, values_from = n)


# Try to split by event id instead
# Ignore for transition events, lump the epoch in with the previous event. 
splitdat <- dat %>% 
    mutate(event = gsub("_.*", "", beh_event_id),
           event = if_else(event == "NA", gsub(".*_", "", beh_event_id), event)) %>%
    unite(bird_beh, ruff_id, majority_behaviour, remove = FALSE) %>% 
    arrange(ruff_id, window_start)

splitdat <- splitdat %>%
    group_by(bird_beh, event) %>%
    # number of windows for this event. Start time of this event
    summarise(n_windows = n(), time = min(window_start)) %>%
    arrange(bird_beh, time) %>% 
    group_by(bird_beh) %>%
    mutate(t70 = quantile(time, 0.7))

# Some duplicate events got made - these are transitions, not a problem. Just 
# need to ensure bird_beh AND event are used when joining back to the orig data.
# splitdat[duplicated(splitdat$event) | duplicated(splitdat$event, fromLast = TRUE),] %>%
#     arrange(event) %>% View()
# Do train-test split first
dat_train <- splitdat %>% filter(time < t70) %>% select(-t70)
dat_test <- splitdat %>% filter(time >= t70) %>% select(-t70)

# # Split train dat into 10-fold CV
# dat_train <- dat_train %>% 
#     group_by(bird_beh) %>%
#     mutate(q = ntile(time, 10), 
#            birdbeh_total_events = n(),
#            birdbeh_total_windows = sum(n_windows),
#            cumsum_n_windows = cumsum(n_windows)) 

# Break it up into different events. Don't worry about weighting events, just
# attempt to balance at the end if the distribution across folds ends up unbalanced.
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

x$pct_deviance <- apply(x[,-1], 1, max) / apply(x[,-1], 1, min) * 100
x
max(x$pct_deviance)

# Add test ids
dat_with_fold <- dat_with_fold %>%
    mutate(fold = if_else(is.na(fold), "test", fold))
# Save fold assignment
dt <- strftime(Sys.time(), format = "%Y-%m-%d_%H%M")
fwrite(dat_with_fold %>% select(recording_id, window_id, fold), 
          file = here("config", glue("timesplit-folds_{dt}.csv")))
