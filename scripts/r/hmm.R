# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'      This script runs supervised hidden markov models in cross-validation  
#'      using the windowed dataset. The full dataset is used for the 'leave some  
#'      individuals out' splits (LSIO), and a training set (70% of data) is used 
#'      for the time-based split. Implementation is adapted from code provided 
#'      in Leos-Barajas et al. 2017
#'
#' Notes: 
#'      Transition windows are not excluded to keep the sequences consolidated
#'      and intact. There is also no generation of synthetic windows through 
#'      SMOTE upsampling.
#'      Feature selection was implemented manually and selected features are 
#'      hard coded in the script 
#'
#' Date Created: 
#'      May 2, 2023
#' 
#' Outputs: 
#'      csv files with predictions for each cross-validation fold exported to:
#'      ./outputs/hmm-results/{datetime}/
#'      
# ~~~~~~~~~~~~~~~ Load packages & Initialization ~~~~~~~~~~~~~~~~~~~~~~~~----
# ---- Start logging ---
log_start_time <- Sys.time()
start_time_string <- format(log_start_time, "%Y-%m-%d_%H%M")
print("------------Loading-----------------")

# ---- Packages ---
library(here)
library(dplyr)
library(data.table, include.only = c("fread", "fwrite"))
library(purrr)
library(glue)
library(fitdistrplus, include.only = "fitdist")
library(doParallel)
source(here("scripts", "r", "hmm-helpers.R"))
select <- dplyr::select

# ---- Script inputs ---
path_windowed_data  <- here("data", "windowed", "windowed-data.csv") 
lsio_fold_spec        <- here("config", "lsio-folds.csv")
timesplit_fold_spec <- here("config", "timesplit-folds.csv")
dist_spec_path <-      here("config", "hmm-distribution-spec.csv")
fcbf_spec_path <-      here("config", "hmm-fcbf-spec.csv")

# Specify "LSIO_fold" or "timesplit_fold" to determine which split method to use
# fold_type <- "timesplit_fold"

# ---- Load data ---
dat <- fread(path_windowed_data, data.table = FALSE)

# Load which windows are assigned the which folds
fold_spec_LSIO <- read.csv(lsio_fold_spec) |> 
    rename(LSIO_fold = fold)
fold_spec_timesplit <- read.csv(timesplit_fold_spec) |> 
    rename(timesplit_fold = fold) |>
    mutate(timesplit_fold = gsub("fold", "", timesplit_fold))

# Load distribution & feature selection specifications
dist_spec <- read.csv(dist_spec_path)
fcbf_spec <- read.csv(fcbf_spec_path)

# Check if all features have a dist specified

if (!all(fcbf_spec$feat %in% dist_spec$feature)){
    stop(glue("features selected by fcbf but missing from distribution spec: 
              {fcbf_spec$feat[!fcbf_spec$feat %in% dist_spec$feature] |>
                    unique() |> paste(collapse = ', ')}"))
}
# ---- Prep dataset for HMM algorithm ---
dat <- dat |> select(-matches("^beh_(?!event)", perl = TRUE))
dat <- left_join(dat, fold_spec_LSIO, by = "ruff_id")
dat <- left_join(dat, fold_spec_timesplit, by = c("recording_id", "window_id"))

# Convert behaviours to integer for the algorithm, add labels
hmmdat <- dat |> 
    mutate(behlab = factor(abbreviate(majority_behaviour, 4)),
           outcome = majority_behaviour) |>
    mutate(outcome = outcome |> factor() |> as.numeric() |> factor())

# Transform variables to get more normal-shaped distributions 
hmmdat <- hmmdat |>
    # Log transforms
    mutate(across(
        .cols = dist_spec |> filter(trans == "log") |> pull(feature), 
        .fns = function(x) log(x+1))) |>
    # Kurtosis reduction transforms
    mutate(across(
        .cols = dist_spec |> filter(trans == "kurt") |> pull(feature),
        .fns = function(x) sqrt(abs(x-mean(x)) + 0.001))) |>
    # Exponential transform
    mutate(across(
        .cols = dist_spec |> filter(trans == "exp") |> pull(feature),
        .fns = exp))

# Make a reference table of class codes versus class labels
beh_labels <- hmmdat |> 
    select(outcome, behlab, majority_behaviour) |> 
    unique() |>
    arrange(desc(outcome))

# ~~~~~~~~~~~~~~~ Implement cross-validation ~~~~~~~~~~~~~~~~~~~~~~~~----
# ---- Initialise ---
print("------------Starting supervised HMM-----------------")
nclasses <- length(unique(hmmdat$outcome))
classes <- levels(hmmdat$outcome)
outputs <- list()
all_preds <- list()
print(glue("Number of classes: {nclasses}"))

# ---- Run cross-validation ---
# Run all 20 cross-validation folds and the test fold in parallel, 21 cores optimal
runs <- expand.grid(fold = 1:10, split = c("LSIO", "timesplit")) |>
    rbind(list(fold="test", split="timesplit"))

cl <- makeCluster(min(parallel::detectCores(), 21))
registerDoParallel(cl)

outputs <- foreach(i = 1:nrow(runs), .packages = c('dplyr','purrr','glue')) %dopar% {
    run <- runs[i,]
    thisfold <- run$fold
    split_type <- as.character(run$split)
    fold_column <- paste0(split_type, "_fold")
    print(glue("run: {split_type} fold {thisfold}"))
    
    # Specify training and validation data for this fold
    traindat <- hmmdat |> 
        filter(.data[[fold_column]] != thisfold & .data[[fold_column]] != "test") |> 
        group_by(segment_id) |>
        filter(n() > 1) |> ungroup()
    valdat <- hmmdat |> 
        filter(.data[[fold_column]] == thisfold) |>
        group_by(segment_id) |>
        filter(n() > 1) |> ungroup()
    
    # ---- Initial probability matrix ----
    #' Initial probability distribution is simply estimated from the proportion 
    #' of windows for which any behaviour is first in the sequence
    initial_probs <- split(traindat, ~segment_id) |> 
        map_dbl(~.x[1,'outcome', drop = TRUE]) |> 
        factor(levels = classes) |>
        table()

    initial_probs <- initial_probs / sum(initial_probs)

    # ---- Emission probability distribution ----
    #' Emission probability distributions give estimates of the feature values
    #' that are expected when in a given behavioural state. A distribution for 
    #' every feature is estimated for every class. The estimates are 
    #' represented as gaussian- or gamma-distributed
    
    # Results holder for all emission dists across all classes
    em_dists <- list() 
    
    feats <- fcbf_spec |> 
        filter(fold == thisfold) |>
        filter(split %in% split_type) |> 
        pull(feat)
    fold_dists <- data.frame(feature = feats) |> left_join(dist_spec)
    # Loop over classes
    for(j in 1:length(classes)){
        thisclass <- classes[j]
        subdat <- traindat |> filter(outcome == thisclass)
        em_dists[[thisclass]] <- list() # em_dists for this class
        # Loop over features within this class and fit respective distribution
        for (k in 1:nrow(fold_dists)){
            thisfeat <- fold_dists[k,'feature']
            thisdist <- fold_dists[k,'dist']
            em_dists[[thisclass]][[thisfeat]] <- 
                fitdistrplus::fitdist(subdat[[thisfeat]], distr = thisdist, method = "mle")
        }
    }
    
    # ---- Transition probability distribution ----
    #' The transition probability distribution gives estimates of how likely a 
    #' bird is to transition from one behaviour to any other behaviour in the 
    #' next window. HMMs require contiguous segments of data (no time gaps 
    #' between windows), so each contiguous segment is processed separately. 
    #' 
    #' In the transition matrix, rows give the target class, columns give the
    #' class in the subsequent window, and values give the estimated probability
    #' of transitioning from the target class to any other class in the 
    #' subsequent window

    tmat <- matrix(0, nrow = nclasses, ncol = nclasses) 
    
    # Loop over every available segment in the training set
    for (seg in unique(traindat$segment_id)){
        # Load the segment
        segdat <- traindat |> filter(segment_id == seg)

        # Get a transition matrix for this segment.
        seg_transitions <- get_trans_mat(segdat$outcome |> as.numeric(), 
                                         classes |> as.numeric())
        # Add result to accumulator for this train fold
        tmat <- tmat + seg_transitions
    }
    # convert the transition matrix from counts to probabilities. 
    tmat_norm <- diag(1/rowSums(tmat)) %*% tmat
    
    # ---- Predict states on validation data ----
    #' Get predictions for windows in the validation fold using the Viterbi 
    #' algorithm. Each segment of data should again be separated due to the time
    #' discontinuities between segments, and the Viterbi algorithm should be run 
    #' on each segment separately. Then predictions from each segment are 
    #' aggregated to represent the full validation fold.

    # Results holder  for this fold
    results <- list()
    
    # Iterate over segments
    for (seg in unique(valdat$segment_id)){
        # Load the segment
        segdat <- valdat |> filter(segment_id == seg)
        
        #' Get Pr(feature-values | behaviour) estimates from the emission 
        #' distribution for each row of the segment.
        #' rows = observations in the validation set, 
        #' cols = behaviours
        #' values = probability estimate of getting these feature values if the
        #'          bird was doing the respective behaviour
        probs <- matrix(NA, nrow = nrow(segdat), ncol = length(classes))
        # Loop over rows of the validation set - TODO: optimizable?
        for (row in 1:nrow(segdat)){
            # Loop over all classes
            for (j in 1:length(classes)){
                thisclass <- classes[j]
                # Store probabilities for this class
                densities <- list()
                # Loop over all features
                for (k in 1:nrow(fold_dists)){
                    thisfeat <- fold_dists[k,'feature']
                    thisdist <- fold_dists[k,'dist']
                    
                    # Get the probability of 
                    if(thisdist == "norm"){
                        densities[[k]] <- dnorm(segdat[[thisfeat]][row], 
                                                em_dists[[j]][[k]]$estimate[1], 
                                                em_dists[[j]][[k]]$estimate[2])
                    } else if(thisdist == "gamma"){
                        densities[[k]] <- dgamma(segdat[[thisfeat]][row], 
                                                 em_dists[[j]][[k]]$estimate[1], 
                                                 em_dists[[j]][[k]]$estimate[2])
                    }
                }
                # Get the product of all densities to estimate the likelihood of 
                # this observation if it was emitted by this class
                probs[row,j] <- unlist(densities) |> reduce(`*`)
                # TODO: vectorise and tidy this process
            }
        }
        
        # Implement the Viterbi algorithm
        preds <- HMM.viterbi(x = segdat,
                             m = nclasses,
                             gamma = tmat_norm,
                             allprobs = probs,
                             delta = initial_probs)
        
        # Collate predictions and truth for this segment
        pred_df <- data.frame(pred = preds, outcome = as.character(preds)) |> 
            tibble() |>
            left_join(beh_labels, by = "outcome") |> 
            # select(-outcome) |>
            rename(pred_behlab = behlab,
                   pred_beh = majority_behaviour)
        truth_df <- segdat |> 
            select(recording_id, window_id,  beh_event_id, window_start,
                   outcome, behlab, majority_behaviour) |>
            rename(truth = outcome, truth_behlab = behlab, 
                   truth_beh = majority_behaviour)
        
        # Store this segments results
        results[[seg]] <- cbind(truth_df, pred_df)
    }
    # Combine results from all segments
    out <- bind_rows(results)
    out$fold <- thisfold 
    if (thisfold != "test") out$fold <- paste0("fold", out$fold)
    out$split <- split_type
    return(out)
}
stopCluster(cl)
registerDoSEQ()

# ---- Save results ----
print("------------Saving results-----------------")

out_dir <- here("outputs", "hmm-results", start_time_string)

# Save results
if(!dir.exists(out_dir)) 
    dir.create(out_dir, recursive = TRUE)
for(i in 1:length(outputs)){
    thisfold <- outputs[[i]]$fold |> unique()
    thissplit <- outputs[[i]]$split |> unique()
    path <- file.path(out_dir, glue("{thissplit}_{thisfold}.csv"))
    fwrite(outputs[[i]], path)
}

print("------------Analysis Complete-----------------")
print(glue("Total Time: {difftime(Sys.time(), log_start_time, units = 'mins') |> round(2)} mins"))

# Quick probe of results
fs <- data.frame()
for(i in 1:length(outputs)){
    thissplit <- outputs[[i]]$split |> unique()
    thisfold <- outputs[[i]]$fold |> unique()
    folddat <- outputs[[i]]
    folddat$pred <- factor(folddat$pred, levels = levels(hmmdat$outcome))
    folddat$truth <- factor(folddat$truth, levels = levels(hmmdat$outcome))
    thisf <- yardstick::f_meas(folddat,pred,truth)$.estimate
    thisresult <- data.frame(split = thissplit,
                             fold = thisfold,
                             f = round(thisf, 2))
    fs <- rbind(fs, thisresult)
}
lsio_cv_f <- fs |> filter(split == "LSIO") |> pull(f) |> mean() |> round(2)
ts_cv_f <- fs |> filter(split == "timesplit" & fold != "test") |> pull(f) |> mean() |> round(2)
ts_test_f <- fs |> filter(split == "timesplit" & fold == "test") |> pull(f) |> mean() |> round(2)

# output a log file: 
# number of classes
# Features selected per fold
# Distributions and transformations for features
# quick results overview
# time elapsed

capture.output(
    file = file.path(out_dir, "log.txt"),
    cat("============= Hmm run log ===================\n"),
    print(glue("Start time: {log_start_time}")),
    print(glue("Run Time: {difftime(Sys.time(), log_start_time, units = 'mins') |> round(2)} mins")),
    cat("\n============= Behaviours ===================\n"),
    print(glue("behaviour class: {unique(hmmdat$majority_behaviour)}")),
    print(glue("n classes: {unique(hmmdat$majority_behaviour) |> length()}")),
    cat("\n============= Overall Results ===================\n"),
    print(glue("LSIO cross-validation f measure:      {lsio_cv_f}")),
    print(glue("timesplit cross-validation f measure: {lsio_cv_f}")),
    print(glue("timesplit testset f measure:          {ts_test_f}")),
    cat("\n========== Distribution Specification ===============\n"),
    print(dist_spec),
    cat("\n======== Feature Selection Specification ==============\n"),
    print(fcbf_spec)
)


