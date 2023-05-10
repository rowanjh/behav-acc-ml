# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' github.com/...
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
#'      ./output/hmm/
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
lsio_fold_spec        <- here("config", "lsio-folds_2022-11-16.csv")
timesplit_fold_spec <- here("config", "timesplit-folds_2023-05-09_2007.csv")
dist_spec_path <-      here("config", "shmm-distributions_2023-01-13.csv")

# Specify "LSIO_fold" or "timesplit_fold" to determine which split method to use
fold_type <- "timesplit_fold"

# ---- Load data ---
dat <- fread(path_windowed_data, data.table = FALSE)

# Load which windows are assigned the which folds
fold_spec_LSIO <- read.csv(lsio_fold_spec) |> 
    rename(LSIO_fold = fold)
fold_spec_timesplit <- read.csv(timesplit_fold_spec) |> 
    rename(timesplit_fold = fold) |>
    mutate(timesplit_fold = gsub("fold", "", timesplit_fold))

# Load distribution specifications
dist_spec <- read.csv(dist_spec_path)

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
        .cols = c(statX_kurtosis,
                  statY_kurtosis,
                  pdynX_mean,
                  ratio_VeDBA_pdynX_median,
                  pwr_top1_freq_Z,
                  sim_XY,
                  pdynX_kurtosis,
                  ratio_VeDBA_pdynZ_min,
                  auc_trap_X,
                  pwr_F25_Y,
                  pwr_skew_Z,
                  accXYZ_min,
                  pwr_top1_freq_X,
                  pwr_top1_freq_Y,
                  ratio_VeDBA_pdynY_median,
                  accZ_max,
                  ratio_VeDBA_pdynY_min,
                  ratio_VeDBA_pdynY_mean,
                  smVeDBA_mean,
                  pwr_top1_freq_Z), 
        .fns = function(x) log(x+1))) |>
    # Kurtosis reduction transform
    mutate(across(
        .cols = c(dynZ_median, difZ_lag2_mean, CSA_prev_YZ, accZ_skewness),
        .fns = function(x) sqrt(abs(x-mean(x)) + 0.001)
    )) |>
    # Exponential transform
    mutate(accZ_median = exp(accZ_median),
           statZ_mean = exp(statZ_mean))

# Make a reference table of class codes versus class labels
beh_labels <- hmmdat |> 
    select(outcome, behlab, majority_behaviour) |> 
    unique() |>
    arrange(desc(outcome))

# ~~~~~~~~~~~~~~~ Implement cross-validation ~~~~~~~~~~~~~~~~~~~~~~~~----
# ---- Initialize ---
print("------------Starting supervised HMM-----------------")
nclasses <- length(unique(hmmdat$outcome))
classes <- levels(hmmdat$outcome)
outputs <- list()
all_preds <- list()
print(glue("Number of classes: {nclasses}"))
print(glue("splitting data by: {fold_type}"))

# Use the appropriate feature set for each CV fold (selected by FCBF)
if(fold_type == "LSIO_fold"){
    feats <- list(
        fold1 = c('accZ_median', 'statX_kurtosis', 'statY_min', 'statY_kurtosis', 'dynZ_skewness', 'pdynX_mean', 'ratio_VeDBA_pdynX_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR2_Y', 'AR2_XYZ', 'AR4_XYZ', 'pwr_MPF_X', 'pwr_top1_freq_Z', 'sim_XY'),
        fold2 = c('statY_min', 'statZ_max', 'dynZ_median', 'pdynX_kurtosis', 'ratio_VeDBA_pdynX_median', 'ratio_VeDBA_pdynZ_min', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR3_X', 'auc_trap_X', 'AR2_Y', 'pwr_F25_Y', 'pwr_top1_freq_Z', 'pwr_F75_Z', 'sim_XY'),
        fold3 = c('accZ_median', 'statX_kurtosis', 'statY_min', 'statY_kurtosis', 'dynZ_skewness', 'pdynX_kurtosis', 'ratio_VeDBA_pdynX_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR3_X', 'auc_trap_X', 'AR4_Z', 'pwr_F25_Y', 'pwr_skew_Z', 'sim_XY'),
        fold4 = c('accXYZ_min', 'statX_kurtosis', 'statY_median', 'statZ_mean', 'dynZ_median', 'ratio_VeDBA_pdynX_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'auc_trap_X', 'AR4_Z', 'AR2_XYZ', 'pwr_top1_freq_X', 'pwr_top1_freq_Y', 'pwr_MPF_Z', 'COR_XZ'),
        fold5 = c('statX_kurtosis', 'statY_min', 'statZ_min', 'dynZ_median', 'pdynX_mean', 'ratio_VeDBA_pdynX_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR2_Y', 'AR4_Z', 'AR2_XYZ', 'pwr_top1_freq_X', 'pwr_MPF_Z', 'COR_YZ', 'sim_XY'),
        fold6 = c('accZ_median', 'statX_kurtosis', 'statY_min', 'dynZ_skewness', 'pdynX_mean', 'pdynX_kurtosis', 'ratio_VeDBA_pdynY_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'difZ_lag2_mean', 'AR2_Y', 'AR4_XYZ', 'pwr_MPF_X', 'pwr_top1_freq_Z', 'CSA_prev_YZ'),
        fold7 = c('accZ_median', 'statX_kurtosis', 'statY_min', 'statY_kurtosis', 'dynZ_skewness', 'pdynX_mean', 'ratio_VeDBA_pdynX_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR2_Y', 'AR4_XYZ', 'pwr_MPF_X', 'pwr_top1_freq_Z', 'jerkRMS_lag1_skewness', 'sim_XY'),
        fold8 = c('accZ_median', 'accZ_max', 'statX_kurtosis', 'statY_min', 'dynZ_skewness', 'pdynX_mean', 'ratio_VeDBA_pdynX_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR2_Y', 'AR2_XYZ', 'pwr_top1_freq_X', 'pwr_MPF_Z', 'COR_XZ', 'CSA_prev_YZ'),
        fold9 = c('accZ_median', 'accZ_max', 'accZ_skewness', 'statX_kurtosis', 'statY_min', 'ratio_VeDBA_pdynX_median', 'ratio_VeDBA_pdynZ_min', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR4_X', 'auc_trap_X', 'AR2_Y', 'AR2_XYZ', 'pwr_MPF_X', 'pwr_top1_freq_Z'),
        fold10 = c('accZ_median', 'accZ_max', 'accZ_skewness', 'statX_kurtosis', 'statY_min', 'pdynX_kurtosis', 'ratio_VeDBA_pdynY_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'auc_trap_X', 'AR2_Y', 'AR2_XYZ', 'pwr_top1_freq_X', 'pwr_MPF_Z', 'COR_XZ'))
} else if(fold_type == "timesplit_fold"){
    feats = list(
        fold1 = c('statX_kurtosis', 'statY_min', 'statZ_max', 'dynZ_median', 'pdynX_mean', 'pdynX_kurtosis', 'ratio_VeDBA_pdynX_median', 'ratio_VeDBA_pdynZ_min', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR2_Y', 'pwr_F25_Y', 'pwr_top1_freq_Z', 'pwr_F75_Z', 'CSA_prev_YZ'),
        fold2 = c('statX_kurtosis', 'statY_min', 'statZ_max', 'dynZ_median', 'ratio_VeDBA_pdynX_median', 'ratio_VeDBA_pdynZ_min', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR3_X', 'AR2_Y', 'pwr_F25_Y', 'pwr_top1_freq_Z', 'pwr_F75_Z', 'smVeDBA_mean', 'CSA_prev_YZ'),
        fold3 = c('accZ_median', 'statX_kurtosis', 'statY_min', 'statY_kurtosis', 'dynZ_skewness', 'pdynX_mean', 'ratio_VeDBA_pdynX_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR2_Y', 'AR2_XYZ', 'AR4_XYZ', 'pwr_MPF_X', 'pwr_top1_freq_Z', 'CSA_prev_YZ'),
        fold4 = c('accZ_median', 'accZ_max', 'statX_kurtosis', 'statY_min', 'dynZ_skewness', 'ratio_VeDBA_pdynY_mean', 'ratio_VeDBA_pdynY_min', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR3_X', 'AR2_Y', 'pwr_F25_Y', 'pwr_top1_freq_Z', 'pwr_F75_Z', 'smVeDBA_mean'),
        fold5 = c('statX_kurtosis', 'statY_min', 'statY_kurtosis', 'statZ_max', 'dynZ_median', 'ratio_VeDBA_pdynY_median', 'difX_lag1_sd', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR3_X', 'AR2_Y', 'pwr_F25_Y', 'pwr_top1_freq_Z', 'pwr_F75_Z', 'sim_XY'),
        fold6 = c('accZ_median', 'accZ_max', 'accZ_skewness', 'statX_kurtosis', 'statY_min', 'statY_kurtosis', 'ratio_VeDBA_pdynX_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'auc_trap_X', 'AR2_Y', 'AR2_XYZ', 'pwr_MPF_X', 'pwr_top1_freq_Z', 'CSA_prev_YZ'),
        fold7 = c('accZ_median', 'statX_kurtosis', 'statY_min', 'dynZ_skewness', 'pdynX_kurtosis', 'ratio_VeDBA_pdynY_mean', 'ratio_VeDBA_pdynY_min', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR3_X', 'pwr_F25_Y', 'pwr_top1_freq_Z', 'pwr_F75_Z', 'smVeDBA_mean', 'CSA_prev_YZ'),
        fold8 = c('accZ_median', 'accZ_max', 'statX_kurtosis', 'statY_min', 'dynZ_skewness', 'pdynX_kurtosis', 'ratio_VeDBA_pdynY_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR2_Y', 'pwr_F25_Y', 'pwr_top1_freq_Z', 'pwr_F75_Z', 'smVeDBA_mean', 'CSA_prev_YZ'),
        fold9 = c('accZ_median', 'accZ_skewness', 'statX_kurtosis', 'statY_min', 'pdynX_kurtosis', 'ratio_VeDBA_pdynY_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'auc_trap_X', 'AR2_Y', 'AR3_Z', 'AR2_XYZ', 'pwr_top1_freq_X', 'pwr_F75_Z', 'CSA_prev_YZ'),
        fold10 = c('accZ_median', 'accZ_max', 'statX_kurtosis', 'statY_min', 'dynZ_skewness', 'pdynX_mean', 'pdynX_kurtosis', 'ratio_VeDBA_pdynX_median', 'difX_lag1_skewness', 'difZ_lag1_skewness', 'AR2_Y', 'pwr_F25_Y', 'pwr_top1_freq_Z', 'pwr_F75_Z', 'CSA_prev_YZ')
    )
}

# ---- Run cross-validation ---
# Run in parallel across folds, 10 cores optimal
cl <- makeCluster(min(parallel::detectCores(), 10))
registerDoParallel(cl)
outputs <- foreach(fold = 1:10, .packages = c('dplyr','purrr','glue')) %dopar% {
    print(glue("fold {fold}"))
    
    # Specify training and validation data for this fold
    traindat <- hmmdat |> 
        filter(.data[[fold_type]] != fold & .data[[fold_type]] != "test") |> 
        group_by(segment_id) |>
        filter(n() > 1) |> ungroup()
    valdat <- hmmdat |> 
        filter(.data[[fold_type]] == fold) |>
        group_by(segment_id) |>
        filter(n() > 1) |> ungroup()
    
    # # To fit on the test set
    # traindat <- hmmdat |> 
    #     filter(.data[[fold_type]] != "test") |> 
    #     group_by(segment_id) |>
    #     filter(n() > 1) |> ungroup()
    # valdat <- hmmdat |> 
    #     filter(.data[[fold_type]] == "test") |>
    #     group_by(segment_id) |>
    #     filter(n() > 1) |> ungroup()
    
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
    
    fold_dists <- data.frame(feature = feats[[fold]]) |> left_join(dist_spec)
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
                # TODO: vectorize and tidy this process
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
            left_join(beh_labels) |> 
            # select(-outcome) |>
            rename(pred_behlab = behlab,
                   pred_beh = majority_behaviour)
        truth_df <- segdat |> 
            select(recording_id, window_id,  beh_event_id, window_start,
                   outcome, behlab, majority_behaviour) |>
            rename(truth = outcome, truth_behlab = behlab, truth_beh = majority_behaviour)
        
        # Store this segments results
        results[[seg]] <- cbind(truth_df, pred_df)
    }
    # Combine results from all segments
    return(bind_rows(results))
}
stopCluster(cl)
registerDoSEQ()

# ---- Save results ----
print("------------Saving results-----------------")

out_dir <- here("outputs", "hmm-results", fold_type, start_time_string)

# Save results
if(!dir.exists(out_dir)) 
    dir.create(out_dir, recursive = TRUE)
for(i in 1:length(outputs)){
    path <- file.path(out_dir, glue("fold{i}.csv"))
    fwrite(outputs[[i]], path)
}

print("------------Analysis Complete-----------------")
print(glue("Total Time: {difftime(Sys.time(), log_start_time, units = 'mins') |> round(2)} mins"))


# # Quick probe of results
# fs <- list()
# for(i in 1:10){
#     folddat <- outputs[[i]]
#     folddat$pred <- factor(folddat$pred, levels = levels(hmmdat$outcome))
#     folddat$truth <- factor(folddat$truth, levels = levels(hmmdat$outcome))
#     thisf <- yardstick::f_meas(folddat,pred,truth)$.estimate
#     print(glue("\nfold {i} f meas: {thisf}"))
#     fs[[i]] <- thisf
# }
# print(glue("mean f-score: {fs |> unlist() |> mean() |> round(2)}"))
