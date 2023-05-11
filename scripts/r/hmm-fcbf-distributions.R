# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' github.com/...
#'
#' Purpose: 
#'      This script does preparation for hidden markov models, including:
#'      1. Runs FCBF on folds with variable thresholds so that 15 features are 
#'         selected. 15 features are selected for convenience, because if there 
#'         was a variable number of features selected the HMM code for emission 
#'         distributions would get more complicated.
#'      2. Explore transformations and distributions of selected features, to 
#'         decide which theoretical probability distributions should be used to
#'         represent each feature in HMMs
#' 
#' Notes:
#'      Room to improve the efficiency of this process, but it gets the job done
#' 
#' Date Created: 
#'      May 2, 2023
# ~~~~~~~~~~~~~~~ Load packages & Initialization ~~~~~~~~~~~~~~~~~~~~~~~~----
library(here)
library(recipes)
library(colino)
library(ggplot2)
library(fitdistrplus, include.only = "fitdist")
library(glue)
library(data.table, include.only = "fread")
library(purrr)
library(gridExtra)
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
    # plots <- map(selected_feats, function(x){
    #     ggplot(folddat, aes_string(x = x)) + geom_histogram()
    # })
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
# ~~~~~~~~~~~~~~~ Look at distributions for every behaviour ~~~~~~~~~~~~~~~~----
#' Distribution should be examined at separately for each behaviour, because
#' if behaviours are pooled strange artifacts may appear (e.g. bimodality)
#' Transformations will be attempted.

# ---- Create plots for untransformed variables ---
all_feats <- c(unlist(features_timesplit), unlist(features_LSIO)) |> unique()

# Loop over features
plots <- list()
dists <- list()
for (feat in all_feats){
    # Look at the distribution for each behaviour for this feature!
    plots[[feat]] <- list()
    dists[[feat]] <- list()
    for (j in 1:length(unique(dat$majority_behaviour))){
        thisbeh <- unique(dat$majority_behaviour)[j]
        # Plot distributions
        plots[[feat]][[thisbeh]] <- 
            dat %>% filter(majority_behaviour == thisbeh) %>%
            ggplot(aes_string(x = feat)) +
            geom_histogram()
        # # Attempt fitting distributions
        # distdat <- dat[dat$majority_behaviour == thisbeh,feat]
        # dists[[feat]][[thisbeh]]$norm <- 
        #     fitdist(distdat, distr = 'norm', method = 'mle')
        # if(all(distdat >= 1)){
        #     dists[[feat]][[thisbeh]]$gamma <- 
        #         fitdist(distdat, distr = 'gamma', method = 'mle')
        # }
    }
}
grid.arrange(grobs = plots[["pdynX_mean"]])#                skewed pos
grid.arrange(grobs = plots[["dynZ_median"]])#               high kurtosis
grid.arrange(grobs = plots[["ratio_VeDBA_pdynX_median"]])#  skewed pos
grid.arrange(grobs = plots[["statY_min"]])#                 norm
grid.arrange(grobs = plots[["ratio_VeDBA_pdynZ_min"]])#     skew pos

grid.arrange(grobs = plots[["statZ_max"]])#                 norm
grid.arrange(grobs = plots[["statX_kurtosis"]]) #           skewed pos
grid.arrange(grobs = plots[["pdynX_kurtosis"]])#            skew pos (+f)
grid.arrange(grobs = plots[["difX_lag1_skewness"]])#        norm
grid.arrange(grobs = plots[["difZ_lag1_skewness"]])#        norm

grid.arrange(grobs = plots[["AR2_Y"]])#                     norm
grid.arrange(grobs = plots[["AR3_Z"]])#                     norm
grid.arrange(grobs = plots[["AR2_XYZ"]])#                   norm
grid.arrange(grobs = plots[["pwr_top1_freq_X"]])#         pos skew
grid.arrange(grobs = plots[["pwr_F75_Z"]])#               norm

grid.arrange(grobs = plots[["AR3_X"]])#                     norm
grid.arrange(grobs = plots[["pwr_F25_Y"]])#               skew pos
grid.arrange(grobs = plots[["pwr_top1_freq_Z"]])#         skew pos (+fudge?)
grid.arrange(grobs = plots[["CSA_prev_YZ"]])#               norm (high kurtosis)
grid.arrange(grobs = plots[["smVeDBA_mean"]])#              pos skew mostly

grid.arrange(grobs = plots[["accZ_median"]]) #              norm ok
grid.arrange(grobs = plots[["accZ_max"]])#                  skew
grid.arrange(grobs = plots[["accZ_skewness"]])#             norm (high kurtosis)
grid.arrange(grobs = plots[["auc_trap_X"]])#                skew pos
grid.arrange(grobs = plots[["AR4_XYZ"]])#                   norm

grid.arrange(grobs = plots[["pwr_MPF_X"]])#               norm
grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_mean"]])#    pos skew                 
grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_min"]])#     pos skew                    
grid.arrange(grobs = plots[["dynZ_skewness"]])#             norm
grid.arrange(grobs = plots[["difX_lag1_sd"]])#              pos skew

grid.arrange(grobs = plots[["sim_XY"]])#                    skew pos (+fudge?)
grid.arrange(grobs = plots[["statY_kurtosis"]])#            skewed pos
grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_median"]])#  pos skew
grid.arrange(grobs = plots[["statZ_kurtosis"]])#            pos skew
grid.arrange(grobs = plots[["pwr_skew_Z"]])#                skew pos

grid.arrange(grobs = plots[["statZ_mean"]])#                norm? maybe transform
grid.arrange(grobs = plots[["statY_median"]])#              norm  
grid.arrange(grobs = plots[["accXYZ_min"]])#                skew pos
grid.arrange(grobs = plots[["AR4_Z"]])#                     norm
grid.arrange(grobs = plots[["pwr_top1_freq_Y"]])#           pos skew

grid.arrange(grobs = plots[["pwr_MPF_Z"]])#                 norm
grid.arrange(grobs = plots[["statZ_min"]])#                 norm
grid.arrange(grobs = plots[["COR_YZ"]])#                    norm
grid.arrange(grobs = plots[["jerkRMS_lag1_skewness"]])#     norm
grid.arrange(grobs = plots[["COR_XZ"]])#                    norm

grid.arrange(grobs = plots[["AR4_X"]])#                     norm

# ---- Transform skewed variables ---
#' Fitting a theoretical probability distribution can be more difficult when  
#' variables are skewed or have other unfavourable distributions Explore 
#' possible transforms that may give data an easier distribution.
#' Apply a log, exponential, or kurtosis reduction transform
dattrans <- dat %>% 
    # Log transforms
    mutate(across(
        .cols = c(statX_kurtosis,
                  statY_kurtosis,
                  statZ_kurtosis,
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
                  pwr_top1_freq_Z,
                  difX_lag1_sd), 
        .fns = function(x) log(x+1))) %>%
    # de-kurtosis transform
    mutate(across(
        .cols = c(difZ_lag2_mean, CSA_prev_YZ, accZ_skewness, dynZ_median),
        .fns = function(x) sqrt(abs(x-mean(x)) + 0.001)
    )) %>%
    # exp transform
    mutate(accZ_median = exp(accZ_median),
           statZ_mean = exp(statZ_mean))

# ---- Create plots of transformed variables ---
plots_trans <- list()
for (feat in all_feats){
    # Look at the distribution for each behaviour for this feature!
    plots_trans[[feat]] <- list()
    for (j in 1:length(unique(dat$majority_behaviour))){
        thisbeh <- unique(dattrans$majority_behaviour)[j]
        # Plot distributions
        plots_trans[[feat]][[thisbeh]] <- 
            dattrans %>% filter(majority_behaviour == thisbeh) %>%
            ggplot(aes_string(x = feat)) +
            geom_histogram()
        # # Attempt fitting distributions
        # distdat <- dat[dat$majority_behaviour == thisbeh,feat]
        # dists[[feat]][[thisbeh]]$norm <- 
        #     fitdist(distdat, distr = 'norm', method = 'mle')
        # if(all(distdat >= 1)){
        #     dists[[feat]][[thisbeh]]$gamma <- 
        #         fitdist(distdat, distr = 'gamma', method = 'mle')
        # }
    }
}
# ---- Select normal or gamma distribution for each variable ---
# Exponential transformed varaibles (skew-reduction)
grid.arrange(grobs = plots[["accZ_median"]]) #           
grid.arrange(grobs = plots_trans[["accZ_median"]]) #    Use transform,Norm dist

# Log transformed variables (Skew-reduction
grid.arrange(grobs = plots[["statX_kurtosis"]]) #           
grid.arrange(grobs = plots_trans[["statX_kurtosis"]]) #    Use transform, Gamma

grid.arrange(grobs = plots[["statZ_kurtosis"]])#            
grid.arrange(grobs = plots_trans[["statZ_kurtosis"]]) #    Use transform, Gamma

grid.arrange(grobs = plots[["pdynX_mean"]])#                
grid.arrange(grobs = plots_trans[["pdynX_mean"]])#   Use transform,  Gamma is probably best
plot(fitdist(dattrans$pdynX_mean[grepl("dynamic squatting",dattrans$majority_behaviour)], "gamma", "mle"))

grid.arrange(grobs = plots[["ratio_VeDBA_pdynX_median"]])# 
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynX_median"]])# Use transform, Gamma

grid.arrange(grobs = plots[["pwr_top1_freq_Z"]])#          
grid.arrange(grobs = plots_trans[["pwr_top1_freq_Z"]])#  Use transform, Gamma (it ain't pretty!)

grid.arrange(grobs = plots[["sim_XY"]])#                    
grid.arrange(grobs = plots_trans[["sim_XY"]])#       Use transform, Normal!

grid.arrange(grobs = plots[["pdynX_kurtosis"]])#             
grid.arrange(grobs = plots_trans[["pdynX_kurtosis"]])#   Use transform, Gamma

grid.arrange(grobs = plots[["ratio_VeDBA_pdynZ_min"]])#     
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynZ_min"]])# Use transform, Gamma

grid.arrange(grobs = plots[["auc_trap_X"]])#                
grid.arrange(grobs = plots_trans[["auc_trap_X"]])#   Use transform,  Normal

grid.arrange(grobs = plots[["pwr_F25_Y"]])#               
grid.arrange(grobs = plots_trans[["pwr_F25_Y"]])#    Use transform,    Normal

grid.arrange(grobs = plots[["pwr_skew_Z"]])#              
grid.arrange(grobs = plots_trans[["pwr_skew_Z"]])#   Use transform,  Normal

grid.arrange(grobs = plots[["accXYZ_min"]])#                
grid.arrange(grobs = plots_trans[["accXYZ_min"]])#   Use transform, Gamma

grid.arrange(grobs = plots[["statZ_mean"]])#     
grid.arrange(grobs = plots_trans[["statZ_mean"]])#   Use transform,  Normal. transform probably doesnt' do much

grid.arrange(grobs = plots[["pwr_top1_freq_X"]])#         
grid.arrange(grobs = plots_trans[["pwr_top1_freq_X"]])#  Use transform, Not pretty at all. Gamma.

grid.arrange(grobs = plots[["pwr_top1_freq_Y"]])#         
grid.arrange(grobs = plots_trans[["pwr_top1_freq_Y"]])#  Use transform, Gamma

grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_median"]])#  
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynY_median"]])# Use transform, Gamma

grid.arrange(grobs = plots[["accZ_max"]])#                  
grid.arrange(grobs = plots_trans[["accZ_max"]])#   Use transform,  normal

# Kurtosis-reduced (sqrt(abs()))
grid.arrange(grobs = plots[["statY_kurtosis"]])#            
grid.arrange(grobs = plots_trans[["statY_kurtosis"]])# Use transform, Gamma

grid.arrange(grobs = plots[["dynZ_median"]])#           
grid.arrange(grobs = plots_trans[["dynZ_median"]])#  Use transform,  Gamma

grid.arrange(grobs = plots[["CSA_prev_YZ"]])#              
grid.arrange(grobs = plots_trans[["CSA_prev_YZ"]])#   Use transform,  Gamma      

grid.arrange(grobs = plots[["accZ_skewness"]])#            
grid.arrange(grobs = plots_trans[["accZ_skewness"]])#  Use transform, Gamma

grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_min"]])#            
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynY_min"]])# Use transform, Gamma        

grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_mean"]])#            
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynY_mean"]])# Use transform, Gamma

grid.arrange(grobs = plots[["smVeDBA_mean"]])#            
grid.arrange(grobs = plots_trans[["smVeDBA_mean"]])# Use transform,  Gamma

grid.arrange(grobs = plots[["difX_lag1_sd"]])#            
grid.arrange(grobs = plots_trans[["difX_lag1_sd"]])# Use transform,  Gamma

# Check: statZ_mean: log or untransformed 
grid.arrange(grobs = plots[["statZ_mean"]])#           
grid.arrange(grobs = plots_trans[["statZ_mean"]])#    

get_fits <- function(df, var, dist){
    unique(df$majority_behaviour) %>%
        map(function(x) 
            fitdist(df |> filter(majority_behaviour == x) |> pull(var), 
                    dist, "mle")$bic) |>
        set_names(unique(df$majority_behaviour)) |>
        unlist()
}
untrans_norm <- get_fits(dat, "statZ_mean", "norm")
trans_norm <- get_fits(dattrans, "statZ_mean", "norm")
trans_gamma <- get_fits(dattrans, "statZ_mean", "gamma")
mean(untrans_norm - trans_norm)         # Looks like untransformed with normal dist is worse
mean(untrans_norm - trans_gamma)
mean(trans_norm - trans_gamma)          # normal and gamma dist very similar, use norm.

# ---- Final decisions about distributions and transformations ---
#' Bit tedious to do this manually, might be helpful to get a second window up.

# helper to paste into the list:
# cat(stringi::stri_pad(all_feats, width = 25, "right"), sep = " = list(dist = "NA", trans = "NA"),\n")

# Assign distributions and transformations
dist_list <- list(
    pdynX_mean                = list(dist = "gamma", trans = "log"),
    dynZ_median               = list(dist = "gamma", trans = "kurt"),
    ratio_VeDBA_pdynX_median  = list(dist = "gamma", trans = "log"),
    statY_min                 = list(dist = "norm", trans = "none"),
    ratio_VeDBA_pdynZ_min     = list(dist = "gamma", trans = "log"),
    statZ_max                 = list(dist = "norm", trans = "none"),
    statX_kurtosis            = list(dist = "gamma",  trans = "log"),
    pdynX_kurtosis            = list(dist = "gamma", trans = "log"),
    difX_lag1_skewness        = list(dist = "norm", trans = "none"),
    difZ_lag1_skewness        = list(dist = "norm", trans = "none"),
    AR2_Y                     = list(dist = "norm", trans = "none"),
    AR3_Z                     = list(dist = "norm", trans = "none"),
    AR2_XYZ                   = list(dist = "norm", trans = "none"),
    pwr_top1_freq_X           = list(dist = "gamma", trans = "log"),
    pwr_F75_Z                 = list(dist = "norm", trans = "none"),
    AR3_X                     = list(dist = "norm", trans = "none"),
    pwr_F25_Y                 = list(dist = "norm", trans = "log"),
    pwr_top1_freq_Z           = list(dist = "gamma", trans = "log"),
    CSA_prev_YZ               = list(dist = "gamma", trans = "kurt"),
    smVeDBA_mean              = list(dist = "gamma", trans = "kurt"),
    accZ_median               = list(dist = "norm", trans = "exp"),
    accZ_max                  = list(dist = "norm", trans = "log"),
    accZ_skewness             = list(dist = "gamma", trans = "kurt"),
    auc_trap_X                = list(dist = "norm", trans = "log"),
    AR4_XYZ                   = list(dist = "norm", trans = "none"),
    pwr_MPF_X                 = list(dist = "norm", trans = "none"),
    ratio_VeDBA_pdynY_mean    = list(dist = "gamma", trans = "kurt"),
    ratio_VeDBA_pdynY_min     = list(dist = "gamma", trans = "kurt"),
    dynZ_skewness             = list(dist = "norm", trans = "none"),
    difX_lag1_sd              = list(dist = "gamma", trans = "kurt"),
    sim_XY                    = list(dist = "norm", trans = "log"),
    statY_kurtosis            = list(dist = "gamma", trans = "kurt"),
    ratio_VeDBA_pdynY_median  = list(dist = "gamma", trans = "log"),
    statZ_kurtosis            = list(dist = "gamma", trans = "log"),
    pwr_skew_Z                = list(dist = "norm", trans = "log"),
    statZ_mean                = list(dist = "norm", trans = "log"),
    statY_median              = list(dist = "norm", trans = "none"),
    accXYZ_min                = list(dist = "gamma", trans = "log"),
    AR4_Z                     = list(dist = "norm", trans = "none"),
    pwr_top1_freq_Y           = list(dist = "gamma", trans = "log"),
    pwr_MPF_Z                 = list(dist = "norm", trans = "none"),
    statZ_min                 = list(dist = "norm", trans = "none"),
    COR_YZ                    = list(dist = "norm", trans = "none"),
    jerkRMS_lag1_skewness     = list(dist = "norm", trans = "none"),
    COR_XZ                    = list(dist = "norm", trans = "none"),
    AR4_X                     = list(dist = "norm", trans = "none")
)

dists <- data.frame(feature = names(dist_list), 
                    dist = map_chr(dist_list, ~.x$dist),
                    trans = map_chr(dist_list, ~.x$trans),
                    row.names = NULL)
write.csv(dists, here("config", "hmm-distribution-spec.csv"), row.names = FALSE)
