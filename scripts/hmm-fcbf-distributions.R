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
#'      Script not as carefully commented or reviewed. TODO: fix this
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
source(here("scripts", "hmm-helpers.R"))

# ---- Script inputs ---
path_windowed_data    <- here("data", "windowed", "windowed_data.csv") 
timesplit_fold_spec   <- here("config", "timesplit-folds_2023-01-03_1541.csv")

# ---- Prep data ---
dat <- fread(path_windowed_data, data.table = FALSE)
fold_spec_timesplit <- read.csv(timesplit_fold_spec) |> 
    rename(timesplit_fold = fold) |>
    mutate(timesplit_fold = gsub("fold", "", timesplit_fold))
dat <- dat |> select(-matches("^beh_(?!event)", perl = TRUE))
dat <- left_join(dat, fold_spec_timesplit, by = c("recording_id", "window_id"))

# ~~~~~~~~~~~~~~~ Run FCBF on each fold ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
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
                    LOSO_fold, timesplit_fold, beh_event_id, new_role = "ID") %>%
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
features_LOSO <- list(
    f1 = get_features_per_fold(1, 0.002, "LOSO_fold"),
    f2 = get_features_per_fold(2, 0.005, "LOSO_fold"),
    f3 = get_features_per_fold(3, 0.0026, "LOSO_fold"),
    f4 = get_features_per_fold(4, 0.003, "LOSO_fold"),
    f5 = get_features_per_fold(5, 0.003, "LOSO_fold"),
    f6 = get_features_per_fold(6, 0.001, "LOSO_fold"),
    f7 = get_features_per_fold(7, 0.0015, "LOSO_fold"),
    f8 = get_features_per_fold(8, 0.00265, "LOSO_fold"),
    f9 = get_features_per_fold(9, 0.003, "LOSO_fold"),
    f10 = get_features_per_fold(10, 0.003, "LOSO_fold")
)
map(features_LOSO, ~unlist(.x))
separator = "', '"
for(i in 1:10){
    print(glue("fold {i}"))
    print(glue("'{paste(features_LOSO[[i]][[1]], collapse = separator)}'"))
    print("")
}

features_timesplit <- list(
    f1 = get_features_per_fold(1, 0.003, "timesplit_fold"),
    f2 = get_features_per_fold(2, 0.00275, "timesplit_fold"),
    f3 = get_features_per_fold(3, 0.0020, "timesplit_fold"),
    f4 = get_features_per_fold(4, 0.004, "timesplit_fold"),
    f5 = get_features_per_fold(5, 0.0024, "timesplit_fold"),
    f6 = get_features_per_fold(6, 0.0026, "timesplit_fold"),
    f7 = get_features_per_fold(7, 0.003, "timesplit_fold"),
    f8 = get_features_per_fold(8, 0.003, "timesplit_fold"),
    f9 = get_features_per_fold(9, 0.0028, "timesplit_fold"),
    f10 = get_features_per_fold(10, 0.0030, "timesplit_fold")
)
map(features_timesplit, ~unlist(.x))

separator = "', '"
for(i in 1:10){
    print(glue("fold {i}"))
    print(glue("'{paste(features_timesplit[[i]][[1]], collapse = separator)}'"))
    print("")
}

# ~~~~~~~~~~~~~~~ Look at distributions for every behaviour ~~~~~~~~~~~~~~~~----
#' Distribution should be looked at separately for every behaviour, or else 
#' weird bimodalities and other deceptive artefacts may appear

# ---- untransformed variables----
dist_spec_path <- here("config", "shmm-distributions_2023-01-13.csv")
dist_spec <- read.csv(dist_spec_path)

# Loop over features
plots <- list()
dists <- list()
for (i in 1:nrow(dist_spec)){
    thisfeat <- dist_spec[i,'feature']
    # Look at the distribution for each behaviour for this feature!
    plots[[thisfeat]] <- list()
    dists[[thisfeat]] <- list()
    for (j in 1:length(unique(dat$majority_behaviour))){
        thisbeh <- unique(dat$majority_behaviour)[j]
        # Plot distributions
        plots[[thisfeat]][[thisbeh]] <- 
            dat %>% filter(majority_behaviour == thisbeh) %>%
            ggplot(aes_string(x = thisfeat)) +
            geom_histogram()
        # # Attempt fitting distributions
        # distdat <- dat[dat$majority_behaviour == thisbeh,thisfeat]
        # dists[[thisfeat]][[thisbeh]]$norm <- 
        #     fitdist(distdat, distr = 'norm', method = 'mle')
        # if(all(distdat >= 1)){
        #     dists[[thisfeat]][[thisbeh]]$gamma <- 
        #         fitdist(distdat, distr = 'gamma', method = 'mle')
        # }
    }
}
grid.arrange(grobs = plots[["accZ_median"]]) #              norm ok
grid.arrange(grobs = plots[["statX_kurtosis"]]) #           skewed pos
grid.arrange(grobs = plots[["statY_min"]])#                 norm
grid.arrange(grobs = plots[["statY_kurtosis"]])#            skewed pos
grid.arrange(grobs = plots[["dynZ_skewness"]])#             norm
grid.arrange(grobs = plots[["pdynX_mean"]])#                skewed pos
grid.arrange(grobs = plots[["ratio_VeDBA_pdynX_median"]])#  skewed pos
grid.arrange(grobs = plots[["difX_lag1_skewness"]])#        norm
grid.arrange(grobs = plots[["difZ_lag1_skewness"]])#        norm
grid.arrange(grobs = plots[["AR2_Y"]])#                     norm
grid.arrange(grobs = plots[["AR2_XYZ"]])#                   norm
grid.arrange(grobs = plots[["AR4_XYZ"]])#                   norm
grid.arrange(grobs = plots[["X.pwr_MPF_X"]])#               norm
grid.arrange(grobs = plots[["Z.pwr_top1_freq_Z"]])#         skew pos (+fudge?)
grid.arrange(grobs = plots[["sim_AB"]])#                    skew pos (+fudge?)
grid.arrange(grobs = plots[["statZ_max"]])#                 norm
grid.arrange(grobs = plots[["dynZ_median"]])#               high kurtosis
grid.arrange(grobs = plots[["pdynX_kurtosis"]])#            skew pos (+f)
grid.arrange(grobs = plots[["ratio_VeDBA_pdynZ_min"]])#     skew pos
grid.arrange(grobs = plots[["AR3_X"]])#                     norm
grid.arrange(grobs = plots[["auc_trap_X"]])#                skew pos
grid.arrange(grobs = plots[["Y.pwr_F25_Y"]])#               skew pos
grid.arrange(grobs = plots[["Z.pwr_F75_Z"]])#               norm
grid.arrange(grobs = plots[["AR4_Z"]])#                     norm
grid.arrange(grobs = plots[["Z.pwr_skew_Z"]])#              skew pos
grid.arrange(grobs = plots[["accXYZ_min"]])#                skew pos
grid.arrange(grobs = plots[["statY_median"]])#              norm  
grid.arrange(grobs = plots[["statZ_mean"]])#                norm? maybe transform
grid.arrange(grobs = plots[["X.pwr_top1_freq_X"]])#         pos skew
grid.arrange(grobs = plots[["Y.pwr_top1_freq_Y"]])#         pos skew
grid.arrange(grobs = plots[["Z.pwr_MPF_Z"]])#               norm
grid.arrange(grobs = plots[["COR_AC"]])#                    norm
grid.arrange(grobs = plots[["statZ_min"]])#                 norm
grid.arrange(grobs = plots[["COR_BC"]])#                    norm
grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_median"]])#  pos skew
grid.arrange(grobs = plots[["difZ_lag2_mean"]])#            norm (high kurtosis)
grid.arrange(grobs = plots[["CSA_prev_BC"]])#               norm (high kurtosis)
grid.arrange(grobs = plots[["jerkRMS_lag1_skewness"]])#     norm/skew?
grid.arrange(grobs = plots[["accZ_max"]])#                  skew
grid.arrange(grobs = plots[["accZ_skewness"]])#             norm (high kurtosis)
grid.arrange(grobs = plots[["AR4_X"]])#                     norm
grid.arrange(grobs = plots[["AR3_Z"]])#                     norm
grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_min"]])#     pos skew                    
grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_mean"]])#    pos skew                 
grid.arrange(grobs = plots[["X.pwr_MPF_X"]])#               norm      
grid.arrange(grobs = plots[["smVeDBA_mean"]])#              pos skew mostly
grid.arrange(grobs = plots[["Z.pwr_top1_freq_Z"]])#         pos skew 
grid.arrange(grobs = plots[["difX_lag1_sd"]])#              pos skew


# ---- Transform skewed variables----
dattrans <- dat %>% 
    # Log transfors
    mutate(across(
        .cols = c(statX_kurtosis,
                    statY_kurtosis,
                    pdynX_mean,
                    ratio_VeDBA_pdynX_median,
                    Z.pwr_top1_freq_Z,
                    sim_XY,
                    pdynX_kurtosis,
                    ratio_VeDBA_pdynZ_min,
                    auc_trap_X,
                    Y.pwr_F25_Y,
                    Z.pwr_skew_Z,
                    accXYZ_min,
                    X.pwr_top1_freq_X,
                    Y.pwr_top1_freq_Y,
                    ratio_VeDBA_pdynY_median,
                    accZ_max,
                  ratio_VeDBA_pdynY_min,
                  ratio_VeDBA_pdynY_mean,
                  smVeDBA_mean,
                  Z.pwr_top1_freq_Z,
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
# Loop over features
plots_trans <- list()
for (i in 1:nrow(dist_spec)){
    thisfeat <- dist_spec[i,'feature']
    # Look at the distribution for each behaviour for this feature!
    plots_trans[[thisfeat]] <- list()
    for (j in 1:length(unique(dat$majority_behaviour))){
        thisbeh <- unique(dattrans$majority_behaviour)[j]
        # Plot distributions
        plots_trans[[thisfeat]][[thisbeh]] <- 
            dattrans %>% filter(majority_behaviour == thisbeh) %>%
            ggplot(aes_string(x = thisfeat)) +
            geom_histogram()
        # # Attempt fitting distributions
        # distdat <- dat[dat$majority_behaviour == thisbeh,thisfeat]
        # dists[[thisfeat]][[thisbeh]]$norm <- 
        #     fitdist(distdat, distr = 'norm', method = 'mle')
        # if(all(distdat >= 1)){
        #     dists[[thisfeat]][[thisbeh]]$gamma <- 
        #         fitdist(distdat, distr = 'gamma', method = 'mle')
        # }
    }
}
# Skew-reduction (exp transform)
grid.arrange(grobs = plots[["accZ_median"]]) #           
grid.arrange(grobs = plots_trans[["accZ_median"]]) #         Norm  

# Skew-reduction (log transform)
grid.arrange(grobs = plots[["statX_kurtosis"]]) #           
grid.arrange(grobs = plots_trans[["statX_kurtosis"]]) #     Gamma

grid.arrange(grobs = plots[["statY_kurtosis"]])#            
grid.arrange(grobs = plots_trans[["statY_kurtosis"]])#  q   Gamma

grid.arrange(grobs = plots[["pdynX_mean"]])#                
grid.arrange(grobs = plots_trans[["pdynX_mean"]])#          Gamma is probably best
plot(fitdist(dattrans$pdynX_mean[grepl("dynamic squatting",dattrans$majority_behaviour)], "gamma", "mle"))

grid.arrange(grobs = plots[["ratio_VeDBA_pdynX_median"]])# 
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynX_median"]])#  Gamma

grid.arrange(grobs = plots[["Z.pwr_top1_freq_Z"]])#          
grid.arrange(grobs = plots_trans[["Z.pwr_top1_freq_Z"]])#       Gamma (it ain't pretty)

grid.arrange(grobs = plots[["sim_AB"]])#                    
grid.arrange(grobs = plots_trans[["sim_AB"]])#                  Normal!

grid.arrange(grobs = plots[["pdynX_kurtosis"]])#             
grid.arrange(grobs = plots_trans[["pdynX_kurtosis"]])#          Gamma

grid.arrange(grobs = plots[["ratio_VeDBA_pdynZ_min"]])#     
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynZ_min"]])#       Gamma

grid.arrange(grobs = plots[["auc_trap_X"]])#                
grid.arrange(grobs = plots_trans[["auc_trap_X"]])#                  Normal

grid.arrange(grobs = plots[["Y.pwr_F25_Y"]])#               
grid.arrange(grobs = plots_trans[["Y.pwr_F25_Y"]])#               Normal

grid.arrange(grobs = plots[["Z.pwr_skew_Z"]])#              
grid.arrange(grobs = plots_trans[["Z.pwr_skew_Z"]])#              Normal

grid.arrange(grobs = plots[["accXYZ_min"]])#                
grid.arrange(grobs = plots_trans[["accXYZ_min"]])#                Gamma

grid.arrange(grobs = plots[["statZ_mean"]])#     
grid.arrange(grobs = plots_trans[["statZ_mean"]])#                  Normal. transform at all???

grid.arrange(grobs = plots[["X.pwr_top1_freq_X"]])#         
grid.arrange(grobs = plots_trans[["X.pwr_top1_freq_X"]])#         Not pretty at all. Gamma.

grid.arrange(grobs = plots[["Y.pwr_top1_freq_Y"]])#         
grid.arrange(grobs = plots_trans[["Y.pwr_top1_freq_Y"]])#             Gamma

grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_median"]])#  
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynY_median"]])#    Gamma

grid.arrange(grobs = plots[["accZ_max"]])#                  
grid.arrange(grobs = plots_trans[["accZ_max"]])#                  normal

# Kurtosis-reduced (sqrt(abs()))
grid.arrange(grobs = plots[["dynZ_median"]])#           
grid.arrange(grobs = plots_trans[["dynZ_median"]])#              Gamma

grid.arrange(grobs = plots[["difZ_lag2_mean"]])#           
grid.arrange(grobs = plots_trans[["difZ_lag2_mean"]])#              Gamma

grid.arrange(grobs = plots[["CSA_prev_BC"]])#              
grid.arrange(grobs = plots_trans[["CSA_prev_BC"]])#                 Gamma      

grid.arrange(grobs = plots[["accZ_skewness"]])#            
grid.arrange(grobs = plots_trans[["accZ_skewness"]])#            Gamma


grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_min"]])#            
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynY_min"]])#    Gamma        

grid.arrange(grobs = plots[["ratio_VeDBA_pdynY_mean"]])#            
grid.arrange(grobs = plots_trans[["ratio_VeDBA_pdynY_mean"]])#    Gamma

grid.arrange(grobs = plots[["smVeDBA_mean"]])#            
grid.arrange(grobs = plots_trans[["smVeDBA_mean"]])#            Gamma

grid.arrange(grobs = plots[["difX_lag1_sd"]])#            
grid.arrange(grobs = plots_trans[["difX_lag1_sd"]])#            Gamma

# Check: statZ_mean and jerkRMS_lag1_skewness. log or untransformed 
grid.arrange(grobs = plots[["statZ_mean"]])#           
grid.arrange(grobs = plots_trans[["statZ_mean"]])#              

grid.arrange(grobs = plots[["jerkRMS_lag1_skewness"]])#           
grid.arrange(grobs = plots_trans[["jerkRMS_lag1_skewness"]])#              

