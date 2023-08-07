# ~~~~~~~~~~~~~~~ Script overview ---------------------------------------
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#'
#' https://github.com/rowanjh/behav-acc-ml
#' 
#' Purpose: 
#'      This script runs alternate machine learning models (e.g. XGboost, SVM)
#'      for comparison to random forest. We do not conduct extensive model 
#'      building and hyperparameter tuning but rather just show a ballpark 
#'      comparison of models using fairly standard parameters.
#'      
#' Date Created: 2023-08-02
#'
# ~~~~~~~~~~~~~~~ 1. Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
(start <- Sys.time())
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

path_windowed_data  <- here("data", "windowed", "windowed-data.csv") 
n_CPU_cores <- min(10, parallel::detectCores())

dat <- fread(path_windowed_data, data.table = FALSE)
dat <- dat %>% select(-matches("^beh_"))
# # Small dummy dataset
# dat <- dat |> filter(majority_behaviour %in% c("being mounted", "copulation attempt", "mounting male", "dynamic squatting"))
dat <- dat %>% mutate(majority_behaviour = factor(majority_behaviour))


# LSIO CV
lsio_fold_spec <- here("config", "lsio-folds.csv")
fold_spec_LSIO <- read.csv(lsio_fold_spec) |> 
    rename(LSIO_fold = fold)
dat <- left_join(dat, fold_spec_LSIO, by = "ruff_id")
LSIO_cvfolds <- custom_cv_split(dat, split_id_col = "LSIO_fold")


# Recipe
rec <- 
    recipe(majority_behaviour ~ ., data = head(dat)) %>%
    update_role(recording_id, ruff_id, segment_id, win_in_segment_id, 
                window_id, loop_j, window_start, window_end, transition, mixed, 
                morph, LSIO_fold, new_role = "ID") %>%
    step_zv(all_predictors()) |>
    step_select_fcbf(all_predictors(), threshold = 0.0035) %>%
    step_normalize(all_numeric_predictors())

# Model
mod_rf <- rand_forest(trees=tune(), 
                      mtry=tune(), 
                      min_n=tune(), 
                      mode = "classification")
mod_xgb <- boost_tree(mtry = tune(), 
                      trees = tune(), 
                      min_n = tune(), 
                      tree_depth = tune(),
                      learn_rate = tune(), 
                      loss_reduction = tune(), 
                      sample_size = tune(),
                      stop_iter = tune(),
                      engine = "xgboost", 
                      mode = "classification")
mod_svm_lin <- svm_linear(cost = tune(), 
                      margin = tune(), 
                      mode = "classification")
mod_svm_rbf <- svm_rbf(cost = tune(), 
                   margin = tune(), 
                   rbf_sigma = tune(), 
                   mode = "classification")
mod_knn <- nearest_neighbor(neighbors = tune(), 
                            weight_func = tune(), 
                            dist_power = tune(),
                            mode = "classification")
mod_logistic <- multinom_reg(penalty = tune(),
                             mode = "classification")
mod_mlp <- mlp(hidden_units = tune(), 
               epochs = tune(), 
               dropout = 0.5, 
               activation = "relu",
               engine = "brulee",
               mode = "classification")

all_models <- 
    workflow_set(
        preproc = list(base = rec),
        models = list(logistic = mod_logistic,
                      knn = mod_knn,
                      svm_rbf = mod_svm_rbf,
                      svm_lin = mod_svm_lin,
                      rf = mod_rf,
                      xgb = mod_xgb,
                      mlp = mod_mlp),
        cross = TRUE
    )

cl <- makeCluster(n_CPU_cores)
registerDoParallel(cl)
parallel::clusterEvalQ(cl, {set.seed(4321)})
output <- workflow_map(object = all_models,
                        fn = "tune_grid", 
                        resamples = LSIO_cvfolds,
                        grid = 10, # number of candidate parameter sets to attempt for each model
                        metrics = metric_set(f_meas))

stopCluster(cl)
registerDoSEQ()

(alternate_models_metrics <- map(output$result, collect_metrics) |> set_names(output$wflow_id))
save(list = "alternate_models_metrics", file = here("outputs", "temp-alternate-model-metrics.Rdata"))
(alternate_models_ranks <- rank_results(output))
save(list = "alternate_models_ranks", file = here("outputs", "temp-alternate-model-ranks.Rdata"))
# (alternate_models_plot <- autoplot(output))
# save(list = "alternate_models_plot", file = here("outputs", "temp-alternate-model-plot.Rdata"))

(end <- Sys.time())
end - start