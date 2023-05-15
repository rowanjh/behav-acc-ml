# ~~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'      Helper functions for random forest models. See rf.R
#'
#' Date created:
#'      May 2, 2023
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#' Create cross-validation splits from a manual specification
#' 
#' Adapted version of rsample::vfold_cv that allows specification of which rows
#' should be put into which folds, rather than random splitting
#' 
#' @param data data.frame containing the dataset to be split into folds
#' @param split_id_col a column with a number between 1 and nfolds giving the
#'      index of the fold that this row should be assigned to
custom_cv_split <- function(data, split_id_col){
    n <- nrow(data)
    # folds <- sample(rep(1:10, length.out = n)) # original rsample code
    folds <- data[[split_id_col]]                # customization here
    idx <- seq_len(n)
    indices <- rsample:::split_unnamed(idx, folds)
    indices <- lapply(indices, rsample:::default_complement, n = n)
    split_objs <- purrr::map(indices, make_splits, data = data, 
                             class = "vfold_split")
    split_objs <- tibble::tibble(splits = split_objs, 
                                 id = names0(length(split_objs), "Fold"))
    split_objs$splits <- map(split_objs$splits, rsample:::rm_out)
    cv_att <- list(v = 10, repeats = 1, strata = !is.null(NULL))
    new_rset(splits = split_objs$splits, 
             ids = split_objs[,grepl("^id", names(split_objs))], 
             attrib = cv_att, 
             subclass = c("vfold_cv", "rset"))
}

#' Convenience function for cross validation
#' 
#' Wrapper that runs tune_grid() with extra conveniences including initialising 
#' parallel computing and saving results
#' 
#' @param workflow
#' @param cv_folds
#' @param tuning_grid
#' @param metrics
#' @param out_dir
#' @param label
#' @param ncores
#' @return NULL
run_CV <- function(workflow, cv_folds, tuning_grid, metrics, 
                   out_dir, label, ncores = min(parallel::detectCores(), 10)){
    message(glue("---Running CV for: {label} on dataset {deparse(substitute(cv_folds))}"))
    
    # Set up multiple workers
    cl <- makeCluster(ncores)
    clusterEvalQ(cl, {library(colino);library(themis)})
    registerDoParallel(cl)
    
    # Put the output in a list with a label
    tuning_output <- 
        tune_grid(
            workflow, 
            resamples = cv_folds, 
            grid = tuning_grid, 
            metrics = metrics,
            control=control_grid(save_pred = TRUE,
                                 parallel_over = 'resamples'))
    stopCluster(cl)
    registerDoSEQ()
    
    # Save output to disk
    save_cv_output(tuning_output, workflow, out_dir, label)
    NULL
}


#' Save tune_grid output to disk
#' 
#' Saves a text summary, workflow object, predictions, and metrics for the 
#' hyperparameter grid.
#'
save_cv_output <- function(tunegrid_obj, wf, out_dir, label){
    # Output dir for this model workflow
    thisdir <- file.path(out_dir, label)
    if (!dir.exists(thisdir))
        dir.create(thisdir, recursive = TRUE)
    
    # Save quick text summary of recipe & model.
    capture.output(
        file = glue("{thisdir}/model_info.txt"),
        wf,
        cat("===================================\n"),
        cat("Variable roles\n"),
        cat("===================================\n"),
        print(wf$pre$actions$recipe$recipe$term_info, n = 500))
    
    # save workflow object (contains full on recipe, not just a basic summary)
    save(wf, file = file.path(thisdir, "workflow_obj.RData"))
    
    # Save predictions
    preds_all <- collect_predictions(tunegrid_obj, summarize = FALSE)
    fwrite(preds_all, file.path(thisdir, "collect_predictions.csv"))
    # Save computed metrics: 
    # summarised over folds
    collect_metrics <- collect_metrics(tunegrid_obj)
    fwrite(collect_metrics, file.path(thisdir, "collect_metrics_summary.csv"))
    # separate for each fold
    collect_metrics_all <- collect_metrics(tunegrid_obj,summarize = FALSE)
    fwrite(collect_metrics_all, file.path(thisdir, "collect_metrics_all.csv"))
}


#' In case backend cleanup is needed during interactive testing
# https://stackoverflow.com/questions/25097729/un-register-a-doparallel-cluster
unregister_dopar <- function() {
    env <- foreach:::.foreachGlobals
    rm(list=ls(name=env), pos=env)
}

