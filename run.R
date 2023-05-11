# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://doi.org/...
#' https://zenodo...
#' https://github.com/rowanjh/behav-acc-ml
#' 
#' Purpose: 
#'      This script shows steps to run all analyses from Aulsebrook et al. (2023).
#'      
#'      Code is provided for all steps in the analyses, including:
#'      1. Calibrations and time synchronisation of raw data
#'      2. Windowing data & feature computation
#'      3. Training hidden markov models, random forests, and neural networks
#'      4. Evaluating results
#'
#' Notes: 
#'      Package management with Renv
#'      
#' Date created:
#'      May 2, 2023
#'      
# ~~~~~~~~~~~~~~  Run Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
library(here)

## ---- Download data ----
download.file("https://zenodo...", here("data", "raw", "ruff-data-db.zip"))
download.file("https://zenodo...", here("data", "raw", "ruff-data-misc.zip"))

unzip(here("data","raw","ruff-data-db.zip"), here("data","raw"))
unzip(here("data","raw","ruff-data-misc.zip"), here("data","raw"))

## ---- Clean data ----

source(here("scripts", "r", "cleaning.R"))

## ---- Window data ----

source(here("scripts", "r", "windowing.R"))

## ---- Split data ----
#' The leave-some-individuals-out (LSIO) and time-based splits need to be 
#' implemented manually. Random stratification is more straightforward and is 
#' implemented in the rf.R script using the tidymodels functions 

source(here("scripts", "r", "misc-cvsplits-slio.R")) 
source(here("scripts", "r", "misc-cvsplits-timesplit.R"))

## ---- Hidden Markov Models (HMMs) ----
#' The HMM script doesn't automatically implement FCBF, so this is conducted in
#' a separate script, and the feature selections for each fold exported to csv.
source(here("scripts", "r", "hmm-fcbf.R"))

#' Each selected feature in the HMM model needs to be estimated using an 
#' appropriate theoretical probability distribution. This script was used to 
#' explore possible distributions and transformations, and the decisions
#' exported to csv.
source(here("scripts", "r", "hmm-specify-distributions.R"))

# Run HMMs
source(here("scripts", "r", "hmm.R"))

## ---- Random Forests (RFs) ----
#' We recommend running the runRF script from a terminal using the Rscript CLI 
#' for better stability (see notes in the runRF.R script)

source(here("scripts", "r", "runRF.R"))

## ---- DeepConvLSTM Neural Network ----
#' DeepConvLSTMs should be run through python, preferably on a machine with GPU
#' available. See instructions for installation and running in ./scripts/py/main.py

## ---- Evaluations ----

