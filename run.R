# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#' 
#' Data are available at:
#' https://edmond.mpdl.mpg.de/privateurl.xhtml?token=3c5da533-d851-4988-994c-f984c5f1b1a4
#' dataset doi: https://doi.org/10.17617/3.KERIBB 
#' 
#' 
#' Purpose: 
#'      This script shows steps to run all analyses from Aulsebrook et al. (2023).
#'      
#'      Code is provided for all steps in the analyses, including:
#'      1. Calibrations and time synchronisation of raw data
#'      2. Windowing data & feature extraction
#'      3. Training hidden markov models, random forests, and neural networks
#'      4. Evaluating results
#'
#' Notes: 
#'      Package management with Renv. Install all required R packages by running
#'      renv::restore().
#'      
#' Date created:
#'      May 2, 2023
#'      
# ~~~~~~~~~~~~~~  Analysis Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
library(here)
library(rmarkdown)

## ---- Download & extract data ----
# Download two zip files containing raw data, save in './data/raw/' directory.
# Download URLs:
# https://edmond.mpdl.mpg.de/api/access/datafile/212351
# https://edmond.mpdl.mpg.de/api/access/datafile/212350

# # Unzip dataset, extract into the './data/raw' directory
unzip(here("data","raw","data-raw-db.zip"), exdir = here("data","raw"))
unzip(here("data","raw","data-raw-other.zip"), exdir = here("data","raw"))

## ---- Clean data ----

source(here("scripts", "r", "cleaning.R"))


## ---- Window data ----

source(here("scripts", "r", "windowing.R"))


## ---- Split data ----
#' The leave-some-individuals-out (LSIO) and time-based splits need to be 
#' implemented manually. Random stratification is more straightforward and is 
#' implemented in the rf.R script using the tidymodels functions 

source(here("scripts", "r", "misc-cvsplits-lsio.R")) 
source(here("scripts", "r", "misc-cvsplits-timesplit.R"))

## ---- Hidden Markov Models (HMMs) ----
#' The HMM script doesn't automatically implement FCBF, so this is conducted in
#' a separate script, and the feature selections for each fold exported to csv.
source(here("scripts", "r", "hmm-fcbf.R"))

#' Each selected feature in the HMM model needs to be estimated using an 
#' appropriate theoretical probability distribution. This script was used to 
#' explore possible distributions and transformations, and the decisions
#' exported to csv. It is designed for interactive work for visualisations 
#' rather than being  sourced()
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
render(here("scripts", "r", "eval-generate-performance-metrics.Rmd"), 
       output_dir = here("outputs", "eval-results"),
       output_file = "eval-generate-performance-metrics.html")
render(here("scripts", "r", "eval-calculate-correlation-coefficients.Rmd"), 
       output_dir = here("outputs", "eval-results"),
       output_file = "eval-calculate-correlation-coefficients.html")
render(here("scripts", "r", "eval-creating-tables-and-figures.Rmd"), 
       output_dir = here("outputs", "tables-and-figures"),
       output_file = "eval-creating-tables-and-figures.html")

## ---- (Optional) quality-assurance notebooks ----
# Check output of cleaning.R
render(here("scripts", "r", "cleaning-qa.Rmd"), 
       output_dir = here("outputs", "qa"),
       output_file = "cleaning-qa.html")

# Check output of windowing.R
render(here("scripts", "r", "windowing-qa.Rmd"), 
       output_dir = here("outputs", "qa"),
       output_file = "windowing-qa.html")

