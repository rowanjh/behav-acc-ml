# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'       This script is used to  explore and create cross-validation folds for 
#'       the 'leave some individuals out' (LSIO) splitting method. 
#' 
#' Notes:
#'       In LSIO, all the data for a given bird is assigned to the same cross-
#'       validation fold. We made 10 folds. 
#'       
#'       In this study there were 15 birds with behaviour scored. 10 had 
#'       continuous scoring, so tended to have examples of most behavioural 
#'       classes. Five birds only received scoring during a supplementary scan
#'       through video data that was specifically  targeting copulation 
#'       attempts and mountings. Therefore they have a much smaller sample size, 
#'       and low diversity of behaviours available. The 10 thoroughly scored 
#'       birds were each assigned to a different fold, and then the five 
#'       remaining birds were assigned to one of those folds in a way that 
#'       balanced the class distributions as well as possible. 
#'       
#' Date Created: 
#'      May 2, 2023
#' 
#' Output:
#'      Specification of the leave-some-individuals-out folds exported to:
#'      ./config/lsio-folds.csv
#'
# ~~~~~~~~~~~~~~~ Load packages & Initialization ~~~~~~~~~~~~~~~~~~~~~~~~----
library(here)
library(data.table, include.only = c("fread"))
library(dplyr)

# library(tidymodels)
dat <- fread(here("data", "windowed", "windowed-data.csv"), data.table = FALSE)

# There were 10 birds with extensive behaviour scoring, and 5 birds with 
# supplementary scoring. 
dat |> group_by(ruff_id) |> count() |> arrange(desc(n))

# Three of the behaviours were especially limited in sample size
dat |> count(majority_behaviour) |> arrange(n)

# See how mounting, copulation attempt, being mounted could be split into folds
dat |> 
    filter(grepl("mount|copulat", majority_behaviour)) |>
    select(ruff_id, majority_behaviour) |>
    table()

# Only 6 individuals had any copulation attempt, and most of the copulation 
# attempts were from just 3 birds. I split all of these across different folds, 
# then balanced mounting/being mounted as well as possible. Making these 
# decisions was a manual process with a bit of trial and error.

# Assign birds to folds
LSIO_assignments <- list(
    "1301"          = 1,
    "7-04-105"      = 2,
    "1361"          = 2,
    "G20-0059-B6.5" = 2,
    "G20-0529-B6.5" = 3,
    "952"           = 4,
    "1331"          = 5,
    "G20-0071-B6.5" = 5,
    "1372"          = 6,
    "1326"          = 7,
    "1399"          = 8,
    "1681"          = 8,
    "G20-0055-B6.5" = 8,
    "1368"          = 9,
    "1333"          = 10
) 

lsio_df <- data.frame(
    ruff_id = names(LSIO_assignments),
    fold = unlist(LSIO_assignments),
    row.names = NULL)
    
# Look at the behaviour breakdown for each fold
left_join(dat, lsio_df, by = "ruff_id") |>
    # filter(grepl("mount|copulat", majority_behaviour)) |>
    select(fold, majority_behaviour) |>
    table()
    
if (!dir.exists(here("config")))
    dir.create(here("config"))

write.csv(lsio_df, file = here("config", "lsio-folds.csv"), row.names = FALSE)
