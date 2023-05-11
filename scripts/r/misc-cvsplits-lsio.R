library(here)
library(data.table, include.only = c("fread"))
library(dplyr)

# library(tidymodels)
dat <- fread(here("data", "windowed", "windowed-data.csv"), data.table = FALSE)

# Focus on the behaviours with most limited sample size
dat |> count(majority_behaviour) |> arrange(n)

# See how mounting, copulation attempt, being mounted can be divided among folds
dat |> 
    filter(grepl("mount|copulat", majority_behaviour)) |>
    select(ruff_id, majority_behaviour) |>
    table()

# Only 6 individuals had any copulation attempt, most were just by 3 birds. 
# Split all of these across different folds, then balance mounting/being mounted
# as well as possible. This is a manual process with a bit of trial and error.

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
    
left_join(dat, lsio_df, by = "ruff_id") |>
    filter(grepl("mount|copulat", majority_behaviour)) |>
    select(fold, majority_behaviour) |>
    table()
    
write.csv(lsio_df, file = here("config", "lsio-folds.csv"), row.names = FALSE)
