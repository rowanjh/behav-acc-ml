# Quantifying mating behaviour using accelerometry and machine learning

------------------------------------------------------------------------

This is the repository for: "Quantifying mating behaviour using accelerometry and machine learning: challenges and opportunities" *Unpublished manuscript*

# Overview

------------------------------------------------------------------------

This analysis uses accelerometry data and machine learning techniques to classify 13 behaviours in captive male shorebirds (Ruffs; *Calidris pugnax*). An end-to-end analysis is presented, including:

1.  Calibration and time synchronisation of raw accelerometer data
2.  Windowing data & feature extraction
3.  Training hidden Markov models, random forests, and neural networks
4.  Evaluating model performance

## Instructions

------------------------------------------------------------------------

1.  Download `code.zip`, extract into project folder
2.  Download `raw-data-db.zip` and `raw-data-other.zip` into directory `./data/raw`, and extract
3.  Open script `run.R`. This script gives the overview of all analyses and is the starting point.

## Package management & installation

------------------------------------------------------------------------

This project uses renv for R package management. Launch the .Rproj file, then simply run `renv::restore()` to install all of the correct packages. This requires build tools that allow compilation of packages from source (e.g. Rtools for Windows). renv will only work if the following three files are in the project: `renv/activate.R` `renv.lock`, and `.Rprofile`.

Neural networks need to be run using python. We recommend a setting up a conda environment and using a GPU if available (see `/scripts/r/py/main.py` for all instructions).

## Notes

------------------------------------------------------------------------

Some parts of the analysis use parallel loops and can have high memory requirements. Use fewer parallel workers to reduce the memory footprint.

Raw accelerometer data are stored in a large sqlite database (\~81GB unzipped), ensure sufficient disk space is available. The database is interfaced and processed through R functions, and intermediate cleaned csv files are produced.

## Project Structure

------------------------------------------------------------------------

-   `scripts/` contains all code for analyses
-   `data/raw` contains all raw accelerometer and behavioural data
-   `data/clean` contains calibrated, time-synchronised data with labels
-   `data/windowed` contains the full windowed dataset with summary features for each window
-   `outputs` contains model results, exploratory plots, and intermediate outputs

# Acknowledgements

------------------------------------------------------------------------

Parts of the code for this analysis were adapted from Leos-Barajs et al. (2017) and Bock et al. (2021).

*Leos-Barajas, V., Photopoulou, T., Langrock, R., Patterson, T. A., Watanbe, Y. Y., Murgatroyd, M., & Papastamatiou, Y. P. (2017). Analysis of animal accelerometer data using hidden Markov models. Methods in Ecology and Evolution, 8(2), 161--173. <https://doi.org/doi:10.1111/2041-210X.12657>*

*Bock, M., Hölzemann, A., Moeller, M., & Van Laerhoven, K. (2021). Improving Deep Learning for HAR with shallow LSTMs. 2021 International Symposium on Wearable Computers, 7--12. <https://doi.org/10.1145/3460421.3480419>; <https://github.com/mariusbock/dl-for-har>*
