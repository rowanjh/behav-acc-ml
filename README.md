# Quantifying mating behaviour using accelerometry and machine learning

------------------------------------------------------------------------

This is the repository for the following paper:

`Aulsebrook, A., Jacques-Hamilton, R., & Kempenaers, B (2024) Quantifying mating behaviour using accelerometry and machine learning: challenges and opportunities, Animal Behaviour (207) p55-76, https://doi.org/10.1016/j.anbehav.2023.10.013`

# Overview

------------------------------------------------------------------------

This analysis uses accelerometry data and machine learning techniques to classify 13 behaviours in captive male shorebirds (Ruffs; *Calidris pugnax*). An end-to-end analysis is presented, including:

1.  Calibration and time synchronisation of raw accelerometer data
2.  Windowing data & feature extraction
3.  Training hidden Markov models, random forests, and neural networks
4.  Evaluating model performance

The dataset is available from a repository on [Edmond](https://edmond.mpdl.mpg.de/privateurl.xhtml?token=3c5da533-d851-4988-994c-f984c5f1b1a4) (<doi:10.17617/3.KERIBB>)

## Instructions

------------------------------------------------------------------------

1.  Download or clone this code repository. The script run.R gives the overview of all analyses and is the starting point.

2.  Install packages (see `Package management & installation`)

3.  Download two zip files containing the data for this analysis: [raw-data-db.zip](https://edmond.mpdl.mpg.de/api/access/datafile/212351) and [raw-data-other.zip](https://edmond.mpdl.mpg.de/api/access/datafile/212350), into directory `./data/raw`.

4.  Extract zipped data files into directory `./data/raw` (code is provided for this within run.R)

5.  Run code in run.R to reproduce all analyses

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

## Dataset

------------------------------------------------------------------------

The dataset is available at [Edmond](https://edmond.mpdl.mpg.de/privateurl.xhtml?token=3c5da533-d851-4988-994c-f984c5f1b1a4). Two zip files are included, which contain the following:

-   `ruff-acc.db` is an sqlite3 database containing raw accelerometer data from 30 birds, each recording having a duration of 5-8 days. Approx 80GB disk space required.
-   `logger_deployment_notes.csv` contains information about each accelerometer deployment (e.g. deployment times, bird measurements)
-   `ruff_behaviour_tidy_2022-11-16.csv` contains bird behavioural data, scored from videos using BORIS software
-   `calibration_recordings_6O_Apr2022.csv` contains information about accelerometer calibration sessions
-   `6O_calibration_files` is a directory containing raw accelerometer data recorded during calibration sessions.

More detail about each file is given in the data readme, which is available on Edmond or at `./data/raw/README-rawdata.md`

# Acknowledgements

------------------------------------------------------------------------

Parts of the code for this analysis were adapted from Leos-Barajs et al. (2017) and Bock et al. (2021).

*Leos-Barajas, V., Photopoulou, T., Langrock, R., Patterson, T. A., Watanbe, Y. Y., Murgatroyd, M., & Papastamatiou, Y. P. (2017). Analysis of animal accelerometer data using hidden Markov models. Methods in Ecology and Evolution, 8(2), 161--173. <https://doi.org/doi:10.1111/2041-210X.12657>*

*Bock, M., Hölzemann, A., Moeller, M., & Van Laerhoven, K. (2021). Improving Deep Learning for HAR with shallow LSTMs. 2021 International Symposium on Wearable Computers, 7--12. <https://doi.org/10.1145/3460421.3480419>; <https://github.com/mariusbock/dl-for-har>*
