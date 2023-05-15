# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'      This script implements a sliding window over labelled accelerometer 
#'      data, and computes features for each window. Time-domain features such
#'      as static and dynamic acceleration are computed, as well as  
#'      frequency-domain features using fast fourier transforms (FFT).
#' 
#'      A csv file is output with windows (epochs) in rows and features in columns.
#'      Windows are assigned a behaviour label according to the majority behaviour 
#'      present in that window
#'
#' Notes: 
#'      A sufficient buffer is required before and after labelled segments of 
#'      data to allow lead-in for FFT and static acceleration calculations.
#'
#' Date Created: 
#'      May 2, 2023
#' 
#' Outputs:
#'      The windowed dataset is output to:
#'      ./data/windowed/windowed-data.csv
#'      
# ~~~~~~~~~~~~~~~ Load packages & Initialization ~~~~~~~~~~~~~~~~~~~~~~~~----
# ---- Start logging ---
log_start_time <- Sys.time()
print(paste0("Windowing data. Start time: ", log_start_time))

# ---- Packages ---
packages <- 
    c("here",
      "dplyr",
      "lubridate",
      "purrr",
      "ggplot2",
      "doParallel",
      "glue",
      "RSQLite",
      "tidyr",
      "data.table",
      "stringr",
      "moments",
      "dtw")
lapply(packages, library, character.only = TRUE)
source(here("scripts", "r", "windowing-helpers.R"))
source(here("scripts", "r", "misc-utils.R"))

# ---- Script Inputs ---
dir_cleaned_data <- here("data", "clean", "data-segments")
beh_file_path <- here("data", "clean", "ruff_behaviours_adjusted_for_drift.csv")

segment_summary <- read.csv(here("data", "clean", "segment_summary.csv"))
segment_summary <- segment_summary |>
    mutate(duration = toposix_ymdhms(seg_stop_real) - 
                          toposix_ymdhms(seg_start_real)) |>
    arrange(desc(duration)) # send long files to the cores first, more optimised

behaviour_labels <- read.csv(beh_file_path) |> pull(behaviour) |> unique()

recording_info <- read.csv(here("data", "clean", "recording_info.csv"))

# ---- Parameters ---
n_CPU_cores <- min(20, parallel::detectCores())
acc_sr <- 50 # sampling rate

# Sliding window parameters
window_size_static <- 2 # rolling window size for static acc (seconds)
window_size_RMS <- 0.25 # window size for smoothing jerk in RMS jerk (seconds)
window_size <- 1 # window size in seconds
window_step <- 1 # how much to step for each window (in seconds)
window_buffer_FFT <- 1 # seconds of acc data from either side of a window to
                       # include in FFT calculations
fft_num_bins <- 10  # number of bins for the fft spectrum.  DC and the 
                    # fundamental are always included in addition to num_bins
fft_max_freq <- 25 # Max frequency to use for FFT, in Hz. (max = nsamples/2)


# ~~~~~~~~~~~~~~~ Implement sliding window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
# Loop over every available segment of scored accelerometer data
cl <- makePSOCKcluster(n_CPU_cores) 
registerDoParallel(cl)
results <- foreach(i = 1:nrow(segment_summary), .packages = packages, 
                   .inorder = FALSE) %dopar% {
    this_rec_id <- segment_summary$recording_id[i]
    this_seg_id <- segment_summary$segment_id[i]
    thisfile <- file.path(dir_cleaned_data, 
                          glue("{this_rec_id}_s{this_seg_id}.csv"))
    
    # Load this segment
    seg_dat <- fread(thisfile, data.table = FALSE, tz = "") 

    # ---- Compute static/dynamic acceleration in 50hz data ---
    seg_dat <- seg_dat |>
        mutate(accXYZ = sqrt(accX^2 + accY^2 + accZ^2)) |>
        get_static(static_window_size = window_size_static, sr = acc_sr) |>
        get_dynamic(sr = acc_sr) |>
        get_jerk(sr = acc_sr, RMS_window = window_size_RMS)   
    
    # ---- Window data ---
    #' Create data frame with 1 row per window, and includes start/end times for
    #' each window
    windowed_data <- 
        get_window_timestamps(sw_start = toposix_ymdhms(seg_dat$datetime[1]),
                              sw_end = toposix_ymdhms(last(seg_dat$datetime)),
                              win_size = window_size, 
                              win_step = window_step)
    # Add id columns 
    windowed_data$recording_id <- this_rec_id
    windowed_data$segment_id <- this_seg_id
    windowed_data$loop_j <- 1:nrow(windowed_data)
    
    # Initialise columns for the proportion of a window scored as each behavior
    windowed_data <- windowed_data %>% 
        cbind(as.data.frame(matrix(0, nrow = nrow(windowed_data), 
                                   ncol = length(behaviour_labels))) %>%
                  set_names(paste0("beh_", behaviour_labels)))
    
    # Print progress message
    cat(paste0("Starting sliding window for:\n ------Recording ", 
                 this_rec_id, ". Segment ", this_seg_id,
                 ". n_windows = ", nrow(windowed_data)))

    # Loop over all windows within this segment, get features and behaviours
    for (j in 1:nrow(windowed_data)){
        cat(glue("{j} "))
        
        # Get the rows of the segment data that correspond to this window
        if(j == 1){
            rows_this_window <- 
                get_rows_for_window(window_index = j,
                                    sample_rate = acc_sr, 
                                    win_size = window_size, 
                                    win_step = window_step)
            data_this_window <- seg_dat[rows_this_window,]
        }
        if(j > 1){
            #' For some features, the next and the previous window's data are 
            #' used in the calculation. Current and prev windows cached 
            #' to avoid re-loading 
            data_prev <- data_this_window
            data_this_window <- data_next # the old data_next from prev loop
            rows_this_window <- rows_next_window # also from prev loop
        } 
        if(j < nrow(windowed_data)){
            # Load the next window, except for the final iteration
            rows_next_window <-
                get_rows_for_window(window_index = j+1, 
                                    sample_rate = acc_sr,
                                    win_size = window_size,
                                    win_step = window_step)
            data_next <- seg_dat[rows_next_window,]
        }

        # If the end of the segment has been reached, we can stop.
        if(any(rows_this_window > nrow(seg_dat))) break
        
        # If there is no behaviour scored, this window is not of interest.
        if(all(data_this_window$behaviour == "")) next
        
        # ---- Compute basic statistical features ----# 
        # Basic statistics for all raw/static/dynamic acceleration columns 
        general_summary_features <- data_this_window |>
            select(-datetime, -behaviour, -beh_event_id, -recording_id) |>
            summarise_all(list(mean=mean, sd=sd, median=median, min=min,
                               max=max, kurtosis=kurtosis,skewness=skewness))
        
        windowed_data[j,names(general_summary_features)] <- 
            general_summary_features
        
        # ---- Features from Human Activity Recognition literature ----# 
        HAR_features <- data_this_window %>%
            get_HAR_features()
        
        windowed_data[j,names(HAR_features)] <- HAR_features
        
        # ---- Compute frequency-based features ----# 
        # Expand FFT window size by adding a buffer to start and end
        FFT_rows <- (min(rows_this_window) - window_buffer_FFT * acc_sr) : 
            (max(rows_this_window) + window_buffer_FFT * acc_sr)
        # Only do FFT computations if the window has missing acceleration data
        if(!any(FFT_rows < 0) & !any(FFT_rows > nrow(seg_dat))) {
            fft_data <- seg_dat[FFT_rows,]
            
            fft_feats <-
                fft_data %>%
                get_power_features(sr = acc_sr,
                                       maxfreq = fft_max_freq,
                                       fft_nbins = fft_num_bins)
            windowed_data[j,names(fft_feats)] <- fft_feats
        }

        
        # ---- Compute CCA and CSA etc. features ----#
        #' Compute features that describe how interaxis correlations and 
        #' interaxis similarities change from the previous window to the present
        #' window, and from the present window to the next window
        if(j > 1 & j < nrow(windowed_data)){
            #' Don't compute if the next window is empty. 
            #' TODO: split next and prev calcualtions so that they aren't 
            #' dependent... should be unnecessary with sufficient buffer around
            #' the segment of accelerometer data.
            if(!any(is.na(data_next$accY))) {
                CCA <- calc_CCA(data_this_window, data_prev, data_next)
                CSA <- calc_CSA(data_this_window, data_prev, data_next)
                context_feats <- unlist(c(CCA, CSA))
                windowed_data[j,names(context_feats)] <- context_feats
            }

        }
        # ---- Get behaviour proportions ----# 
        #' Get the proportion of time spent on each behaviour, then work
        #' out what the majority behaviour was.
        behaviours <- table(data_this_window$behaviour) / acc_sr
        
        # Add majority behaviour and transitions
        majority_beh <- names(sort(behaviours, decreasing = TRUE))[1]
        # If the majority of the window has no behaviour label
        if(majority_beh == "") 
            majority_beh <- "beh_unknown"
        windowed_data[j, "majority_behaviour"] <- majority_beh
        windowed_data[j, "mixed"] <- as.integer(length(behaviours) > 1)
        
        # Add proportions for all behaviours
        for (k in 1:length(behaviours)){
            this_beh <- names(behaviours)[k]
            if(this_beh == "") next
            windowed_data[j, paste0("beh_", this_beh)] <-
                behaviours[k]
        }
        # Add event id
        windowed_data[j, "beh_event_id"] <- unique(data_this_window$beh_event_id) %>%
            paste(., collapse = "_")
    }
    # Remove any rows with no behaviour label
    windowed_data <- windowed_data %>% filter(!is.na(majority_behaviour))
    
    # ---- Mark transitions ----# 
    #' The first and last window in a set of consecutive windows with the same 
    #' majority behaviour is a transition. Any window with multiple event IDs
    #' is also a transition.
    windowed_data <- windowed_data %>% 
        mutate(transition = majority_behaviour != lag(majority_behaviour, default = "none") | 
                   majority_behaviour != lead(majority_behaviour, default = "none") |
                   grepl("_", beh_event_id))
    
    # Add window id for this segment
    windowed_data$win_in_segment_id <- 1:nrow(windowed_data)
    # Add proportion of epoch with behaviour unknown
    windowed_data$beh_unknown <- 1 - rowSums(windowed_data %>% select(matches("^beh_(?!event)", perl = TRUE)), na.rm=TRUE)
    # return windowed_data from the foreach loop
    windowed_data
}
stopCluster(cl)
registerDoSEQ()

# ---- Save final dataset ----# 
# Combine results from different compute nodes
all_windows <- bind_rows(results)

# Add a unique window id. 
all_windows$window_id <- 1:nrow(all_windows)

# Add morph and ruff_id as column
all_windows <- all_windows %>% 
    left_join(recording_info %>% select(recording_id, ruff_id, morph), by = "recording_id")

# Reorder columns
all_windows <- all_windows %>% 
    select(recording_id, ruff_id, segment_id, win_in_segment_id, loop_j, window_id, 
           beh_event_id, window_start, window_end, transition, majority_behaviour, morph,
           matches("beh"), matches("[XYZ]"), everything())

# Remove any windows with beh_unknown as majority. These aren't used in analysis
all_windows <- all_windows %>% filter(majority_behaviour != "beh_unknown")

out_path <- here("data", "windowed", "windowed-data.csv")
if (!dir.exists(here("data", "windowed")))
    dir.create(here("data", "windowed"))

print(paste0("Data windowing finished. Duration: ", 
             round(difftime(Sys.time(), log_start_time, units = 'mins'), 1), 
             " minutes."))
print(paste(out_path))

fwrite(all_windows, out_path)
