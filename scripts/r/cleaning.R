# ~~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'      This script preprocesses raw data, and exports csv files with scaled and 
#'      labelled accelerometer data. The following steps are included:
#'      1. Calculate accelerometer clock error
#'      2. Adjust behaviour observation times to match accelerometer clock times
#'      3. Calculate 6-O calibrations factors
#'      4. Calculate Static bias calibration factors
#'      5. Label contiguous segments of accelerometer data with behaviour scoring
#'
#' Notes:
#'      We extract segments of accelerometer data that have behaviour labels. 
#'      Then X/Y/Z values are scaled according to a 6-orientation gravity calibration
#'      following Garde et al. 2022 (doi: 10.1111/2041-210X.13804).
#'      Acceleration bias (e.g. due to differences in harness fit and orientation) is
#'      also corrected by centering X/Y/Z values on the median.
#'      All timestamps in raw data are in internet-synchronised 'real time', 
#'      except for the raw accelerometer data which subject to clock drift.
#'      Behaviour timings are adjusted for accelerometer drift, so timestamps 
#'      from behaviours match the accelerometer.
#'      Behaviour data is then appended to accelerometer data and saved, so output
#'      consists of files with columns: timestamp; accX; accY; accZ; Behaviour.
#'      Each intact segment (contiguous, uninterrupted) of labelled accelerometer 
#'      data is saved in a separate csv file.
#'
#' Date created:
#'      May 2, 2023
#'      
#' Outputs:
#'      >200 csv files with processed accelerometer data segments are exported to
#'      ./data/clean/data-segments/
#'      
#'      Useful information about each data segment exported to
#'      ./data/clean/segment_summary.csv
#'
#'      Useful information about each accelerometer recording exported to 
#'      ./data/clean/recording_info.csv
#'      
#'      Behaviour data with drift-adjusted timestamps appended (mainly used in
#'      quality assurance scripts) exported to
#'      ./data/clean/ruff_behaviours_adjusted_for_drift.csv
#'      
# ~~~~~~~~~~~~~~~ Load packages & Initialization ~~~~~~~~~~~~~~~~~~~~~~~~----
# ---- Start logging ---
log_start_time <- Sys.time()
print(paste0("Cleaning data. Start time: ", log_start_time))

# ---- Packages ---
packages <- 
    c("dplyr",
      "lubridate",
      "purrr",
      "here",
      "ggplot2",
      "doParallel",
      "glue",
      "RSQLite",
      "tidyr",
      "data.table")
lapply(packages, require, character.only = TRUE)
source(here("scripts", "r", "cleaning-helpers.R"))
source(here("scripts", "r", "misc-utils.R"))

# ---- Script Inputs ---
path_raw_beh       <- here("data", "raw", "ruff_behaviour_tidy_2022-11-16.csv")
path_deploy_notes  <- here("data", "raw", "logger_deployment_notes.csv")
path_raw_acc_db    <- here("data", "raw", "ruff-acc.db")
path_6O_calib_info <- here("data", "raw", "calibration_recordings_6O_Apr2022.csv")
dir_acc_calib_data <- here("data", "raw", "6O_calibration_files")

# # For development #TODO remove this
# path_raw_acc_db    <- "~/ruff-acc.db"

# ---- Load deployment notes ---
deploy_notes <- read.csv(path_deploy_notes)

# Disregard recordings that had no video observations
deploy_notes <- deploy_notes |>
    filter(recording_id %in% (read.csv(path_raw_beh) |> pull(recording_id) |> unique()))

# ---- Parameters ---
# Automatically use up to 30 cores
n_CPU_cores <- min(30, parallel::detectCores())
acc_sr <- 50 # accelerometer sampling rate (Hz, samples per second)
segment_buffer <- 2.5 # number of seconds before and after acc segments e.g. to
                      # allow lead-in for FFT and static acc calculations
# 6-O scaling parameters
scaling_window <- 5 # number of seconds of data used for scaling


# ~~~~~~~~~~~~~~~ 1. Calculate clock error & drift rate ~~~~~~~~~~~~~~~~~~~~----
# ---- Compute clock error at calibration points----
## Get clock error at calibration points 1 and 3, to help calculate drift rates.
#' Clock error is given in seconds AFTER the corresponding real time. So the 
#' error is the number of seconds you need to ADD to real time to get the 
#' corresponding accelerometer time. 
#' 
#' There were 4 calibration times but 1 and 3 were the most reliable
#' Total drift (at any given time point) = drift_rate (seconds per day) * 
#'      days since the recording started + drift_yintercept (start drift). 

# Output folders for drift visualizations
if(!dir.exists(here("outputs", "drift-plots", "calib1")))
    dir.create(here("outputs", "drift-plots", "calib1"), recursive = TRUE)
if(!dir.exists(here("outputs", "drift-plots", "calib3")))
    dir.create(here("outputs", "drift-plots", "calib3"), recursive = TRUE)
print(paste0("Saving drift plots to: ", here("outputs", "drift-plots")))

# Estimate clock error for every accelerometer recording
deploy_notes$clock_error_1 <- NA
deploy_notes$clock_error_3 <- NA
    
for (i in 1:nrow(deploy_notes)){
    note <- deploy_notes[i,]
    
    # Get the clock error for this recording at each calibration timepoint
    cal1 <- get_clock_error(note$recording_id, note$cal1_time)
    cal3 <- get_clock_error(note$recording_id, note$cal3_time)
    
    # Save clock error to deployment notes
    deploy_notes$clock_error_1[i] <- cal1$clock_error_secs
    deploy_notes$clock_error_3[i] <- cal3$clock_error_secs
        
    # Save drift plots 
    ggsave(filename = here("outputs", "drift-plots", "calib1", 
                           paste0(note$recording_id, ".png")),
           plot = cal1$plot, width = 18, height = 10, units = "cm")
    ggsave(filename = here("outputs", "drift-plots", "calib3", 
                           paste0(note$recording_id, ".png")), 
           plot = cal3$plot, width = 18, height = 10, units = "cm")
}

# Manually fix wrong clock errors (based on visual examination of plots)
# 1st calib
deploy_notes[deploy_notes$recording_id == "1651_L12_2","clock_error_1"] <- 1.00
deploy_notes[deploy_notes$recording_id == "952_L43_3","clock_error_1"] <- -0.38
deploy_notes[deploy_notes$recording_id == "1372_L48_2","clock_error_1"] <- 3.46
#3rd calib
deploy_notes[deploy_notes$recording_id == "1347_L49_1","clock_error_3"] <- NA
deploy_notes[deploy_notes$recording_id == "1399_L33_1","clock_error_3"] <- -6.50
deploy_notes[deploy_notes$recording_id == "1372_L48_2","clock_error_3"] <- 6.5

# Estimate drift rate (seconds drift per day) between two calibration points 
deploy_notes$drift_rate <- 
    get_drift_rate(
        timeA = toposix_dmyhms(deploy_notes$cal3_time),
        timeB = toposix_dmyhms(deploy_notes$cal1_time),
        timeA_error = deploy_notes$clock_error_3,
        timeB_error = deploy_notes$clock_error_1)

# Estimate initial clock error (y intercept)
deploy_notes$drift_yintercept <- 
    estimate_clock(target_time = deploy_notes$start_dt,
                   calib_time = deploy_notes$cal1_time,
                   calib_error = deploy_notes$clock_error_1,
                   drift_rate = deploy_notes$drift_rate)

# ---- Get total recording duration ----
#' Get the total duration of a recording for the purpose of summarising data
#' and implementing the scaling factor calculation later. Is not adjusted for 
#' drift but is close enough for an overview.
deploy_notes$recording_duration_days <- NA
deploy_notes$acc_first_dt <- NA
deploy_notes$acc_last_dt <- NA

for (i in 1:nrow(deploy_notes)){
    start <- get_raw_acc_first_row(recording_id = deploy_notes$recording_id[i])$datetime
    end <- get_raw_acc_last_row(recording_id = deploy_notes$recording_id[i])$datetime
    deploy_notes$recording_duration_days[i] <- 
        as.numeric(difftime(
            strptime(end, format = "%Y-%m-%d %H:%M:%OS", tz = "CET"), 
            strptime(start, format = "%Y-%m-%d %H:%M:%OS", tz = "CET"), 
            units = 'days'))
    deploy_notes$acc_first_dt[i] <- start
    deploy_notes$acc_last_dt[i] <- end
}

# ~~~~~~~~~~~~~~~ 2. Get 6-O tag scaling factors ~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' For every accelerometer, a 6 orientation calibration was conducted 
#' following the procedure in Garde et al. (2022). Briefly, the logger began 
#' the calibration at rest, and then was rotated through 6 different 
#' orientations at prespecified timepoints. The total acceleration (the 
#' modulus/vectorial sum of the three axes) should equal ~1 (i.e. due to 
#' gravity) in every orientation
#' 
#' 6-orientation calibrations were conducted in a separate short recordings 
#' and are stored in csv files
# ---- Load 6O scaling info ----
# Load 6O calibration data
cal_6O_rec <- read.csv(path_6O_calib_info)
cal_6O_rec$start_dt <- paste(cal_6O_rec$start_date, cal_6O_rec$start_time)
cal_6O_rec$path <- file.path(dir_acc_calib_data, cal_6O_rec$filename)

# Convert all datetime columns to to POSIX
cal_6O_rec <- cal_6O_rec |>
    mutate(across(matches("dt"), toposix_dmyhms))

# ---- Get median vectorial sum for each axis ----
#' The overall acceleration (the modulus/vectorial sum of the three axes) is 
#' measured during calibration at each orientation.
#' Unlike the original paper, which uses the maximum acceleration for the 
#' calibration, we use median acceleration, which was more robust to small 
#' accidental movements, small errors in time adjustments, or other artefacts

# Create object to hold calibration values
calib_vals <- data.frame(
    logger_id = cal_6O_rec$logger_id,
    x1 = NA, x2 = NA, y1 = NA, y2 = NA, z1 = NA, z2 = NA)

for (i in 1:nrow(cal_6O_rec)){
    for (axis in c("x1", "x2", "y1", "y2", "z1", "z2")){
        calib_vals[i, axis] <- get_median_vec_sum(
            dat_path = cal_6O_rec[i, 'path'], 
            calib_time = cal_6O_rec[i, paste0(axis, "_cal_dt")],
            start_time = cal_6O_rec[i, 'start_dt'])
    }
}

scaling_factors_6O <- calib_vals |>
    rowwise() |>
    mutate(scale_x_6O = 1 / (mean(c(x1, x2))),
           scale_y_6O = 1 / (mean(c(y1, y2))),
           scale_z_6O = 1 / (mean(c(z1, z2)))) |>
    ungroup() |>
    select(logger_id, scale_x_6O, scale_y_6O, scale_z_6O)

# Start populating recording info df
recording_info <- left_join(deploy_notes, scaling_factors_6O, by = "logger_id")
recording_info <- recording_info |> 
    rename(ruff_id = ruff_id_number,
           deploy_time_start = start_dt,
           deploy_time_end = stop_dt)

# ~~~~~~~~~~~~~~~ 3. Adjust behavioural data for drift ~~~~~~~~~~~~~~~~~~~~~----
#' Behavioural observations are stored in a csv file, with timestamps that are
#' synchronised to internet time servers. Because the accelerometer data have 
#' imprecise timestamps, the times of these two data sources are misaligned, and
#' this needs to be corrected. It is easier to adjust the behaviour labels so 
#' that they match the accelerometer's clock, due to the smaller size of the 
#' behaviour dataset.
#'
# ---- Prepare behaviour data ----
# Read behaviour data 
beh_data <- read.csv(path_raw_beh) |>
    rename(beh_start_real = start_dt_real,
           beh_stop_real = stop_dt_real)

# Add drift information as new columns
beh_data <- left_join(
    beh_data, 
    recording_info |>
        select(recording_id, deploy_time_start, drift_rate, drift_yintercept), 
    by='recording_id')

# ---- Implement time adjustments ----
# Add new columns for accelerometer clock time with suffix _acc
beh_data <- beh_data |>
    mutate(beh_start_acc = 
               adjust_time(toposix_ymdhms(beh_start_real), 
                           toposix_dmyhms(deploy_time_start), 
                           drift_rate, drift_yintercept),
           beh_stop_acc = 
               adjust_time(toposix_ymdhms(beh_stop_real), 
                           toposix_dmyhms(deploy_time_start), 
                           drift_rate, drift_yintercept))


# ~~~~~~~~~~~~~~~ 4. Get deployment median calibrations (static acc bias) ~~----
#' An additional scaling factor is applied based on the median static 
#' acceleration value for each axis. This adjustment helps account for
#' biases in static acceleration that are due to how the logger is fitted. 
#' True differences in posture or activity levels could also contribute to the
#' static acceleration bias, but the adjustment was overall beneficial for model 
#' performance. Some other bias adjustment methods were tested, but using the 
#' global median was just as effective and simpler.

# ---- Compute scaling factor for every data file ----
#' Note: assumes parallel computing is available.
cl <- makePSOCKcluster(n_CPU_cores) 
registerDoParallel(cl)
scales_staticbias <- foreach(i = 1:nrow(recording_info), .packages = packages) %dopar% {
    #' Load all accelerometer data from a recording, and calculate the median
    #' of each axis. Exclude first and last 40h of recording (logger may not be 
    #' deployed on bird yet + some initial adjustment period is allowed)
    secs_exclude <- (40 * 60 * 60)
    start <- toposix_ymdhms(recording_info$acc_first_dt[i])
    end <- toposix_ymdhms(recording_info$acc_last_dt[i])
    
    dat <- get_raw_acc_segment(recording_info$recording_id[i],
                               segment_start = start + secs_exclude,
                               segment_end = end - secs_exclude)
    # # replace with actual recording_id
    # dat$recording_id <- recording_info$recording_id[i]
    
    # Exclude nighttime hours because only daytime behaviours were observed 
    # and scored, nighttime behaviours were not examined in this study.
    
    dat <- dat |> 
        mutate(datetime = strptime(datetime, format = "%Y-%m-%d %H:%M:%OS", 
                                   tz = "CET")) |>
        filter(datetime$hour > 08 & datetime$hour < 17)

    # Re-scale by 6O scaling factor
    dat$accX <- dat$accX * recording_info$scale_x_6O[i]
    dat$accY <- dat$accY * recording_info$scale_y_6O[i]
    dat$accZ <- dat$accZ * recording_info$scale_z_6O[i]
    
    # Compute static acc (rolling mean for each axis with 2s window)
    dat$statX <- 
        frollmean(dat$accX, 100, algo = "fast", align = "center", na.rm = TRUE)
    dat$statY <- 
        frollmean(dat$accY, 100, algo = "fast", align = "center", na.rm = TRUE)
    dat$statZ <- 
        frollmean(dat$accZ, 100, algo = "fast", align = "center", na.rm = TRUE)
    
    # Get the median static for all datapoints by recording.
    return(list(recording_id = recording_info$recording_id[i],
                scale_x_deploybias = median(dat$statX, na.rm = TRUE), 
                scale_y_deploybias = median(dat$statY, na.rm = TRUE),
                scale_z_deploybias = median(dat$statZ, na.rm = TRUE)))
}
stopCluster(cl)
registerDoSEQ()

recording_info <- left_join(recording_info, bind_rows(scales_staticbias), 
                            by = "recording_id")

# ~~~~~~~~~~~~~~~ 5. Segment and annotate accelerometer data ~~~~~~~~~~~~~~~----
#' Here we identify accelerometer data that has concurrent video behaviour 
#' scoring available. Behaviour labels are appended to the data as a new column. 
#' X/Y/Z acceleration values are adjusted according to the scaling factors. 
#' 
#' Behaviour scoring was not always contiguous, i.e. there are some large gaps. 
#' To prevent unnecessary processing of unlabelled data, and because our Hidden 
#' Markov Models use contiguous sequences, we split the data into contiguous
#' segments, process it, and save each segment in a separate csv file. Each 
#' accelerometer recording may therefore produce multiple segments of data

# ---- Find contiguous segments of data ----
#' Get uninterrupted segments of video-scored behaviour (max time gap between 
#' behaviours of 0.5 seconds

beh_data <- beh_data |>
    group_by(recording_id) |>
    arrange(toposix_ymdhms(beh_start_real), .by_group = TRUE) |>
    ungroup() |>
    mutate(segment_id = find_beh_segments(recording_id,
                                          toposix_ymdhms(beh_start_real), 
                                          toposix_ymdhms(beh_stop_real), 
                                          max_gap = 0.5))

# Get the start time, end time, and other useful info about each segment
segment_summary <- beh_data |> 
    group_by(segment_id) |>
    summarise(recording_id = first(recording_id),
              seg_start_real = first(beh_start_real),
              seg_start_acc = first(beh_start_acc),
              seg_stop_real = last(beh_stop_real),
              seg_stop_acc = last(beh_stop_acc))

segment_summary <- segment_summary |> 
    left_join(recording_info |> 
                  select(recording_id, filename, matches("scale"), 
                         acc_first_dt, acc_last_dt))

# ---- Attach behaviours to segments, export to csv ----
# For every segment, load every behaviour one by one, concatenate them, and 
# join them back to the segment
# Load each behaviour individually, then stick them together
segment_dir <- here("data", "clean", "data-segments")
if(!dir.exists(segment_dir))
    dir.create(segment_dir, recursive = TRUE)

cl <- makePSOCKcluster(n_CPU_cores) 
registerDoParallel(cl)
foreach(i = 1:nrow(segment_summary), .packages = packages, .inorder = FALSE) %dopar% {
    #' Loop over every segment. Load the raw accdata, scale the acceleration 
    #' values, append the behaviour label, then save as csv. A small read buffer
    #' of unlabelled data is included either side of the target segment to 
    #' facilitate static acc FFT-based calculations later which need a lead-in

    this_rec_id <- segment_summary$recording_id[i]
    this_seg_behs <- beh_data |> 
        filter(segment_id == segment_summary$segment_id[i])

    # Load the accelerometer data for this segment.
    seg_dat <- get_raw_acc_segment(recording_id = segment_summary$recording_id[i],
                                   segment_start = segment_summary$seg_start_acc[i],
                                   segment_end = segment_summary$seg_stop_acc[i], 
                                   buffer_secs = segment_buffer)
    
    # Rescale data with 6O scaling factors
    seg_dat <- seg_dat |> 
        mutate(accX = accX * segment_summary$scale_x_6O[i],
               accY = accY * segment_summary$scale_y_6O[i],
               accZ = accZ * segment_summary$scale_z_6O[i])
    
    # Rescale with static bias
    seg_dat <- seg_dat |> 
        mutate(accX = accX - segment_summary$scale_x_deploybias[i],
               accY = accY - segment_summary$scale_y_deploybias[i],
               accZ = accZ - segment_summary$scale_z_deploybias[i])
    
    #' The following process is used to append labels to the accelerometer data.
    #' The start_time and stop_times given in the behaviour scoring dataset need 
    #' to be transformed into a dataframe with 3 columns: timestamp, behaviour,
    #' and behavioural event id. Timestamps should match the accelerometer 
    #' data's timestamps precisely. 
    #' 
    #' Matching the timestamps from behaviour data to accelerometer data got a 
    #' bit awkward with floating-point precision and fractional seconds, and 
    #' matching up the correct event id to but a fast & simple process is to 
    #' to select rows from the accelerometer database that correspond to the 
    #' start/stop of the respective behaviour. Then apply the behaviour label to 
    #' all of those timestamps. Now the timestamps are an exact match and easily 
    #' can be joined back to the accelerometer data. 

    beh_timestamps <- vector("list", nrow(this_seg_behs))
    for(j in 1:nrow(this_seg_behs)){
        # Fetch only the datetime column from the database for rows that 
        # overlap with this the timing of behaviour from this recording.
        # Bit awkward to double-read, but this
        dat_this_beh <- 
            get_raw_acc_segment(recording_id = segment_summary$recording_id[i],
                                segment_start = this_seg_behs$beh_start_acc[j],
                                segment_end = this_seg_behs$beh_stop_acc[j],
                                select_col = "datetime")

        # Add annotations to data
        dat_this_beh$behaviour <- this_seg_behs$behaviour[j]
        dat_this_beh$beh_event_id <- this_seg_behs$beh_event_id[j]
        beh_timestamps[[j]] <- dat_this_beh
    }
    
    # concatenate the annotations
    beh_timestamps <- bind_rows(beh_timestamps)
    
    # Join the labels to the accelerometer data from this segment
    seg_dat_labelled <- left_join(seg_dat, beh_timestamps, by = "datetime")
    
    # Write out csv file
    seg_dat_labelled$recordingid <- NULL
    seg_dat_labelled$recording_id <- this_rec_id
    fwrite(seg_dat_labelled, 
           file = file.path(segment_dir, paste0(this_rec_id, "_s", i, ".csv")))
}
stopCluster(cl)
registerDoSEQ()

# Other outputs
outpath_beh_adjusted <- here("data", "clean", "ruff_behaviours_adjusted_for_drift.csv")
outpath_recording_info <- here("data", "clean", "recording_info.csv")
outpath_segment_summary <- here("data", "clean", "segment_summary.csv")

# Convert POSIX times to character with customised rounding of fractional secs
beh_data <- beh_data |> 
    mutate_at(c('beh_start_acc', 'beh_stop_acc'), ~prtime(., 2))

write.csv(beh_data, file=outpath_beh_adjusted, row.names=FALSE)
write.csv(recording_info, outpath_recording_info, row.names = FALSE, na = "")
write.csv(segment_summary, file = outpath_segment_summary, row.names = FALSE)

print(paste0("Finished data cleaning. Duration: ", 
             round(difftime(Sys.time(), log_start_time, units = "mins"), 1), 
             " minutes"))
