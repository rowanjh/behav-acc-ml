---
title: "Data cleaning quality check"
author: "Rowan Jacques-Hamilton"
date: "2023-05-04"
output: 
    html_document
---

This notebook checks for bugs in the data cleaning, including domains of: 
* Behaviour label alignment & drift rates 
* Accelerometer data scaling (6-O desk scaling, and on-bird median scaling)

It was mainly for internal use during development and is not as well polished or
commented as the main analysis scripts.


```r
library(here)
library(ggplot2)
library(dplyr)
library(purrr)
library(glue)
library(tidyr)
# library(gridExtra)
library(doParallel)
library(RSQLite)
library(data.table)
library(lubridate)
library(stringi)
library(patchwork)
source(here("scripts","cleaning-helpers.R"))

# Critical dated output directory: change to latest version
processed_data_dir <- here("data","clean","data-segments")

# Raw files
path_raw_acc_db <- "~/ruff-acc.db"
path_deploy_notes <- here("data", "raw", "logger_deployment_notes.csv")
path_6O_calib_info <- here("data", "raw", "calibration_recordings_6O_Apr2022.csv")
dir_acc_calib_data <- here("data", "raw", "6O_calibration_files")

# Processed files
beh_obs_file <- here("data", "clean", "ruff_behaviours_adjusted_for_drift.csv")
path_recording_info <- here("data", "clean", "recording_info.csv")
segment_summary_file <- here("data", "clean", "segment_summary.csv")

## Parameters
num_cores <- min(20, parallel::detectCores())
options(digits.secs = 4)
acc_sr = 50 # sampling rate, Hz (samples / second), used by some helpers

# Helpers
last <- function(x) {return(x[length(x)])}

## Load beh data
beh_obs_data <- fread(beh_obs_file, data.table = FALSE, tz = "")

# Convert times to useful format, subset to useful columns
beh_obs_data <- beh_obs_data |> 
    mutate(beh_start_acc = toposix_ymdhms(beh_start_acc),
           beh_stop_acc = toposix_ymdhms(beh_stop_acc), 
           beh_start_real = toposix_ymdhms(beh_start_real), 
           beh_stop_real = toposix_ymdhms(beh_stop_real)) |>
  select(recording_id, beh_event_id, behaviour, beh_start_real, beh_stop_real, 
         beh_start_acc, beh_stop_acc, duration_secs)

# Load segment summary: shows segments of continuously scored behaviour
segment_summary <- read.csv(segment_summary_file) |>
    mutate(seg_start_real = toposix_ymdhms(seg_start_real), 
           seg_start_acc = toposix_ymdhms(seg_start_acc), 
           seg_stop_real = toposix_ymdhms(seg_stop_real),
           seg_stop_acc = toposix_ymdhms(seg_stop_acc))

## Load deployment notes
deploy_notes <- read.csv(path_deploy_notes)

## Load recording info
recording_info <- read.csv(path_recording_info)
```
Plots for clock error calculations

```r
# Get bird acc segments at calibration points
# Calculate drift rates from that
for(i in 1:nrow(recording_info)){
    this_recording_info <- recording_info[i,]
    this_id <- this_recording_info$recording_id
    this_deploy <- deploy_notes |> filter(recording_id %in% this_id)
    this_c1time <- strptime(this_deploy$cal1_time, "%d/%m/%Y %H:%M:%S")
    this_c3time <- strptime(this_deploy$cal3_time, "%d/%m/%Y %H:%M:%S")
    db <- dbConnect(SQLite(), path_raw_acc_db)
    thisID <- as.numeric(dbGetQuery(db, glue("SELECT id FROM recordings WHERE ", 
                                             "recording_id IS '{this_id}'")))
    this_c1_error <- this_recording_info$clock_error_1
    this_c3_error <- this_recording_info$clock_error_3
    c1_data <- dbGetQuery(db, glue(
            "SELECT * 
             FROM acc 
             WHERE datetime > '{this_c1time - 15}' 
                AND datetime <= '{this_c1time + 15}' 
                AND recording_id IS {thisID}")
    ) |> mutate(datetime = as.POSIXct(datetime, format = "%Y-%m-%d %H:%M:%OS", tz = "CET"))
    
    c3_data <- dbGetQuery(db, glue(
            "SELECT * 
             FROM acc 
             WHERE datetime > '{this_c3time - 15}' 
                AND datetime <= '{this_c3time + 15}' 
                AND recording_id IS {thisID}")
    ) |> mutate(datetime = as.POSIXct(datetime, format = "%Y-%m-%d %H:%M:%OS", tz = "CET"))
    dbDisconnect(db)

    c1_data |>
        ggplot(aes(x = datetime, y = accZ)) +
        geom_point() +
        geom_vline(aes(xintercept = this_c1time + this_c1_error), 
                   size = 1, alpha = 0.5) +
        scale_x_datetime(labels = scales::date_format("%Y-%m-%d %H:%M:%OS2", tz = "CET")) +
        labs(title = glue("i={i} id={this_id} calibration point 1")) +
    (c3_data |>
        ggplot(aes(x = datetime, y = accZ)) +
        geom_point() +
        geom_vline(aes(xintercept = this_c3time + this_c3_error ), 
                   size = 1, alpha = 0.5) +
        scale_x_datetime(labels = scales::date_format("%Y-%m-%d %H:%M:%OS2", tz = "CET")) +
        labs(title = "calibration point 3"))
}
```

Check drift rate & intercept calculations

```r
any_problems <- FALSE
for(i in 1:length(unique(beh_obs_data$recording_id))){
    cat(glue("{i}."))
    thisid <- unique(beh_obs_data$recording_id)[i]
    this_deploy <- deploy_notes |> filter(recording_id %in% thisid)
    this_rec_info <- fread(path_recording_info) |> filter(recording_id %in% thisid)
    
    this_recording_origin_time <- toposix_dmyhms(this_deploy$start_dt)
    this_c1time <- toposix_dmyhms(this_deploy$cal1_time)
    this_c3time <- toposix_dmyhms(this_deploy$cal3_time)
    this_c1_error <- this_rec_info$clock_error_1
    this_c3_error <- this_rec_info$clock_error_3
    this_drift_rate <- this_rec_info$drift_rate
    this_drift_yintercept <- this_rec_info$drift_yintercept
    
    recalculated_drift_rate <- (this_c3_error - this_c1_error) / 
        as.numeric(difftime(this_c3time, this_c1time, units = "days"))
    recalculated_drift_yint <- as.numeric(difftime(this_c1time, this_recording_origin_time, units = "days")) * 
        recalculated_drift_rate + this_c1_error
    # small tolerance due to rounding dts
    if(!all(abs(this_drift_rate - recalculated_drift_rate) < 0.01, 
            abs(this_drift_yintercept - recalculated_drift_yint) < 0.01)){
        print(glue("i={i}, {thisid} drift rate or y intercept calcualtion did not match"))
        any_problems <- TRUE
    }
}
```

```
## 1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.
```

```r
if(!any_problems) print("All OK")
```

```
## [1] "All OK"
```

Check conversion of behaviours real time to accelerometer time

```r
# Pick a few times, reconvert behaviour times to accelerometer times. Check
#   that the notated time matches the expected time
any_problems <- FALSE
for(i in 1:length(unique(beh_obs_data$recording_id))){
    cat(glue("{i}."))
    thisid <- unique(beh_obs_data$recording_id)[i]
    this_rec_info <- fread(path_recording_info, tz = "") |> 
        filter(recording_id %in% thisid) |>
        mutate(deploy_time_start = toposix_dmyhms(deploy_time_start),
               deploy_time_end = toposix_dmyhms(deploy_time_end),
               acc_first_dt = toposix_ymdhms(acc_first_dt))
    behs_this_id <- beh_obs_data |> filter(recording_id == thisid)
    
    # rows <- seq(1, nrow(behs_this_id), 50)
    
    time_elapsed <- as.numeric(difftime(behs_this_id$beh_start_real,
                                        this_rec_info$deploy_time_start, 
                                        units = "days"))
    behs_this_id$start_acc_recalculated <- 
        behs_this_id$beh_start_real + 
        this_rec_info$drift_rate * time_elapsed + 
        this_rec_info$drift_yintercept
    behs_this_id$stop_acc_recalculated <- 
        behs_this_id$beh_stop_real + 
        this_rec_info$drift_rate * time_elapsed + 
        this_rec_info$drift_yintercept
    
    starttime_diff <- 
        abs(as.numeric(difftime(behs_this_id$start_acc_recalculated, 
                                behs_this_id$beh_start_acc, "secs")))
    endtime_diff <- 
        abs(as.numeric(difftime(behs_this_id$stop_acc_recalculated,
                                behs_this_id$beh_stop_acc, "secs")))
    start_match <- starttime_diff < 0.01
    end_match <- endtime_diff < 0.01

    if (!all(start_match & end_match)){
        print(glue("\n{thisid} behaviour real to acc time adjustments mismatched, ",
                   "{sum(!start_match)} start problems, {sum(!end_match)} end problems"))
        print(behs_this_id[!start_match|!end_match,] |> select(matches("start_acc|stop_acc")))
        any_problems <- TRUE
    }
}
```

```
## 1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.
```

```r
if (!any_problems) print("All OK")
```

```
## [1] "All OK"
```

Check behaviour observation segmentation

```r
# Behaviour observations were not continuous and were divided into segments
# Checks for segments:
#       Is the full segment indeed in the behaviour file? 
#       Does the end of the segment truly correspond to a gap in the behaviour file? 
#       Are there unnecessary small gaps in behaviour scoring that might be able
#          to be filled in?

# Load behaviour data but add a couple of columns and split it by id
any_problems <- FALSE
temp <- beh_obs_data
beh_by_id <- split(temp, temp$recording_id)
beh_by_id <- map(beh_by_id, function(x){
    x <- x |> arrange(beh_start_real)
    
    x$elapsed_before_me <- 
        c(99999, as.numeric(difftime(x$beh_start_real[-1],
                                  x$beh_stop_real[-nrow(x)],
                                  units = "secs")))
    x$gap_before_me <- x$elapsed_before_me > 0.5
    x$same_beh_as_prev_row <- 
        c(FALSE, x$behaviour[-1] == x$behaviour[-length(x$behaviour)])

    return(x)})

# Check segments (top-down)
cat("\n---------------------Segment contiguity check-------------------------")
```

```
## 
## ---------------------Segment contiguity check-------------------------
```

```r
# -- No gaps seen in the middle of a segment? --
for (i in 1:nrow(segment_summary)){
    cat(glue("{i}."))
    thisseg <- segment_summary[i,]
    thisid <- thisseg$recording_id
    behs_thisid <- beh_by_id[[thisid]] |> arrange(beh_start_real)
    # Find gaps in the behaviour scoring sequence (max gap 0.5 is hard coded for now)
    gapsinthisseg <- behs_thisid |> filter(beh_start_real > thisseg$seg_start_real &
                               beh_stop_real < thisseg$seg_stop_real) |> pull(gap_before_me)
    if(any(gapsinthisseg)){
        print(glue("{this_id} segment {thisseg$segment_id} had gaps within it"))
        any_problems <- TRUE
    }
}
```

```
## 1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.29.30.31.32.33.34.35.36.37.38.39.40.41.42.43.44.45.46.47.48.49.50.51.52.53.54.55.56.57.58.59.60.61.62.63.64.65.66.67.68.69.70.71.72.73.74.75.76.77.78.79.80.81.82.83.84.85.86.87.88.89.90.91.92.93.94.95.96.97.98.99.100.101.102.103.104.105.106.107.108.109.110.111.112.113.114.115.116.117.118.119.120.121.122.123.124.125.126.127.128.129.130.131.132.133.134.135.136.137.138.139.140.141.142.143.144.145.146.147.148.149.150.151.152.153.154.155.156.157.158.159.160.161.162.163.164.165.166.167.168.169.170.171.172.173.174.175.176.177.178.179.180.181.182.183.184.185.186.187.188.189.190.191.192.193.194.195.196.197.198.199.200.201.202.203.204.205.206.207.208.209.210.211.212.213.214.215.216.217.218.219.220.221.
```

```r
if (!any_problems){
    cat("\nAll ok.")
} 
```

```
## 
## All ok.
```

```r
any_problems <- FALSE

# -- Gaps between segments should have no 'floating' behavioral data --
cat("\n--------------------Floating behaviours check-----------------------------")
```

```
## 
## --------------------Floating behaviours check-----------------------------
```

```r
for (i in 1:length(unique(beh_obs_data$recording_id))){
    cat(glue("{i}."))
    thisid <- unique(beh_obs_data$recording_id)[i]
    behs_thisid <- beh_by_id[[thisid]]
    segs_thisid <- segment_summary |> filter(recording_id == thisid) |> arrange(seg_start_real)
    if (nrow(segs_thisid) == 1) next
    segment_gaps <-
        data.frame(gap_start = segs_thisid$seg_stop_real[-nrow(segs_thisid)],
                   gap_stop = segs_thisid$seg_start_real[-1])
    # go over each gap
    for(i in 1:nrow(segment_gaps)){
        # Check if there are any behaviours in this time
        floating_behaviours <- 
            behs_thisid |> filter(beh_start_real > segment_gaps$gap_start[i] & 
                                   beh_stop_real < segment_gaps$gap_stop[i])
        if (nrow(floating_behaviours) > 0){
            print(glue("{thisid} segment {segs_thisid$segment_id[i]}, i={i} had an" , 
                       " adjacent floating behaviour"))
            any_problems <- TRUE
        }
    }
}
```

```
## 1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.
```

```r
if (!any_problems){
    cat("\nAll ok.")
} 
```

```
## 
## All ok.
```

```r
any_problems <- FALSE

cat("\n---------------------Check segment start times-----------------------------")
```

```
## 
## ---------------------Check segment start times-----------------------------
```

```r
# Go over all ids and get info
for (i in 1:length(unique(beh_obs_data$recording_id))){
    cat(glue("{i}."))
    thisid <- unique(beh_obs_data$recording_id)[i]
    # -- Behaviours with a gap before them should be the start of a new segment --
    behs_thisid <- beh_by_id[[thisid]]
    segs_thisid <- segment_summary |> filter(recording_id == thisid) |> arrange(seg_start_real)
    
    seg_starts_recalculated <- behs_thisid[behs_thisid$gap_before_me, 'beh_start_real']
    seg_starts_from_outputfile <- segs_thisid$seg_start_real
    a <- prtime(seg_starts_recalculated) 
    b <- prtime(seg_starts_from_outputfile)
    mismatch <- !all(a == b)
    if(any(mismatch)){
        print(glue("{this_id} had unverified segment times issues: segmented ", 
                   "behaviour times do not seem to match up with behaviour gaps"))
        any_problems <- TRUE
    }
}
```

```
## 1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.
```

```r
if (!any_problems) cat("\nAll ok.")
```

```
## 
## All ok.
```

```r
## Design decision was made to allow successive identical behaviours
# print("---------------Check for successive identical behaviours---------------")
# # Check if there are two behaviours in a row with the same label. These should be
# combined into a single behaviour
# for (i in 1:length(beh_by_id)){
#     cat(glue("{i}."))
#     thisid <- unique(beh_obs_data$recording_id)[i]
#     behs_thisid <- beh_by_id[[thisid]]
#     
#     repeated_behaviours <- behs_thisid$same_beh_as_prev_row & !behs_thisid$gap_before_me
# 
#     if(any(repeated_behaviours)){
#         cat("\n")
#         print(glue("i={i} {thisid} had repeated subsequent behaviours\n"))
#     }
# }
```

Check label assignments

```r
# Pick out random behaviours from beh_acc. Get the start and stop time for this 
# beh and check that the labelled data file has the correct behaviour labels
# between these times.
any_problems <- FALSE
nsamples <- 300
random_behs <- beh_obs_data[sample(1:nrow(beh_obs_data), nsamples),]
random_behs$assigned_seg <- NA

# First, get the data segment corresponding to the randomly chosen behaviour. 
# Incidentally, check that there is a segment whose time overlaps with this
# behaviour, and only 1
for (i in 1:nrow(random_behs)){
    cat(glue("{i}."))
    thisbeh <- random_behs[i,]
    assigned_seg <- segment_summary |> 
        filter(recording_id == thisbeh$recording_id) |> 
        filter(seg_start_real <= thisbeh$beh_start_real) |>
        filter(seg_stop_real  >= thisbeh$beh_stop_real) |>
        pull(segment_id)
    
    if(length(assigned_seg) != 1){
        print(glue("Behaviour {thisbeh$beh_event_id}: beh times overlaps with ",
                   "multiple or zero segments"))
            any_problems <- TRUE
    }
    random_behs$assigned_seg[i] <- assigned_seg
}
```

```
## 1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.29.30.31.32.33.34.35.36.37.38.39.40.41.42.43.44.45.46.47.48.49.50.51.52.53.54.55.56.57.58.59.60.61.62.63.64.65.66.67.68.69.70.71.72.73.74.75.76.77.78.79.80.81.82.83.84.85.86.87.88.89.90.91.92.93.94.95.96.97.98.99.100.101.102.103.104.105.106.107.108.109.110.111.112.113.114.115.116.117.118.119.120.121.122.123.124.125.126.127.128.129.130.131.132.133.134.135.136.137.138.139.140.141.142.143.144.145.146.147.148.149.150.151.152.153.154.155.156.157.158.159.160.161.162.163.164.165.166.167.168.169.170.171.172.173.174.175.176.177.178.179.180.181.182.183.184.185.186.187.188.189.190.191.192.193.194.195.196.197.198.199.200.201.202.203.204.205.206.207.208.209.210.211.212.213.214.215.216.217.218.219.220.221.222.223.224.225.226.227.228.229.230.231.232.233.234.235.236.237.238.239.240.241.242.243.244.245.246.247.248.249.250.251.252.253.254.255.256.257.258.259.260.261.262.263.264.265.266.267.268.269.270.271.272.273.274.275.276.277.278.279.280.281.282.283.284.285.286.287.288.289.290.291.292.293.294.295.296.297.298.299.300.
```

```r
# Then, fetch the labelled data segment that this behaviour should be present in. 
# Check that the correct part of the data file labelled. 
# note: caching used so the same segment isn't loaded multiple times
random_behs <- random_behs |> arrange(assigned_seg)
cached_seg <- -1 # initialize with placeholder number
for (i in 1:nrow(random_behs)){
    # for each behaviour
    cat(glue("{i}."))
    thisbeh <- random_behs[i,]

    # Find which segmented file this behaviour should be represented in
    thisid <- thisbeh$recording_id
    thisseg <- thisbeh$assigned_seg
    
    # Get the datafile generated for this segment, if it's not already loaded
    if(thisseg != cached_seg){
        thisseg_data <- 
            fread(file.path(processed_data_dir, glue("{thisid}_s{thisseg}.csv")),
                  tz = "") |> 
            mutate(datetime = toposix_ymdhms(datetime))
        cached_seg <- thisseg
    }

    # Check the expected rows of the file, and see if any neighbouring behaviours
    # are intruding, or perhaps if the whole thing is outright incorrectly labelled
    matching_data <- thisseg_data |> 
        filter(datetime > thisbeh$beh_start_acc & datetime < thisbeh$beh_stop_acc)
    
    if(length(unique(matching_data$behaviour)) > 1){
        print(glue("\ni={i}Behaviour {thisbeh$beh_event_id} segment {thisseg}: matching",
                   " part of annotated data was not pure"))
            any_problems <- TRUE
    } else if(thisbeh$behaviour != unique(matching_data$behaviour)){
        print(glue("\ni={i}Behaviour {thisbeh$beh_event_id} segment {thisseg}: matching",
                   " annotated data had wrong label"))
            any_problems <- TRUE
    }
    
    data_subsequent_rows <- thisseg_data |> 
        filter(datetime > thisbeh$beh_stop_acc & datetime < thisbeh$beh_stop_acc + 0.25)
    data_prior_rows <- thisseg_data |> 
        filter(datetime > thisbeh$beh_start_acc - 0.25 & datetime < thisbeh$beh_start_acc)
    
    subsequent_bleeding <- data_subsequent_rows$behaviour[1] %in% matching_data$behaviour
    prior_bleeding <- last(data_prior_rows$behaviour) %in% matching_data$behaviour
    # If any of the target behaviour appears in the nearby ~quarter of second,
    # there may be a labeling problem. Check the beahiviour scoring and see if
    # the next/previous behaviour is truly a repetition or not. These repetitions
    # came about sometimes due to changing behaviour labels earlier in the process.
    if(subsequent_bleeding){
        # It's not bleeding if the next event is truly the same behaviour
        next_event <- thisbeh$beh_event_id+1
        if (!thisbeh$behaviour == beh_obs_data[beh_obs_data$beh_event_id == next_event, "behaviour"]){
            print(glue("\ni={i} Behaviour {thisbeh$beh_event_id} segment {thisseg}: bled ",
                       " into from subsequent behaviour event"))
            any_problems <- TRUE
        }
    }
    if(prior_bleeding){
        # It's not bleeding if the prev event is truly the same behaviour
        prev_event <- thisbeh$beh_event_id-1
        if (!thisbeh$behaviour == beh_obs_data[beh_obs_data$beh_event_id == prev_event, "behaviour"]){
            print(glue("\ni={i} Behaviour {thisbeh$beh_event_id} segment {thisseg}: bled ",
                       "into by prior behaviour event"))
            any_problems <- TRUE
        }
    }
}
```

```
## 1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23.24.25.26.27.28.29.30.31.32.33.34.35.36.37.38.39.40.41.42.43.44.45.46.47.48.49.50.51.52.53.54.55.56.57.58.59.60.61.62.63.64.65.66.67.68.69.70.71.72.73.74.75.76.77.78.79.80.81.82.83.84.85.86.87.88.89.90.91.92.93.94.95.96.97.98.99.100.101.102.103.104.105.106.107.108.109.110.111.112.113.114.115.116.117.118.119.120.121.122.123.124.125.126.127.128.129.130.131.132.133.134.135.136.137.138.139.140.141.142.143.144.145.146.147.148.149.150.151.152.153.154.155.156.157.158.159.160.161.162.163.164.165.166.167.168.169.170.171.172.173.174.175.176.177.178.179.180.181.182.183.184.185.186.187.188.189.190.191.192.193.194.195.196.197.198.199.200.201.202.203.204.205.206.207.208.209.210.211.212.213.214.215.216.217.218.219.220.221.222.223.224.225.226.227.228.229.230.231.232.233.234.235.236.237.238.239.240.241.242.243.244.245.246.247.248.249.250.251.252.253.254.255.256.257.258.259.260.261.262.263.264.265.266.267.268.269.270.271.272.273.274.275.276.277.278.279.280.281.282.283.284.285.286.287.288.289.290.291.292.293.294.295.296.297.298.299.300.
```

```r
if (!any_problems) print("All OK")
```

```
## [1] "All OK"
```

Check 6O scaling factors

```r
# Recalculate 6-O factors and check that it matches expected values
any_problems <- FALSE
info_6O <- fread(path_6O_calib_info, tz = "")
recording_info <- fread(path_recording_info, tz = "")
# Select wanted columns
info_6O$start_dt <- paste(info_6O$start_date, info_6O$start_time)
info_6O <- info_6O[,.(logger_id, filename, start_dt, z1_cal_dt, z2_cal_dt, 
                      y1_cal_dt, y2_cal_dt, x1_cal_dt, x2_cal_dt)]

# Convert to datetimes
cols <- names(info_6O)[grepl("dt", names(info_6O))]
info_6O[, (cols) := lapply(.SD, function(x)dmy_hms(x, tz = "CET")), .SDcols = cols]

get_norm <- function(datatable){
    sqrt(datatable$accX^2 + datatable$accY^2 + datatable$accZ^2)
}

plots <- list()

# For each recording, recalculate and check the 6O scaling factors. Also 
# apply scaling factor to calibration data and check it results in acc ~= 1
for (i in 1:nrow(recording_info)){
    thisrecinfo <- recording_info[i,]
    this6Oinfo <- info_6O[info_6O$logger_id == thisrecinfo$logger_id,]
    thisx1dt <- this6Oinfo$x1_cal_dt
    thisx2dt <- this6Oinfo$x2_cal_dt
    thisy1dt <- this6Oinfo$y1_cal_dt
    thisy2dt <- this6Oinfo$y2_cal_dt
    thisz1dt <- this6Oinfo$z1_cal_dt
    thisz2dt <- this6Oinfo$z2_cal_dt
    
    thisdat <- fread(file.path(dir_acc_calib_data, this6Oinfo$filename))
    thisdat$dt <- dmy_hms(paste(thisdat$Date, thisdat$Time), tz = "CET")
    thisdat <- thisdat |> filter(dt > min(thisx1dt, thisx2dt, thisy1dt, 
                                           thisy2dt, thisz1dt, thisz2dt) - 5)
    thisdat <- thisdat |> filter(dt < max(thisx1dt, thisx2dt, thisy1dt, 
                                           thisy2dt, thisz1dt, thisz2dt) + 5)
    x1 <- thisdat[dt > thisx1dt & dt < thisx1dt + 5] |> get_norm() |> median()
    x2 <- thisdat[dt > thisx2dt & dt < thisx2dt + 5] |> get_norm() |> median()
    y1 <- thisdat[dt > thisy1dt & dt < thisy1dt + 5] |> get_norm() |> median()
    y2 <- thisdat[dt > thisy2dt & dt < thisy2dt + 5] |> get_norm() |> median()
    z1 <- thisdat[dt > thisz1dt & dt < thisz1dt + 5] |> get_norm() |> median()  
    z2 <- thisdat[dt > thisz2dt & dt < thisz2dt + 5] |> get_norm() |> median()

    if (((1 / mean(c(x1, x2))) - thisrecinfo$scale_x_6O) > 0.01){
        print(glue("recording {thisrecinfo$recording_id} X scaling didn't match. ",
                   "orig {round(thisrecinfo$scale_x_6O, 3)}, ",
                   "recalculated{round(1/mean(c(x1,x2)),3)}"))
        any_problems <- TRUE
    }
    if (((1 / mean(c(y1, y2))) - thisrecinfo$scale_y_6O) > 0.01){
        print(glue("recording {thisrecinfo$recording_id} Y scaling didn't match. ",
                   "orig {round(thisrecinfo$scale_y_6O, 3)}, ",
                   "recalculated {round(1/mean(c(y1,y2)),3)}"))
        any_problems <- TRUE
    }
    if (((1 / mean(c(z1, z2))) - thisrecinfo$scale_z_6O) > 0.01){
        print(glue("recording {thisrecinfo$recording_id} Z scaling didn't match. ",
                   "orig {round(thisrecinfo$scale_z_6O, 3)}, ",
                   "recalculated {round(1/mean(c(z1,z2)),3)}"))
        any_problems <- TRUE
    }
    # Apply scaling factor to data and see if it equals 1.
    back_dat <- thisdat
    back_dat$accX <- back_dat$accX * thisrecinfo$scale_x_6O
    back_dat$accY <- back_dat$accY * thisrecinfo$scale_y_6O
    back_dat$accZ <- back_dat$accZ * thisrecinfo$scale_z_6O
    x1 <- back_dat[dt > thisx1dt & dt < thisx1dt + 5] |> get_norm() |> median()
    x2 <- back_dat[dt > thisx2dt & dt < thisx2dt + 5] |> get_norm() |> median()
    y1 <- back_dat[dt > thisy1dt & dt < thisy1dt + 5] |> get_norm() |> median()
    y2 <- back_dat[dt > thisy2dt & dt < thisy2dt + 5] |> get_norm() |> median()
    z1 <- back_dat[dt > thisz1dt & dt < thisz1dt + 5] |> get_norm() |> median()
    z2 <- back_dat[dt > thisz2dt & dt < thisz2dt + 5] |> get_norm() |> median()
    if(abs(x1 - 1) > 0.01|abs(x2 - 1) > 0.01|abs(y1 - 1) > 0.01|abs(y2 - 1) > 0.01|
       abs(z1 - 1) > 0.01|abs(z2 - 1) > 0.01){
        print(glue("recording {thisrecinfo$recording_id} applying scaling ",
                   "factor to data does not yield g = 1. Double-check vals:\n", 
                   "     x1 : {round(x1,2)}    x2 : {round(x2,2)}  ",
                   "   y1 : {round(y1,2)}    y2 : {round(y2,2)}  ",
                   "   z1 : {round(z1,2)}    z2 : {round(z2,2)}"))
    }
    rects <- data.frame(var = c('accX', 'accX', 'accY','accY','accZ','accZ'),
                        xmin = c(thisx1dt, thisx2dt, thisy1dt, thisy2dt, 
                                 thisz1dt, thisz2dt), ymax = Inf, ymin = -Inf)
    rects$xmax = rects$xmin + 5
    points <- data.frame(var = c('accX', 'accX', 'accY','accY','accZ','accZ'),
                        x = c(thisx1dt, thisx2dt, thisy1dt, thisy2dt, 
                                 thisz1dt, thisz2dt), 
                        y = c(-1/ thisrecinfo$scale_x_6O, 1/ thisrecinfo$scale_x_6O,
                              -1/ thisrecinfo$scale_y_6O, 1/ thisrecinfo$scale_y_6O,
                              -1/ thisrecinfo$scale_z_6O, 1/ thisrecinfo$scale_z_6O))
    points$x = points$x + 2.5
    plots[[i]] <- thisdat |> 
        pivot_longer(cols = c(accX, accY, accZ), 
                     names_to = "var", values_to = "val") |>
        ggplot() + 
        geom_line(aes(x = dt, y = val)) + 
        facet_wrap(~var, ncol = 1, scales = "free") + 
        geom_rect(data = rects, aes(xmin = xmin, xmax = xmax, ymin = ymin, 
                                    ymax = ymax, col = var), fill = NA) +
        geom_point(data = points, aes(x = x, y = y, col = var)) +
        coord_cartesian(ylim = c(-2.5, 2.5))
}
```

```
## recording 1372_L24_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 0.99    x2 : 1     y1 : 1.04    y2 : 0.96     z1 : 1.02    z2 : 0.98
## recording 1301_L25_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 0.99    x2 : 1.02     y1 : 1    y2 : 1     z1 : 1.03    z2 : 0.97
## recording 952_L27_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.02    x2 : 0.98     y1 : 1.02    y2 : 0.98     z1 : 1.01    z2 : 0.98
## recording 1326_L29_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 0.99    x2 : 1.02     y1 : 1    y2 : 1     z1 : 1.02    z2 : 0.98
## recording 1399_L33_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.02    x2 : 0.98     y1 : 0.99    y2 : 1     z1 : 1.03    z2 : 0.97
## recording G20-0055_L38_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.06    x2 : 0.94     y1 : 1.03    y2 : 0.97     z1 : 1.02    z2 : 0.98
## recording 1361_L42_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1    x2 : 1     y1 : 1    y2 : 1     z1 : 1.04    z2 : 0.96
## recording 1301_L28_2 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.05    x2 : 0.96     y1 : 1.02    y2 : 0.98     z1 : 1.02    z2 : 0.98
## recording 1326_L31_2 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.03    x2 : 0.97     y1 : 1    y2 : 1     z1 : 1.02    z2 : 0.98
## recording 1331_L34_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 0.98    x2 : 1.03     y1 : 1    y2 : 1     z1 : 1.05    z2 : 0.96
## recording G20-0055_L36_2 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 0.99    x2 : 1.01     y1 : 1    y2 : 1     z1 : 1.02    z2 : 0.98
## recording 1361_L39_2 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.02    x2 : 0.98     y1 : 1    y2 : 0.99     z1 : 1.01    z2 : 1
## recording 952_L43_2 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1    x2 : 1     y1 : 1.01    y2 : 0.99     z1 : 1.03    z2 : 0.97
## recording 1399_L46_2 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1    x2 : 1     y1 : 1.03    y2 : 0.98     z1 : 1.03    z2 : 0.97
## recording 1372_L48_2 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1    x2 : 1     y1 : 0.98    y2 : 1.02     z1 : 1.01    z2 : 0.99
## recording 1333_L51_2 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.01    x2 : 0.99     y1 : 1.02    y2 : 0.98     z1 : 1.02    z2 : 0.98
## recording 1301_L25_3 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 0.99    x2 : 1.02     y1 : 1    y2 : 1     z1 : 1.03    z2 : 0.97
## recording 1368_L27_3 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.02    x2 : 0.98     y1 : 1.02    y2 : 0.98     z1 : 1.01    z2 : 0.98
## recording G20-0059_L29_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 0.99    x2 : 1.02     y1 : 1    y2 : 1     z1 : 1.02    z2 : 0.98
## recording 1326_L31_3 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.03    x2 : 0.97     y1 : 1    y2 : 1     z1 : 1.02    z2 : 0.98
## recording 7-04-105_L32_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.01    x2 : 0.99     y1 : 1    y2 : 1     z1 : 1.03    z2 : 0.97
## recording 1681_L33_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.02    x2 : 0.98     y1 : 0.99    y2 : 1     z1 : 1.03    z2 : 0.97
## recording G20-0071_L34_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 0.98    x2 : 1.03     y1 : 1    y2 : 1     z1 : 1.05    z2 : 0.96
## recording G20-0529_L37_1 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.02    x2 : 0.98     y1 : 1    y2 : 1.01     z1 : 1.04    z2 : 0.97
## recording 1361_L39_3 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1.02    x2 : 0.98     y1 : 1    y2 : 0.99     z1 : 1.01    z2 : 1
## recording 1333_L41_3 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 0.99    x2 : 1.01     y1 : 1.01    y2 : 0.99     z1 : 1.04    z2 : 0.96
## recording 952_L43_3 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1    x2 : 1     y1 : 1.01    y2 : 0.99     z1 : 1.03    z2 : 0.97
## recording 1372_L48_3 applying scaling factor to data does not yield g = 1. Double-check vals:
## x1 : 1    x2 : 1     y1 : 0.98    y2 : 1.02     z1 : 1.01    z2 : 0.99
```


```r
plots
```

```
## [[1]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-1.png)

```
## 
## [[2]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-2.png)

```
## 
## [[3]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-3.png)

```
## 
## [[4]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-4.png)

```
## 
## [[5]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-5.png)

```
## 
## [[6]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-6.png)

```
## 
## [[7]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-7.png)

```
## 
## [[8]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-8.png)

```
## 
## [[9]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-9.png)

```
## 
## [[10]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-10.png)

```
## 
## [[11]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-11.png)

```
## 
## [[12]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-12.png)

```
## 
## [[13]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-13.png)

```
## 
## [[14]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-14.png)

```
## 
## [[15]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-15.png)

```
## 
## [[16]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-16.png)

```
## 
## [[17]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-17.png)

```
## 
## [[18]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-18.png)

```
## 
## [[19]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-19.png)

```
## 
## [[20]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-20.png)

```
## 
## [[21]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-21.png)

```
## 
## [[22]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-22.png)

```
## 
## [[23]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-23.png)

```
## 
## [[24]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-24.png)

```
## 
## [[25]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-25.png)

```
## 
## [[26]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-26.png)

```
## 
## [[27]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-27.png)

```
## 
## [[28]]
```

![plot of chunk Show plots for 6O scaling](figure/Show plots for 6O scaling-28.png)


```r
# Check that the 6-O scaled data results in static acceleration values 
# that average to  g = 1

for (i in 1:nrow(recording_info)){
    thisinfo <- recording_info[i,]
    thisinfo$cal1_time <- toposix_dmyhms(thisinfo$cal1_time)
    thisinfo$cal3_time <- toposix_dmyhms(thisinfo$cal3_time)
    thisinfo$deploy_time_start <- toposix_dmyhms(thisinfo$deploy_time_start)
    setDT(thisinfo)
    cols <- names(thisinfo)[grepl("dt|time", names(thisinfo))]
    thisinfo[, (cols) := lapply(.SD, function(x) toposix_dmyhms(x)), .SDcols = cols]
    # Find a time when the logger was being calibrated before bird deployment,
    # it should have been nice and stationary just before being calibrated
    calib1time <- adjust_time(target_time = thisinfo$cal1_time[1], 
                              origin = thisinfo$deploy_time_start,
                              dr = thisinfo$drift_rate,
                              init_error = thisinfo$drift_yintercept)
    calib3time <- adjust_time(target_time = thisinfo$cal3_time[1], 
                              origin = thisinfo$deploy_time_start,
                              dr = thisinfo$drift_rate,
                              init_error = thisinfo$drift_yintercept)

    # Get 5 hour buffer around this time
    calib1_data <- get_raw_acc_segment(gsub(".csv", "", thisinfo$recording_id),
                        segment_start = calib1time - 30, 
                        duration_secs = 15,
                        path_db = path_raw_acc_db)
    calib3_data <- get_raw_acc_segment(gsub(".csv", "", thisinfo$recording_id),
                        segment_start = calib3time - 30, 
                        duration_secs = 15,
                        path_db = path_raw_acc_db)
    calib1_data$accX <- calib1_data$accX * thisinfo$scale_x_6O
    calib1_data$accY <- calib1_data$accY * thisinfo$scale_y_6O
    calib1_data$accZ <- calib1_data$accZ * thisinfo$scale_z_6O
    calib1_data$norm <- sqrt(calib1_data$accX^2 + calib1_data$accY^2 + calib1_data$accZ^2)
    med_1 <- median(calib1_data$norm)
    
    calib3_data$accX <- calib3_data$accX * thisinfo$scale_x_6O
    calib3_data$accY <- calib3_data$accY * thisinfo$scale_y_6O
    calib3_data$accZ <- calib3_data$accZ * thisinfo$scale_z_6O
    calib3_data$norm <- sqrt(calib3_data$accX^2 + calib3_data$accY^2 + calib3_data$accZ^2)
    med_3 <- median(calib3_data$norm)
    if (med_1 > 1.05 | med_1 < 0.95 | med_3 > 1.05 | med_3 < 0.95){
        cat("PROBLEM?-")
        any_problems <- TRUE
    } else {
        cat("---------")
    }
    print(glue("Recording {stri_pad_right(thisinfo$recording_id, 14, ' ')} : ",
       "g for calibration 1;3 (should be near 1) = {sprintf('%.2f', round(med_1,2))};",
       "{sprintf('%.2f',round(med_3,2))}"))
}
```

```
## ---------Recording 1372_L24_1     : g for calibration 1;3 (should be near 1) = 0.98;0.98
## ---------Recording 1301_L25_1     : g for calibration 1;3 (should be near 1) = 0.97;0.96
## ---------Recording 952_L27_1      : g for calibration 1;3 (should be near 1) = 0.97;0.97
## ---------Recording 1326_L29_1     : g for calibration 1;3 (should be near 1) = 0.97;0.95
## ---------Recording 1399_L33_1     : g for calibration 1;3 (should be near 1) = 0.97;0.97
## ---------Recording G20-0055_L38_1 : g for calibration 1;3 (should be near 1) = 0.98;0.95
## PROBLEM?-Recording 1361_L42_1     : g for calibration 1;3 (should be near 1) = 0.95;0.95
## ---------Recording 1301_L28_2     : g for calibration 1;3 (should be near 1) = 0.96;0.96
## ---------Recording 1326_L31_2     : g for calibration 1;3 (should be near 1) = 0.98;0.98
## ---------Recording 1331_L34_1     : g for calibration 1;3 (should be near 1) = 0.95;0.96
## ---------Recording G20-0055_L36_2 : g for calibration 1;3 (should be near 1) = 0.99;0.99
## ---------Recording 1361_L39_2     : g for calibration 1;3 (should be near 1) = 1.01;1.01
## ---------Recording 952_L43_2      : g for calibration 1;3 (should be near 1) = 0.97;0.95
## ---------Recording 1399_L46_2     : g for calibration 1;3 (should be near 1) = 0.97;0.96
## ---------Recording 1372_L48_2     : g for calibration 1;3 (should be near 1) = 1.00;0.99
## ---------Recording 1333_L51_2     : g for calibration 1;3 (should be near 1) = 0.99;0.98
## ---------Recording 1301_L25_3     : g for calibration 1;3 (should be near 1) = 0.97;0.96
## ---------Recording 1368_L27_3     : g for calibration 1;3 (should be near 1) = 0.97;1.00
## ---------Recording G20-0059_L29_1 : g for calibration 1;3 (should be near 1) = 0.97;0.96
## ---------Recording 1326_L31_3     : g for calibration 1;3 (should be near 1) = 0.97;0.98
## ---------Recording 7-04-105_L32_1 : g for calibration 1;3 (should be near 1) = 0.97;0.98
## ---------Recording 1681_L33_1     : g for calibration 1;3 (should be near 1) = 0.96;0.96
## ---------Recording G20-0071_L34_1 : g for calibration 1;3 (should be near 1) = 0.97;0.95
## PROBLEM?-Recording G20-0529_L37_1 : g for calibration 1;3 (should be near 1) = 0.95;0.96
## ---------Recording 1361_L39_3     : g for calibration 1;3 (should be near 1) = 1.00;1.01
## ---------Recording 1333_L41_3     : g for calibration 1;3 (should be near 1) = 0.95;0.95
## ---------Recording 952_L43_3      : g for calibration 1;3 (should be near 1) = 0.97;0.96
## ---------Recording 1372_L48_3     : g for calibration 1;3 (should be near 1) = 0.99;1.00
```

Check deployment bias (median adjustment)

```r
# Median scaling factors: calculate a rougher median scaling 
#   factors (e.g. with the subsample of data rather than full recording) 
#   and check they are similar enough to the originally calculated scaling factor

print(glue("Recalculating median scaling factors using an estimate. Recalculated",
           "values should be in the same ballpark"))
```

```
## Recalculating median scaling factors using an estimate. Recalculatedvalues should be in the same ballpark
```

```r
for (i in 1:nrow(recording_info)){
    thisinfo <- recording_info[i,]
    firstbeh <- beh_obs_data |> filter(recording_id == thisinfo$recording_id) |>
        arrange(beh_start_real)
    firstbehtime <- firstbeh$beh_start_acc[1]
    # Get some  data from middle of their recording
    calib1_data <- get_raw_acc_segment(gsub(".csv", "", thisinfo$recording_id),
                        segment_start = firstbehtime - 60 * 60, # 1 hour before first beh
                        duration_secs = 60 * 60 * 5, # 5 hours of behaviour
                        path_db = path_raw_acc_db)
    
    # Do 6O correction
    calib1_data$accX <- calib1_data$accX * thisinfo$scale_x_6O
    calib1_data$accY <- calib1_data$accY * thisinfo$scale_y_6O
    calib1_data$accZ <- calib1_data$accZ * thisinfo$scale_z_6O
    # See how recalculated medians compare to the original processed value
    xdiff <- abs(median(calib1_data$accX) - thisinfo$scale_x_deploybias)
    ydiff <- abs(median(calib1_data$accY) - thisinfo$scale_y_deploybias)
    zdiff <- abs(median(calib1_data$accZ) - thisinfo$scale_z_deploybias)
    
    if (xdiff > 0.05 | ydiff > 0.05 | zdiff > 0.05){
        cat("PROBLEM?-")
        any_problems <- TRUE
    } else{
        cat("---------")
    }
    print(glue("Recording {stri_pad_right(thisinfo$recording_id, 14, ' ')} ",
          "median scales (new/orig). ", 
          "X: {sprintf('%.2f', round(median(calib1_data$accX),2))}/",
          "{sprintf('%.2f', round(thisinfo$scale_x_deploybias,2))}",
          " Y: {sprintf('%.2f', round(median(calib1_data$accY),2))}/",
          "{sprintf('%.2f', round(thisinfo$scale_y_deploybias,2))}",
          " Z: {sprintf('%.2f', round(median(calib1_data$accZ),2))}/",
          "{sprintf('%.2f', round(thisinfo$scale_z_deploybias,2))}"))
}
```

```
## PROBLEM?-Recording 1372_L24_1     median scales (new/orig). X: 0.58/0.64 Y: -0.12/-0.12 Z: 0.71/0.76
## ---------Recording 1301_L25_1     median scales (new/orig). X: 0.61/0.59 Y: 0.00/-0.01 Z: 0.77/0.80
## PROBLEM?-Recording 952_L27_1      median scales (new/orig). X: 0.63/0.51 Y: -0.03/-0.04 Z: 0.76/0.85
## ---------Recording 1326_L29_1     median scales (new/orig). X: 0.73/0.68 Y: -0.03/-0.04 Z: 0.69/0.73
## ---------Recording 1399_L33_1     median scales (new/orig). X: 0.46/0.46 Y: 0.00/-0.04 Z: 0.86/0.87
## PROBLEM?-Recording G20-0055_L38_1 median scales (new/orig). X: 0.66/0.60 Y: 0.02/0.00 Z: 0.70/0.76
## ---------Recording 1361_L42_1     median scales (new/orig). X: 0.44/0.45 Y: -0.06/-0.07 Z: 0.86/0.85
## ---------Recording 1301_L28_2     median scales (new/orig). X: 0.67/0.70 Y: -0.03/-0.06 Z: 0.67/0.65
## ---------Recording 1326_L31_2     median scales (new/orig). X: 0.58/0.61 Y: -0.05/-0.02 Z: 0.79/0.77
## PROBLEM?-Recording 1331_L34_1     median scales (new/orig). X: 0.41/0.34 Y: -0.01/0.00 Z: 0.87/0.90
## ---------Recording G20-0055_L36_2 median scales (new/orig). X: 0.66/0.68 Y: 0.01/-0.00 Z: 0.73/0.73
## ---------Recording 1361_L39_2     median scales (new/orig). X: 0.44/0.41 Y: -0.08/-0.08 Z: 0.89/0.91
## ---------Recording 952_L43_2      median scales (new/orig). X: 0.65/0.62 Y: 0.03/-0.02 Z: 0.69/0.73
## PROBLEM?-Recording 1399_L46_2     median scales (new/orig). X: 0.59/0.51 Y: 0.07/0.12 Z: 0.77/0.82
## ---------Recording 1372_L48_2     median scales (new/orig). X: 0.53/0.58 Y: 0.01/-0.04 Z: 0.83/0.82
## PROBLEM?-Recording 1333_L51_2     median scales (new/orig). X: 0.44/0.36 Y: -0.06/-0.06 Z: 0.88/0.93
## ---------Recording 1301_L25_3     median scales (new/orig). X: 0.69/0.72 Y: 0.00/0.01 Z: 0.69/0.66
## PROBLEM?-Recording 1368_L27_3     median scales (new/orig). X: 0.82/0.76 Y: -0.03/-0.05 Z: 0.56/0.63
## PROBLEM?-Recording G20-0059_L29_1 median scales (new/orig). X: 0.67/0.71 Y: -0.12/-0.05 Z: 0.75/0.69
## ---------Recording 1326_L31_3     median scales (new/orig). X: 0.67/0.70 Y: 0.00/-0.01 Z: 0.72/0.68
## PROBLEM?-Recording 7-04-105_L32_1 median scales (new/orig). X: 0.43/0.37 Y: 0.06/0.06 Z: 0.87/0.90
## PROBLEM?-Recording 1681_L33_1     median scales (new/orig). X: 0.74/0.58 Y: 0.16/0.16 Z: 0.62/0.77
## ---------Recording G20-0071_L34_1 median scales (new/orig). X: 0.62/0.60 Y: -0.03/-0.05 Z: 0.75/0.76
## ---------Recording G20-0529_L37_1 median scales (new/orig). X: 0.54/0.55 Y: 0.00/-0.01 Z: 0.79/0.79
## ---------Recording 1361_L39_3     median scales (new/orig). X: 0.41/0.43 Y: -0.10/-0.11 Z: 0.89/0.89
## ---------Recording 1333_L41_3     median scales (new/orig). X: 0.54/0.57 Y: -0.08/-0.13 Z: 0.80/0.77
## PROBLEM?-Recording 952_L43_3      median scales (new/orig). X: 0.72/0.68 Y: 0.01/-0.04 Z: 0.63/0.68
## PROBLEM?-Recording 1372_L48_3     median scales (new/orig). X: 0.47/0.57 Y: -0.01/-0.00 Z: 0.89/0.82
```

```r
if (!any_problems) print("All OK")
```

Check for NA values?

```r
# Not implemented
```

Check for duplicated timestamps?

```r
# Not implemented
```
