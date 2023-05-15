# ~~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'      Helper functions for preprocessing accelerometer data. See cleaning.R
#'
#' Date created:
#'      May 2, 2023
#'      
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(dplyr)
library(ggplot2)
library(glue)
library(data.table)
library(RSQLite)
#' Fetch accelerometer data accelerometer segment from database
#' 
#' This helper function reads a segment of raw accelerometer data from the
#' database, according to a start time and a duration or end time.
#' 
#' NOTE: time stamps in raw accelerometer data are subject to clock drift.
#'
#' @param segment_start (POSIXct) start time of the target segment, in 
#'      accelerometer clock time. Will attempt to parse plain text.
#' @param duration_secs (integer) how many seconds of data to read. Only provide
#'      one of segment_end or duration_secs. 
#' @param segment_end (POSIXct) end time of the target segment, in accelerometer 
#'      clock time. Ignored if duration_secs is provided
#' @param path_db (character) path to SQLite database file containing the raw 
#'      acc data.
#' @param sr (integer) sampling rate in Hz (samples per minute).
#' @param buffer_secs (integer) number of seconds before and after the segment 
#'      to load, e.g. for computation lead-in
#' @param select_cols (character) columns to extract from the database's acc 
#'      table using the SQL SELECT query, defaults to all columns "*". 
#'
#' @return data.frame with columns recording_id, datetime, accX, accY, accZ
get_raw_acc_segment <- 
    function(recording_id, segment_start, segment_end = NA, duration_secs = NA, 
             path_db = path_raw_acc_db, sr = acc_sr, buffer_secs = 0, 
             select_cols = "*"){
        # Input checking
        if(length(segment_end) != 1 | length(segment_start) != 1 | 
           length(duration_secs) != 1 | length(sr) != 1){
            stop(paste0("Only a single value can be provided for arguments ", 
                        "segment_start, segment_end, duration, and sr"))
        }
        if(!is.na(segment_end) && !is.na(duration_secs)){
            stop("Only one of segment_end or duration_secs can be provided")
        }
        if(is.na(segment_end) && is.na(duration_secs)){
            stop("One of segment_end or duration_secs must be provided")
        }
        
        # Attempt conversion to POSIXct if in plain text format
        if(!is.POSIXct(segment_start)){
            segment_start <- 
                as.POSIXct(segment_start, format="%Y-%m-%d %H:%M:%OS", tz="CET")
            if(is.na(segment_start)) 
                stop("Failed to parse segment_start time, provide as POSIXt")
        }
        if(!is.na(segment_end)){
            if(!is.POSIXct(segment_end)){
                segment_end <- 
                    as.POSIXct(segment_end, format="%d/%m/%Y %H:%M:%OS", tz="CET")
                if(is.na(segment_end)) 
                    stop("Failed to parse segment_start time, provide as POSIXt")
            } 
        }
        
        # Connect to db
        mydb <- dbConnect(RSQLite::SQLite(), path_db)
        # Get the required recording_id's numeric database code
        thisid <- 
            dbGetQuery(mydb, glue("SELECT id FROM recordings ", 
                                  "WHERE recording_id = '{recording_id}'")) |>
            as.numeric()
        if(is.na(thisid)) 
            stop(paste0("Could not find id in database: ", recording_id))
        
        # Select rows from database according to requested start and end times
        if(!is.na(duration_secs)){
            segment_end <- segment_start + duration_secs
        }
        if(buffer_secs > 0){
            segment_end <- segment_end + buffer_secs
            segment_start <- segment_start - buffer_secs
        }
        # Convert to string for database comparison
        segment_start <- strftime(segment_start, 
                                  format = "%Y-%m-%d %H:%M:%OS3", tz = "CET")
        segment_end <- strftime(segment_end, 
                                format = "%Y-%m-%d %H:%M:%OS3", tz = "CET")
        result <- dbGetQuery(mydb, glue(
        "SELECT {select_cols} 
         FROM acc 
         WHERE datetime > '{segment_start}' 
            AND datetime <= '{segment_end}' 
            AND recording_id IS {thisid}")
        )
        dbDisconnect(mydb)
        if(nrow(result) == 0) 
            stop(paste0("Extracted 0 rows of data for ID ", 
                        recording_id, " try increasing buffer?"))
        return(result)
    }

#' Fetch first row of accelerometer data for a given recording ID from database
#' 
#' NOTE: time stamps in raw accelerometer data are subject to clock drift.
#' 
#' @param recording_id (character) recording ID to fetch data for.
#' @param path_db (character) path to SQLite database file containing the raw 
#'      acc data.
#' @return data.frame with columns recording_id, datetime, accX, accY, accZ
get_raw_acc_first_row <-
    function(recording_id, path_db = path_raw_acc_db){
        # Connect to db
        mydb <- dbConnect(RSQLite::SQLite(), path_db)
        # get the required filename's numeric code
        thisid <- 
            dbGetQuery(mydb, glue("SELECT id FROM recordings ", 
                                  "WHERE recording_id = '{recording_id}'")) |>
            as.numeric()
        if(is.na(thisid)) 
            stop(paste0("Could not find id in database: ", recording_id))
        
        # Select first row for for this recording from the database
        result <- dbGetQuery(mydb, glue(
            'SELECT * 
         FROM acc 
         WHERE recording_id = {thisid}
         LIMIT 1')
        )
        dbDisconnect(mydb)
        return(result)
    }

#' Fetch last row of accelerometer data for a given recording ID from database
#' 
#' NOTE: time stamps in raw accelerometer data are subject to clock drift.
#' 
#' @param recording_id (character) recording ID to fetch data for.
#' @param path_db (character) path to SQLite database file containing the raw 
#'      acc data.
#' @return data.frame with columns recording_id, datetime, accX, accY, accZ
get_raw_acc_last_row <-
    function(recording_id, path_db = path_raw_acc_db){
        # Connect to db
        mydb <- dbConnect(RSQLite::SQLite(), path_db)
        # get the required recording_id's numeric code
        thisid <- 
            dbGetQuery(mydb, glue("SELECT id FROM recordings ", 
                                  "WHERE recording_id = '{recording_id}'")) |>
            as.numeric()
        if(is.na(thisid)) 
            stop(paste0("Could not find id in database: ", recording_id))
        
        # Select last row for this recording from the database
        result <- dbGetQuery(mydb, glue(
            'SELECT * 
         FROM acc 
         WHERE recording_id = {thisid} 
         AND datetime = 
            (SELECT max(datetime) FROM acc WHERE recording_id = {thisid})
         LIMIT 1')
        )
        dbDisconnect(mydb)
        return(result)
    }

#' Fetch all accelerometer data for a given recording ID from database
#' 
#' NOTE: time stamps in raw accelerometer data are subject to clock drift.
#' 
#' @param recording_id (character) recording ID to fetch data for.
#' @param path_db (character) path to SQLite database file containing the raw 
#'      acc data.
#' @return data.frame with columns recording_id, datetime, accX, accY, accZ
get_raw_acc_full <-
    function(recording_id, path_db = path_raw_acc_db){
        # Connect to db
        mydb <- dbConnect(RSQLite::SQLite(), path_db)
        # get the required recording_id's numeric code
        thisid <- 
            dbGetQuery(mydb, glue("SELECT id FROM recordings ", 
                                  "WHERE recording_id = '{recording_id}'")) |>
            as.numeric()
        if(is.na(thisid)) 
            stop(paste0("Could not find id in database: ", recording_id))
        
        # Pull all data for this recording
        result <- dbGetQuery(mydb, glue(
            'SELECT * 
         FROM acc 
         WHERE recording_id = {thisid}')
        )
        dbDisconnect(mydb)
        return(result)
    }

#' Fetch all recording IDs from accelerometer database
#' 
#' @param path_db (character) path to SQLite database file containing the raw 
#'      acc data.
#' @return data.frame with columns 'id' (database id code) and 'recording_id'
get_db_ids <- function(path_db = path_raw_acc_db){
    # Connect to db
    mydb <- dbConnect(RSQLite::SQLite(), path_db)
    # Get the recording IDs 
    result <- dbGetQuery(mydb, glue("SELECT id, recording_id FROM recordings"))
    dbDisconnect(mydb)
    return(result)
}

#' Get clock error from accelerometer data
#' 
#' Scans a 2 minute window of accelerometer data for a calibration event, then
#' computes the clock error. A calibration event consists of a logger sitting
#' flat on the desk for an extended duration of time, then it is moved suddenly
#' at the calibration time. 
#'
#' @param recording_id (character) identifier for target accelerometer recording
#' @param calib_time (character) time of the calibration event according to
#'      reference time (from https://time.is) in format "%d/%m/%Y %H:%M:%S"
#' @param plot (logical) whether to export a plot of the calibration error
#' @return A list with two elements, the plot (if requested), and the number of 
#'      seconds calibration error (numeric). A positive error indicates that 
#'      accelerometer clock is too early/fast, and a negative error indicates 
#'      the accelerometer clock was too late/slow
get_clock_error <- function(recording_id, calib_time, plot = TRUE) {
    # Convert time stamp (plain text) to datetime
    calib_time <- 
        as.POSIXct(calib_time, format="%d/%m/%Y %H:%M:%S", tz="CET")
    
    # Check input
    if(is.na(calib_time)){
        stop(glue("Missing/bad calibration time for {recording_id}"))
    }
    
    # Fetch data for the target accelerometer recording at the calibration time
    acc <- get_raw_acc_segment(
        recording_id = recording_id,
        segment_start = calib_time,
        duration_secs = 60,
        buffer_secs = 30)
    
    acc$datetime <- 
        as.POSIXct(acc$datetime, format="%Y-%m-%d %H:%M:%OS", tz = "CET") 
    
    # Find the resting Z acceleration value (logger begins at rest on the desk)
    resting_z <- acc |> filter(datetime < calib_time) |> pull(accZ) |> median()
    
    # Find the first large deviation from this resting value, indicating the 
    # calibration motion occurred
    impulse_time <- acc |> 
        filter(abs(accZ - resting_z) > 0.2) |> 
        first() |>
        pull(datetime)
    
    # Clock error is the difference between expected calibration time, and 
    # timing of the calibration event. 
    clock_error_secs <- difftime(impulse_time, calib_time, units = 'secs') |>
        as.numeric()
    
    plot <- vis_clock_error(acc, clock_error_secs, calib_time)
    return(list(clock_error_secs = clock_error_secs, plot = plot))
}

#' Visualise clock error and automated calibration event detection
#' 
#' Helper function that plots accelerometry against time since calibration for 
#' a given data frame
#' 
#' @param df data.frame with a column called secs_since_cal 
#'     (seconds since calibration point) and a column with the y variable
#' @param clock_error (numeric) delay between accelerometer-presumed event time 
#'      and detected event time (i.e. clock error)
#' @param y (character) determines the y variable plotted (name of column in the 
#'     dataframe df; usually accX or accZ)
#'
#' @return ggplot object
vis_clock_error <- function(df, clock_error, cal_time) {
    df |> 
        mutate(t = difftime(
            datetime, cal_time, units = 'secs') |> as.numeric()) |>
        ggplot(aes(x=t, y=accZ)) +
        geom_line() + 
        theme_classic()+
        scale_y_continuous(n.breaks=4) +
        scale_x_continuous(n.breaks=30) +
        geom_vline(xintercept=0, colour='red') +
        geom_vline(xintercept=clock_error, colour='blue') +
        theme(plot.title = element_text(size = 12)) +
        coord_cartesian(xlim = c(-10, 10)) +
        ggtitle(label = 
                    paste0("Red line = expected time of the calibration event according",
                           " to the accelerometer clock\n", 
                           "Blue line = detected calibration event from",
                           " acceleration values.")) +
        xlab("\nTime (s) since expected calibration event") +
        ylab(paste("z-axis acceleration (raw values)\n"))
}

#' Estimate accelerometer clock error at a specified time
#' 
#' Clock error at a specific time of interest can be estimated from a 
#' calibration point by assuming a linear rate of clock drift
#'
#' @param target_time (POSIXct) time at which clock error will be estimated
#' @param calib_time (POSIXct) the time at which the calibration 
#'      occurred (e.g. from an atomic clock-synchronised time server).
#' @param calib_error accelerometer clock error (seconds) at the calibration
#' 
#' Times should be provided as POSIXct.
#' Clock error value should be in seconds
#' Drift rate should be seconds of drift per day. 
estimate_clock <- function(target_time, calib_time, calib_error, drift_rate){
    
    # Get days differece with the target time
    days_elapsed <- 
        difftime(calib_time, target_time, units = 'days') |>
        as.numeric()
    # Calculate how many seconds drift occurred during this time
    target_clockerror <- calib_error + drift_rate * days_elapsed 
    return(target_clockerror)
}

#' Calculate clock drift rate
#' 
#' Calculates linear drift rate from two time referencing points. Reference 
#' times should ideally come from an atomic clock for maximum precision. 
#' Function is vectorizable
#' 
#' @param timeA (POSIXct) Time of the first reference (from timeserver)
#' @param timeB (POSIXct) Time of the second reference as determined from a timeserver
#' @param timeA_error (numeric) accelerometer clock error in seconds at first 
#'      reference point
#' @param timeB_error (numeric) accelerometer clock error in seconds at second 
#'      reference point.
#' @return a number giving the drift rate in seconds per day.
get_drift_rate <- function(timeA, timeB, timeA_error, timeB_error){
    # Days elapsed during the calibration period
    days_elapsed = as.numeric(difftime(timeB, timeA, units = 'days'))
    # How much the click drifted during this period
    secs_drift <- timeB_error - timeA_error
    # Drift rate in seconds per day
    drift_rate <- secs_drift / days_elapsed
    return(drift_rate)
}


get_recording_end <- function(filename, estimated_end_time){
    if(!is.POSIXct(recording_start_time)){
        recording_start_time <- 
            as.POSIXct(recording_start_time, format="%d/%m/%Y %H:%M:%S", tz="CET")
    } 
    
}

#' Implement the sing scaling method for 6O scaling
#'
#' This function calculates the median vectorial sum from a short segment of 
#' raw acc data during a calibration. The force of gravity should result in 
#' a vectorial sum of 1 if correctly scaled.
#'
#' @param dat_path (character) the full path to the target csv file containing
#'      calibration data
#' @param calib_time (POSIXct) is the desired time to be used as a reference for scaling.
#' @param start_time (POSIXct) time when the recording started, used in 
#'      combination with the target_time to calculate which rows from the data 
#'      file contains the target time
#' @param seconds (integer) amount of data used to calculate the median
#' @param sample_rate (integer) the sampling rate of the accelerometer in Hz
#' @return single numeric value representing the scale of the accelerometer data
#'      (1 = scaled accurately, <1 values are too low, >1 values are too high)
get_median_vec_sum <- function(dat_path, start_time, calib_time, 
                               seconds = scaling_window, sample_rate = acc_sr){
    if(is.na(calib_time)){
        return(as.numeric(NA))
    }
    # Load segment of accelerometer data from csv file
    buffer <- sample_rate * 2 # add 2s buffer after start time
    n_skip <- 
        as.numeric(difftime(calib_time, start_time, units="secs")) * sample_rate + buffer
    # Read some 'seconds' of data, beginning at start_time
    subbed_data <- 
        fread(dat_path, skip=n_skip, nrows=seconds*sample_rate, header=FALSE) |>
        set_names(names(fread(dat_path, nrows=1)))
    
    # Compute the scale on the target segment of accelerometer data
    median(sqrt(subbed_data$accX^2 + subbed_data$accY^2 + subbed_data$accZ^2))
}

last <- function(vector){vector[length(vector)]}

#' Identify contiguous segments of video-scored behaviour data.
#'
#' Discontinuities in behaviour scoring occur when the recording_id changes, or
#' there is a large time gap between the end time and start time of subsequent
#' behaviours.
#' 
#' @beh_recording_ids (character vector) a sequence of recording id values
#' @beh_start_times (POSIXct) a sequence of behaviour start times
#' @beh_stop_times (POSIXct) a sequence of behaviour stop times
#' @max_gap (numeric) the maximum gap in seconds allowed between subsequent 
#'      behaviours before splitting the behaviours into different segments
#' @return vector of segmentIDs which increments if there is time gap between
#'       behaviours larger than max_gap, or if there is a new recording_id
find_beh_segments <- function(beh_recording_ids, beh_start_times, beh_stop_times, 
                              max_gap = 0.5){
    if(length(beh_recording_ids) != length(beh_start_times) |
       length(beh_recording_ids) != length(beh_stop_times)){
        stop("recording_ids, start times, and stop times should be vectors with
             the same length")
    }
    # Work out number of seconds difference between the end of previous behaviour
    # and the start of the current behaviour
    beh_lag <- beh_start_times[-1] - beh_stop_times[-length(beh_stop_times)]
    beh_lag <- c(0, round(as.numeric(beh_lag), 2))
    
    # Determine if there was a large gap between this behaviour and the previous
    big_gap <- abs(beh_lag) > max_gap
    
    # Determine if this behaviour is associated with a different recording ID
    # than the previous behaviour
    new_recordingid <- c(FALSE, beh_recording_ids[-1] != 
                             beh_recording_ids[-length(beh_recording_ids)])
    
    # Return out a segment id for every row, which increments whenever there was 
    # a gap, or the behaviour was from a new recording ID
    return(cumsum(big_gap | new_recordingid) + 1)
}

#' Adjust timestamps to match accelerometer's clock
#' 
#' Adjusts timestamps from internet-synchronized ('real') time to accelerometer 
#' clock time. Vectorizable
#'
#' @target: the time(s) to be transformed from real time to accelerometer time
#'       (e.g. placing scored behaviour onto the accelerometer timeline)
#' @origin: real time at which the accelerometer was switched on
#' @dr: drift rate of accelerometer clock (seconds per day)
#' @init_error: initial accelerometer clock error (in seconds)
#'
#' @return times(s) transformed to the accelerometer clock, thus aligning 
#'      behaviours with accelerometer data
adjust_time <- function(target_time, origin, dr, init_error){
    # Calcualte difference between origin and target time
    days_elapsed <- as.numeric(difftime(target_time, origin, units="days"))
    # Calculate the clock error (how different the accelerometer is from real 
    # time) expected at that target time
    target_time_error <- dr * days_elapsed + init_error
    return(target_time + target_time_error)
}


