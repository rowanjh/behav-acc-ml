# ~~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'      Helper functions for implementing sliding windows. See windowing.R
#'
#' Date created:
#'      May 2, 2023
#'      
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(dtw)
library(moments)
library(data.table)
library(purrr)
#' Compute static acceleration and derived values from raw accelerometer data
#' 
#' Static X Y Z, as well as pitch and roll are computed.
#'
#' @param df (data.frame) df assumed to have accX, accY, and accZ columns 
#'      representing raw acceleration values
#' @param static_window_size (numeric) the size of the rolling window over which
#'      to compute static acceleration, in seconds
#' @param sr (integer) sampling rate (in Hz)
#' @return data.frame with new columns appended
get_static <- function(df, static_window_size, sr){
    df$statX <- frollmean(df$accX, sr * static_window_size, fill=NA, 
                          algo="fast", align="center", na.rm=TRUE)
    df$statY <- frollmean(df$accY, sr * static_window_size, fill=NA, 
                          algo="fast", align="center", na.rm=TRUE)
    df$statZ <- frollmean(df$accZ, sr * static_window_size, fill=NA, 
                          algo="fast", align="center", na.rm=TRUE)
    df$pitch <- atan(-df$statX/sqrt(df$statY^2+df$statZ^2))*180/pi
    df$roll <- atan2(df$statY, df$statZ)*180/pi
    return(df)
}

#' Computes dynamic acceleration and derived values on raw accelerometer data
#' 
#' Dynamic X/Y/Z, PDBA, ODBA, VeDBA, PDBA to VeDBA ratio are calculated
#' @param df (data.frame) df assumed to have columns accX/Y/Z (raw acceleration)
#'      and statX/Y/Z (static acceleration)
#' @param sr (integer) sampling rate (in Hz)
#' @return data.frame with new columns appended
get_dynamic <- function(df, sr){
    # Assumes the data has columns accX/Y/Z, and statX/Y/Z
    # 
    
    # Dynamic acceleration
    df$dynX <- df$accX - df$statX
    df$dynY <- df$accY - df$statY
    df$dynZ <- df$accZ - df$statZ
    df$dynXYZ <- sqrt(df$dynX^2 + df$dynY^2 + df$dynZ^2) # used in HAR features
    
    # Partial dynamic acceleration (PDBA)
    df$pdynX <- abs(df$dynX)
    df$pdynY <- abs(df$dynY) 
    df$pdynZ <- abs(df$dynZ)
    
    # Overall dynamic body acceleration (ODBA)
    df$ODBA <- df$pdynX + df$pdynY + df$pdynZ
    
    # Vectorial dynamic body acceleration (VeDBA) ('total' dynamic acc)
    df$VeDBA <- (df$pdynX^2 + df$pdynY^2 + df$pdynZ^2)^0.5
    
    # Smoothed VeDBA (over 1 second)
    df$smVeDBA <- frollmean(df$VeDBA, sr, fill = NA, 
                            algo = "fast", align = "center", na.rm = TRUE)
    
    df$smVeDBA[is.na(df$VeDBA)] <- NA
    # Only take smVeDBA values when VeDBA is not NA (in other words the 
    # smoothing window should only be calculated if it is at least half full - 
    # if there are 49 missing values and 1 valid value in the window, don't keep it)
    
    # PDBA to VeDBA ratio
    df$ratio_VeDBA_pdynX <- df$VeDBA / pmax(0.001, df$pdynX)
    df$ratio_VeDBA_pdynY <- df$VeDBA / pmax(0.001, df$pdynY)
    df$ratio_VeDBA_pdynZ <- df$VeDBA / pmax(0.001, df$pdynZ)
    return(df)
}

#' Computes differential acceleration and derived values (jerk)
#' 
#' Differential acceleration (jerk) is the rate of change in acceleration. 
#' Calculated as the difference in acceleration between between the 
#' prior and the next sample (n+1 and n-1), divided by the number of 
#' samples. A lag-2 version is also calculated in case that is more performant.
#' 
#' @param df (data.frame) df assumed to have columns 
#' @param sr (integer) sampling rate (in Hz)
#' @param RMS_window (numeric) size of the window (in seconds) over which to 
#'      smooth jerk for the RMS jerk calculation
#' @return data.frame with new columns appended
get_jerk <- function(df, sr, RMS_window){
    # Differential acceleration: rate of change in acc based on n+1 and n-1 samples,
    # or n+2 and n-2 samples. Trying 2 different lags to compare performance
    
    # Pad start and end with NAs. Divide by time  to get rate of change
    df$difX_lag1 <- c(NA, diff(df$accX, lag = 2), NA) / (2/sr)
    df$difY_lag1 <- c(NA, diff(df$accY, lag = 2), NA) / (2/sr)
    df$difZ_lag1 <- c(NA, diff(df$accZ, lag = 2), NA) / (2/sr)
    
    df$difX_lag2 <- c(NA, NA, diff(df$accX, lag = 4), NA, NA) / (4/sr)
    df$difY_lag2 <- c(NA, NA, diff(df$accY, lag = 4), NA, NA) / (4/sr)
    df$difZ_lag2 <- c(NA, NA, diff(df$accZ, lag = 4), NA, NA) / (4/sr)
    
    # Jerk (instantaneous jerk, over 2 or 4 samples)
    # i.e. differential acceleration across all 3 axes
    df$jerk_lag1 <- (df$difX_lag1^2 + df$difY_lag1^2 + df$difZ_lag1^2)^0.5
    df$jerk_lag2 <- (df$difX_lag2^2 + df$difY_lag2^2 + df$difZ_lag2^2)^0.5
    
    # Jerk windowed (RMS jerk across 250ms) 
    # Specify target window to smooth jerk over in seconds (250ms = 0.25
    # Calculate number of samples needed to sum over to achieve target window (approx)
    samples_needed <- round(RMS_window / (1/sr)) 
    
    df$jerkRMS_lag1 <- 
        frollapply(df$jerk_lag1, samples_needed, FUN = function(x){sqrt(sum(x^2))}, 
                   fill=NA, align="center")
    
    df$jerkRMS_lag2 <- 
        frollapply(df$jerk_lag2, samples_needed, FUN = function(x){sqrt(sum(x^2))}, 
                   fill=NA, align="center")
    return(df)
}

#' Get timestamps for a sliding window implementation
#' 
#' A series of sliding windows will be created between sw_start and sw_end. The
#' number of windows and window spacing depends on the window size and step. If
#' window step is smaller than window size, there will be overlapping windows.
#'
#' @param sw_start (POSIXct) The start time of the series of sliding windows
#' @param sw_end (POSIXct) The end time of the series of sliding windows
#' @param win_size (numeric) the duration of each window, in seconds
#' @param win_step (numeric) the interval between the start time of a window, 
#'      and the start time of the next window.
#' @return data.frame with 1 row per window, and window start/window end columns
get_window_timestamps <- 
    function(sw_start, sw_end, win_size, win_step){
        ## Calculate how many windows can be extracted from a sequence of data
        # Checks
        if (!is.POSIXt(sw_start) | !is.POSIXt(sw_end)){
            stop(paste0("seg_start and seg_end must be provided as POSIXct",
                        "with fractional seconds"))
        }
        
        # Allow partial windows at the end of the sequence
        window_duration <- as.numeric(sw_end) - as.numeric(sw_start)
        nwindows <- ceiling(window_duration / win_step)
        
        window_df <- data.frame(window_start = rep(NA, nwindows), 
                                window_end = rep(NA, nwindows))
        
        starts <- (0:(nwindows-1)) * win_step
        ends <- starts + win_size
        window_df$window_start <- prtime(starts + sw_start,2)
        window_df$window_end <- prtime(ends + sw_start,2)
        return(window_df)
    }

#' Determine which rows of the raw accelerometer data file belong to this window
#' 
#' This function helps to subset a raw acceleration data frame more quickly than
#' filtering by window datetime.
#' @param window_index (integer) The window number (first, second, third etc.)
#' @param sample_rate (integer) sampling rate (in Hz)
#' @param win_size (numeric) the duration of each window, in seconds
#' @param win_step (numeric) the interval between the start time of a window, 
#' @return 
get_rows_for_window <- function(window_index, sample_rate, win_size, win_step){
    # epoch number = first, second, third epoch etc. 
    # sample_rate = sampling rate in Hz
    # win_size = epoch size in seconds
    # win_step = difference in start time between successive windows (in seconds)
    
    n_per_window <- win_size * sample_rate # number of samples (rows) in each epoch
    n_per_step <- win_step * sample_rate
    
    row_start <- 1 + round((window_index-1) * n_per_step) # rounds down if odd sample rate
    row_end <- row_start + n_per_window -1
    
    return(row_start:row_end)
}

#' Compute various features from Human Activity Recognition literature
#'
#' All rows in the input dataset will be summarised, resulting in a single 
#' output value for each feature
#'
#' @param df (data.frame) dataset assumed to contain the columns dynX, dynY, 
#'      dynZ, and dynXYZ
#' @return named numeric vector, containing a value for each feature computed 
get_HAR_features <- function(df){
    results <- list()
    for(axis in c('X','Y','Z','XYZ')){
        this_var <- paste0('dyn', axis)
        this_series <- df[[this_var]]
        if(any(is.na(this_series))) next
        var <- var(this_series)
        RMS <- sqrt(mean(this_series^2))
        IQR <- IQR(this_series)
        P2P <- max(this_series) - min(this_series) # peak-to-peak
        MAV <- mean(abs(this_series)) # Mean absolute value
        WL <- sum(abs(this_series[-1] - this_series[-length(this_series)])) #Waveform length
        LD <- exp(mean(log(abs(this_series)))) #Log detector
        if(!var(this_series) == 0){
            ARs <- ar(this_series, aic = FALSE, order.max = 4)$ar # Autoregressive coefficients
        } else{
            ARs <- c(0,0,0,0)
        }
        AR1 <- ARs[1]
        AR2 <- ARs[2]
        AR3 <- ARs[3]
        AR4 <- ARs[4]
        MMAV <- calc_mmav(this_series) # Modified mean absolute value
        sig_pwr <- sum(this_series^2) # Just RMS squared. Sometimes mean rather than sum is used ('energy')
        CV <- sd(this_series) / mean(this_series) # Coefficient of variation
        pct_10 <- quantile(this_series, 0.1)
        pct_25 <- quantile(this_series, 0.25)
        pct_50 <- quantile(this_series, 0.5)
        pct_75 <- quantile(this_series, 0.75)
        pct_90 <- quantile(this_series, 0.9)
        # Trapezoidal_num_integration, step = 1, formula from:
        # https://stackoverflow.com/questions/4954507/calculate-the-area-under-a-curve
        auc_trap <- sum(diff(1:length(this_series)) * (head(abs(this_series),-1)+tail(abs(this_series),-1)))/2
        # Median Crossings
        MC <- sum(diff(this_series < median(this_series)) != 0)
        
        results[[axis]] <- c(var, RMS, IQR, P2P, MAV, WL, LD, AR1, AR2, AR3, AR4, MMAV, 
                             sig_pwr, CV, pct_10, pct_25, pct_50, pct_75, pct_90, auc_trap, MC)
        names(results[[axis]]) <- 
            paste0(c('var', 'RMS', 'IQR', 'P2P', 'MAV', 'WL', 
                     'LD', 'AR1', 'AR2', 'AR3', 'AR4', 'MMAV',
                     'sig_pwr', 'CV', 'pct_10', 'pct_25', 'pct_50', 
                     'pct_75', 'pct_90', 'auc_trap', 'MC'), '_', axis)
    }
    names(results) <- NULL # Trick to prevent unlist from adding X. Y. Z. prefixes later
    
    # Signal magnitude area
    results$SMA <- mean(abs(df$dynX) + 
                            abs(df$dynY) + 
                            abs(df$dynZ)) 
    return(unlist(results))
}

#' Calculate modified mean absolute value 
#'
#' helper function for get_HAR_features
#' @param series (numeric) a univariate series of acceleration values (e.g. 50
#'      consecutive samples of dynamic acceleration on the X-axis)
#' @return A numeric value giving the modified mean absolute value
calc_mmav <- function(series){
    N <- length(series)
    w <- 1:N <= 0.75*N & 1:N >= 0.25 * N
    result <- mean(w*abs(series))
    return(result)
}

#' Calculate change of correlation values
#' 
#' Calculates the pairwise correlation between values on the X, Y, and Z axes, 
#' and the degree of change in these correlations across successive windows.
#' 
#' @param thisdat (data.frame) accelerometer data for the target window, assumed 
#'      to contain accX, accY, and accZ columns
#' @param prevdat (data.frame) accelerometer data for the previous window,
#'       assumed to contain accX, accY, and accZ columns
#' @param nextdat (data.frame) accelerometer data for the next window, assumed 
#'      to contain accX, accY, and accZ columns
#' @return a list containing pairwise correlations for the target window, and 
#'      the change in correlation over successive windows
calc_CCA <- function(thisdat, prevdat, nextdat){
    # Get pairwise correlations between axes
    this_cor_XY <- cor(thisdat$accX, thisdat$accY)
    this_cor_XZ <- cor(thisdat$accX, thisdat$accZ)
    this_cor_YZ <- cor(thisdat$accY, thisdat$accZ)
    prev_cor_XY <- cor(prevdat$accX, prevdat$accY)
    prev_cor_XZ <- cor(prevdat$accX, prevdat$accZ)
    prev_cor_YZ <- cor(prevdat$accY, prevdat$accZ)
    next_cor_XY <- cor(nextdat$accX, nextdat$accY)
    next_cor_XZ <- cor(nextdat$accX, nextdat$accZ)
    next_cor_YZ <- cor(nextdat$accY, nextdat$accZ)
    
    # Get how the correlation between axes changed over time
    CCA_prev_XY <- this_cor_XY - prev_cor_XY
    CCA_prev_XZ <- this_cor_XZ - prev_cor_XZ
    CCA_prev_YZ <- this_cor_YZ - prev_cor_YZ
    CCA_next_XY <- next_cor_XY - this_cor_XY
    CCA_next_XZ <- next_cor_XZ - this_cor_XZ
    CCA_next_YZ <- next_cor_YZ - this_cor_YZ
    
    results <- list(COR_XY = this_cor_XY,
                    COR_XZ = this_cor_XZ,
                    COR_YZ = this_cor_YZ,
                    CCA_prev_XY = CCA_prev_XY, 
                    CCA_prev_XZ = CCA_prev_XZ, 
                    CCA_prev_YZ = CCA_prev_YZ, 
                    CCA_next_XY = CCA_next_XY, 
                    CCA_next_XZ = CCA_next_XZ, 
                    CCA_next_YZ = CCA_next_YZ)
    return(results)
}

#' Calculate change of similarity values
#' 
#' Calculates the pairwise similarities between values on the X, Y, and Z axes 
#' using dynamic time warping. Calculate the degree of change in these 
#' similarity values across succesive windows.
#' 
#' @param thisdat (data.frame) accelerometer data for the target window, assumed 
#'      to contain accX, accY, and accZ columns
#' @param prevdat (data.frame) accelerometer data for the previous window,
#'       assumed to contain accX, accY, and accZ columns
#' @param nextdat (data.frame) accelerometer data for the next window, assumed 
#'      to contain accX, accY, and accZ columns
#' @return a list containing pairwise similarities for the target window, and 
#'      the change in similarities over successive windows
calc_CSA <- function(thisdat, prevdat, nextdat){
    # Compute similarities between X-Y, X-Z, Y-Z time series in the current 
    # time window and previous/next windows
    this_sim_XY <- dtw(thisdat$accX, thisdat$accY, distance.only = TRUE)$distance
    this_sim_XZ <- dtw(thisdat$accX, thisdat$accZ, distance.only = TRUE)$distance
    this_sim_YZ <- dtw(thisdat$accY, thisdat$accZ, distance.only = TRUE)$distance
    prev_sim_XY <- dtw(prevdat$accX, prevdat$accY, distance.only = TRUE)$distance
    prev_sim_XZ <- dtw(prevdat$accX, prevdat$accZ, distance.only = TRUE)$distance
    prev_sim_YZ <- dtw(prevdat$accY, prevdat$accZ, distance.only = TRUE)$distance
    next_sim_XY <- dtw(nextdat$accX, nextdat$accY, distance.only = TRUE)$distance
    next_sim_XZ <- dtw(nextdat$accX, nextdat$accZ, distance.only = TRUE)$distance
    next_sim_YZ <- dtw(nextdat$accY, nextdat$accZ, distance.only = TRUE)$distance
    
    # Compute changes in X-Y-Z similarity across windows
    CSA_prev_XY <- this_sim_XY - prev_sim_XY
    CSA_prev_XZ <- this_sim_XZ - prev_sim_XZ
    CSA_prev_YZ <- this_sim_YZ - prev_sim_YZ
    CSA_next_XY <- next_sim_XY - this_sim_XY
    CSA_next_XZ <- next_sim_XZ - this_sim_XZ
    CSA_next_YZ <- next_sim_YZ - this_sim_YZ
    
    results <- list(sim_XY = this_sim_XY,
                    sim_XZ = this_sim_XZ,
                    sim_YZ = this_sim_YZ,
                    CSA_prev_XY = CSA_prev_XY, 
                    CSA_prev_XZ = CSA_prev_XZ, 
                    CSA_prev_YZ = CSA_prev_YZ, 
                    CSA_next_XY = CSA_next_XY, 
                    CSA_next_XZ = CSA_next_XZ, 
                    CSA_next_YZ = CSA_next_YZ)
    return(results)
}

### ======= Frequency-based features -------------
# Note: the FFT only uses 2^n points, so only the first 128 points from the time  
# series.. might be able to optimise instead of pulling out 150.

#' Compute frequency-based features derived from a fast fourier transform
#'
#' All rows in the input dataset will be summarised, resulting in a single 
#' output value for each feature
#'
#' @param df (data.frame) accelerometer dataset assumed to have columns accX, 
#'      accY, and accZ
#' @param sr (integer) sampling rate
#' @param maxfreq maximum frequency that will be extracted from the spectrum.
#'      This frequency and all lower frequencies will be extracted
#' @param fft_nbins (integer) number of bins used to split the spectrum and
#'      to compute power density for. DC is always used, then the remainder
#'      of the extracted spectrum will be averaged over bins 
#'      (e.g. 0.0Hz-2.5Hz 2.5Hz-5.0Hz, 5.0Hz-7.5Hz, etc.)
#' @return named numeric vector, containing a value for each feature computed 
get_power_features <- function(df, sr, maxfreq, fft_nbins){
    # Maxfreq is the maximum allowed frequency for the FFT to consider (in Hz).
    # This is important for variable-sized FFT windows
    power_results_holder <- list(X = NULL, Y = NULL, Z = NULL)
    # Loop over X Y and Z accelerometer axes
    for(this_axis in c('X', 'Y', 'Z')){
        this_var <- paste0("acc", this_axis)
        # Pull out acc data from this axis
        this_series <- df[[this_var]]
        # Get PSD for this time series
        PSD <- this_series |> detrend_time_series() |> fft() |>
            FFT_to_PSD(sr = sr)
        
        # Compute basic summaries of the PSD
        top <- PSD %>% top_n(2, power)
        pwr_top1_freq <- top$freq_Hz[1]
        pwr_top1_pwr <- top$power[1]
        pwr_top2_freq <- top$freq_Hz[2]
        pwr_top2_pwr <- top$power[2]
        pwr_total <- sum(PSD$power)
        pwr_variance <- sd(PSD$power)
        
        # Additional frequency features from HAR
        pwr_entropy <- -sum((PSD$power / sum(PSD$power)) *
                                log(PSD$power / sum(PSD$power)+0.000001))
        # 1/4 power in PSD below this frequency, 3/4 power above
        F25 <- calc.mdf(PSD, 0.25)
        # Median frequency: half of the power above and below this frequency
        MDF <- calc.mdf(PSD) 
        # 3/4 power below, 1/4 power above this frequency
        F75 <- calc.mdf(PSD, 0.75)
        # Mean power frequency
        MPF <- sum(PSD$freq_Hz * PSD$power) / sum(PSD$power) 
        # Power spectral density skewness
        pwr_skew <- moments::skewness(PSD$power)
        # Power spectral density kurtosis
        pwr_kurt <- moments::kurtosis(PSD$power)
        
        # Create binned features of the PSD
        PSD_binned <- bin_PSD(PSD, n_bins = fft_nbins, maxfreq = maxfreq)
        
        # Create a list that is friendly for adding to the data frame
        power_features_thisaxis <- c(
            list(pwr_top1_freq = pwr_top1_freq, pwr_top1_pwr = pwr_top1_pwr, 
                 pwr_top2_freq = pwr_top2_freq, pwr_top2_pwr = pwr_top2_pwr, 
                 pwr_total=pwr_total, pwr_variance=pwr_variance,
                 pwr_entropy = pwr_entropy, pwr_F25 = F25, pwr_MDF = MDF,
                 pwr_F75 = F75, pwr_MPF = MPF, 
                 pwr_skew = pwr_skew, pwr_kurt = pwr_kurt),
            # Make the binned PSD output list-friendly
            as.list(PSD_binned$powerAv) %>% 
                set_names(paste0("pwr_", PSD_binned$bin))
        )
        names(power_features_thisaxis) <-
            paste0(names(power_features_thisaxis), "_", this_axis)
        
        # Now put the results into the results holder object
        power_results_holder[[this_axis]] <- power_features_thisaxis
    }
    names(power_results_holder) <- NULL
    power_results_holder <- unlist(power_results_holder)
    return(power_results_holder)
}

#' Helper functions to de-trend a timeseries
#'
#' Detrending removes any linearly increasing or decreasing trend from 
#' the series. It is a recommended step to be done before computing the power 
#' spectral density (PSD). Code from Fehlmann et al. 2017
#' @param series
#' @return
detrend_time_series <- function(series){
    alpha <- 1:length(series)
    detrended_series <- lm(series ~ alpha)$residuals
    return(detrended_series)
}

#' Compute power spectral density
#' 
#' Computes power spectral density from the output of the fft() function. The 
#' first element of fft() output is the constant/DC component, so labels for 
#' the PSD start at 0Hz. Code from Fehlmann et al. 2017's supplement
#' 
#' Usage: 
#'  PSD <- my_series |> detrend_time_series |> fft |> FFT_to_PSD(acc_sr)
#' 
#' @param FFT output from fft() function
#' @param sr Sampling rate (in Hz)
#' @return data frame with two columns: frequency, and power for that frequency
FFT_to_PSD <- function(FFT, sr){
    seq_seconds <- length(FFT) / sr
    PSD <- data.frame(freq_Hz = 0:(length(FFT)-1) / seq_seconds, # Convert to Hz
                      power = (sqrt(Re(FFT)^2+Im(FFT)^2)*2/length(FFT))^2)
    
    # Remove the second half of the spectrum, as it is redundant
    PSD <- PSD[1:floor(nrow(PSD)/2),]
    return(PSD)
}


#' Bin the power spectral density estimates
#' 
#' Split the power spectrum into bins, and calculate the average power in 
#' each bin.
#' 
#' @param PSD (data.frame) power spectral density estimates, assumed to have 
#'      columns 'freq_Hz' and 'power'
#' @param n_bins number of bins used to split the power spectral density. The 
#'      DC component is always included, and the rest of the spectrum is binned
#' @param maxfreq A maximum frequency to include (in Hz). Any higher frequencies 
#'      are excluded prior to binning. 
#' @return a data.frame with two columns: bin (factor), and average power in
#'      that bin (numeric)
bin_PSD <- function(psd, n_bins, maxfreq){
    # Work out cutpoints for bins
    binwidth <- maxfreq/n_bins
    cutpoints <- seq(from = 0, to = maxfreq, by = binwidth)
    
    # Assign frequencies to bins
    bins <- cut(psd$freq_Hz[-1], breaks = cutpoints,
                labels = paste0(
                    sprintf("%.1f", cutpoints[-length(cutpoints)]), "Hz-",
                    sprintf("%.1f", cutpoints[-1]), "Hz")) |>
        as.character() %>%
        c("0Hz", .)
    
    # Add column. Converting to factor gives output nicer ordering
    psd$bin <- factor(bins, levels = unique(bins))
    # Get mean for each bin
    PSD_binned <- psd |>
        group_by(bin) |>
        summarise(powerAv = mean(power))
    return(PSD_binned)
}

#' Calculating median frequency
#' 
#' Helper function to compute the frequency which sits at the middle of the PSD; 
#' one that has equal amounts of power above and below it in the PSD.
#'
#' @param psd (data.frame) power spectral density estimates, assumed to have 
#'      columns 'freq_Hz' and 'power'
#' @param prop (numeric) density quantile to estimate. E.g. prop 0.25 gives the 
#'      frequency which has 75% of power above it and 25% of power below it
#' @return a number giving the median (or other target quantile) frequency in Hz
calc.mdf <- function (psd, prop = 0.5){
    if(prop >=1| prop <=0) stop("prop must be between 0 and 1")
    total_power <- sum(psd$power)

    psd[which.min(abs(cumsum(psd$power) - total_power*prop)),'freq_Hz']
}

