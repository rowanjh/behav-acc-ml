# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'       Utility functions used in various parts of the project
#' 
#' Date Created: 
#'      May 2, 2023
#'      
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#' Round fractional seconds and print
#' 
#' Due to floating point precision and the way that print methods round down
#' fractional seconds, datetimes can get messed up when converted from POSIXct 
#' to character (e.g. when exporting to csv). See more detail in this thread
#' https://stackoverflow.com/questions/7726034/how-r-formats-posixct-with-fractional-seconds
#' This function is still buggy sometimes but is an improvement.
#' 
#' @param x (POSIXct) a vector of datetime values to round.
#' @param digits number of decimal places to be used.
prtime <- function(x, digits=0) {
    x2 <- round(unclass(x), digits)
    attributes(x2) <- attributes(x)
    x <- as.POSIXlt(x2)
    x$sec <- round(x$sec, digits)
    # Add a fudge factor 
    x$sec <- x$sec + 10^(-digits-1)
    format.POSIXlt(x, paste("%Y-%m-%d %H:%M:%OS",digits,sep=""))
}

#' Convenience functions to convert timestamps from character to POSIXct
toposix_ymdhms <- function(time){
    as.POSIXct(time, format = "%Y-%m-%d %H:%M:%OS", tz = "CET")
}
toposix_dmyhms <- function(time){
    # no fractional seconds
    as.POSIXct(time, format = "%d/%m/%Y %H:%M:%S", tz = "CET") 
}
