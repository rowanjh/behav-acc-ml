# ~~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' https://github.com/rowanjh/behav-acc-ml
#'
#' Purpose: 
#'      Helper functions for hidden markov models. See hmm.R
#'
#' Date created:
#'      May 2, 2023
#' 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#' Get transition matrix from class label sequence
#'
#' @param class_seq (factor) a sequence of temporally consecutive labels 
#'          (i.e. no time gaps)
get_trans_mat <- function(class_seq, classes){
    l <- length(class_seq)
    transitions <- list()
    for (i in classes){
        # Where does this behaviour appear in the sequence
        w <- which(class_seq[1:(l-1)] == classes[i])
        #' Get the state that came beforehand, take a difference which says which
        #' class it was (-1 = the class before in terms of factor ordering, 
        #' which is 1 line above it in the transition matrix)
        s <- class_seq[w] - class_seq[w + 1]
        transitions[[i]] <- s
    }
    # Sum up all of the transitions into a matrix for this segment
    # Loop over each row and column of the transition matrix
    tmat <- matrix(0, nrow=length(classes), ncol = length(classes))
    for (i in classes){
        for (j in classes){
            # for state i in row i, find how often it transitioned to other 
            # candidate classes j. 
            # i-j means 
            tmat[i,j] <- sum(transitions[[i]] == (i-j))
        }
    }
    return(tmat)
}

#' Run Viterbi Algorithm
#' 
#' Function taken from Leos-Barajas et al. 2017. 
#' 
#' @param x the input sequence (sequence of feature values)
#' @param m the number of possible states (behaviours)
#' @param gamma the transition matrix
#' @param allprobs the emission probabilities
#' @param delta the initial probabilities
HMM.viterbi <- function(x, m, gamma, allprobs, delta=NULL, ...)
{
    if(is.null(delta)) delta <- solve(t(diag(m) - gamma + 1), rep(1, m))
    n <- dim(x)[1]
    
    xi <- matrix(0, n, m)
    foo <- delta*allprobs[1,] # Get the most likely initial state given data and initial probabilities
    xi[1,] <- foo/sum(foo) # Put the initial probs in the matrix
    for(i in 2:n)
    {
        foo <- apply(xi[i-1,]*gamma, 2, max)*allprobs[i,] # multiply transition probs by emission probs
        xi[i,] <- foo/sum(foo)
    }
    
    iv <- numeric(n)
    iv[n] <- which.max(xi[n,])
    
    for(i in (n-1):1)
        iv[i] <- which.max(gamma[, iv[i+1]]*xi[i,])
    
    return(iv)
}
