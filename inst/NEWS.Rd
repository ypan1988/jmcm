\name{NEWS}
\title{jmcm News}
\encoding{UTF-8}

\section{CHANGES IN VERSION 0.2.0}{
  \subsection{USER-VISIBLE CHANGES}{
    \itemize{
      \item add option \code{optim.method} in function \code{jmcm()}: 
      by default, \code{optim.method} is 'default' so that the BFGS algorithm 
      we implemented internally is used; alternatively, we can set 
      \code{optim.method} as 'BFGS' so that \code{vmmin()} (the underlying C 
      function used in \code{stats::optim()} for BFGS algorithm) will be 
      called through R package roptim.
    }
  }
  \subsection{NEW FEATURES}{
    \itemize{
      \item class \code{JmcmBase} is implemented as the base class for class 
      \code{MCD}, \code{ACD} and \code{HPC} to reduce code duplication.
      \item class \code{JmcmFit} is used for fitting of the three
      Cholesky-based joint mean-covariance models (i.e., MCD, ACD and HPC).
    }
  }
}