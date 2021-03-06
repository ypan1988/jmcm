jmcm: Joint Mean-Covariance Models in R.
====

[![Build Status](https://travis-ci.org/ypan1988/jmcm.svg?branch=master)](https://travis-ci.org/ypan1988/jmcm)
[![cran version](http://www.r-pkg.org/badges/version/jmcm)](https://cran.r-project.org/web/packages/jmcm)
[![downloads](http://cranlogs.r-pkg.org/badges/jmcm)](http://cranlogs.r-pkg.org/badges/jmcm)
[![total downloads](http://cranlogs.r-pkg.org/badges/grand-total/jmcm)](http://cranlogs.r-pkg.org/badges/grand-total/jmcm)
[![Research software impact](http://depsy.org/api/package/cran/jmcm/badge.svg)](http://depsy.org/package/r/jmcm)

## Features

* Efficient for large data sets, using algorithms from the
[Armadillo](http://arma.sourceforge.net/) linear algebra package via the
[RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo/index.html)
interface layer.
* Fits joint mean-covariance models based on three Cholesky decomposition-type
covariance structure modelling methods, namely modified Cholesky decomposition
(MCD), alternative Cholesky decomposition (ACD) and hyperpherical
parameterization of Cholesky factor (HPC).

## Citation
To cite **jmcm** in publications use:

Pan J and Pan Y (2017). “jmcm: An R Package for Joint Mean-Covariance Modeling of Longitudinal Data.” _Journal of Statistical Software_, 82(9), pp. 1–29. doi: 10.18637/jss.v082.i09.

Corresponding BibTeX entry:

    @Article{,
      title = {{jmcm}: An {R} Package for Joint Mean-Covariance Modeling
        of Longitudinal Data},
      author = {Jianxin Pan and Yi Pan},
      journal = {Journal of Statistical Software},
      year = {2017},
      volume = {82},
      number = {9},
      pages = {1--29},
      doi = {10.18637/jss.v082.i09},
    }

## Installation

Get the development version from github:
```R
install.packages("devtools")
library(devtools)
devtools::install_github("ypan1988/jmcm", dependencies=TRUE)
```

Or the released version from CRAN:
```R
install.packages("jmcm")
```

## Usage

Fit a joint mean-covariance model to longitudinal data, via maximum likelihood:
```R
jmcm(formula, data = NULL, triple = c(3, 3, 3), cov.method = c("mcd", "acd", "hpc"),
  optim.method = c("default", "BFGS"), control = jmcmControl(), start = NULL)
```

* formula:
a two-sided linear formula object describing the covariates for both the mean and covariance matrix part of the model, with the response, the corresponding subject id and measurement time on the left of a operator~, divided by vertical bars ("|").

* data:
a data frame containing the variables named in formula.

* triple:
an integer vector of length three containing the degrees of the three polynomial functions for the mean structure, the log innovation -variances and the autoregressive or moving average coefficients when 'mcd' or 'acd' is specified for cov.method. It refers to the degree for the mean structure, variances and angles when 'hpc' is specified for cov.method.

* cov.method:
covariance structure modelling method, choose 'mcd' (Pourahmadi 1999), 'acd' (Chen and Dunson 2013) or 'hpc' (Zhang et al. 2015).

* optim.method:
optimization method, choose 'default' or 'BFGS' (vmmin in R).

* control:
a list (of correct class, resulting from jmcmControl()) containing control parameters, see the jmcmControl documentation for details.

* start:
starting values for the parameters in the model.

Check the JSS paper with the corresponding R replication code [v82i09.R](https://github.com/ypan1988/jmcm-demo/releases/download/v1.0/v82i09.R) (this file is part of [jmcm-demo](https://github.com/ypan1988/jmcm-demo)) for details.

## Benchmarks

For benchmarking of jmcm package, use the R script [jmcm-benchmark.R](https://github.com/ypan1988/jmcm-demo/releases/download/v1.0/jmcm-benchmark.R) (this file is part of [jmcm-demo](https://github.com/ypan1988/jmcm-demo)) with the following command in `R`:
```R
source("~/jmcm-benchmark.R") # you may need to change the directory
```