jmcm: Joint Mean-Covariance Models in R.
====

[![Build Status](https://travis-ci.org/ypan1988/jmcm.svg?branch=master)](https://travis-ci.org/ypan1988/jmcm)
[![cran version](http://www.r-pkg.org/badges/version/jmcm)](https://cran.r-project.org/web/packages/jmcm)
[![downloads](http://cranlogs.r-pkg.org/badges/jmcm)](http://cranlogs.r-pkg.org/badges/jmcm)
[![total downloads](http://cranlogs.r-pkg.org/badges/grand-total/jmcm)](http://cranlogs.r-pkg.org/badges/grand-total/jmcm)

## Features

* Efficient for large data sets, using algorithms from the
[Armadillo](http://arma.sourceforge.net/) linear algebra package via the
[RcppArmadillo](https://cran.r-project.org/web/packages/RcppArmadillo/index.html)
interface layer.
* Fits joint mean-covariance models based on three Cholesky decomposition-type
covariance structure modelling methods, namely modified Cholesky decomposition
(MCD), alternative Cholesky decomposition (ACD) and hyperpherical
parameterization of Cholesky factor (HPC).

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
