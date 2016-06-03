// arma_util.h: Utility functions for extension of Armadillo
//
// Copyright (C) 2015-2016 The University of Manchester
//
// Written by Yi Pan - ypan1988@gmail.com

#ifndef JMCM_ARMA_UTIL_H_
#define JMCM_ARMA_UTIL_H_

#include <RcppArmadillo.h>

namespace pan {

// vector to upper triangular matrix with column-major order (Fortran style)
arma::mat VecToUpperTrimatCol(int n, const arma::vec& x, bool diag = false);

// vector to lower triangular matrix with column-major order (Fortran style)
arma::mat VecToLowerTrimatCol(int n, const arma::vec& x, bool diag = false);

arma::mat ltrimat(int n, const arma::vec& x, bool diag = false,
                  bool byrow = true);

// upper triangular matrix to vector with column-major order (Fortran style)
arma::vec UpperTrimatToVecCol(const arma::mat& X, bool diag = false);

// lower triangular matrix to vector with column-major order (Fortran style)
arma::vec LowerTrimatToVecCol(const arma::mat& X, bool diag = false);

arma::vec lvectorise(const arma::mat& X, bool diag = false, bool byrow = true);
}

#endif  // JMCM_ARMA_UTIL_H_
