// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// arma_util.h: implementation of utility functions for extension of Armadillo
//
// Copyright (C) 2015 Yi Pan and Jianxin Pan
//
// This file is part of jmcm.

#include "arma_util.h"

namespace pan {

arma::mat VecToUpperTrimatCol(int n, const arma::vec& x, bool diag) {
  arma::mat X = arma::eye<arma::mat>(n, n);

  // make empty matrices
  arma::mat RowIdx(n, n, arma::fill::zeros);
  arma::mat ColIdx(n, n, arma::fill::zeros);

  // fill matrices with integers
  arma::vec idx = arma::linspace<arma::vec>(1, n, n);
  RowIdx.each_col() += idx;
  ColIdx.each_row() += trans(idx);

  // assign upper triangular elements
  // the >= allows inclusion of diagonal elements
  if (diag)
    X.elem(arma::find(RowIdx <= ColIdx)) = x;
  else
    X.elem(arma::find(RowIdx < ColIdx)) = x;

  return X;
}

arma::mat VecToLowerTrimatCol(int n, const arma::vec& x, bool diag) {
  arma::mat X = arma::eye<arma::mat>(n, n);

  // make empty matrices
  arma::mat RowIdx(n, n, arma::fill::zeros);
  arma::mat ColIdx(n, n, arma::fill::zeros);

  // fill matrices with integers
  arma::vec idx = arma::linspace<arma::vec>(1, n, n);
  RowIdx.each_col() += idx;
  ColIdx.each_row() += trans(idx);

  // assign upper triangular elements
  // the >= allows inclusion of diagonal elements
  if (diag)
    X.elem(arma::find(RowIdx >= ColIdx)) = x;
  else
    X.elem(arma::find(RowIdx > ColIdx)) = x;

  return X;
}

arma::mat ltrimat(int n, const arma::vec& x, bool diag, bool byrow) {
  arma::mat X;

  if (byrow)
    X = arma::trans(VecToUpperTrimatCol(n, x, diag));
  else
    X = VecToLowerTrimatCol(n, x, diag);

  return X;
}

arma::vec UpperTrimatToVecCol(const arma::mat& X, bool diag) {
  int n = X.n_rows;
  arma::vec x;

  // make empty matrices
  arma::mat RowIdx(n, n, arma::fill::zeros);
  arma::mat ColIdx(n, n, arma::fill::zeros);

  // fill matrices with integers
  arma::vec idx = arma::linspace<arma::vec>(1, n, n);
  RowIdx.each_col() += idx;
  ColIdx.each_row() += trans(idx);

  // assign upper triangular elements
  // the >= allows inclusion of diagonal elements
  if (diag)
    x = X.elem(arma::find(RowIdx <= ColIdx));
  else
    x = X.elem(arma::find(RowIdx < ColIdx));

  return x;
}

arma::vec LowerTrimatToVecCol(const arma::mat& X, bool diag) {
  int n = X.n_rows;
  arma::vec x;

  // make empty matrices
  arma::mat RowIdx(n, n, arma::fill::zeros);
  arma::mat ColIdx(n, n, arma::fill::zeros);

  // fill matrices with integers
  arma::vec idx = arma::linspace<arma::vec>(1, n, n);
  RowIdx.each_col() += idx;
  ColIdx.each_row() += trans(idx);

  // assign upper triangular elements
  // the >= allows inclusion of diagonal elements
  if (diag)
    x = X.elem(arma::find(RowIdx >= ColIdx));
  else
    x = X.elem(arma::find(RowIdx > ColIdx));

  return x;
}

arma::vec lvectorise(const arma::mat& X, bool diag, bool byrow) {
  arma::vec x;

  if (byrow)
    x = UpperTrimatToVecCol(X.t(), diag);
  else
    x = LowerTrimatToVecCol(X, diag);

  return x;
}
}
