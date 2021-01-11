//  mcd.h: joint mean-covariance models based on modified Cholesky
//         decomposition (MCD) of the covariance matrix
//  This file is part of jmcm.
//
//  Copyright (C) 2015-2021 Yi Pan <ypan1988@gmail.com>
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  A copy of the GNU General Public License is available at
//  https://www.R-project.org/Licenses/

#ifndef JMCM_SRC_MCD_H_
#define JMCM_SRC_MCD_H_

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>

#include "arma_util.h"
#include "jmcm_base.h"

namespace jmcm {

class MCD : public JmcmBase {
 public:
  MCD() = delete;
  MCD(const MCD&) = delete;
  ~MCD() = default;

  MCD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
      const arma::mat& Z, const arma::mat& W);

  void UpdateLambda(const arma::vec& x) override { set_lambda(x); }
  void UpdateGamma() override;

  arma::mat get_D(arma::uword i) const override {
    return arma::diagmat(arma::exp(Zlmd_.subvec(cumsum_m_(i), cumsum_m_(i+1) - 1)));
  }
  arma::mat get_T(arma::uword i) const override {
    return m_(i) == 1 ? arma::eye(m_(i), m_(i)) :
           pan::ltrimat(m_(i), -Wgma_.subvec(cumsum_trim_(i), cumsum_trim_(i+1) - 1));
  }
  arma::mat get_Sigma(arma::uword i) const override;
  arma::mat get_Sigma_inv(arma::uword i) const override;

  arma::vec Grad2() const override;
  arma::vec Grad3() const override;

  void UpdateModel() override;

  double CalcLogDetSigma() const override { return arma::sum(Zlmd_); }

 private:
  arma::mat G_;
  arma::vec TResid_;

  arma::mat get_G(arma::uword i) const {
    return G_.rows(cumsum_m_(i), cumsum_m_(i+1) - 1);
  }
  arma::vec get_TResid(arma::uword i) const {
    return TResid_.subvec(cumsum_m_(i), cumsum_m_(i+1) - 1);
  }

  void UpdateG();
  void UpdateTResid();

};  // class MCD

inline MCD::MCD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
                const arma::mat& Z, const arma::mat& W)
    : JmcmBase(m, Y, X, Z, W, 0) {
  arma::uword N = Y_.n_rows;
  arma::uword n_gma = W_.n_cols;

  G_ = arma::zeros<arma::mat>(N, n_gma);
  TResid_ = arma::zeros<arma::vec>(N);
}

inline void MCD::UpdateGamma() {
  arma::uword i, n_sub = m_.n_elem, n_gma = W_.n_cols;
  arma::mat GDG = arma::zeros<arma::mat>(n_gma, n_gma);
  arma::vec GDr = arma::zeros<arma::vec>(n_gma);

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi = get_G(i);
    arma::vec ri = get_Resid(i);
    arma::mat Di = get_D(i);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    GDG += Gi.t() * Di_inv * Gi;
    GDr += Gi.t() * (Di_inv * ri);
  }

  arma::vec gamma = GDG.i() * GDr;

  set_gamma(gamma);
}

inline arma::mat MCD::get_Sigma(arma::uword i) const {
  arma::mat Ti = get_T(i);
  arma::mat Ti_inv = arma::pinv(Ti);
  arma::mat Di = get_D(i);

  return Ti_inv * Di * Ti_inv.t();
}

inline arma::mat MCD::get_Sigma_inv(arma::uword i) const {
  arma::mat Ti = get_T(i);
  arma::mat Di = get_D(i);
  arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

  return Ti.t() * Di_inv * Ti;
}

inline arma::vec MCD::Grad2() const {
  arma::uword i, n_sub = m_.n_elem, n_lmd = Z_.n_cols;
  arma::vec grad2 = arma::zeros<arma::vec>(n_lmd);

  for (i = 0; i < n_sub; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);

    arma::mat Di = get_D(i);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    arma::vec ei = arma::pow(get_TResid(i), 2);

    grad2 += 0.5 * Zi.t() * (Di_inv * ei - one);
  }

  return (-2 * grad2);
}

inline arma::vec MCD::Grad3() const {
  arma::uword i, n_sub = m_.n_elem, n_gma = W_.n_cols;
  arma::vec grad3 = arma::zeros<arma::vec>(n_gma);

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi = get_G(i);

    arma::mat Di = get_D(i);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    arma::vec Tiri = get_TResid(i);

    grad3 += Gi.t() * (Di_inv * Tiri);
  }

  return (-2 * grad3);
}

inline void MCD::UpdateModel() {
  switch (free_param_) {
    case 0:
      UpdateG();
      UpdateTResid();
      break;

    case 1:
      UpdateG();
      UpdateTResid();
      break;

    case 2: break;

    case 3:
      UpdateTResid();
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline void MCD::UpdateG() {
  arma::uword i, j, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi = arma::zeros<arma::mat>(m_(i), W_.n_cols);

    arma::mat Wi = get_W(i);
    arma::vec ri = get_Resid(i);
    for (j = 1; j != m_(i); ++j) {
      arma::uword index = 0;
      if (j != 1) index = (j-1) * j /2;
      Gi.row(j) = ri.subvec(0, j - 1).t() * Wi.rows(index, index + j - 1);
    }

    arma::uword first_index = cumsum_m_(i);
    arma::uword last_index = cumsum_m_(i+1) - 1;

    G_.rows(first_index, last_index) = Gi;
  }
}

inline void MCD::UpdateTResid() {
  arma::uword i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Tiri = get_T(i) * get_Resid(i);

    arma::uword first_index = cumsum_m_(i);
    arma::uword last_index = cumsum_m_(i+1) - 1;

    TResid_.subvec(first_index, last_index) = Tiri;
  }
}

}  // namespace jmcm

#endif  // JMCM_SRC_MCD_H_
