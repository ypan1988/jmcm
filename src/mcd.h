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
  void UpdateModel() override;

  arma::vec Grad2() const override;
  arma::vec Grad3() const override;

  double CalcLogDetSigma() const override { return arma::sum(Zlmd_); }
  arma::mat get_Sigma(arma::uword i) const override {
    arma::mat Ti_inv = get_invT(i), Di = get_D(i);
    return Ti_inv * Di * Ti_inv.t();
  }
  arma::mat get_Sigma_inv(arma::uword i) const override {
    arma::mat Ti = get_T(i), Di_inv = get_invD(i);
    return Ti.t() * Di_inv * Ti;
  }

  arma::mat get_D(arma::uword i) const override {
    return arma::diagmat(
        arma::exp(Zlmd_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1)));
  }
  arma::mat get_invD(arma::uword i) const {
    return arma::diagmat(
        arma::exp(-Zlmd_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1)));
  }
  arma::mat get_T(arma::uword i) const override {
    return m_(i) == 1 ? arma::eye(m_(i), m_(i))
                      : get_ltrimatrix(m_(i),
                                       -Wgma_.subvec(cumsum_trim_(i),
                                                     cumsum_trim_(i + 1) - 1),
                                       false);
  }
  arma::mat get_invT(arma::uword i) const { return arma::pinv(get_T(i)); }

 private:
  arma::mat G_;
  arma::vec TResid_;

  arma::mat get_G(arma::uword i) const {
    return G_.rows(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }
  arma::vec get_TResid(arma::uword i) const {
    return TResid_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }

  void UpdateG();
  void UpdateTResid();

};  // class MCD

inline MCD::MCD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
                const arma::mat& Z, const arma::mat& W)
    : JmcmBase(m, Y, X, Z, W, 0) {
  G_ = arma::zeros<arma::mat>(N_, n_gma_);
  TResid_ = arma::zeros<arma::vec>(N_);
}

inline void MCD::UpdateGamma() {
  arma::mat GDG = arma::zeros<arma::mat>(n_gma_, n_gma_);
  arma::vec GDr = arma::zeros<arma::vec>(n_gma_);

  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Gi = get_G(i);
    arma::vec ri = get_Resid(i);
    arma::mat Di_inv = get_invD(i);

    GDG += Gi.t() * Di_inv * Gi;
    GDr += Gi.t() * (Di_inv * ri);
  }

  set_gamma(GDG.i() * GDr);
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

    case 2:
      break;

    case 3:
      UpdateTResid();
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline arma::vec MCD::Grad2() const {
  arma::vec grad2 = arma::zeros<arma::vec>(n_lmd_);
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);
    arma::mat Di_inv = get_invD(i);
    arma::vec ei = arma::pow(get_TResid(i), 2);

    grad2 += Zi.t() * (Di_inv * ei - one);
  }

  return (-grad2);  // 2 is cancelled with the 0.5 in the for loop
}

inline arma::vec MCD::Grad3() const {
  arma::vec grad3 = arma::zeros<arma::vec>(n_gma_);
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Gi = get_G(i);
    arma::mat Di_inv = get_invD(i);
    arma::vec Tiri = get_TResid(i);

    grad3 += Gi.t() * (Di_inv * Tiri);
  }

  return (-2 * grad3);
}

inline void MCD::UpdateG() {
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Gi = arma::zeros<arma::mat>(m_(i), n_gma_);

    arma::mat Wi = get_W(i);
    arma::vec ri = get_Resid(i);
    for (arma::uword j = 1; j != m_(i); ++j) {
      arma::uword index = 0;
      if (j != 1) index = (j - 1) * j / 2;
      Gi.row(j) = ri.subvec(0, j - 1).t() * Wi.rows(index, index + j - 1);
    }

    arma::uword first_index = cumsum_m_(i);
    arma::uword last_index = cumsum_m_(i + 1) - 1;

    G_.rows(first_index, last_index) = Gi;
  }
}

inline void MCD::UpdateTResid() {
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Tiri = get_T(i) * get_Resid(i);

    arma::uword first_index = cumsum_m_(i);
    arma::uword last_index = cumsum_m_(i + 1) - 1;

    TResid_.subvec(first_index, last_index) = Tiri;
  }
}

}  // namespace jmcm

#endif  // JMCM_SRC_MCD_H_
