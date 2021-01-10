//  acd.h: joint mean-covariance models based on alternative Cholesky
//         decomposition (ACD) of the covariance matrix
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

#ifndef JMCM_SRC_ACD_H_
#define JMCM_SRC_ACD_H_

#include <algorithm>  // std::equal

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>

#include "arma_util.h"
#include "jmcm_base.h"

namespace jmcm {

class ACD : public JmcmBase {
 public:
  ACD() = delete;
  ACD(const ACD&) = delete;
  ~ACD() = default;

  ACD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
      const arma::mat& Z, const arma::mat& W);

  void UpdateLambdaGamma(const arma::vec& x) override { set_lmdgma(x); }

  arma::mat get_D(arma::uword i) const override {
    return arma::diagmat(arma::exp(Zlmd_.subvec(cumsum_m_(i), cumsum_m_(i+1) - 1)/2));
  }
  arma::mat get_T(arma::uword i) const override {
    return m_(i) == 1 ? arma::eye(m_(i), m_(i)) :
           pan::ltrimat(m_(i), Wgma_.subvec(cumsum_trim_(i), cumsum_trim_(i+1) - 1), false);
  }
  arma::mat get_invT(arma::uword i) const {
    return m_(i) == 1 ? arma::eye(m_(i), m_(i)) :
           pan::ltrimat(m_(i), invTelem_.subvec(cumsum_trim2_(i), cumsum_trim2_(i+1) - 1), true);
  }

  arma::mat get_Sigma(arma::uword i) const override;
  arma::mat get_Sigma_inv(arma::uword i) const override;

  double operator()(const arma::vec& x) override;
  void Gradient(const arma::vec& x, arma::vec& grad) override;
  void Grad2(arma::vec& grad2);

  void UpdateJmcm(const arma::vec& x) override;
  void UpdateParam(const arma::vec& x);
  void UpdateModel();

 private:
  arma::vec invTelem_;
  arma::vec TDResid_;
  arma::vec TDResid2_;

  arma::vec get_TDResid(arma::uword i) const {
    return TDResid_.subvec(cumsum_m_(i), cumsum_m_(i+1) - 1);
  }
  arma::vec get_TDResid2(arma::uword i) const {
    return TDResid2_.subvec(cumsum_m_(i), cumsum_m_(i+1) - 1);
  }

  void UpdateTelem();
  void UpdateTDResid();

  arma::vec CalcTijkDeriv(arma::uword i, arma::uword j, arma::uword k) { return Wijk(i, j, k); }
  arma::mat CalcTransTiDeriv(arma::uword i);
};  // class ACD

inline ACD::ACD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
                const arma::mat& Z, const arma::mat& W)
    : JmcmBase(m, Y, X, Z, W, 1) {
  arma::uword N = Y_.n_rows;

  invTelem_ = arma::zeros<arma::vec>(W_.n_rows + N);
  TDResid_ = arma::zeros<arma::vec>(N);
  TDResid2_ = arma::zeros<arma::vec>(N);
}

inline arma::mat ACD::get_Sigma(arma::uword i) const {
  arma::mat DiTi = get_D(i) * get_T(i);

  return DiTi * DiTi.t();
}

inline arma::mat ACD::get_Sigma_inv(arma::uword i) const {
  arma::mat Di = get_D(i);
  arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));
  arma::mat Ti_inv = get_invT(i);
  arma::mat Ti_inv_Di_inv = Ti_inv * Di_inv;

  return Ti_inv_Di_inv.t() * Ti_inv_Di_inv;
}

inline double ACD::operator()(const arma::vec& x) {
  UpdateJmcm(x);

  arma::uword i, n_sub = m_.n_elem;
  double result = 0.0;

  for (i = 0; i < n_sub; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Sigmai_inv = get_Sigma_inv(i);
    result += arma::as_scalar(ri.t() * (Sigmai_inv * ri));
  }

  result += 2 * arma::sum(arma::log(arma::exp(Zlmd_ / 2)));

  return result;
}

inline void ACD::Gradient(const arma::vec& x, arma::vec& grad) {
  UpdateJmcm(x);

  arma::uword n_bta = X_.n_cols, n_lmd = Z_.n_cols, n_gma = W_.n_cols;

  arma::vec grad1, grad2, grad3;

  switch (free_param_) {
    case 0:

      Grad1(grad1);
      Grad2(grad2);

      grad = arma::zeros<arma::vec>(theta_.n_rows);
      grad.subvec(0, n_bta - 1) = grad1;
      grad.subvec(n_bta, n_bta + n_lmd + n_gma - 1) = grad2;

      break;

    case 1:
      Grad1(grad);
      break;

    case 23:
      Grad2(grad);
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline void ACD::Grad2(arma::vec& grad2) {
  arma::uword i, n_sub = m_.n_elem, n_lmd = Z_.n_cols, n_gma = W_.n_cols;
  grad2 = arma::zeros<arma::vec>(n_lmd + n_gma);
  arma::vec grad2_lmd = arma::zeros<arma::vec>(n_lmd);
  arma::vec grad2_gma = arma::zeros<arma::vec>(n_gma);

  for (i = 0; i < n_sub; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);
    arma::vec hi = get_TDResid2(i);

    grad2_lmd += 0.5 * Zi.t() * (hi - one);

    arma::mat Ti = get_T(i);
    arma::mat Ti_inv = get_invT(i);

    arma::vec ei = get_TDResid(i);

    arma::mat Ti_trans_deriv = CalcTransTiDeriv(i);

    grad2_gma += arma::kron(ei.t(), arma::eye(n_gma, n_gma)) * Ti_trans_deriv *
                 Ti_inv.t() * ei;
  }
  grad2.subvec(0, n_lmd - 1) = grad2_lmd;
  grad2.subvec(n_lmd, n_lmd + n_gma - 1) = grad2_gma;

  grad2 *= -2;
}

inline void ACD::UpdateJmcm(const arma::vec& x) {
  arma::uword debug = 0;
  bool update = true;

  switch (free_param_) {
    case 0:
      if (std::equal(x.cbegin(), x.cend(), theta_.cbegin())) update = false;
      break;

    case 1:
      if (std::equal(x.cbegin(), x.cend(), beta_.cbegin())) update = false;
      break;

    case 23:
      if (std::equal(x.cbegin(), x.cend(), lmdgma_.cbegin())) update = false;
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }

  if (update) {
    UpdateParam(x);
    UpdateModel();
  } else {
    if (debug) Rcpp::Rcout << "Hey, I did save some time!:)" << std::endl;
  }
}

inline void ACD::UpdateParam(const arma::vec& x) {
  arma::uword n_bta = X_.n_cols;
  arma::uword n_lmd = Z_.n_cols;
  arma::uword n_gma = W_.n_cols;

  switch (free_param_) {
    case 0:
      theta_ = x;
      beta_ = x.rows(0, n_bta - 1);
      lambda_ = x.rows(n_bta, n_bta + n_lmd - 1);
      gamma_ = x.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1);
      lmdgma_ = x.rows(n_bta, n_bta + n_lmd + n_gma - 1);
      break;

    case 1:
      theta_.rows(0, n_bta - 1) = x;
      beta_ = x;
      break;

    case 23:
      theta_.rows(n_bta, n_bta + n_lmd + n_gma - 1) = x;
      lambda_ = x.rows(0, n_lmd - 1);
      gamma_ = x.rows(n_lmd, n_lmd + n_gma - 1);
      lmdgma_ = x;
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline void ACD::UpdateModel() {
  switch (free_param_) {
    case 0:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = X_ * beta_;

      Zlmd_ = Z_ * lambda_;
      Wgma_ = W_ * gamma_;
      Resid_ = Y_ - Xbta_;

      UpdateTelem();
      UpdateTDResid();

      break;

    case 1:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = X_ * beta_;
      Resid_ = Y_ - Xbta_;

      UpdateTDResid();

      break;

    case 23:
      Zlmd_ = Z_ * lambda_;
      Wgma_ = W_ * gamma_;

      UpdateTelem();
      UpdateTDResid();

      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline void ACD::UpdateTelem() {
  arma::uword i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Ti = get_T(i);
    arma::mat Ti_inv;
    if (!arma::inv(Ti_inv, Ti)) Ti_inv = arma::pinv(Ti);

    arma::uword first_index = cumsum_trim2_(i);
    arma::uword last_index = cumsum_trim2_(i+1)  - 1;
    invTelem_.subvec(first_index, last_index) = pan::lvectorise(Ti_inv, true);
  }
}

inline void ACD::UpdateTDResid() {
  arma::uword i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::vec ri = get_Resid(i);

    arma::mat Ti = get_T(i);
    arma::mat Ti_inv = get_invT(i);

    arma::mat Di = get_D(i);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    arma::vec TiDiri = Ti_inv * Di_inv * ri;
    arma::vec TiDiri2 = arma::diagvec(Ti_inv.t() * Ti_inv * Di_inv * ri *
                                      ri.t() * Di_inv);  // hi

    arma::uword first_index = cumsum_m_(i);
    arma::uword last_index = cumsum_m_(i+1) - 1;

    TDResid_.subvec(first_index, last_index) = TiDiri;
    TDResid2_.subvec(first_index, last_index) = TiDiri2;
  }
}

inline arma::mat ACD::CalcTransTiDeriv(arma::uword i) {
  arma::uword n_gma = W_.n_cols;

  arma::mat result = arma::zeros<arma::mat>(n_gma * m_(i), m_(i));
  for (arma::uword k = 1; k != m_(i); ++k) {
    for (arma::uword j = 0; j <= k; ++j) {
      result.submat(j * n_gma, k, (j * n_gma + n_gma - 1), k) =
          CalcTijkDeriv(i, k, j);
    }
  }

  return result;
}

}  // namespace jmcm

#endif  // JMCM_ACD_H_
