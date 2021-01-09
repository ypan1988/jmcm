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

#include <algorithm>  // std::equal

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

  void UpdateLambda(const arma::vec& x) override;
  void UpdateGamma() override;

  arma::mat get_D(arma::uword i) const override;
  arma::mat get_T(arma::uword i) const override;
  arma::vec get_mu(arma::uword i) const override;
  arma::mat get_Sigma(arma::uword i) const override;
  arma::mat get_Sigma_inv(arma::uword i) const override;
  arma::vec get_Resid(arma::uword i) const override;

  double operator()(const arma::vec& x) override;
  void Gradient(const arma::vec& x, arma::vec& grad) override;
  void Grad1(arma::vec& grad1);
  void Grad2(arma::vec& grad2);
  void Grad3(arma::vec& grad3);

  void UpdateJmcm(const arma::vec& x) override;
  void UpdateParam(const arma::vec& x);
  void UpdateModel();

 private:
  arma::mat G_;
  arma::vec TResid_;

  arma::mat get_G(arma::uword i) const;
  arma::vec get_TResid(arma::uword i) const;
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

inline void MCD::UpdateLambda(const arma::vec& x) { set_lambda(x); }

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
    GDr += Gi.t() * Di_inv * ri;
  }

  arma::vec gamma = GDG.i() * GDr;

  set_gamma(gamma);
}

inline arma::mat MCD::get_D(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return arma::diagmat(arma::exp(Zlmd_.subvec(first_index, last_index)));
}

inline arma::mat MCD::get_T(arma::uword i) const {
  arma::mat Ti = arma::eye(m_(i), m_(i));
  if (m_(i) != 1) {
    arma::uword first_index = cumsum_trim_(i);
    arma::uword last_index = cumsum_trim_(i+1) - 1;

    Ti = pan::ltrimat(m_(i), -Wgma_.subvec(first_index, last_index));
  }
  return Ti;
}

inline arma::vec MCD::get_mu(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return Xbta_.subvec(first_index, last_index);
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

inline arma::vec MCD::get_Resid(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return Resid_.subvec(first_index, last_index);
}

inline double MCD::operator()(const arma::vec& x) {
  UpdateJmcm(x);

  arma::uword i, n_sub = m_.n_elem;
  double result = 0.0;
  for (i = 0; i < n_sub; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Sigmai_inv = get_Sigma_inv(i);
    result += arma::as_scalar(ri.t() * (Sigmai_inv * ri));
  }

  result += arma::sum(arma::log(arma::exp(Zlmd_)));
  return result;
}

inline void MCD::Gradient(const arma::vec& x, arma::vec& grad) {
  UpdateJmcm(x);

  arma::uword n_bta = X_.n_cols, n_lmd = Z_.n_cols, n_gma = W_.n_cols;

  arma::vec grad1, grad2, grad3;

  switch (free_param_) {
    case 0:

      Grad1(grad1);
      Grad2(grad2);
      Grad3(grad3);

      grad = arma::zeros<arma::vec>(theta_.n_rows);
      grad.subvec(0, n_bta - 1) = grad1;
      grad.subvec(n_bta, n_bta + n_lmd - 1) = grad2;
      grad.subvec(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1) = grad3;

      break;

    case 1:
      Grad1(grad);
      break;

    case 2:
      Grad2(grad);
      break;

    case 3:
      Grad3(grad);
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline void MCD::Grad1(arma::vec& grad1) {
  arma::uword i, n_sub = m_.n_elem, n_bta = X_.n_cols;
  grad1 = arma::zeros<arma::vec>(n_bta);

  for (i = 0; i < n_sub; ++i) {
    arma::mat Xi = get_X(i);
    arma::vec ri = get_Resid(i);
    arma::mat Sigmai_inv = get_Sigma_inv(i);
    grad1 += Xi.t() * (Sigmai_inv * ri);
  }

  grad1 *= -2;
}

inline void MCD::Grad2(arma::vec& grad2) {
  arma::uword i, n_sub = m_.n_elem, n_lmd = Z_.n_cols;
  grad2 = arma::zeros<arma::vec>(n_lmd);

  for (i = 0; i < n_sub; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);

    arma::mat Di = get_D(i);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    arma::vec ei = arma::pow(get_TResid(i), 2);

    grad2 += 0.5 * Zi.t() * (Di_inv * ei - one);
  }

  grad2 *= -2;
}

inline void MCD::Grad3(arma::vec& grad3) {
  arma::uword i, n_sub = m_.n_elem, n_gma = W_.n_cols;
  grad3 = arma::zeros<arma::vec>(n_gma);

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi = get_G(i);

    arma::mat Di = get_D(i);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    arma::vec Tiri = get_TResid(i);

    grad3 += Gi.t() * (Di_inv * Tiri);
  }

  grad3 *= -2;
}

inline void MCD::UpdateJmcm(const arma::vec& x) {
  arma::uword debug = 0;
  bool update = true;

  switch (free_param_) {
    case 0:
      if (std::equal(x.cbegin(), x.cend(), theta_.cbegin())) update = false;
      break;

    case 1:
      if (std::equal(x.cbegin(), x.cend(), beta_.cbegin())) update = false;
      break;

    case 2:
      if (std::equal(x.cbegin(), x.cend(), lambda_.cbegin())) update = false;
      break;

    case 3:
      if (std::equal(x.cbegin(), x.cend(), gamma_.cbegin())) update = false;
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

inline void MCD::UpdateParam(const arma::vec& x) {
  arma::uword n_bta = X_.n_cols;
  arma::uword n_lmd = Z_.n_cols;
  arma::uword n_gma = W_.n_cols;

  switch (free_param_) {
    case 0:
      theta_ = x;
      beta_ = x.rows(0, n_bta - 1);
      lambda_ = x.rows(n_bta, n_bta + n_lmd - 1);
      gamma_ = x.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1);
      break;

    case 1:
      theta_.rows(0, n_bta - 1) = x;
      beta_ = x;
      break;

    case 2:
      theta_.rows(n_bta, n_bta + n_lmd - 1) = x;
      lambda_ = x;
      break;

    case 3:
      theta_.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1) = x;
      gamma_ = x;
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline void MCD::UpdateModel() {
  switch (free_param_) {
    case 0:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = X_ * beta_;

      Zlmd_ = Z_ * lambda_;
      Wgma_ = W_ * gamma_;
      Resid_ = Y_ - Xbta_;

      UpdateG();
      UpdateTResid();

      break;

    case 1:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = X_ * beta_;

      Resid_ = Y_ - Xbta_;

      UpdateG();
      UpdateTResid();

      break;

    case 2:
      Zlmd_ = Z_ * lambda_;

      break;

    case 3:
      Wgma_ = W_ * gamma_;

      UpdateTResid();

      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline arma::mat MCD::get_G(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return G_.rows(first_index, last_index);
}

inline arma::vec MCD::get_TResid(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return TResid_.subvec(first_index, last_index);
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
