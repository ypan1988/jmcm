//  jmcm_base.h: base class for three joint mean-covariance models (MCD/ACD/HPC)
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

#ifndef JMCM_SRC_JMCM_BASE_H_
#define JMCM_SRC_JMCM_BASE_H_

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>

#include "roptim.h"

namespace jmcm {

class JmcmBase : public roptim::Functor {
 public:
  JmcmBase() = delete;
  JmcmBase(const JmcmBase&) = delete;
  virtual ~JmcmBase() = default;

  JmcmBase(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
           const arma::mat& Z, const arma::mat& W, const arma::uword method_id);

  arma::uword get_method_id() const { return method_id_; }

  arma::vec get_m() const { return m_; }
  arma::vec get_Y() const { return Y_; }
  arma::mat get_X() const { return X_; }
  arma::mat get_Z() const { return Z_; }
  arma::mat get_W() const { return W_; }

  arma::uword get_m(arma::uword i) const;
  arma::vec get_Y(arma::uword i) const;
  arma::mat get_X(arma::uword i) const;
  arma::mat get_Z(arma::uword i) const;
  arma::mat get_W(arma::uword i) const;
  arma::vec Wijk(arma::uword i, arma::uword j, arma::uword k);

  arma::vec get_theta() const { return theta_; }
  arma::vec get_beta() const { return beta_; }
  arma::vec get_lambda() const { return lambda_; }
  arma::vec get_gamma() const { return gamma_; }
  arma::uword get_free_param() const { return free_param_; }

  void set_theta(const arma::vec& x);
  void set_beta(const arma::vec& x);
  void set_lambda(const arma::vec& x);
  void set_gamma(const arma::vec& x);
  void set_lmdgma(const arma::vec& x);
  void set_free_param(arma::uword n) { free_param_ = n; }

  void UpdateBeta();
  void Grad1(arma::vec& grad1);

  arma::vec get_mu(arma::uword i) const;
  arma::vec get_Resid(arma::uword i) const;

  virtual void UpdateLambda(const arma::vec&) {}
  virtual void UpdateGamma() {}
  virtual void UpdateLambdaGamma(const arma::vec&) {}

  virtual arma::mat get_D(arma::uword i) const = 0;
  virtual arma::mat get_T(arma::uword i) const = 0;
  virtual arma::mat get_Sigma(arma::uword i) const = 0;
  virtual arma::mat get_Sigma_inv(arma::uword i) const = 0;

  // virtual double operator()(const arma::vec& x) = 0;
  virtual void Gradient(const arma::vec& x, arma::vec& grad) = 0;
  virtual void UpdateJmcm(const arma::vec& x) = 0;

  void set_mean(const arma::vec& mean) {
    cov_only_ = true;
    mean_ = mean;
  }

 protected:
  arma::vec m_, Y_;
  arma::mat X_, Z_, W_;
  arma::uword method_id_;

  arma::vec theta_, beta_, lambda_, gamma_, lmdgma_;
  arma::vec Xbta_, Zlmd_, Wgma_, Resid_;

  arma::vec cumsum_m_;
  arma::vec cumsum_trim_;
  arma::vec cumsum_trim2_;

  // free_param_ == 0  ---- beta + lambda + gamma
  // free_param_ == 1  ---- beta
  // free_param_ == 2  ---- lambda
  // free_param_ == 3  ---- gamma
  // free_param_ == 23 -----lambda + gamma
  arma::uword free_param_;

  bool cov_only_;
  arma::vec mean_;
};

inline JmcmBase::JmcmBase(const arma::vec& m, const arma::vec& Y,
                          const arma::mat& X, const arma::mat& Z,
                          const arma::mat& W, const arma::uword method_id)
    : m_(m),
      Y_(Y),
      X_(X),
      Z_(Z),
      W_(W),
      method_id_(method_id),
      free_param_(0),
      cov_only_(false),
      mean_(Y) {
  arma::uword N = Y_.n_rows;
  arma::uword n_sub = m_.size();
  arma::uword n_bta = X_.n_cols;
  arma::uword n_lmd = Z_.n_cols;
  arma::uword n_gma = W_.n_cols;

  theta_ = arma::zeros<arma::vec>(n_bta + n_lmd + n_gma);
  beta_ = arma::zeros<arma::vec>(n_bta);
  lambda_ = arma::zeros<arma::vec>(n_lmd);
  gamma_ = arma::zeros<arma::vec>(n_gma);
  lmdgma_ = arma::zeros<arma::vec>(n_lmd + n_gma);

  Xbta_ = arma::zeros<arma::vec>(N);
  Zlmd_ = arma::zeros<arma::vec>(N);
  Wgma_ = arma::zeros<arma::vec>(W_.n_rows);
  Resid_ = arma::zeros<arma::vec>(N);

  cumsum_m_ = arma::zeros<arma::vec>(n_sub+1);
  cumsum_m_.tail(n_sub) = arma::cumsum(m_);

  cumsum_trim_ = arma::zeros<arma::vec>(n_sub+1);
  cumsum_trim_.tail(n_sub) = arma::cumsum(m_%(m_-1)/2);

  cumsum_trim2_ = arma::zeros<arma::vec>(n_sub+1);
  cumsum_trim2_.tail(n_sub) = arma::cumsum(m_%(m_+1)/2);
}

inline arma::uword JmcmBase::get_m(arma::uword i) const { return m_(i); }

inline arma::vec JmcmBase::get_Y(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return Y_.subvec(first_index, last_index);
}

inline arma::mat JmcmBase::get_X(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return X_.rows(first_index, last_index);
}

inline arma::mat JmcmBase::get_Z(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return Z_.rows(first_index, last_index);
}

inline arma::mat JmcmBase::get_W(arma::uword i) const {
  arma::mat Wi = arma::zeros<arma::mat>(m_(i), W_.n_cols);

  if (m_(i) != 1) {
    arma::uword first_index = cumsum_trim_(i);
    arma::uword last_index = cumsum_trim_(i+1) - 1;

    Wi = W_.rows(first_index, last_index);
  }

  return Wi;
}

inline arma::vec JmcmBase::Wijk(arma::uword i, arma::uword j, arma::uword k) {
  if (j <= k) return arma::zeros<arma::vec>(W_.n_cols);
  return W_.row(cumsum_trim_(i) + j * (j - 1) / 2 + k).t();
}

inline void JmcmBase::set_theta(const arma::vec& x) {
  arma::uword fp2 = free_param_;
  free_param_ = 0;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::set_beta(const arma::vec& x) {
  arma::uword fp2 = free_param_;
  free_param_ = 1;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::set_lambda(const arma::vec& x) {
  arma::uword fp2 = free_param_;
  free_param_ = 2;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::set_gamma(const arma::vec& x) {
  arma::uword fp2 = free_param_;
  free_param_ = 3;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::set_lmdgma(const arma::vec& x) {
  arma::uword fp2 = free_param_;
  free_param_ = 23;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::UpdateBeta() {
  arma::uword i, n_sub = m_.n_elem, n_bta = X_.n_cols;
  arma::mat XSX = arma::zeros<arma::mat>(n_bta, n_bta);
  arma::vec XSY = arma::zeros<arma::vec>(n_bta);

  for (i = 0; i < n_sub; ++i) {
    arma::mat Xi = get_X(i);
    arma::vec Yi = get_Y(i);
    arma::mat Sigmai_inv = get_Sigma_inv(i);

    XSX += Xi.t() * Sigmai_inv * Xi;
    XSY += Xi.t() * Sigmai_inv * Yi;
  }

  arma::vec beta = XSX.i() * XSY;

  arma::uword fp2 = free_param_;
  free_param_ = 1;
  UpdateJmcm(beta);  // template method
  free_param_ = fp2;
}

inline void JmcmBase::Grad1(arma::vec& grad1) {
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

inline arma::vec JmcmBase::get_mu(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return Xbta_.subvec(first_index, last_index);
}

inline arma::vec JmcmBase::get_Resid(arma::uword i) const {
  arma::uword first_index = cumsum_m_(i);
  arma::uword last_index = cumsum_m_(i+1) - 1;

  return Resid_.subvec(first_index, last_index);
}


}  // namespace jmcm

#endif
