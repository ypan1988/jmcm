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

#include <algorithm>

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

  arma::vec get_m() const { return m_; }
  arma::vec get_Y() const { return Y_; }
  arma::mat get_X() const { return X_; }
  arma::mat get_Z() const { return Z_; }
  arma::mat get_W() const { return W_; }

  arma::uword get_m(arma::uword i) const { return m_(i); }
  arma::vec get_Y(arma::uword i) const {
    return Y_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }
  arma::mat get_X(arma::uword i) const {
    return X_.rows(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }
  arma::mat get_Z(arma::uword i) const {
    return Z_.rows(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }
  arma::mat get_W(arma::uword i) const {
    return m_(i) == 1
               ? arma::zeros<arma::mat>(m_(i), n_gma_)
               : arma::mat(W_.rows(cumsum_trim_(i), cumsum_trim_(i + 1) - 1));
  }
  arma::vec Wijk(arma::uword i, arma::uword j, arma::uword k) const {
    return j <= k
               ? arma::zeros<arma::vec>(n_gma_)
               : arma::vec(W_.row(cumsum_trim_(i) + j * (j - 1) / 2 + k).t());
  }

  arma::uword get_method_id() const { return method_id_; }
  void set_free_param(arma::uword n) { free_param_ = n; }

  void set_mean(const arma::vec& mean) {
    cov_only_ = true;
    mean_ = mean;
  }

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

  void UpdateBeta();
  void UpdateLambda(const arma::vec& x) { set_lambda(x); }
  void UpdateLambdaGamma(const arma::vec& x) { set_lmdgma(x); }
  virtual void UpdateGamma() {}

  void UpdateJmcm(const arma::vec& x);
  virtual void UpdateModel() = 0;

  double operator()(const arma::vec& x) override;
  void Gradient(const arma::vec& x, arma::vec& grad) override;

  arma::vec Grad1() const;
  arma::vec Grad23() const { return arma::join_cols(Grad2(), Grad3()); }
  virtual arma::vec Grad2() const = 0;
  virtual arma::vec Grad3() const = 0;

  arma::vec get_mu(arma::uword i) const {
    return Xbta_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }
  arma::vec get_Resid(arma::uword i) const {
    return Resid_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }

  virtual double CalcLogDetSigma() const = 0;
  virtual arma::mat get_Sigma(arma::uword i) const = 0;
  virtual arma::mat get_Sigma_inv(arma::uword i) const = 0;

  // Matrix D refers to the diagonal matrix in MCD/ACD/HPC
  // Matrix T refers to the lower-triangular matrix in MCD/ACD/HPC
  virtual arma::mat get_D(arma::uword i) const = 0;
  virtual arma::mat get_T(arma::uword i) const = 0;

 protected:
  const arma::vec m_, Y_;
  const arma::mat X_, Z_, W_;
  const arma::uword N_, n_sub_, n_bta_, n_lmd_, n_gma_, n_lmdgma_;

  // method_id_ == 0 ---- MCD
  // method_id_ == 1 ---- ACD
  // method_id_ == 2 ---- HPC
  arma::uword method_id_;

  // free_param_ == 0  ---- beta + lambda + gamma
  // free_param_ == 1  ---- beta
  // free_param_ == 2  ---- lambda
  // free_param_ == 3  ---- gamma
  // free_param_ == 23 -----lambda + gamma
  //-----------------------------------------------------------------
  // Note: this variable is used to change the behaviour of the
  // Functor (i.e., MCD/ACD/HPC).  For example: If free_param_ == 1
  // then the first parameter (i.e., beta) is free to change and the
  // value of lambda and gamma will be fixed.  If free_param_ == 23
  // then the second and third parameter (i.e., lambda and gamma) are
  // free to change and the value of beta will be fixed.
  //-----------------------------------------------------------------
  arma::uword free_param_;

  // Do we have a pre-specified mean_? (it can be set by set_mean())
  // If Yes, only covariance matrix will be modelled (cov_only_=true)
  // and the value of mean_ will be fixed with the specified vector.
  // If No (default), both mean and covariace matrix will be modelled.
  // Here, we have cov_only_ = false and mean_ = Y_ (Obviously, the
  // values of mean_ are useless as they are never used in this case).
  bool cov_only_;
  arma::vec mean_;

  arma::vec theta_, beta_, lambda_, gamma_, lmdgma_;
  arma::vec Xbta_, Zlmd_, Wgma_, Resid_;

  // Some useful data members to avoid duplicate index calculation.
  // cumsum_m_ == {0, m(0), m(0)+m(1), m(0)+m(1)+m(2), ...}
  // cumsum_trim_ == {0, m(0)*(m(0)-1)/2, m(0)*(m(0)-1)/2+m(1)*(m(1)-1)/2, ...}
  // cumsum_trim2_ == {0, m(0)*(m(0)+1)/2, m(0)*(m(0)+1)/2+m(1)*(m(1)+1)/2, ...}
  // cumsum_param_ == {0, n_bta_, n_bta_ + n_lmd_, n_bta_ + n_lmd_ + n_gma_}
  arma::vec cumsum_m_;
  arma::vec cumsum_trim_;
  arma::vec cumsum_trim2_;
  arma::uvec cumsum_param_;

 protected:
  // Return a column vector containing the elements that form the
  // lower triangle part (include diagonal elements) of matrix M.
  arma::vec get_lower_part(const arma::mat& M) const {
    return arma::mat(M.t())(arma::trimatu_ind(arma::size(M)));
  }

  // Construct an n x n lower triangular matrix M with vector x.
  // Should the diagonal be included?
  // diag == true ---- YES
  // diag == false ---- NO
  // Note: if diag == false, the diagonal elements of M are 1.
  arma::mat get_ltrimatrix(int n, const arma::vec& x, bool diag) const {
    int k = diag ? 0 : 1;
    arma::mat M = arma::eye<arma::mat>(n, n);
    M(arma::trimatu_ind(arma::size(M), k)) = x;
    return M.t();
  }

 private:
  bool is_same(const arma::vec v1, const arma::vec v2) const {
    return std::equal(v1.cbegin(), v1.cend(), v2.cbegin());
  }
};

inline JmcmBase::JmcmBase(const arma::vec& m, const arma::vec& Y,
                          const arma::mat& X, const arma::mat& Z,
                          const arma::mat& W, const arma::uword method_id)
    : m_(m),
      Y_(Y),
      X_(X),
      Z_(Z),
      W_(W),
      N_(Y_.n_rows),
      n_sub_(m_.n_elem),
      n_bta_(X_.n_cols),
      n_lmd_(Z_.n_cols),
      n_gma_(W_.n_cols),
      n_lmdgma_(n_lmd_ + n_gma_),
      method_id_(method_id),
      free_param_(0),
      cov_only_(false),
      mean_(Y) {
  theta_ = arma::zeros<arma::vec>(n_bta_ + n_lmd_ + n_gma_);
  beta_ = arma::zeros<arma::vec>(n_bta_);
  lambda_ = arma::zeros<arma::vec>(n_lmd_);
  gamma_ = arma::zeros<arma::vec>(n_gma_);
  lmdgma_ = arma::zeros<arma::vec>(n_lmd_ + n_gma_);

  Xbta_ = arma::zeros<arma::vec>(N_);
  Zlmd_ = arma::zeros<arma::vec>(N_);
  Wgma_ = arma::zeros<arma::vec>(W_.n_rows);
  Resid_ = arma::zeros<arma::vec>(N_);

  cumsum_m_ = arma::zeros<arma::vec>(n_sub_ + 1);
  cumsum_m_.tail(n_sub_) = arma::cumsum(m_);

  cumsum_trim_ = arma::zeros<arma::vec>(n_sub_ + 1);
  cumsum_trim_.tail(n_sub_) = arma::cumsum(m_ % (m_ - 1) / 2);

  cumsum_trim2_ = arma::zeros<arma::vec>(n_sub_ + 1);
  cumsum_trim2_.tail(n_sub_) = arma::cumsum(m_ % (m_ + 1) / 2);

  cumsum_param_ = arma::cumsum(arma::uvec({0, n_bta_, n_lmd_, n_gma_}));
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
  arma::mat XSX = arma::zeros<arma::mat>(n_bta_, n_bta_);
  arma::vec XSy = arma::zeros<arma::vec>(n_bta_);
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Xi = get_X(i);
    arma::vec yi = get_Y(i);
    arma::mat Sigmai_inv = get_Sigma_inv(i);

    XSX += Xi.t() * Sigmai_inv * Xi;
    XSy += Xi.t() * (Sigmai_inv * yi);
  }

  set_beta(XSX.i() * XSy);
}

inline void JmcmBase::UpdateJmcm(const arma::vec& x) {
  switch (free_param_) {
    case 0:
      if (!is_same(x, theta_)) {
        theta_ = x;
        beta_ = x.rows(cumsum_param_(0), cumsum_param_(1) - 1);
        lambda_ = x.rows(cumsum_param_(1), cumsum_param_(2) - 1);
        gamma_ = x.rows(cumsum_param_(2), cumsum_param_(3) - 1);

        if (cov_only_)
          Xbta_ = mean_;
        else
          Xbta_ = X_ * beta_;

        Zlmd_ = Z_ * lambda_;
        Wgma_ = W_ * gamma_;
        Resid_ = Y_ - Xbta_;
        UpdateModel();
      }
      break;

    case 1:
      if (!is_same(x, beta_)) {
        theta_.rows(cumsum_param_(0), cumsum_param_(1) - 1) = x;
        beta_ = x;

        if (cov_only_)
          Xbta_ = mean_;
        else
          Xbta_ = X_ * beta_;

        Resid_ = Y_ - Xbta_;
        UpdateModel();
      }
      break;

    case 2:
      if (!is_same(x, lambda_)) {
        theta_.rows(cumsum_param_(1), cumsum_param_(2) - 1) = x;
        lambda_ = x;

        Zlmd_ = Z_ * lambda_;
        UpdateModel();
      }
      break;

    case 3:
      if (!is_same(x, gamma_)) {
        theta_.rows(cumsum_param_(2), cumsum_param_(3) - 1) = x;
        gamma_ = x;

        Wgma_ = W_ * gamma_;
        UpdateModel();
      }
      break;

    case 23:
      if (!is_same(x, lmdgma_)) {
        theta_.rows(cumsum_param_(1), cumsum_param_(3) - 1) = x;
        lambda_ = x.rows(0, n_lmd_ - 1);
        gamma_ = x.rows(n_lmd_, n_lmdgma_ - 1);
        lmdgma_ = x;

        Zlmd_ = Z_ * lambda_;
        Wgma_ = W_ * gamma_;
        UpdateModel();
      }
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline double JmcmBase::operator()(const arma::vec& x) {
  UpdateJmcm(x);

  double result = 0.0;
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Sigmai_inv = get_Sigma_inv(i);
    result += arma::as_scalar(ri.t() * (Sigmai_inv * ri));
  }

  result += CalcLogDetSigma();
  return result;
}

inline void JmcmBase::Gradient(const arma::vec& x, arma::vec& grad) {
  UpdateJmcm(x);

  switch (free_param_) {
    case 0:
      grad = arma::zeros<arma::vec>(theta_.n_rows);
      grad.subvec(cumsum_param_(0), cumsum_param_(1) - 1) = Grad1();
      grad.subvec(cumsum_param_(1), cumsum_param_(2) - 1) = Grad2();
      grad.subvec(cumsum_param_(2), cumsum_param_(3) - 1) = Grad3();
      break;

    case 1:
      grad = Grad1();
      break;

    case 2:
      grad = Grad2();
      break;

    case 3:
      grad = Grad3();
      break;

    case 23:
      grad = Grad23();
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline arma::vec JmcmBase::Grad1() const {
  arma::vec grad1 = arma::zeros<arma::vec>(n_bta_);
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Xi = get_X(i);
    arma::vec ri = get_Resid(i);
    arma::mat Sigmai_inv = get_Sigma_inv(i);
    grad1 += Xi.t() * (Sigmai_inv * ri);
  }

  return (-2 * grad1);
}

}  // namespace jmcm

#endif
