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

#ifndef _JMCM_BASE_H_
#define _JMCM_BASE_H_

#include "jmcm_config.h"

namespace jmcm {

class JmcmBase : public roptim::Functor {
 public:
  JmcmBase(const arma::uvec& m, const arma::vec& Y, const arma::mat& X,
           const arma::mat& Z, const arma::mat& W, const arma::uword method_id);

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
    if (m_(i) == 1) return arma::zeros<arma::mat>(m_(i), n_gma_);
    return W_.rows(cumsum_trim_(i), cumsum_trim_(i + 1) - 1);
  }
  arma::vec Wijk(arma::uword i, arma::uword j, arma::uword k) const {
    if (j <= k) return arma::zeros<arma::vec>(n_gma_);
    return W_.row(cumsum_trim_(i) + j * (j - 1) / 2 + k).t();
  }

  arma::uword get_free_param() const { return free_param_; }
  void set_free_param(arma::uword n) { free_param_ = n; }

  void set_mean(const arma::vec& mean) { cov_only_ = true, mean_ = mean; }

  // clang-format off
  // A unified function to get parameters. fp is used as a temp value
  // for free_param_ to specify the parameter you want to get.
  arma::vec get_param(int fp) const {
    switch (fp) {
      case  0: return theta_;
      case  1: return theta_.rows(cumsum_param_(0), cumsum_param_(1) - 1); // beta
      case  2: return theta_.rows(cumsum_param_(1), cumsum_param_(2) - 1); // lambda
      case  3: return theta_.rows(cumsum_param_(2), cumsum_param_(3) - 1); // gamma
      case 23: return theta_.rows(cumsum_param_(1), cumsum_param_(3) - 1); // lambda + gamma
      default: arma::get_cerr_stream() << "Wrong fp value" << std::endl;
    }
    return arma::vec();
  }
  // clang-format on

  // A unified function to set parameters. fp is used as a temp value
  // for free_param_ to specify the parameter you want to change.
  void set_param(const arma::vec& x, int fp) {
    arma::uword fp2 = free_param_;
    free_param_ = fp;
    UpdateJmcm(x);
    free_param_ = fp2;
  }

  arma::vec get_mu(arma::uword i) const {
    return Xbta_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }
  arma::vec get_Zlmd(arma::uword i) const {
    return Zlmd_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }
  arma::vec get_Wgma(arma::uword i) const {
    return Wgma_.subvec(cumsum_trim_(i), cumsum_trim_(i + 1) - 1);
  }
  arma::vec get_Resid(arma::uword i) const {
    return Resid_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }

  void UpdateBeta();
  virtual void UpdateGamma() {}

  // Preparation work needed for the Functors to:
  // + Calculate the objective function,
  // + Calculate the gradient.
  void UpdateJmcm(const arma::vec& x);
  // Extra preparation work (MCD/ACD/HPC specific).
  virtual void UpdateModel() = 0;

  // Core functions of Functor
  double operator()(const arma::vec& x) override;
  void Gradient(const arma::vec& x, arma::vec& grad) override;

  arma::vec Grad1() const;
  virtual arma::vec Grad2() const = 0;
  virtual arma::vec Grad3() const = 0;

  virtual double CalcLogDetSigma() const = 0;
  virtual arma::mat get_Sigma(arma::uword i, bool inv = false) const = 0;

  // Matrix D refers to the diagonal matrix in MCD/ACD/HPC
  // Matrix T refers to the lower-triangular matrix in MCD/ACD/HPC
  virtual arma::mat get_D(arma::uword i, bool inv = false) const = 0;
  virtual arma::mat get_T(arma::uword i, bool inv = false) const = 0;

 public:
  const arma::uvec m_;         // number of measurements
  const arma::vec Y_;          // longitudinal measurements
  const arma::mat X_, Z_, W_;  // three model matrices

  //     N_: number of all measurements m(0) + m(1) + m(2) + ... + m(n_sub-1)
  // n_sub_: number of subjects
  // n_bta_: number of elements in parameter beta
  // n_lmd_: number of elements in parameter lambda
  // n_gma_: number of elements in parameter gamma
  const arma::uword N_, n_sub_, n_bta_, n_lmd_, n_gma_;

  // Some useful data members to avoid duplicate index calculation.
  // cumsum_m_     == {0, m(0), m(0)+m(1), m(0)+m(1)+m(2), ...}
  // cumsum_trim_  == {0, m(0)*(m(0)-1)/2, m(0)*(m(0)-1)/2+m(1)*(m(1)-1)/2, ...}
  // cumsum_trim2_ == {0, m(0)*(m(0)+1)/2, m(0)*(m(0)+1)/2+m(1)*(m(1)+1)/2, ...}
  // cumsum_param_ == {0, n_bta_, n_bta_ + n_lmd_, n_bta_ + n_lmd_ + n_gma_}
  const arma::uvec cumsum_m_, cumsum_trim_, cumsum_trim2_, cumsum_param_;

  // method_id_ == 0 ---- MCD
  // method_id_ == 1 ---- ACD
  // method_id_ == 2 ---- HPC
  const arma::uword method_id_;

 protected:
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

  // theta_ == c(beta, lambda, gamma)
  // Xbta_  == X * beta
  // Zlmd_  == Z * lambda
  // Wgma_  == W * gamma
  // Resid_ == Y - X * beta
  arma::vec theta_, Xbta_, Zlmd_, Wgma_, Resid_;

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
};

// clang-format off
inline JmcmBase::JmcmBase(const arma::uvec& m, const arma::vec& Y,
                          const arma::mat& X, const arma::mat& Z,
                          const arma::mat& W, const arma::uword method_id)
    : m_(m), Y_(Y), X_(X), Z_(Z), W_(W), N_(Y_.n_rows), n_sub_(m_.n_elem),
      n_bta_(X_.n_cols), n_lmd_(Z_.n_cols), n_gma_(W_.n_cols),
      cumsum_m_(arma::cumsum(arma::join_cols(arma::uvec({0}), m_))),
      cumsum_trim_(arma::join_cols(arma::uvec({0}), arma::cumsum(m_%(m_-1)/2))),
      cumsum_trim2_(arma::join_cols(arma::uvec({0}), arma::cumsum(m_%(m_+1)/2))),
      cumsum_param_(arma::cumsum(arma::uvec({0, n_bta_, n_lmd_, n_gma_}))),
      method_id_(method_id), free_param_(0), cov_only_(false), mean_(Y),
      theta_(n_bta_ + n_lmd_ + n_gma_, arma::fill::zeros),
      Xbta_(N_, arma::fill::zeros), Zlmd_(N_, arma::fill::zeros),
      Wgma_(W_.n_rows, arma::fill::zeros), Resid_(N_, arma::fill::zeros) {}
// clang-format on

inline void JmcmBase::UpdateBeta() {
  arma::mat XSX = arma::zeros<arma::mat>(n_bta_, n_bta_);
  arma::vec XSy = arma::zeros<arma::vec>(n_bta_);
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Xi = get_X(i);
    arma::vec yi = get_Y(i);
    arma::mat Sigmai_inv = get_Sigma(i, true);

    XSX += Xi.t() * Sigmai_inv * Xi;
    XSy += Xi.t() * (Sigmai_inv * yi);
  }

  set_param(XSX.i() * XSy, 1);
}

inline void JmcmBase::UpdateJmcm(const arma::vec& x) {
  if (arma::all(x - get_param(free_param_) == 0.0)) return;

  switch (free_param_) {
    case 0:
      theta_ = x;
      Xbta_ = cov_only_ ? mean_ : X_ * get_param(1);
      Zlmd_ = Z_ * get_param(2);
      Wgma_ = W_ * get_param(3);
      Resid_ = Y_ - Xbta_;
      break;

    case 1:
      theta_.rows(cumsum_param_(0), cumsum_param_(1) - 1) = x;
      Xbta_ = cov_only_ ? mean_ : X_ * get_param(1);
      Resid_ = Y_ - Xbta_;
      break;

    case 2:
      theta_.rows(cumsum_param_(1), cumsum_param_(2) - 1) = x;
      Zlmd_ = Z_ * get_param(2);
      break;

    case 3:
      theta_.rows(cumsum_param_(2), cumsum_param_(3) - 1) = x;
      Wgma_ = W_ * get_param(3);
      break;

    case 23:
      theta_.rows(cumsum_param_(1), cumsum_param_(3) - 1) = x;
      Zlmd_ = Z_ * get_param(2);
      Wgma_ = W_ * get_param(3);
      break;

    default:
      arma::get_cerr_stream() << "Wrong value for free_param_" << std::endl;
  }
  UpdateModel();  // specific MCD/ACD/HPC preparation work
}

inline double JmcmBase::operator()(const arma::vec& x) {
  UpdateJmcm(x);

  double result = 0.0;
#pragma omp parallel for reduction(+ : result)
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Sigmai_inv = get_Sigma(i, true);
    result += arma::as_scalar(ri.t() * (Sigmai_inv * ri));
  }

  return result + CalcLogDetSigma();
}

// clang-format off
inline void JmcmBase::Gradient(const arma::vec& x, arma::vec& grad) {
  UpdateJmcm(x);

  switch (free_param_) {
    case  0: { grad = arma::join_cols(Grad1(), Grad2(), Grad3()); break; }
    case  1: { grad = Grad1(); break; }
    case  2: { grad = Grad2(); break; }
    case  3: { grad = Grad3(); break; }
    case 23: { grad = arma::join_cols(Grad2(), Grad3()); break; }
    default: arma::get_cerr_stream() << "Wrong value for free_param_" << std::endl;
  }
}

inline arma::vec JmcmBase::Grad1() const {
  arma::vec grad1 = arma::zeros<arma::vec>(n_bta_);
#pragma omp parallel
{
  arma::vec local = arma::zeros<arma::vec>(n_bta_);
#pragma omp for
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Xi = get_X(i);
    arma::vec ri = get_Resid(i);
    arma::mat Sigmai_inv = get_Sigma(i, true);
    local += Xi.t() * (Sigmai_inv * ri);
  }
#pragma omp critical
  grad1 += local;
}

  return (-2 * grad1);
}
// clang-format on

}  // namespace jmcm

#endif
