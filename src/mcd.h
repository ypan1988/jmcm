// mcd.h: joint mean-covariance models based on modified Cholesky decgomposition
//        (MCD) of the covariance matrix
//
// Copyright (C) 2015-2017 Yi Pan
//
// This file is part of jmcm.

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

  void set_free_param(const int n) { free_param_ = n; }
  void set_theta(const arma::vec& x);
  void set_beta(const arma::vec& x);
  void set_lambda(const arma::vec& x);
  void set_gamma(const arma::vec& x);

  void UpdateBeta();
  void UpdateLambda(const arma::vec& x);
  void UpdateGamma();

  arma::mat get_D(const int i) const override;
  arma::mat get_T(const int i) const override;
  arma::vec get_mu(const int i) const override;
  arma::mat get_Sigma(const int i) const override;
  arma::mat get_Sigma_inv(const int i) const override;
  arma::vec get_Resid(const int i) const;

  void get_D(const int i, arma::mat& Di) const;
  void get_T(const int i, arma::mat& Ti) const;
  void get_Sigma_inv(const int i, arma::mat& Sigmai_inv) const;
  void get_Resid(const int i, arma::vec& ri) const;

  double operator()(const arma::vec& x);
  void Gradient(const arma::vec& x, arma::vec& grad);
  void Grad1(arma::vec& grad1);
  void Grad2(arma::vec& grad2);
  void Grad3(arma::vec& grad3);

  void UpdateJmcm(const arma::vec& x);
  void UpdateParam(const arma::vec& x);
  void UpdateModel();

  void set_mean(const arma::vec& mean) {
    cov_only_ = true;
    mean_ = mean;
  }

 private:
  arma::mat G_;
  arma::vec TResid_;

  int free_param_;
  bool cov_only_;
  arma::vec mean_;

  arma::mat get_G(const int i) const;
  arma::vec get_TResid(const int i) const;
  void get_G(const int i, arma::mat& Gi) const;
  void get_TResid(const int i, arma::vec& Tiri) const;
  void UpdateG();
  void UpdateTResid();

};  // class MCD

inline MCD::MCD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
                const arma::mat& Z, const arma::mat& W)
    : JmcmBase(m, Y, X, Z, W, 0) {
  int debug = 0;

  if (debug) Rcpp::Rcout << "Creating MCD object" << std::endl;
  // m_.print("m = ");

  int N = Y_.n_rows;
  int n_bta = X_.n_cols;
  int n_lmd = Z_.n_cols;
  int n_gma = W_.n_cols;

  G_ = arma::zeros<arma::mat>(N, n_gma);
  TResid_ = arma::zeros<arma::vec>(N);

  free_param_ = 0;

  cov_only_ = false;
  mean_ = Y_;

  if (debug) Rcpp::Rcout << "MCD object created" << std::endl;
}

inline void MCD::set_theta(const arma::vec& x) {
  int fp2 = free_param_;
  free_param_ = 0;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void MCD::set_beta(const arma::vec& x) {
  int fp2 = free_param_;
  free_param_ = 1;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void MCD::set_lambda(const arma::vec& x) {
  int fp2 = free_param_;
  free_param_ = 2;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void MCD::set_gamma(const arma::vec& x) {
  int fp2 = free_param_;
  free_param_ = 3;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void MCD::UpdateBeta() {
  int i, n_sub = m_.n_elem, n_bta = X_.n_cols;
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

  set_beta(beta);
}

inline void MCD::UpdateLambda(const arma::vec& x) { set_lambda(x); }

inline void MCD::UpdateGamma() {
  int i, n_sub = m_.n_elem, n_gma = W_.n_cols;
  arma::mat GDG = arma::zeros<arma::mat>(n_gma, n_gma);
  arma::vec GDr = arma::zeros<arma::vec>(n_gma);

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi;
    get_G(i, Gi);
    arma::vec ri;
    get_Resid(i, ri);
    arma::mat Di;
    get_D(i, Di);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    GDG += Gi.t() * Di_inv * Gi;
    GDr += Gi.t() * Di_inv * ri;
  }

  arma::vec gamma = GDG.i() * GDr;

  set_gamma(gamma);
}

inline arma::mat MCD::get_D(const int i) const {
  arma::mat Di = arma::eye(m_(i), m_(i));
  if (i == 0)
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(0, m_(0) - 1)));
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(index, index + m_(i) - 1)));
  }
  return Di;
}

inline void MCD::get_D(const int i, arma::mat& Di) const {
  Di = arma::eye(m_(i), m_(i));
  if (i == 0)
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(0, m_(0) - 1)));
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(index, index + m_(i) - 1)));
  }
}

inline arma::mat MCD::get_T(const int i) const {
  arma::mat Ti = arma::eye(m_(i), m_(i));
  if (m_(i) != 1) {
    if (i == 0) {
      int first_index = 0;
      int last_index = m_(0) * (m_(0) - 1) / 2 - 1;

      Ti = pan::ltrimat(m_(0), -Wgma_.subvec(first_index, last_index));
    } else {
      int first_index = 0;
      for (int idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

      Ti = pan::ltrimat(m_(i), -Wgma_.subvec(first_index, last_index));
    }
  }
  return Ti;
}

inline void MCD::get_T(const int i, arma::mat& Ti) const {
  Ti = arma::eye(m_(i), m_(i));
  if (m_(i) != 1) {
    if (i == 0) {
      int first_index = 0;
      int last_index = m_(0) * (m_(0) - 1) / 2 - 1;

      Ti = pan::ltrimat(m_(0), -Wgma_.subvec(first_index, last_index));
    } else {
      int first_index = 0;
      for (int idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

      Ti = pan::ltrimat(m_(i), -Wgma_.subvec(first_index, last_index));
    }
  }
}

inline arma::vec MCD::get_mu(const int i) const {
  arma::vec mui;
  if (i == 0)
    mui = Xbta_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    mui = Xbta_.subvec(index, index + m_(i) - 1);
  }
  return mui;
}

inline arma::mat MCD::get_Sigma(const int i) const {
  int debug = 0;

  arma::mat Ti = get_T(i);
  arma::mat Ti_inv = arma::pinv(Ti);
  arma::mat Di = get_D(i);

  if (debug) {
    Ti.print("Ti = ");
    Di.print("Di = ");
  }
  return Ti_inv * Di * Ti_inv.t();
}

inline arma::mat MCD::get_Sigma_inv(const int i) const {
  int debug = 0;

  arma::mat Ti = get_T(i);
  arma::mat Di = get_D(i);
  arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

  if (debug) {
    Ti.print("Ti = ");
    Di.print("Di = ");
  }
  return Ti.t() * Di_inv * Ti;
}

inline void MCD::get_Sigma_inv(const int i, arma::mat& Sigmai_inv) const {
  int debug = 0;

  arma::mat Ti;
  get_T(i, Ti);
  arma::mat Di;
  get_D(i, Di);
  arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

  if (debug) {
    Ti.print("Ti = ");
    Di.print("Di = ");
  }
  Sigmai_inv = Ti.t() * Di_inv * Ti;
}

inline arma::vec MCD::get_Resid(const int i) const {
  arma::vec ri;
  if (i == 0)
    ri = Resid_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    ri = Resid_.subvec(index, index + m_(i) - 1);
  }
  return ri;
}

inline void MCD::get_Resid(const int i, arma::vec& ri) const {
  if (i == 0)
    ri = Resid_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    ri = Resid_.subvec(index, index + m_(i) - 1);
  }
}

inline double MCD::operator()(const arma::vec& x) {
  int debug = 0;

  if (debug) Rcpp::Rcout << "UpdateJmcm(x)" << std::endl;
  UpdateJmcm(x);

  int i, n_sub = m_.n_elem;
  double result = 0.0;

  // arma::wall_clock timer;
  // timer.tic();

  if (debug) Rcpp::Rcout << "before for loop" << std::endl;
  //	#pragma omp parallel for reduction(+:result)
  for (i = 0; i < n_sub; ++i) {
    // arma::vec ri = get_Resid(i);
    arma::vec ri;
    get_Resid(i, ri);
    // arma::mat Sigmai_inv = get_Sigma_inv(i);
    arma::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    result += arma::as_scalar(ri.t() * Sigmai_inv * ri);
  }

  if (debug) Rcpp::Rcout << "after for loop" << std::endl;

  result += arma::sum(arma::log(arma::exp(Zlmd_)));

  // double n = timer.toc();
  // Rcpp::Rcout << "number of seconds: " << n << std::endl;

  return result;
}

inline void MCD::Gradient(const arma::vec& x, arma::vec& grad) {
  UpdateJmcm(x);

  int n_bta = X_.n_cols, n_lmd = Z_.n_cols, n_gma = W_.n_cols;

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
  int debug = 0;

  int i, n_sub = m_.n_elem, n_bta = X_.n_cols;
  grad1 = arma::zeros<arma::vec>(n_bta);

  if (debug) Rcpp::Rcout << "Update grad1" << std::endl;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Xi = get_X(i);
    arma::vec ri;
    get_Resid(i, ri);
    arma::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    grad1 += Xi.t() * Sigmai_inv * ri;
  }

  grad1 *= -2;
}

inline void MCD::Grad2(arma::vec& grad2) {
  int debug = 0;

  int i, n_sub = m_.n_elem, n_lmd = Z_.n_cols;
  grad2 = arma::zeros<arma::vec>(n_lmd);

  if (debug) Rcpp::Rcout << "Update grad2" << std::endl;

  for (i = 0; i < n_sub; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);
    //	    arma::mat Gi = get_G(i);

    arma::mat Di;
    get_D(i, Di);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    //	    arma::vec ri = get_Resid(i);
    //      arma::vec ei = arma::pow(ri - Gi * gamma_, 2);
    arma::vec ei = arma::pow(get_TResid(i), 2);

    grad2 += 0.5 * Zi.t() * (Di_inv * ei - one);
  }

  grad2 *= -2;
}

inline void MCD::Grad3(arma::vec& grad3) {
  int debug = 0;

  int i, n_sub = m_.n_elem, n_gma = W_.n_cols;
  grad3 = arma::zeros<arma::vec>(n_gma);

  if (debug) Rcpp::Rcout << "Update grad3" << std::endl;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi;
    get_G(i, Gi);

    arma::mat Di;
    get_D(i, Di);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    // arma::vec ri = get_Resid(i);

    // grad3 += Gi.t() * Di_inv * (ri - Gi * gamma_);

    arma::vec Tiri;
    get_TResid(i, Tiri);

    grad3 += Gi.t() * Di_inv * Tiri;
  }

  grad3 *= -2;
}

inline void MCD::UpdateJmcm(const arma::vec& x) {
  int debug = 0;
  bool update = true;

  switch (free_param_) {
    case 0:
      if (arma::min(x == theta_) == 1) update = false;

      break;

    case 1:
      if (arma::min(x == beta_) == 1) update = false;

      break;

    case 2:
      if (arma::min(x == lambda_) == 1) update = false;

      break;

    case 3:
      if (arma::min(x == gamma_) == 1) update = false;

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
  int n_bta = X_.n_cols;
  int n_lmd = Z_.n_cols;
  int n_gma = W_.n_cols;

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
  int debug = 0;

  if (debug) Rcpp::Rcout << "update Xbta Zlmd Wgam r" << std::endl;

  switch (free_param_) {
    case 0:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = X_ * beta_;
      Zlmd_ = Z_ * lambda_;
      Wgma_ = W_ * gamma_;
      Resid_ = Y_ - Xbta_;

      if (debug) Rcpp::Rcout << "UpdateG(x)" << std::endl;
      UpdateG();
      if (debug) Rcpp::Rcout << "UpdateTResid(x)" << std::endl;
      UpdateTResid();
      if (debug) Rcpp::Rcout << "Update Finished.." << std::endl;

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

inline arma::mat MCD::get_G(const int i) const {
  arma::mat Gi;
  if (i == 0)
    Gi = G_.rows(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Gi = G_.rows(index, index + m_(i) - 1);
  }
  return Gi;
}

inline void MCD::get_G(const int i, arma::mat& Gi) const {
  if (i == 0)
    Gi = G_.rows(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Gi = G_.rows(index, index + m_(i) - 1);
  }
}

inline arma::vec MCD::get_TResid(const int i) const {
  arma::vec Tiri;
  if (i == 0)
    Tiri = TResid_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Tiri = TResid_.subvec(index, index + m_(i) - 1);
  }
  return Tiri;
}

inline void MCD::get_TResid(const int i, arma::vec& Tiri) const {
  if (i == 0)
    Tiri = TResid_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Tiri = TResid_.subvec(index, index + m_(i) - 1);
  }
}

inline void MCD::UpdateG() {
  int i, j, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi = arma::zeros<arma::mat>(m_(i), W_.n_cols);

    arma::mat Wi = get_W(i);
    arma::vec ri;
    get_Resid(i, ri);
    for (j = 1; j != m_(i); ++j) {
      int index = 0;
      if (j == 1)
        index = 0;
      else {
        for (int idx = 1; idx < j; ++idx) index += idx;
      }
      Gi.row(j) = ri.subvec(0, j - 1).t() * Wi.rows(index, index + j - 1);
    }
    if (i == 0)
      G_.rows(0, m_(0) - 1) = Gi;
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      G_.rows(index, index + m_(i) - 1) = Gi;
    }
  }
}

inline void MCD::UpdateTResid() {
  int i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::vec ri;
    get_Resid(i, ri);
    arma::mat Ti;
    get_T(i, Ti);
    arma::mat Tiri = Ti * ri;
    if (i == 0)
      TResid_.subvec(0, m_(0) - 1) = Tiri;
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      TResid_.subvec(index, index + m_(i) - 1) = Tiri;
    }
  }
}

}  // namespace jmcm

#endif  // JMCM_SRC_MCD_H_
