// acd.h: joint mean-covariance models based on alternative Cholesky
//        decomposition (ACD) of the covariance matrix
//
// Copyright (C) 2015-2017 Yi Pan
//
// This file is part of jmcm.

#ifndef JMCM_SRC_ACD_H_
#define JMCM_SRC_ACD_H_

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

  void set_free_param(arma::uword n) { free_param_ = n; }
  void set_theta(const arma::vec& x);
  void set_beta(const arma::vec& x);
  void set_lmdgma(const arma::vec& x);

  void UpdateBeta();
  void UpdateLambdaGamma(const arma::vec& x);

  arma::mat get_D(arma::uword i) const override;
  arma::mat get_T(arma::uword i) const override;
  arma::vec get_mu(arma::uword i) const override;
  arma::mat get_Sigma(arma::uword i) const override;
  arma::mat get_Sigma_inv(arma::uword i) const override;
  arma::vec get_Resid(arma::uword i) const override;

  void get_D(arma::uword i, arma::mat& Di) const;
  void get_T(arma::uword i, arma::mat& Ti) const;
  void get_invT(arma::uword i, arma::mat& Ti_inv) const;
  void get_Sigma_inv(arma::uword i, arma::mat& Sigmai_inv);

  double operator()(const arma::vec& x);
  void Gradient(const arma::vec& x, arma::vec& grad);
  void Grad1(arma::vec& grad1);
  void Grad2(arma::vec& grad2);

  void UpdateJmcm(const arma::vec& x);
  void UpdateParam(const arma::vec& x);
  void UpdateModel();

  void set_mean(const arma::vec& mean) {
    cov_only_ = true;
    mean_ = mean;
  }

 private:
  arma::vec lmdgma_;
  arma::vec invTelem_;
  arma::vec TDResid_;
  arma::vec TDResid2_;

  int free_param_;
  bool cov_only_;
  arma::vec mean_;

  arma::vec get_TDResid(arma::uword i) const;
  arma::vec get_TDResid2(arma::uword i) const;
  void get_TDResid(arma::uword i, arma::vec& TiDiri) const;
  void get_TDResid2(arma::uword i, arma::vec& TiDiri2) const;

  void UpdateTelem();
  void UpdateTDResid();

  arma::vec Wijk(arma::uword i, arma::uword j, arma::uword k);
  arma::vec CalcTijkDeriv(arma::uword i, arma::uword j, arma::uword k);
  arma::mat CalcTransTiDeriv(arma::uword i);
};  // class ACD

inline ACD::ACD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
                const arma::mat& Z, const arma::mat& W)
    : JmcmBase(m, Y, X, Z, W, 1) {
  int debug = 0;

  if (debug) Rcpp::Rcout << "Creating ACD object" << std::endl;

  int N = Y_.n_rows;
  int n_bta = X_.n_cols;
  int n_lmd = Z_.n_cols;
  int n_gma = W_.n_cols;

  lmdgma_ = arma::zeros<arma::vec>(n_lmd + n_gma);
  invTelem_ = arma::zeros<arma::vec>(W_.n_rows + arma::sum(m_));
  TDResid_ = arma::zeros<arma::vec>(N);
  TDResid2_ = arma::zeros<arma::vec>(N);

  free_param_ = 0;

  cov_only_ = false;
  mean_ = Y_;

  if (debug) Rcpp::Rcout << "ACD object created" << std::endl;
}

inline void ACD::set_theta(const arma::vec& x) {
  int fp2 = free_param_;
  free_param_ = 0;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void ACD::set_beta(const arma::vec& x) {
  int fp2 = free_param_;
  free_param_ = 1;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void ACD::set_lmdgma(const arma::vec& x) {
  int fp2 = free_param_;
  free_param_ = 2;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void ACD::UpdateBeta() {
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

inline void ACD::UpdateLambdaGamma(const arma::vec& x) { set_lmdgma(x); }

inline arma::mat ACD::get_D(arma::uword i) const {
  arma::mat Di = arma::eye(m_(i), m_(i));
  if (i == 0)
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(0, m_(0) - 1) / 2));
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(index, index + m_(i) - 1) / 2));
  }
  return Di;
}

inline arma::mat ACD::get_T(arma::uword i) const {
  arma::mat Ti = arma::ones<arma::mat>(m_(i), m_(i));
  if (m_(i) != 1) {
    if (i == 0) {
      int first_index = 0;
      int last_index = m_(0) * (m_(0) - 1) / 2 - 1;

      Ti = pan::ltrimat(m_(0), Wgma_.subvec(first_index, last_index), false);
    } else {
      int first_index = 0;
      for (int idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

      Ti = pan::ltrimat(m_(i), Wgma_.subvec(first_index, last_index), false);
    }
  }
  return Ti;
}

inline arma::vec ACD::get_mu(arma::uword i) const {
  arma::vec mui;
  if (i == 0)
    mui = Xbta_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    mui = Xbta_.subvec(index, index + m_(i) - 1);
  }
  return mui;
}

inline arma::mat ACD::get_Sigma(arma::uword i) const {
  arma::mat Ti = get_T(i);
  arma::mat Di = get_D(i);

  return Di * Ti * Ti.t() * Di;
}

inline arma::mat ACD::get_Sigma_inv(arma::uword i) const {
  arma::mat Ti = get_T(i);
  //	    arma::mat Ti_inv = arma::pinv(Ti);
  arma::mat Ti_inv = Ti.i();

  arma::mat Di = get_D(i);
  arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

  return Di_inv * Ti_inv.t() * Ti_inv * Di_inv;
}

inline arma::vec ACD::get_Resid(arma::uword i) const {
  arma::vec ri;
  if (i == 0)
    ri = Resid_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    ri = Resid_.subvec(index, index + m_(i) - 1);
  }
  return ri;
}

inline void ACD::get_D(arma::uword i, arma::mat& Di) const {
  Di = arma::eye(m_(i), m_(i));
  if (i == 0)
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(0, m_(0) - 1) / 2));
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Di = arma::diagmat(arma::exp(Zlmd_.subvec(index, index + m_(i) - 1) / 2));
  }
}

inline void ACD::get_T(arma::uword i, arma::mat& Ti) const {
  Ti = arma::eye<arma::mat>(m_(i), m_(i));
  if (m_(i) != 1) {
    if (i == 0) {
      int first_index = 0;
      int last_index = m_(0) * (m_(0) - 1) / 2 - 1;

      Ti = pan::ltrimat(m_(0), Wgma_.subvec(first_index, last_index), false);
    } else {
      int first_index = 0;
      for (int idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

      Ti = pan::ltrimat(m_(i), Wgma_.subvec(first_index, last_index), false);
    }
  }
}

inline void ACD::get_invT(arma::uword i, arma::mat& Ti_inv) const {
  Ti_inv = arma::eye(m_(i), m_(i));
  if (m_(i) != 1) {
    if (i == 0) {
      int first_index = 0;
      int last_index = m_(0) * (m_(0) + 1) / 2 - 1;

      Ti_inv =
          pan::ltrimat(m_(0), invTelem_.subvec(first_index, last_index), true);
    } else {
      int first_index = 0;
      for (int idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) + 1) / 2;
      }
      int last_index = first_index + m_(i) * (m_(i) + 1) / 2 - 1;

      Ti_inv =
          pan::ltrimat(m_(i), invTelem_.subvec(first_index, last_index), true);
    }
  }
}

inline void ACD::get_Sigma_inv(arma::uword i, arma::mat& Sigmai_inv) {
  arma::mat Ti;
  get_T(i, Ti);
  //	    arma::mat Ti_inv = arma::pinv(Ti);
  arma::mat Ti_inv;
  get_invT(i, Ti_inv);

  arma::mat Di;
  get_D(i, Di);
  arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

  Sigmai_inv = Di_inv * Ti_inv.t() * Ti_inv * Di_inv;
}

inline double ACD::operator()(const arma::vec& x) {
  int debug = 0;
  if (debug) Rcpp::Rcout << "UpdateJmcm..." << std::endl;
  UpdateJmcm(x);

  if (debug) Rcpp::Rcout << "UpdateJmcm finished..." << std::endl;

  int i, n_sub = m_.n_elem;
  double result = 0.0;

  //#pragma omp parallel for reduction(+:result)
  for (i = 0; i < n_sub; ++i) {
    arma::vec ri = get_Resid(i);
    //	    arma::mat Sigmai_inv = get_Sigma_inv(i);
    arma::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    result += arma::as_scalar(ri.t() * Sigmai_inv * ri);
    if (result < 0) {
      Rcpp::Rcout << "result = " << result << std::endl;
    }
  }
  if (debug) Rcpp::Rcout << "After for loop" << std::endl;

  result += 2 * arma::sum(arma::log(arma::exp(Zlmd_ / 2)));

  return result;
}

inline void ACD::Gradient(const arma::vec& x, arma::vec& grad) {
  UpdateJmcm(x);

  int n_bta = X_.n_cols, n_lmd = Z_.n_cols, n_gma = W_.n_cols;

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

    case 2:
      Grad2(grad);
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline void ACD::Grad1(arma::vec& grad1) {
  int debug = 0;

  int n_sub = m_.n_elem, n_bta = X_.n_cols;
  grad1 = arma::zeros<arma::vec>(n_bta);

  if (debug) Rcpp::Rcout << "Update grad1" << std::endl;

  for (auto i = 0; i < n_sub; ++i) {
    arma::mat Xi = get_X(i);
    arma::vec ri = get_Resid(i);
    // arma::mat Sigmai_inv = get_Sigma_inv(i);
    arma::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    grad1 += Xi.t() * Sigmai_inv * ri;
  }

  grad1 *= -2;
}

inline void ACD::Grad2(arma::vec& grad2) {
  int debug = 0;

  int i, n_sub = m_.n_elem, n_lmd = Z_.n_cols, n_gma = W_.n_cols;
  grad2 = arma::zeros<arma::vec>(n_lmd + n_gma);
  arma::vec grad2_lmd = arma::zeros<arma::vec>(n_lmd);
  arma::vec grad2_gma = arma::zeros<arma::vec>(n_gma);

  if (debug) Rcpp::Rcout << "Update grad2" << std::endl;

  for (i = 0; i < n_sub; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);
    arma::vec hi;
    get_TDResid2(i, hi);

    grad2_lmd += 0.5 * Zi.t() * (hi - one);

    arma::mat Ti;
    get_T(i, Ti);
    //	    arma::mat Ti_inv = arma::pinv(Ti);
    // arma::mat Ti_inv = Ti.i();
    arma::mat Ti_inv;
    get_invT(i, Ti_inv);

    arma::vec ei;
    get_TDResid(i, ei);

    arma::mat Ti_trans_deriv = CalcTransTiDeriv(i);

    grad2_gma += arma::kron(ei.t(), arma::eye(n_gma, n_gma)) * Ti_trans_deriv *
                 Ti_inv.t() * ei;
  }
  grad2.subvec(0, n_lmd - 1) = grad2_lmd;
  grad2.subvec(n_lmd, n_lmd + n_gma - 1) = grad2_gma;

  grad2 *= -2;
}

inline void ACD::UpdateJmcm(const arma::vec& x) {
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
      if (arma::min(x == lmdgma_) == 1) update = false;
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }

  if (update) {
    if (debug) Rcpp::Rcout << "UpdateParam..." << std::endl;
    UpdateParam(x);
    if (debug) Rcpp::Rcout << "UpdateModel..." << std::endl;
    UpdateModel();
    if (debug) Rcpp::Rcout << "Update Finished..." << std::endl;
  } else {
    if (debug) Rcpp::Rcout << "Hey, I did save some time!:)" << std::endl;
  }
}

inline void ACD::UpdateParam(const arma::vec& x) {
  int n_bta = X_.n_cols;
  int n_lmd = Z_.n_cols;
  int n_gma = W_.n_cols;

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

    case 2:
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

    case 2:
      Zlmd_ = Z_ * lambda_;
      Wgma_ = W_ * gamma_;

      UpdateTelem();
      UpdateTDResid();

      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline arma::vec ACD::get_TDResid(arma::uword i) const {
  arma::vec TiDiri;
  if (i == 0)
    TiDiri = TDResid_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    TiDiri = TDResid_.subvec(index, index + m_(i) - 1);
  }
  return TiDiri;
}

inline void ACD::get_TDResid(arma::uword i, arma::vec& TiDiri) const {
  if (i == 0)
    TiDiri = TDResid_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    TiDiri = TDResid_.subvec(index, index + m_(i) - 1);
  }
}

inline arma::vec ACD::get_TDResid2(arma::uword i) const {
  arma::vec TiDiri2;
  if (i == 0)
    TiDiri2 = TDResid2_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    TiDiri2 = TDResid2_.subvec(index, index + m_(i) - 1);
  }
  return TiDiri2;
}

inline void ACD::get_TDResid2(arma::uword i, arma::vec& TiDiri2) const {
  if (i == 0)
    TiDiri2 = TDResid2_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    TiDiri2 = TDResid2_.subvec(index, index + m_(i) - 1);
  }
}

inline void ACD::UpdateTelem() {
  int i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Ti;
    get_T(i, Ti);
    // arma::mat Ti_inv = arma::pinv(Ti);
    // arma::mat Ti_inv = Ti.i();
    // arma::mat Ti_inv = pinv(Ti);
    arma::mat Ti_inv;
    // bool is_Ti_pd = arma::inv(Ti_inv, Ti);
    // if (!is_Ti_pd) Ti_inv = arma::pinv(Ti);
    if (!arma::inv(Ti_inv, Ti)) Ti_inv = arma::pinv(Ti);

    if (i == 0) {
      int first_index = 0;
      int last_index = m_(0) * (m_(0) + 1) / 2 - 1;

      invTelem_.subvec(first_index, last_index) = pan::lvectorise(Ti_inv, true);
    } else {
      int first_index = 0;
      for (int idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) + 1) / 2;
      }
      int last_index = first_index + m_(i) * (m_(i) + 1) / 2 - 1;

      invTelem_.subvec(first_index, last_index) = pan::lvectorise(Ti_inv, true);
    }
  }
}

inline void ACD::UpdateTDResid() {
  int i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::vec ri = get_Resid(i);

    arma::mat Ti = get_T(i);
    //	    arma::mat Ti_inv = arma::pinv(Ti);
    // arma::mat Ti_inv = Ti.i();
    arma::mat Ti_inv;
    get_invT(i, Ti_inv);

    arma::mat Di = get_D(i);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    arma::vec TiDiri = Ti_inv * Di_inv * ri;
    arma::vec TiDiri2 = arma::diagvec(Ti_inv.t() * Ti_inv * Di_inv * ri *
                                      ri.t() * Di_inv);  // hi

    if (i == 0) {
      TDResid_.subvec(0, m_(0) - 1) = TiDiri;
      TDResid2_.subvec(0, m_(0) - 1) = TiDiri2;
    } else {
      int index = arma::sum(m_.subvec(0, i - 1));
      TDResid_.subvec(index, index + m_(i) - 1) = TiDiri;
      TDResid2_.subvec(index, index + m_(i) - 1) = TiDiri2;
    }
  }
}

inline arma::vec ACD::Wijk(arma::uword i, arma::uword j, arma::uword k) {
  int n_sub = m_.n_rows;
  int n_gma = W_.n_cols;

  int W_rowindex = 0;
  bool indexfound = false;
  arma::vec result = arma::zeros<arma::vec>(n_gma);
  for (int ii = 0; ii != n_sub && !indexfound; ++ii) {
    for (int jj = 0; jj != m_(ii) && !indexfound; ++jj) {
      for (int kk = 0; kk != jj && !indexfound; ++kk) {
        if (ii == i && jj == j && kk == k) {
          indexfound = true;
          result = W_.row(W_rowindex).t();
        }
        ++W_rowindex;
      }
    }
  }
  return result;
}

inline arma::vec ACD::CalcTijkDeriv(arma::uword i, arma::uword j,
                                    arma::uword k) {
  arma::vec result = Wijk(i, j, k);

  return result;
}

inline arma::mat ACD::CalcTransTiDeriv(arma::uword i) {
  int n_gma = W_.n_cols;

  arma::mat result = arma::zeros<arma::mat>(n_gma * m_(i), m_(i));
  for (int k = 1; k != m_(i); ++k) {
    for (int j = 0; j <= k; ++j) {
      result.submat(j * n_gma, k, (j * n_gma + n_gma - 1), k) =
          CalcTijkDeriv(i, k, j);
    }
  }

  return result;
}

}  // namespace jmcm

#endif  // JMCM_ACD_H_
