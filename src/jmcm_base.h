// jmcm_base.h: base class for three joint mean-covariance models (MCD/ACD/HPC)
//
// Copyright (C) 2015-2017 Yi Pan
//
// This file is part of jmcm.

#ifndef JMCM_SRC_JMCM_BASE_H_
#define JMCM_SRC_JMCM_BASE_H_

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>

namespace jmcm {
class JmcmBase {
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

  arma::vec get_theta() const { return theta_; }
  arma::vec get_beta() const { return beta_; }
  arma::vec get_lambda() const { return lambda_; }
  arma::vec get_gamma() const { return gamma_; }

  virtual void UpdateBeta() {}
  virtual void UpdateLambda(const arma::vec&) {}
  virtual void UpdateGamma() {}
  virtual void UpdateLambdaGamma(const arma::vec&) {}

  virtual arma::mat get_D(arma::uword i) const = 0;
  virtual arma::mat get_T(arma::uword i) const = 0;
  virtual arma::vec get_mu(arma::uword i) const = 0;
  virtual arma::mat get_Sigma(arma::uword i) const = 0;
  virtual arma::mat get_Sigma_inv(arma::uword i) const = 0;
  virtual arma::vec get_Resid(arma::uword i) const = 0;

  virtual double operator()(const arma::vec& x) = 0;
  virtual void Gradient(const arma::vec& x, arma::vec& grad) = 0;
  virtual void UpdateJmcm(const arma::vec& x) = 0;

 protected:
  arma::vec m_, Y_;
  arma::mat X_, Z_, W_;
  arma::uword method_id_;

  arma::vec theta_, beta_, lambda_, gamma_;
  arma::vec Xbta_, Zlmd_, Wgma_, Resid_;
};

inline JmcmBase::JmcmBase(const arma::vec& m, const arma::vec& Y,
                          const arma::mat& X, const arma::mat& Z,
                          const arma::mat& W, const arma::uword method_id)
    : m_(m), Y_(Y), X_(X), Z_(Z), W_(W), method_id_(method_id) {
  arma::uword N = Y_.n_rows;
  arma::uword n_bta = X_.n_cols;
  arma::uword n_lmd = Z_.n_cols;
  arma::uword n_gma = W_.n_cols;

  theta_ = arma::zeros<arma::vec>(n_bta + n_lmd + n_gma);
  beta_ = arma::zeros<arma::vec>(n_bta);
  lambda_ = arma::zeros<arma::vec>(n_lmd);
  gamma_ = arma::zeros<arma::vec>(n_gma);

  Xbta_ = arma::zeros<arma::vec>(N);
  Zlmd_ = arma::zeros<arma::vec>(N);
  Wgma_ = arma::zeros<arma::vec>(W_.n_rows);
  Resid_ = arma::zeros<arma::vec>(N);
}

inline arma::uword JmcmBase::get_m(arma::uword i) const { return m_(i); }

inline arma::vec JmcmBase::get_Y(arma::uword i) const {
  arma::vec Yi;
  if (i == 0)
    Yi = Y_.subvec(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Yi = Y_.subvec(index, index + m_(i) - 1);
  }
  return Yi;
}

inline arma::mat JmcmBase::get_X(arma::uword i) const {
  arma::mat Xi;
  if (i == 0)
    Xi = X_.rows(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Xi = X_.rows(index, index + m_(i) - 1);
  }
  return Xi;
}

inline arma::mat JmcmBase::get_Z(arma::uword i) const {
  arma::mat Zi;
  if (i == 0)
    Zi = Z_.rows(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Zi = Z_.rows(index, index + m_(i) - 1);
  }
  return Zi;
}

inline arma::mat JmcmBase::get_W(arma::uword i) const {
  arma::mat Wi;
  if (m_(i) != 1) {
    if (i == 0) {
      int first_index = 0;
      int last_index = m_(0) * (m_(0) - 1) / 2 - 1;
      Wi = W_.rows(first_index, last_index);
    } else {
      int first_index = 0;
      for (arma::uword idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

      Wi = W_.rows(first_index, last_index);
    }
  }

  return Wi;
}

}  // namespace jmcm

#endif
