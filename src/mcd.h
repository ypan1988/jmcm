// mcd.h: joint mean-covariance models based on modified Cholesky decgomposition
//        (MCD) of the covariance matrix
//
// Copyright (C) 2015-2017 Yi Pan
//
// This file is part of jmcm.

#ifndef JMCM_SRC_MCD_H_
#define JMCM_SRC_MCD_H_

#include "arma_util.h"
#include "jmcm_base.h"
#include <RcppArmadillo.h>

namespace jmcm {

class MCD : public JmcmBase {
 public:
  MCD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
      const arma::mat& Z, const arma::mat& W);
  ~MCD();

  void set_free_param(const int n) { free_param_ = n; }

  void set_theta(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 0;
    UpdateJmcm(x);
    free_param_ = fp2;
  }

  void set_beta(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 1;
    UpdateJmcm(x);
    free_param_ = fp2;
  }

  void UpdateBeta() {
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

  void set_lambda(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 2;
    UpdateJmcm(x);
    free_param_ = fp2;
  }

  void set_gamma(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 3;
    UpdateJmcm(x);
    free_param_ = fp2;
  }

  arma::mat get_D(const int i) const {
    arma::mat Di = arma::eye(m_(i), m_(i));
    if (i == 0)
      Di = arma::diagmat(arma::exp(Zlmd_.subvec(0, m_(0) - 1)));
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Di = arma::diagmat(arma::exp(Zlmd_.subvec(index, index + m_(i) - 1)));
    }
    return Di;
  }

  void get_D(const int i, arma::mat& Di) const {
    Di = arma::eye(m_(i), m_(i));
    if (i == 0)
      Di = arma::diagmat(arma::exp(Zlmd_.subvec(0, m_(0) - 1)));
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Di = arma::diagmat(arma::exp(Zlmd_.subvec(index, index + m_(i) - 1)));
    }
  }

  arma::mat get_T(const int i) const {
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

  void get_T(const int i, arma::mat& Ti) const {
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

  arma::mat get_Sigma(const int i) const {
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

  arma::mat get_Sigma_inv(const int i) const override {
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

  void get_Sigma_inv(const int i, arma::mat& Sigmai_inv) const {
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

  arma::vec get_mu(const int i) const {
    arma::vec mui;
    if (i == 0)
      mui = Xbta_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      mui = Xbta_.subvec(index, index + m_(i) - 1);
    }
    return mui;
  }

  arma::vec get_Resid(const int i) const {
    arma::vec ri;
    if (i == 0)
      ri = Resid_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      ri = Resid_.subvec(index, index + m_(i) - 1);
    }
    return ri;
  }

  void get_Resid(const int i, arma::vec& ri) const {
    if (i == 0)
      ri = Resid_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      ri = Resid_.subvec(index, index + m_(i) - 1);
    }
  }

  double operator()(const arma::vec& x);
  void Gradient(const arma::vec& x, arma::vec& grad);
  void Grad1(arma::vec& grad1);
  void Grad2(arma::vec& grad2);
  void Grad3(arma::vec& grad3);

  void UpdateJmcm(const arma::vec& x);
  void UpdateParam(const arma::vec& x);
  void UpdateModel();

  void UpdateLambda(const arma::vec& x);
  void UpdateGamma();

  void set_mean(const arma::vec& mean) {
    cov_only_ = true;
    mean_ = mean;
  }

  // void CalcMeanCovmati(const arma::vec& x, int i,
  // 		     arma::vec& mui, arma::mat& Sigmai);
  // void SimResp(int n, const arma::vec& x, arma::mat& resp);

 private:
  arma::mat G_;
  arma::vec TResid_;

  int free_param_;

  bool cov_only_;
  arma::vec mean_;

  arma::mat get_G(const int i) const {
    arma::mat Gi;
    if (i == 0)
      Gi = G_.rows(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Gi = G_.rows(index, index + m_(i) - 1);
    }
    return Gi;
  }

  void get_G(const int i, arma::mat& Gi) const {
    if (i == 0)
      Gi = G_.rows(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Gi = G_.rows(index, index + m_(i) - 1);
    }
  }

  arma::vec get_TResid(const int i) const {
    arma::vec Tiri;
    if (i == 0)
      Tiri = TResid_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Tiri = TResid_.subvec(index, index + m_(i) - 1);
    }
    return Tiri;
  }

  void get_TResid(const int i, arma::vec& Tiri) const {
    if (i == 0)
      Tiri = TResid_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Tiri = TResid_.subvec(index, index + m_(i) - 1);
    }
  }

  void UpdateG();
  void UpdateTResid();

};  // class MCD

}  // namespace jmcm

#endif  // JMCM_SRC_MCD_H_
