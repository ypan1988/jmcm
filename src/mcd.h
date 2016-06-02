// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// mcd.h: implementation of joint mean-covariance models based on
//        modified Cholesky decomposition(M.CD) of the covariance matrix
//
// Copyright (C) 2015 Yi Pan and Jianxin Pan
//
// This file is part of jmcm.

#ifndef JMCM_MCD_H_
#define JMCM_MCD_H_

#include <RcppArmadillo.h>
#include "arma_util.h"

namespace jmcm {

class MCD {
 public:
  MCD(arma::vec& m, arma::vec& Y, arma::mat& X, arma::mat& Z, arma::mat& W);
  ~MCD();

  arma::vec get_m() const { return m_; }
  arma::vec get_Y() const { return Y_; }
  arma::mat get_X() const { return X_; }
  arma::mat get_Z() const { return Z_; }
  arma::mat get_W() const { return W_; }

  int get_m(const int i) const { return m_(i); }

  arma::vec get_Y(const int i) const {
    arma::vec Yi;
    if (i == 0)
      Yi = Y_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Yi = Y_.subvec(index, index + m_(i) - 1);
    }
    return Yi;
  }

  void get_Y(const int i, arma::vec& Yi) const {
    if (i == 0)
      Yi = Y_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Yi = Y_.subvec(index, index + m_(i) - 1);
    }
  }

  arma::mat get_X(const int i) const {
    arma::mat Xi;
    if (i == 0)
      Xi = X_.rows(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Xi = X_.rows(index, index + m_(i) - 1);
    }
    return Xi;
  }

  void get_X(const int i, arma::mat& Xi) const {
    if (i == 0)
      Xi = X_.rows(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Xi = X_.rows(index, index + m_(i) - 1);
    }
  }

  arma::mat get_Z(const int i) const {
    arma::mat Zi;
    if (i == 0)
      Zi = Z_.rows(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Zi = Z_.rows(index, index + m_(i) - 1);
    }
    return Zi;
  }

  void get_Z(const int i, arma::mat& Zi) const {
    if (i == 0)
      Zi = Z_.rows(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Zi = Z_.rows(index, index + m_(i) - 1);
    }
  }

  arma::mat get_W(const int i) const {
    arma::mat Wi;
    if (m_(i) != 1) {
      if (i == 0) {
        int first_index = 0;
        int last_index = m_(0) * (m_(0) - 1) / 2 - 1;
        Wi = W_.rows(first_index, last_index);
      } else {
        int first_index = 0;
        for (int idx = 0; idx != i; ++idx) {
          first_index += m_(idx) * (m_(idx) - 1) / 2;
        }
        int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

        Wi = W_.rows(first_index, last_index);
      }
    }

    return Wi;
  }

  void get_W(const int i, arma::mat& Wi) const {
    if (m_(i) != 1) {
      if (i == 0) {
        int first_index = 0;
        int last_index = m_(0) * (m_(0) - 1) / 2 - 1;
        Wi = W_.rows(first_index, last_index);
      } else {
        int first_index = 0;
        for (int idx = 0; idx != i; ++idx) {
          first_index += m_(idx) * (m_(idx) - 1) / 2;
        }
        int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

        Wi = W_.rows(first_index, last_index);
      }
    }
  }

  void set_free_param(const int n) { free_param_ = n; }

  void set_theta(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 0;
    UpdateMCD(x);
    free_param_ = fp2;
  }

  arma::vec get_theta() const { return theta_; }

  void set_beta(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 1;
    UpdateMCD(x);
    free_param_ = fp2;
  }

  arma::vec get_beta() const { return beta_; }

  void set_lambda(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 2;
    UpdateMCD(x);
    free_param_ = fp2;
  }

  arma::vec get_lambda() const { return lambda_; }

  void set_gamma(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 3;
    UpdateMCD(x);
    free_param_ = fp2;
  }

  arma::vec get_gamma() const { return gamma_; }

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

  arma::mat get_Sigma_inv(const int i) const {
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

  void UpdateMCD(const arma::vec& x);
  void UpdateParam(const arma::vec& x);
  void UpdateModel();

  void UpdateBeta();
  void UpdateLambda(const arma::vec& x);
  void UpdateGamma();

  // void CalcMeanCovmati(const arma::vec& x, int i,
  // 		     arma::vec& mui, arma::mat& Sigmai);
  // void SimResp(int n, const arma::vec& x, arma::mat& resp);

 private:
  arma::vec m_;
  arma::vec Y_;
  arma::mat X_;
  arma::mat Z_;
  arma::mat W_;

  arma::vec theta_;
  arma::vec beta_;
  arma::vec lambda_;
  arma::vec gamma_;

  arma::vec Xbta_;
  arma::vec Zlmd_;
  arma::vec Wgma_;
  arma::vec Resid_;

  arma::mat G_;
  arma::vec TResid_;

  int free_param_;

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

#endif  // JMCM_MCD_H_
