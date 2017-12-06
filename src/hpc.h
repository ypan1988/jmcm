// hpc.h: joint mean-covariance models based on standard Cholesky decomposition
//        of the correlation matrix R and the hyperspherical parametrization
//        (HPC) of its Cholesky factor
//
// Copyright (C) 2015-2017 Yi Pan
//
// This file is part of jmcm.

#ifndef JMCM_HPC_H_
#define JMCM_HPC_H_

#define ARMA_DONT_PRINT_ERRORS
#include "arma_util.h"
#include "jmcm_base.h"
#include <RcppArmadillo.h>

namespace jmcm {

class HPC : public JmcmBase {
 public:
  HPC(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
      const arma::mat& Z, const arma::mat& W);
  ~HPC();

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

  void set_lmdgma(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 2;
    UpdateJmcm(x);
    free_param_ = fp2;
  }

  arma::mat get_D(const int i) const {
    arma::mat Di = arma::eye(m_(i), m_(i));
    if (i == 0)
      Di = arma::diagmat(arma::exp(Zlmd_.subvec(0, m_(0) - 1) / 2));
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Di = arma::diagmat(arma::exp(Zlmd_.subvec(index, index + m_(i) - 1) / 2));
    }
    return Di;
  }

  void get_D(const int i, arma::mat& Di) const {
    Di = arma::eye(m_(i), m_(i));
    if (i == 0)
      Di = arma::diagmat(arma::exp(Zlmd_.subvec(0, m_(0) - 1) / 2));
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Di = arma::diagmat(arma::exp(Zlmd_.subvec(index, index + m_(i) - 1) / 2));
    }
  }

  arma::mat get_Phi(const int i) const {
    arma::mat Phii = arma::zeros<arma::mat>(m_(i), m_(i));
    if (m_(i) != 1) {
      if (i == 0) {
        int first_index = 0;
        int last_index = m_(0) * (m_(0) - 1) / 2 - 1;

        Phii =
            pan::ltrimat(m_(0), Wgma_.subvec(first_index, last_index), false);
      } else {
        int first_index = 0;
        for (int idx = 0; idx != i; ++idx) {
          first_index += m_(idx) * (m_(idx) - 1) / 2;
        }
        int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

        Phii =
            pan::ltrimat(m_(i), Wgma_.subvec(first_index, last_index), false);
      }
    }
    return Phii;
  }

  void get_Phi(const int i, arma::mat& Phii) const {
    Phii = arma::zeros<arma::mat>(m_(i), m_(i));
    if (m_(i) != 1) {
      if (i == 0) {
        int first_index = 0;
        int last_index = m_(0) * (m_(0) - 1) / 2 - 1;

        Phii =
            pan::ltrimat(m_(0), Wgma_.subvec(first_index, last_index), false);
      } else {
        int first_index = 0;
        for (int idx = 0; idx != i; ++idx) {
          first_index += m_(idx) * (m_(idx) - 1) / 2;
        }
        int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

        Phii =
            pan::ltrimat(m_(i), Wgma_.subvec(first_index, last_index), false);
      }
    }
  }

  arma::mat get_T(const int i) const {
    arma::mat Ti = arma::eye(m_(i), m_(i));
    if (m_(i) != 1) {
      if (i == 0) {
        int first_index = 0;
        int last_index = m_(0) * (m_(0) + 1) / 2 - 1;

        Ti = pan::ltrimat(m_(0), Telem_.subvec(first_index, last_index), true);
      } else {
        int first_index = 0;
        for (int idx = 0; idx != i; ++idx) {
          first_index += m_(idx) * (m_(idx) + 1) / 2;
        }
        int last_index = first_index + m_(i) * (m_(i) + 1) / 2 - 1;

        Ti = pan::ltrimat(m_(i), Telem_.subvec(first_index, last_index), true);
      }
    }
    return Ti;
  }

  void get_T(const int i, arma::mat& Ti) const {
    Ti = arma::eye(m_(i), m_(i));
    if (m_(i) != 1) {
      if (i == 0) {
        int first_index = 0;
        int last_index = m_(0) * (m_(0) + 1) / 2 - 1;

        Ti = pan::ltrimat(m_(0), Telem_.subvec(first_index, last_index), true);
      } else {
        int first_index = 0;
        for (int idx = 0; idx != i; ++idx) {
          first_index += m_(idx) * (m_(idx) + 1) / 2;
        }
        int last_index = first_index + m_(i) * (m_(i) + 1) / 2 - 1;

        Ti = pan::ltrimat(m_(i), Telem_.subvec(first_index, last_index), true);
      }
    }
  }

  void get_invT(const int i, arma::mat& Ti_inv) const {
    Ti_inv = arma::eye(m_(i), m_(i));
    if (m_(i) != 1) {
      if (i == 0) {
        int first_index = 0;
        int last_index = m_(0) * (m_(0) + 1) / 2 - 1;

        Ti_inv = pan::ltrimat(m_(0), invTelem_.subvec(first_index, last_index),
                              true);
      } else {
        int first_index = 0;
        for (int idx = 0; idx != i; ++idx) {
          first_index += m_(idx) * (m_(idx) + 1) / 2;
        }
        int last_index = first_index + m_(i) * (m_(i) + 1) / 2 - 1;

        Ti_inv = pan::ltrimat(m_(i), invTelem_.subvec(first_index, last_index),
                              true);
      }
    }
  }

  arma::mat get_R(const int i) const {
    arma::mat Ti = get_T(i);

    return Ti * Ti.t();
  }

  void get_R(const int i, arma::mat& Ri) const {
    arma::mat Ti = get_T(i);

    Ri = Ti * Ti.t();
  }

  arma::mat get_Sigma(const int i) const override {
    arma::mat Ti = get_T(i);
    arma::mat Di = get_D(i);

    return Di * Ti * Ti.t() * Di;
  }

  arma::mat get_Sigma_inv(const int i) const override {
    arma::mat Ti = get_T(i);
    // arma::mat Ti_inv = arma::pinv(Ti);
    arma::mat Ti_inv = Ti.i();

    arma::mat Di = get_D(i);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    return Di_inv * Ti_inv.t() * Ti_inv * Di_inv;
  }

  void get_Sigma_inv(const int i, arma::mat& Sigmai_inv) const {
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

  arma::vec get_Resid(const int i) {
    arma::vec ri;
    if (i == 0)
      ri = Resid_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      ri = Resid_.subvec(index, index + m_(i) - 1);
    }
    return ri;
  }

  void get_Resid(const int i, arma::vec& ri) {
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

  void UpdateJmcm(const arma::vec& x);
  void UpdateParam(const arma::vec& x);
  void UpdateModel();

  void UpdateLambdaGamma(const arma::vec& x);

  void set_mean(const arma::vec& mean) {
    cov_only_ = true;
    mean_ = mean;
  }

  // void CalcMeanCovmati(const arma::vec& x, int i,
  // 		     arma::vec& mui, arma::mat& Sigmai);
  // void SimResp(int n, const arma::vec& x, arma::mat& resp);

 private:
  arma::vec lmdgma_;
  arma::vec Telem_;  // elements for the lower triangular matrix T
  arma::vec invTelem_;

  arma::vec TDResid_;
  arma::vec TDResid2_;

  int free_param_;

  bool cov_only_;
  arma::vec mean_;

  arma::vec get_TDResid(const int i) const {
    arma::vec TiDiri;
    if (i == 0)
      TiDiri = TDResid_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      TiDiri = TDResid_.subvec(index, index + m_(i) - 1);
    }
    return TiDiri;
  }

  void get_TDResid(const int i, arma::vec& TiDiri) const {
    if (i == 0)
      TiDiri = TDResid_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      TiDiri = TDResid_.subvec(index, index + m_(i) - 1);
    }
  }

  arma::vec get_TDResid2(const int i) const {
    arma::vec TiDiri2;
    if (i == 0)
      TiDiri2 = TDResid2_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      TiDiri2 = TDResid2_.subvec(index, index + m_(i) - 1);
    }
    return TiDiri2;
  }

  void get_TDResid2(const int i, arma::vec& TiDiri2) const {
    if (i == 0)
      TiDiri2 = TDResid2_.subvec(0, m_(0) - 1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      TiDiri2 = TDResid2_.subvec(index, index + m_(i) - 1);
    }
  }

  void UpdateTelem();
  void UpdateTDResid();

  arma::vec Wijk(const int i, const int j, const int k);
  arma::vec CalcTijkDeriv(const int i, const int j, const int k,
                          const arma::mat& Phii, const arma::mat& Ti);
  arma::mat CalcTransTiDeriv(const int i, const arma::mat& Phii,
                             const arma::mat& Ti);

};  // class HPC

}  // namespace jmcm

#endif  // JMCM_HPC_H_
