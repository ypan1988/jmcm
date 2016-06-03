// hpc.h: implementation of joint mean-covariance models based on standard
//        Cholesky decomposition of the correlation matrix R and the
//        hyperspherical parametrization(HPC) of its Cholesky factor
//
// Copyright (C) 2015-2016 The University of Manchester
//
// Written by Yi Pan - ypan1988@gmail.com

#ifndef JMCM_HPC_H_
#define JMCM_HPC_H_

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>
#include "arma_util.h"

namespace jmcm {

class HPC {
 public:
  HPC(arma::vec& m, arma::vec& Y, arma::mat& X, arma::mat& Z, arma::mat& W);
  ~HPC();

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

  arma::mat get_X(int i) const {
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
    UpdateHPC(x);
    free_param_ = fp2;
  }

  arma::vec get_theta() const { return theta_; }

  void set_beta(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 1;
    UpdateHPC(x);
    free_param_ = fp2;
  }

  arma::vec get_beta() const { return beta_; }

  void set_lmdgma(const arma::vec& x) {
    int fp2 = free_param_;
    free_param_ = 2;
    UpdateHPC(x);
    free_param_ = fp2;
  }

  arma::vec get_lambda() const { return lambda_; }

  arma::vec get_gamma() const { return gamma_; }

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

  arma::mat get_Sigma(const int i) const {
    arma::mat Ti = get_T(i);
    arma::mat Di = get_D(i);

    return Di * Ti * Ti.t() * Di;
  }

  arma::mat get_Sigma_inv(const int i) const {
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

  void UpdateHPC(const arma::vec& x);
  void UpdateParam(const arma::vec& x);
  void UpdateModel();

  void UpdateBeta();
  void UpdateLambdaGamma(const arma::vec& x);

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
  arma::vec lmdgma_;

  arma::vec Xbta_;
  arma::vec Zlmd_;
  arma::vec Wgma_;
  arma::vec Telem_;  // elements for the lower triangular matrix T
  arma::vec invTelem_;
  arma::vec Resid_;

  arma::vec TDResid_;
  arma::vec TDResid2_;

  int free_param_;

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
