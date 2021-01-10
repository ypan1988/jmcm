//  hpc.h: joint mean-covariance models based on standard Cholesky decomposition
//         of the correlation matrix R and the hyperspherical parametrization
//         (HPC) of its Cholesky factor
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

#ifndef JMCM_SRC_HPC_H_
#define JMCM_SRC_HPC_H_

#include <cmath>

#include <algorithm>  // std::equal

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>

#include "arma_util.h"
#include "jmcm_base.h"

namespace jmcm {

class HPC : public JmcmBase {
 public:
  HPC() = delete;
  HPC(const HPC&) = delete;
  ~HPC() = default;

  HPC(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
      const arma::mat& Z, const arma::mat& W);

  void UpdateLambdaGamma(const arma::vec& x) override { set_lmdgma(x); }

  arma::mat get_Phi(arma::uword i) const {
    return m_(i) == 1 ? arma::zeros<arma::mat>(m_(i), m_(i)) :
           pan::ltrimat(m_(i), Wgma_.subvec(cumsum_trim_(i), cumsum_trim_(i+1) - 1), false);
  }
  arma::mat get_R(arma::uword i) const {
    arma::mat Ti = get_T(i);
    return Ti * Ti.t();
  }

  arma::mat get_D(arma::uword i) const override {
    return arma::diagmat(arma::exp(Zlmd_.subvec(cumsum_m_(i), cumsum_m_(i+1) - 1) / 2));
  }
  arma::mat get_T(arma::uword i) const override {
    return m_(i) == 1 ? arma::eye(m_(i), m_(i)) :
           pan::ltrimat(m_(i), Telem_.subvec(cumsum_trim2_(i), cumsum_trim2_(i+1) - 1), true);
  }
  arma::mat get_invT(arma::uword i) const {
    return m_(i) == 1 ? arma::eye(m_(i), m_(i)) :
           pan::ltrimat(m_(i), invTelem_.subvec(cumsum_trim2_(i), cumsum_trim2_(i+1) - 1), true);
  }

  arma::mat get_Sigma(arma::uword i) const override;
  arma::mat get_Sigma_inv(arma::uword i) const override;

  void Grad2(arma::vec& grad2) override;
  void Grad3(arma::vec& grad3) override;

  void UpdateModel() override;

  double CalcLogDetSigma() const override {
    double result = 0.0;
    for (arma::uword i = 0; i < m_.n_elem; ++i) {
      result += 2 * arma::sum(arma::log(get_T(i).diag()));
    }
    result += 2 * arma::sum(arma::log(arma::exp(Zlmd_ / 2)));
    return result;
  }

 private:
  arma::vec Telem_;  // elements for the lower triangular matrix T
  arma::vec invTelem_;
  arma::vec TDResid_;
  arma::vec TDResid2_;

  arma::vec get_TDResid(arma::uword i) const {
    return TDResid_.subvec(cumsum_m_(i), cumsum_m_(i+1) - 1);
  }
  arma::vec get_TDResid2(arma::uword i) const {
    return TDResid2_.subvec(cumsum_m_(i), cumsum_m_(i+1) - 1);
  }

  void UpdateTelem();
  void UpdateTDResid();

  arma::vec CalcTijkDeriv(arma::uword i, arma::uword j, arma::uword k,
                          const arma::mat& Phii, const arma::mat& Ti);
  arma::mat CalcTransTiDeriv(arma::uword i, const arma::mat& Phii,
                             const arma::mat& Ti);

};  // class HPC

inline HPC::HPC(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
                const arma::mat& Z, const arma::mat& W)
    : JmcmBase(m, Y, X, Z, W, 2) {
  arma::uword N = Y_.n_rows;

  Telem_ = arma::zeros<arma::vec>(W_.n_rows + N);
  invTelem_ = arma::zeros<arma::vec>(W_.n_rows + N);

  TDResid_ = arma::zeros<arma::vec>(N);
  TDResid2_ = arma::zeros<arma::vec>(N);
}

inline arma::mat HPC::get_Sigma(arma::uword i) const {
  arma::mat DiTi = get_D(i) * get_T(i);

  return DiTi * DiTi.t();
}

inline arma::mat HPC::get_Sigma_inv(arma::uword i) const {
  arma::mat Di = get_D(i);
  arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));
  arma::mat Ti_inv = get_invT(i);
  arma::mat Ti_inv_Di_inv = Ti_inv * Di_inv;

  return Ti_inv_Di_inv.t() * Ti_inv_Di_inv;
}

inline void HPC::Grad2(arma::vec& grad2) {
  arma::uword i, n_sub = m_.n_elem, n_lmd = Z_.n_cols;
  grad2 = arma::zeros<arma::vec>(n_lmd);

  for (i = 0; i < n_sub; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);
    arma::vec hi = get_TDResid2(i);

    grad2 += 0.5 * Zi.t() * (hi - one);
  }

  grad2 *= -2;
}

inline void HPC::Grad3(arma::vec& grad3) {
  arma::uword i, n_sub = m_.n_elem, n_gma = W_.n_cols;
  grad3 = arma::zeros<arma::vec>(n_gma);

  for (i = 0; i < n_sub; ++i) {
    arma::mat Phii = get_Phi(i);
    arma::mat Ti = get_T(i);
    arma::mat Ti_inv = get_invT(i);
    arma::vec ei = get_TDResid(i);

    arma::mat Ti_trans_deriv = CalcTransTiDeriv(i, Phii, Ti);
    for (arma::uword j = 0; j != m_(i); ++j) {
      grad3 += -1 / Ti(j, j) * CalcTijkDeriv(i, j, j, Phii, Ti);
    }
    grad3 += arma::kron(ei.t(), arma::eye(n_gma, n_gma)) * Ti_trans_deriv *
      Ti_inv.t() * ei;
  }

  grad3 *= -2;
}

inline void HPC::UpdateModel() {
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

    case 23:
      Zlmd_ = Z_ * lambda_;
      Wgma_ = W_ * gamma_;

      UpdateTelem();
      UpdateTDResid();

      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

inline void HPC::UpdateTelem() {
  arma::uword i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Phii = get_Phi(i);
    arma::mat Ti = arma::eye(m_(i), m_(i));

    Ti(0, 0) = 1;
    for (arma::uword j = 1; j != m_(i); ++j) {
      Ti(j, 0) = std::cos(Phii(j, 0));
      double cumsin=std::sin(Phii(j, 0));
      for (arma::uword l = 1; l != j; ++l) {
        Ti(j, l) = std::cos(Phii(j, l)) * cumsin;
        cumsin *= std::sin(Phii(j, l));
      }
      Ti(j, j) = cumsin;
    }

    arma::mat Ti_inv;
    if (!arma::inv(Ti_inv, Ti)) Ti_inv = arma::pinv(Ti);

    arma::uword first_index = cumsum_trim2_(i);
    arma::uword last_index = cumsum_trim2_(i+1)  - 1;

    Telem_.subvec(first_index, last_index) = pan::lvectorise(Ti, true);
    invTelem_.subvec(first_index, last_index) = pan::lvectorise(Ti_inv, true);
  }
}

inline void HPC::UpdateTDResid() {
  arma::uword i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Ti = get_T(i);
    arma::mat Ti_inv = get_invT(i);
    arma::mat Di = get_D(i);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    arma::vec TiDiri = Ti_inv * Di_inv * ri;
    arma::vec TiDiri2 = arma::diagvec(Ti_inv.t() * Ti_inv * Di_inv * ri *
                                      ri.t() * Di_inv);  // hi

    arma::uword first_index = cumsum_m_(i);
    arma::uword last_index = cumsum_m_(i+1) - 1;

    TDResid_.subvec(first_index, last_index) = TiDiri;
    TDResid2_.subvec(first_index, last_index) = TiDiri2;
  }
}

inline arma::vec HPC::CalcTijkDeriv(arma::uword i, arma::uword j, arma::uword k,
                                    const arma::mat& Phii,
                                    const arma::mat& Ti) {
  arma::uword n_gma = W_.n_cols;

  arma::vec result = arma::zeros<arma::vec>(n_gma);
  if (k < j) {
    result = Ti(j, k) * (-std::tan(Phii(j, k)) * Wijk(i, j, k));
    for (arma::uword l = 0; l != k; ++l) {
      result += Ti(j, k) * Wijk(i, j, l) / std::tan(Phii(j, l));
    }
    return result;
  } else if (k == j) {
    for (arma::uword l = 0; l != k; ++l) {
      result += Ti(j, k) * Wijk(i, j, l) / std::tan(Phii(j, l));
    }
    return result;
  }

  return result;
}

inline arma::mat HPC::CalcTransTiDeriv(arma::uword i, const arma::mat& Phii,
                                       const arma::mat& Ti) {
  arma::uword n_gma = W_.n_cols;

  arma::mat result = arma::zeros<arma::mat>(n_gma * m_(i), m_(i));
  for (arma::uword k = 1; k != m_(i); ++k) {
    for (arma::uword j = 0; j <= k; ++j) {
      result.submat(j * n_gma, k, (j * n_gma + n_gma - 1), k) =
          CalcTijkDeriv(i, k, j, Phii, Ti);
    }
  }

  return result;
}

}  // namespace jmcm

#endif  // JMCM_SRC_HPC_H_
