//  acd.h: joint mean-covariance models based on alternative Cholesky
//         decomposition (ACD) of the covariance matrix
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

#ifndef _JMCM_ACD_H_
#define _JMCM_ACD_H_

#include "jmcm_base.h"
#include "jmcm_config.h"

namespace jmcm {

class ACD : public JmcmBase {
 public:
  ACD(const arma::uvec& m, const arma::vec& Y, const arma::mat& X,
      const arma::mat& Z, const arma::mat& W);

  void UpdateModel() override;

  arma::vec Grad2() const override;
  arma::vec Grad3() const override;

  double CalcLogDetSigma() const override { return arma::sum(Zlmd_); }

  // Sigma = Di * Ti * Ti.t() * Di
  // Sigma_inv = Di_inv * Ti_inv.t() * Ti_inv * Di_inv
  arma::mat get_Sigma(arma::uword i, bool inv = false) const override {
    arma::mat Di = get_D(i, inv), Ti = get_T(i, inv);
    arma::mat tmp = inv ? Ti * Di : Di * Ti;
    return inv ? arma::mat(tmp.t() * tmp) : arma::mat(tmp * tmp.t());
  }

  arma::mat get_D(arma::uword i, bool inv = false) const override {
    arma::vec elem = inv ? -get_Zlmd(i) : get_Zlmd(i);
    return arma::diagmat(arma::exp(elem / 2));
  }
  arma::mat get_T(arma::uword i, bool inv = false) const override {
    if (m_(i) == 1) return arma::eye(m_(i), m_(i));
    arma::vec elem =
        inv ? invTelem_.subvec(cumsum_trim2_(i), cumsum_trim2_(i + 1) - 1)
            : get_Wgma(i);
    return get_ltrimatrix(m_(i), elem, inv);
  }

 private:
  arma::vec invTelem_, TDResid_, TDResid2_;

  arma::vec get_TDResid(arma::uword i) const {
    return TDResid_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }
  arma::vec get_TDResid2(arma::uword i) const {
    return TDResid2_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }

  void UpdateTelem();
  void UpdateTDResid();

  arma::vec CalcTijkDeriv(arma::uword i, arma::uword j, arma::uword k) const {
    return Wijk(i, j, k);
  }
  arma::mat CalcTransTiDeriv(arma::uword i) const;
};  // class ACD

inline ACD::ACD(const arma::uvec& m, const arma::vec& Y, const arma::mat& X,
                const arma::mat& Z, const arma::mat& W)
    : JmcmBase(m, Y, X, Z, W, 1),
      invTelem_(W_.n_rows + N_, arma::fill::zeros),
      TDResid_(N_, arma::fill::zeros),
      TDResid2_(N_, arma::fill::zeros) {}

// clang-format off
inline void ACD::UpdateModel() {
  switch (get_free_param()) {
    case  1: { break; }
    case  0:
    case 23: { UpdateTelem(); break; }
    default: { arma::get_cerr_stream() << "Wrong value for free_param_" << std::endl; }
  }
  UpdateTDResid();
}
// clang-format on

inline arma::vec ACD::Grad2() const {
  arma::vec grad2 = arma::zeros<arma::vec>(n_lmd_);
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);
    arma::vec hi = get_TDResid2(i);

    grad2 += Zi.t() * (hi - one);  // cancel the 0.5 in front and 2 in return
  }

  return (-grad2);
}

inline arma::vec ACD::Grad3() const {
  arma::vec grad3 = arma::zeros<arma::vec>(n_gma_);
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::vec ei = get_TDResid(i);
    arma::mat Ti_trans_deriv = CalcTransTiDeriv(i);
    arma::mat Ti_inv = get_T(i, true);

    grad3 += arma::kron(ei.t(), arma::eye(n_gma_, n_gma_)) *
             (Ti_trans_deriv * (Ti_inv.t() * ei));
  }

  return (-2 * grad3);
}

inline void ACD::UpdateTelem() {
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Ti = get_T(i);
    arma::mat Ti_inv;
    if (!arma::inv(Ti_inv, Ti)) Ti_inv = arma::pinv(Ti);
    invTelem_.subvec(cumsum_trim2_(i), cumsum_trim2_(i + 1) - 1) =
        get_lower_part(Ti_inv);
  }
}

inline void ACD::UpdateTDResid() {
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Ti_inv = get_T(i, true);
    arma::mat Di_inv = get_D(i, true);

    arma::vec Diri = Di_inv * ri;
    arma::vec TiDiri = Ti_inv * Diri;
    arma::vec TiDiri2 = arma::diagvec(Ti_inv.t() * TiDiri * Diri.t());

    TDResid_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1) = TiDiri;
    TDResid2_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1) = TiDiri2;
  }
}

inline arma::mat ACD::CalcTransTiDeriv(arma::uword i) const {
  arma::mat result = arma::zeros<arma::mat>(n_gma_ * m_(i), m_(i));
  for (arma::uword k = 1; k < m_(i); ++k) {
    for (arma::uword j = 0; j <= k; ++j) {
      result.submat(j * n_gma_, k, (j * n_gma_ + n_gma_ - 1), k) =
          CalcTijkDeriv(i, k, j);
    }
  }

  return result;
}

}  // namespace jmcm

#endif  // _JMCM_ACD_H_
