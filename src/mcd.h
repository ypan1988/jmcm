//  mcd.h: joint mean-covariance models based on modified Cholesky
//         decomposition (MCD) of the covariance matrix
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

#ifndef _JMCM_MCD_H_
#define _JMCM_MCD_H_

#include "jmcm_base.h"
#include "jmcm_config.h"

namespace jmcm {

class MCD : public JmcmBase {
 public:
  MCD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
      const arma::mat& Z, const arma::mat& W);

  void UpdateGamma() override;
  void UpdateModel() override;

  arma::vec Grad2() const override;
  arma::vec Grad3() const override;

  double CalcLogDetSigma() const override { return arma::sum(Zlmd_); }

  // Sigma = Ti_inv * Di * Ti_inv.t()
  // Sigma_inv = Ti.t() * Di_inv * Ti
  arma::mat get_Sigma(arma::uword i, bool inv = true) const override {
    arma::mat Ti = get_T(i, !inv), Di = get_D(i, inv);
    return inv ? arma::mat(Ti.t() * Di * Ti) : arma::mat(Ti * Di * Ti.t());
  }

  arma::mat get_D(arma::uword i, bool inv = false) const override {
    arma::vec elem = inv ? -get_Zlmd(i) : get_Zlmd(i);
    return arma::diagmat(arma::exp(elem));
  }
  arma::mat get_T(arma::uword i, bool inv = false) const override {
    if (m_(i) == 1) return arma::eye(m_(i), m_(i));
    arma::mat Ti = get_ltrimatrix(m_(i), -get_Wgma(i), false);
    return inv ? arma::pinv(Ti) : Ti;
  }

 private:
  arma::mat G_;
  arma::vec TResid_;

  arma::mat get_G(arma::uword i) const {
    return G_.rows(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }
  arma::vec get_TResid(arma::uword i) const {
    return TResid_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1);
  }

  void UpdateG();
  void UpdateTResid();

};  // class MCD

inline MCD::MCD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
                const arma::mat& Z, const arma::mat& W)
    : JmcmBase(m, Y, X, Z, W, 0),
      G_(N_, n_gma_, arma::fill::zeros),
      TResid_(N_, arma::fill::zeros) {}

inline void MCD::UpdateGamma() {
  arma::mat GDG = arma::zeros<arma::mat>(n_gma_, n_gma_);
  arma::vec GDr = arma::zeros<arma::vec>(n_gma_);

  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Di_inv = get_D(i, true);
    arma::mat Gi = get_G(i);
    arma::mat Gi_Di_inv = Gi.t() * Di_inv;

    GDG += Gi_Di_inv * Gi;
    GDr += Gi_Di_inv * ri;
  }

  set_param(GDG.i() * GDr, 3);
}

// clang-format off
inline void MCD::UpdateModel() {
  switch (get_free_param()) {
    case 0 :
    case 1 : { UpdateG(); UpdateTResid(); break; }
    case 2 : { break; }
    case 3 : { UpdateTResid(); break; }
    default: { arma::get_cerr_stream() << "Wrong value for free_param_" << std::endl; }
  }
}
// clang-format on

inline arma::vec MCD::Grad2() const {
  arma::vec grad2 = arma::zeros<arma::vec>(n_lmd_);
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);
    arma::mat Di_inv = get_D(i, true);
    arma::vec ei = arma::pow(get_TResid(i), 2);

    grad2 += Zi.t() * (Di_inv * ei - one);
  }

  return (-grad2);  // 2 is cancelled with the 0.5 in the for loop
}

inline arma::vec MCD::Grad3() const {
  arma::vec grad3 = arma::zeros<arma::vec>(n_gma_);
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Gi = get_G(i);
    arma::mat Di_inv = get_D(i, true);
    arma::vec Tiri = get_TResid(i);

    grad3 += Gi.t() * (Di_inv * Tiri);
  }

  return (-2 * grad3);
}

inline void MCD::UpdateG() {
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Gi = arma::zeros<arma::mat>(m_(i), n_gma_);

    arma::mat Wi = get_W(i);
    arma::vec ri = get_Resid(i);
    for (arma::uword j = 1; j < m_(i); ++j) {
      arma::uword index = (j - 1) * j / 2;
      Gi.row(j) = ri.subvec(0, j - 1).t() * Wi.rows(index, index + j - 1);
    }

    G_.rows(cumsum_m_(i), cumsum_m_(i + 1) - 1) = Gi;
  }
}

inline void MCD::UpdateTResid() {
  for (arma::uword i = 0; i < n_sub_; ++i) {
    arma::mat Tiri = get_T(i) * get_Resid(i);
    TResid_.subvec(cumsum_m_(i), cumsum_m_(i + 1) - 1) = Tiri;
  }
}

}  // namespace jmcm

#endif  // _JMCM_MCD_H_
