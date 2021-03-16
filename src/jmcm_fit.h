//  jmcm_fit.h: model fitting for three joint mean-covariance models
//
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

#ifndef JMCM_SRC_JMCM_FIT_H_
#define JMCM_SRC_JMCM_FIT_H_

#include <string>

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>

#include "bfgs.h"
#include "roptim.h"

template <typename JMCM>
class JmcmFit {
 public:
  JmcmFit() = delete;
  JmcmFit(const JmcmFit&) = delete;
  ~JmcmFit() = default;

  JmcmFit(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
          const arma::mat& Z, const arma::mat& W, arma::vec start,
          arma::vec mean, bool trace = false, bool profile = true,
          bool errormsg = false, bool covonly = false,
          std::string optim_method = "default")
      : jmcm_(m, Y, X, Z, W),
        start_(start),
        mean_(mean),
        trace_(trace),
        profile_(profile),
        errormsg_(errormsg),
        covonly_(covonly),
        optim_method_(optim_method) {
    method_id_ = jmcm_.get_method_id();
    f_min_ = 0.0;
    n_iters_ = 0;
  }

  arma::vec Optimize();
  double get_f_min() const { return f_min_; }
  arma::uword get_n_iters() const { return n_iters_; }

 private:
  JMCM jmcm_;
  arma::uword method_id_;
  arma::vec start_, mean_;
  bool trace_, profile_, errormsg_, covonly_;
  std::string optim_method_;
  const std::string line_ =
      "--------------------------------------------------";

  double f_min_;
  arma::uword n_iters_;
};

// clang-format off
template <typename JMCM>
arma::vec JmcmFit<JMCM>::Optimize() {
  int n_bta = jmcm_.n_bta_;
  int n_lmd = jmcm_.n_lmd_;
  int n_gma = jmcm_.n_gma_;

  if (covonly_) {
    if ((jmcm_.N_ != mean_.n_rows) && errormsg_)
      Rcpp::Rcerr << "The size of the responses Y does not match the size of "
                     "the given mean"
                  << std::endl;
    jmcm_.set_mean(mean_);
  }

  pan::BFGS<JMCM> bfgs;
  bfgs.set_trace(trace_);
  bfgs.set_message(errormsg_);

  roptim::Roptim<JMCM> optim("BFGS");
  optim.control.trace = trace_;

  arma::vec x = start_;

  if (profile_) {
    const int n_pars = x.n_rows;  // number of parameters

    double f = jmcm_(x);
    arma::vec grad;
    jmcm_.Gradient(x, grad);

    // Initialize the inverse Hessian to a unit matrix
    arma::mat hess_inv = arma::eye<arma::mat>(n_pars, n_pars);

    // Initialize Newton Step
    arma::vec h = -hess_inv * grad;

    // Calculate the maximum step length
    double sum = sqrt(arma::dot(x, x));
    const double delta = bfgs.kScaStepMax_ * std::max(sum, double(n_pars));

    // Main loop over the iterations
    for (int iter = 0; iter != bfgs.kIterMax_; ++iter) {
      n_iters_ = iter;

      double h_norm = arma::norm(h, 2);
      if (h_norm > delta) h *= delta / h_norm;
      double alpha = bfgs.linesearch(jmcm_, x, h, f, grad);

      arma::vec x2 = x;  // Save the old point
      x += alpha * h;

      f = jmcm_(x);  // Update function value
      h = x - x2;    // Update line direction
      x2 = x;
      f_min_ = f;

      if (trace_) {
        Rcpp::Rcout << std::setw(5) << iter << ": " << std::setw(10) << jmcm_(x) << ": ";
        x.t().print();
      }

      // Test for convergence on Delta x
      if (bfgs.test_diff_x(x, h)) break;

      arma::vec grad2 = grad;   // Save the old gradient
      jmcm_.Gradient(x, grad);  // Get the new gradient

      // Test for convergence on zero gradient
      if (bfgs.test_grad(x, f, grad)) break;

      if (!covonly_) jmcm_.UpdateBeta();

      arma::vec param = (method_id_ == 0) ? x.rows(n_bta, n_bta + n_lmd - 1) : x.rows(n_bta, n_bta + n_lmd + n_gma - 1);
      if (trace_) {
        std::string str = "Updating ";
        switch (method_id_) {
          case 0: { str += "Innovation Variance "; break; }
          case 1: { str += "Innovation Variance and Moving Average "; break; }
          case 2: { str += "Variance and Angle "; break;}
          default:;
        }
        str += "Parameters...";
        Rcpp::Rcout << line_ << "\n" << str << std::endl;
      }

      arma::uword free_param = (method_id_ == 0) ? 2 : 23;
      jmcm_.set_free_param(free_param);
      optim_method_ == "default" ? bfgs.minimize(jmcm_, param) : optim.minimize(jmcm_, param);
      jmcm_.set_free_param(0);

      if (trace_) Rcpp::Rcout << line_ << std::endl;

      if (method_id_ == 0) {
        jmcm_.set_param(param, 2);
        jmcm_.UpdateGamma();
      } else {
        jmcm_.set_param(param, 23);
      }

      arma::vec xnew = jmcm_.get_param(0);
      h = xnew - x;
    }
  } else {
    if (optim_method_ == "default") {
      bfgs.minimize(jmcm_, x);
      f_min_ = bfgs.f_min();
      n_iters_ = bfgs.n_iters();
    } else {
      optim.minimize(jmcm_, x);
      f_min_ = optim.value();
    }
  }

  return x;
}
// clang-format on

#endif  // JMCM_SRC_JMCM_FIT_H_