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
  bfgs.set_message(errormsg_);
  roptim::Roptim<JMCM> optim;

  if (optim_method_ == "default") {
  } else if (optim_method_ == "Nelder-Mead" || optim_method_ == "BFGS" ||
             optim_method_ == "CG" || optim_method_ == "L-BFGS-B") {
    optim.set_method(optim_method_);
  }

  arma::vec x = start_;

  if (profile_) {
    bfgs.set_trace(trace_);
    bfgs.set_message(errormsg_);

    optim.control.trace = trace_;

    const int n_pars = x.n_rows;  // number of parameters

    double f = jmcm_(x);
    arma::vec grad;
    jmcm_.Gradient(x, grad);

    // Initialize the inverse Hessian to a unit matrix
    arma::mat hess_inv = arma::eye<arma::mat>(n_pars, n_pars);

    // Initialize Newton Step
    arma::vec p = -hess_inv * grad;

    // Calculate the maximum step length
    double sum = sqrt(arma::dot(x, x));
    const double delta = bfgs.kScaStepMax_ * std::max(sum, double(n_pars));

    // Main loop over the iterations
    for (int iter = 0; iter != bfgs.kIterMax_; ++iter) {
      n_iters_ = iter;

      arma::vec x2 = x;  // Save the old point

      double h_norm = arma::norm(p, 2);
      if (h_norm > delta) p *= delta / h_norm;
      bfgs.linesearch(jmcm_, x, p, f, grad);

      f = jmcm_(x);  // Update function value
      p = x - x2;    // Update line direction
      x2 = x;
      f_min_ = f;

      if (trace_) {
        Rcpp::Rcout << std::setw(5) << iter << ": " << std::setw(10) << jmcm_(x)
                    << ": ";
        x.t().print();
      }

      // Test for convergence on Delta x
      if (bfgs.test_diff_x(x, p)) break;

      arma::vec grad2 = grad;   // Save the old gradient
      jmcm_.Gradient(x, grad);  // Get the new gradient

      // Test for convergence on zero gradient
      if (bfgs.test_grad(x, f, grad)) break;

      if (!covonly_) jmcm_.UpdateBeta();

      if (method_id_ == 0) {
        arma::vec lmd = x.rows(n_bta, n_bta + n_lmd - 1);

        if (trace_) {
          Rcpp::Rcout << line_
                      << "\n Updating Innovation Variance Parameters..."
                      << std::endl;
        }

        jmcm_.set_free_param(2);
        if (optim_method_ == "default")
          bfgs.minimize(jmcm_, lmd);
        else
          optim.minimize(jmcm_, lmd);
        jmcm_.set_free_param(0);

        if (trace_) {
          Rcpp::Rcout << line_ << std::endl;
        }

        jmcm_.UpdateLambda(lmd);
        jmcm_.UpdateGamma();

      } else if (method_id_ == 1 || method_id_ == 2) {
        arma::vec lmdgma = x.rows(n_bta, n_bta + n_lmd + n_gma - 1);

        if (trace_) {
          switch (method_id_) {
            case 1: {
              Rcpp::Rcout << line_
                          << "\n Updating Innovation Variance Parameters"
                          << " and Moving Average Parameters..." << std::endl;
              break;
            }
            case 2: {
              Rcpp::Rcout << line_ << "\n Updating Variance Parameters"
                          << " and Angle Parameters..." << std::endl;
              break;
            }
            default: {
            }
          }
        }
        jmcm_.set_free_param(23);
        if (optim_method_ == "default")
          bfgs.minimize(jmcm_, lmdgma);
        else
          optim.minimize(jmcm_, lmdgma);
        jmcm_.set_free_param(0);
        if (trace_) {
          Rcpp::Rcout << line_ << std::endl;
        }

        jmcm_.UpdateLambdaGamma(lmdgma);
      }

      arma::vec xnew = jmcm_.get_param(0);

      p = xnew - x;
    }
  } else {
    if (optim_method_ == "default") {
      bfgs.set_trace(trace_);
      bfgs.set_message(errormsg_);
      bfgs.minimize(jmcm_, x);
      f_min_ = bfgs.f_min();
      n_iters_ = bfgs.n_iters();
    } else {
      optim.control.trace = trace_;
      optim.minimize(jmcm_, x);
      f_min_ = optim.value();
    }
  }

  return x;
}

#endif  // JMCM_SRC_JMCM_FIT_H_