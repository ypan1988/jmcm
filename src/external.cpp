//  external.cpp: externally .Call'able functions in jmcm
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

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// clang-format off
#include "mcd.h"
#include "acd.h"
#include "hpc.h"
#include "jmcm_fit.h"
// clang-format on

template <typename JMCM>
Rcpp::List jmcm_estimation(arma::uvec m, arma::vec Y, arma::mat X, arma::mat Z,
                           arma::mat W, arma::vec start, arma::vec mean,
                           bool trace = false, bool profile = true,
                           bool errormsg = false, bool covonly = false,
                           std::string optim_method = "default") {
  JmcmFit<JMCM> fit(m, Y, X, Z, W, start, mean, trace, profile, errormsg,
                    covonly, optim_method);
  arma::vec x = fit.Optimize();
  double f_min = fit.get_f_min();
  arma::uword n_iters = fit.get_n_iters();

  arma::vec beta = fit.jmcm_.get_param(1);
  arma::vec lambda = fit.jmcm_.get_param(2);
  arma::vec gamma = fit.jmcm_.get_param(3);

  int n_par = x.n_elem;
  int n_sub = m.n_rows;

  return Rcpp::List::create(
      Rcpp::Named("par") = x, Rcpp::Named("beta") = beta,
      Rcpp::Named("lambda") = lambda, Rcpp::Named("gamma") = gamma,
      Rcpp::Named("loglik") = -f_min / 2,
      Rcpp::Named("BIC") =
          f_min / n_sub + n_par * log(static_cast<double>(n_sub)) / n_sub,
      Rcpp::Named("iter") = n_iters);
}

//'@title Fit Joint Mean-Covariance Models based on MCD
//'@description Fit joint mean-covariance models based on MCD.
//'@param m an integer vector of numbers of measurements for subject.
//'@param Y a vector of responses for all subjects.
//'@param X model matrix for the mean structure model.
//'@param Z model matrix for the diagonal matrix.
//'@param W model matrix for the lower triangular matrix.
//'@param start starting values for the parameters in the model.
//'@param mean when covonly is true, it is used as the given mean.
//'@param trace the values of the objective function and the parameters are
//'       printed for all the trace'th iterations.
//'@param profile whether parameters should be estimated sequentially using the
//'       idea of profile likelihood or not.
//'@param errormsg whether or not the error message should be print.
//'@param covonly estimate the covariance structure only, and use given mean.
//'@param optim_method optimization method, choose "default" or "BFGS"(vmmin in
//'       R).
//'@seealso \code{\link{acd_estimation}} for joint mean covariance model fitting
//'         based on ACD, \code{\link{hpc_estimation}} for joint mean covariance
//'         model fitting based on HPC.
//'@export
// [[Rcpp::export]]
Rcpp::List mcd_estimation(arma::uvec m, arma::vec Y, arma::mat X, arma::mat Z,
                          arma::mat W, arma::vec start, arma::vec mean,
                          bool trace = false, bool profile = true,
                          bool errormsg = false, bool covonly = false,
                          std::string optim_method = "default") {
  return jmcm_estimation<jmcm::MCD>(m, Y, X, Z, W, start, mean, trace, profile,
                                    errormsg, covonly);
}

//'@title Fit Joint Mean-Covariance Models based on ACD
//'@description Fit joint mean-covariance models based on ACD.
//'@param m an integer vector of numbers of measurements for subject.
//'@param Y a vector of responses for all subjects.
//'@param X model matrix for the mean structure model.
//'@param Z model matrix for the diagonal matrix.
//'@param W model matrix for the lower triangular matrix.
//'@param start starting values for the parameters in the model.
//'@param mean when covonly is true, it is used as the given mean.
//'@param trace the values of the objective function and the parameters are
//'       printed for all the trace'th iterations.
//'@param profile whether parameters should be estimated sequentially using the
//'       idea of profile likelihood or not.
//'@param errormsg whether or not the error message should be print.
//'@param covonly estimate the covariance structure only, and use given mean.
//'@param optim_method optimization method, choose "default" or "BFGS"(vmmin in
//'       R).
//'@seealso \code{\link{mcd_estimation}} for joint mean covariance model fitting
//'         based on MCD, \code{\link{hpc_estimation}} for joint mean covariance
//'         model fitting based on HPC.
//'@export
// [[Rcpp::export]]
Rcpp::List acd_estimation(arma::uvec m, arma::vec Y, arma::mat X, arma::mat Z,
                          arma::mat W, arma::vec start, arma::vec mean,
                          bool trace = false, bool profile = true,
                          bool errormsg = false, bool covonly = false,
                          std::string optim_method = "default") {
  return jmcm_estimation<jmcm::ACD>(m, Y, X, Z, W, start, mean, trace, profile,
                                    errormsg, covonly);
}

//'@title Fit Joint Mean-Covariance Models based on HPC
//'@description Fit joint mean-covariance models based on HPC.
//'@param m an integer vector of numbers of measurements for subject.
//'@param Y a vector of responses for all subjects.
//'@param X model matrix for the mean structure model.
//'@param Z model matrix for the diagonal matrix.
//'@param W model matrix for the lower triangular matrix.
//'@param start starting values for the parameters in the model.
//'@param mean when covonly is true, it is used as the given mean.
//'@param trace the values of the objective function and the parameters are
//'       printed for all the trace'th iterations.
//'@param profile whether parameters should be estimated sequentially using the
//'       idea of profile likelihood or not.
//'@param errormsg whether or not the error message should be print.
//'@param covonly estimate the covariance structure only, and use given mean.
//'@param optim_method optimization method, choose "default" or "BFGS"(vmmin in
//'       R).
//'@seealso \code{\link{mcd_estimation}} for joint mean covariance model fitting
//'         based on MCD, \code{\link{acd_estimation}} for joint mean covariance
//'         model fitting based on ACD.
//'@export
// [[Rcpp::export]]
Rcpp::List hpc_estimation(arma::uvec m, arma::vec Y, arma::mat X, arma::mat Z,
                          arma::mat W, arma::vec start, arma::vec mean,
                          bool trace = false, bool profile = true,
                          bool errormsg = false, bool covonly = false,
                          std::string optim_method = "default") {
  return jmcm_estimation<jmcm::HPC>(m, Y, X, Z, W, start, mean, trace, profile,
                                    errormsg, covonly);
}

template <typename JMCM>
SEXP JMCM__new(SEXP m_, SEXP Y_, SEXP X_, SEXP Z_, SEXP W_) {
  arma::uvec m = Rcpp::as<arma::uvec>(m_);
  arma::vec Y = Rcpp::as<arma::vec>(Y_);
  arma::mat X = Rcpp::as<arma::mat>(X_);
  arma::mat Z = Rcpp::as<arma::mat>(Z_);
  arma::mat W = Rcpp::as<arma::mat>(W_);

  Rcpp::XPtr<jmcm::JmcmBase> ptr(new JMCM(m, Y, X, Z, W), true);

  return ptr;
}

RcppExport SEXP MCD__new(SEXP m_, SEXP Y_, SEXP X_, SEXP Z_, SEXP W_) {
  return JMCM__new<jmcm::MCD>(m_, Y_, X_, Z_, W_);
}

RcppExport SEXP ACD__new(SEXP m_, SEXP Y_, SEXP X_, SEXP Z_, SEXP W_) {
  return JMCM__new<jmcm::ACD>(m_, Y_, X_, Z_, W_);
}

RcppExport SEXP HPC__new(SEXP m_, SEXP Y_, SEXP X_, SEXP Z_, SEXP W_) {
  return JMCM__new<jmcm::HPC>(m_, Y_, X_, Z_, W_);
}

RcppExport SEXP get_m(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;
  return Rcpp::wrap(ptr->get_m(i));
}

RcppExport SEXP get_Y(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;
  return Rcpp::wrap(ptr->get_Y(i));
}

RcppExport SEXP get_X(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;
  return Rcpp::wrap(ptr->get_X(i));
}

RcppExport SEXP get_Z(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;
  return Rcpp::wrap(ptr->get_Z(i));
}

RcppExport SEXP get_W(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;
  return Rcpp::wrap(ptr->get_W(i));
}

RcppExport SEXP get_D(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;
  ptr->UpdateJmcm(x);
  return Rcpp::wrap(ptr->get_D(i));
}

RcppExport SEXP get_T(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;
  ptr->UpdateJmcm(x);
  return Rcpp::wrap(ptr->get_T(i));
}

RcppExport SEXP get_mu(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;
  ptr->UpdateJmcm(x);
  return Rcpp::wrap(ptr->get_mu(i));
}

RcppExport SEXP get_Sigma(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;
  ptr->UpdateJmcm(x);
  return Rcpp::wrap(ptr->get_Sigma(i));
}

RcppExport SEXP n2loglik(SEXP xp, SEXP x_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  arma::vec x = Rcpp::as<arma::vec>(x_);
  double result = ptr->operator()(x);
  return Rcpp::wrap(result);
}

RcppExport SEXP grad(SEXP xp, SEXP x_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  arma::vec x = Rcpp::as<arma::vec>(x_);
  arma::vec grad;
  ptr->Gradient(x, grad);
  return Rcpp::wrap(grad);
}

RcppExport SEXP hess(SEXP xp, SEXP x_) {
  Rcpp::XPtr<jmcm::JmcmBase> ptr(xp);
  arma::vec x = Rcpp::as<arma::vec>(x_);
  arma::mat hess;
  ptr->Hessian(x, hess);
  return Rcpp::wrap(hess);
}
