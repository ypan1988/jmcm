// external.cpp: implementing some R functions in C++
//
// Copyright (C) 2015-2016 The University of Manchester
//
// Written by Yi Pan - ypan1988@gmail.com

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "bfgs.h"
#include "mcd.h"
#include "acd.h"
#include "hpc.h"

//'@title Fit Joint Mean-Covariance Models based on MCD
//'@description Fit joint mean-covariance models based on MCD.
//'@param m an integer vector of numbers of measurements for subject.
//'@param Y a vector of responses for all subjects.
//'@param X model matrix for the mean structure model.
//'@param Z model matrix for the diagonal matrix.
//'@param W model matrix for the lower triangular matrix.
//'@param start starting values for the parameters in the model.
//'@param trace the values of the objective function and the parameters are
//'       printed for all the trace'th iterations.
//'@param profile whether parameters should be estimated sequentially using the
//'       idea of profile likelihood or not.
//'@param errormsg whether or not the error message should be print.
//'@seealso \code{\link{acd_estimation}} for joint mean covariance model fitting
//'         based on ACD, \code{\link{hpc_estimation}} for joint mean covariance
//'         model fitting based on HPC.
//'@export
// [[Rcpp::export]]
Rcpp::List mcd_estimation(arma::vec m, arma::vec Y, arma::mat X, arma::mat Z,
                          arma::mat W, arma::vec start, bool trace = false,
                          bool profile = true, bool errormsg = false) {
  int debug = 0;
  int debug2 = 0;

  int n_bta = X.n_cols;
  int n_lmd = Z.n_cols;
  int n_gma = W.n_cols;

  jmcm::MCD mcd(m, Y, X, Z, W);
  pan::BFGS<jmcm::MCD> bfgs;
  pan::LineSearch<jmcm::MCD> linesearch;
  linesearch.set_message(errormsg);
  
  arma::vec x = start;

  double f_min = 0.0;
  int n_iters = 0;
  if (profile) {
    bfgs.set_trace(trace);
    bfgs.set_message(errormsg);

    if (debug) {
      Rcpp::Rcout << "Start profile opt ..." << std::endl;
      x.print("start value: ");
    }

    // Maximum number of iterations
    const int kIterMax = 200;

    // Machine precision
    const double kEpsilon = std::numeric_limits<double>::epsilon();

    // Convergence criterion on x values
    const double kTolX = 4 * kEpsilon;

    // Scaled maximum step length allowed in line searches
    const double kScaStepMax = 100;

    const double grad_tol = 1e-6;

    const int n_pars = x.n_rows;  // number of parameters

    double f = mcd(x);
    arma::vec grad;
    mcd.Gradient(x, grad);

    // Initialize the inverse Hessian to a unit matrix
    arma::mat hess_inv = arma::eye<arma::mat>(n_pars, n_pars);

    // Initialize Newton Step
    arma::vec p = -hess_inv * grad;

    // Calculate the maximum step length
    double sum = sqrt(arma::dot(x, x));
    const double kStepMax = kScaStepMax * std::max(sum, double(n_pars));

    if (debug) Rcpp::Rcout << "Before for loop" << std::endl;

    // Main loop over the iterations
    for (int iter = 0; iter != kIterMax; ++iter) {
      if (debug) Rcpp::Rcout << "iter " << iter << ":" << std::endl;

      n_iters = iter;

      arma::vec x2 = x;  // Save the old point

      linesearch.GetStep(mcd, x, p, kStepMax);

      f = mcd(x);  // Update function value
      p = x - x2;  // Update line direction
      x2 = x;
      f_min = f;

      if (trace) {
        Rcpp::Rcout << std::setw(5) << iter << ": " << std::setw(10) << mcd(x)
                    << ": ";
        x.t().print();
      }

      if (debug) Rcpp::Rcout << "Checking convergence..." << std::endl;
      // Test for convergence on Delta x
      double test = 0.0;
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(p(i)) / std::max(std::abs(x(i)), 1.0);
        if (temp > test) test = temp;
      }
      if (debug2) Rcpp::Rcout << "test1 = " << test << std::endl;
      if (test < kTolX) {
        if (debug)
          Rcpp::Rcout << "Test for convergence on Delta x: converged."
                      << std::endl;
        break;
      }

      arma::vec grad2 = grad;  // Save the old gradient
      mcd.Gradient(x, grad);   // Get the new gradient

      // Test for convergence on zero gradient
      test = 0.0;
      double den = std::max(f, 1.0);
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(grad(i)) * std::max(std::abs(x(i)), 1.0) / den;
        if (temp > test) test = temp;
      }
      if (debug2) Rcpp::Rcout << "test2 = " << test << std::endl;
      if (test < grad_tol) {
        if (debug)
          Rcpp::Rcout << "Test for convergence on zero gradient: converged."
                      << std::endl;
        break;
      }

      if (debug) Rcpp::Rcout << "Update beta..." << std::endl;
      mcd.UpdateBeta();

      if (debug) Rcpp::Rcout << "Update lambda..." << std::endl;
      arma::vec lmd = x.rows(n_bta, n_bta + n_lmd - 1);

      if (trace) {
        Rcpp::Rcout << "--------------------------------------------------"
                    << "\n Updating Innovation Variance Parameters..."
                    << std::endl;
      }

      mcd.set_free_param(2);
      bfgs.Optimize(mcd, lmd);
      mcd.set_free_param(0);

      if (trace) {
        Rcpp::Rcout << "--------------------------------------------------"
                    << std::endl;
      }

      mcd.UpdateLambda(lmd);

      if (debug) Rcpp::Rcout << "Update gamma..." << std::endl;
      mcd.UpdateGamma();

      if (debug) Rcpp::Rcout << "Update theta..." << std::endl;
      arma::vec xnew = mcd.get_theta();

      p = xnew - x;
    }
  } else {
    bfgs.set_trace(trace);
    bfgs.set_message(errormsg);
    bfgs.Optimize(mcd, x);
    f_min = bfgs.f_min();
    n_iters = bfgs.n_iters();
  }

  arma::vec beta = x.rows(0, n_bta - 1);
  arma::vec lambda = x.rows(n_bta, n_bta + n_lmd - 1);
  arma::vec gamma = x.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1);

  int n_par = n_bta + n_lmd + n_gma;
  int n_sub = m.n_rows;

  return Rcpp::List::create(
      Rcpp::Named("par") = x, Rcpp::Named("beta") = beta,
      Rcpp::Named("lambda") = lambda, Rcpp::Named("gamma") = gamma,
      Rcpp::Named("loglik") = -f_min / 2,
      Rcpp::Named("BIC") =
          f_min / n_sub + n_par * log(static_cast<double>(n_sub)) / n_sub,
      Rcpp::Named("iter") = n_iters);
}

//'@title Fit Joint Mean-Covariance Models based on ACD
//'@description Fit joint mean-covariance models based on ACD.
//'@param m an integer vector of numbers of measurements for subject.
//'@param Y a vector of responses for all subjects.
//'@param X model matrix for the mean structure model.
//'@param Z model matrix for the diagonal matrix.
//'@param W model matrix for the lower triangular matrix.
//'@param start starting values for the parameters in the model.
//'@param trace the values of the objective function and the parameters are
//'       printed for all the trace'th iterations.
//'@param profile whether parameters should be estimated sequentially using the
//'       idea of profile likelihood or not.
//'@param errormsg whether or not the error message should be print.
//'@seealso \code{\link{mcd_estimation}} for joint mean covariance model fitting
//'         based on MCD, \code{\link{hpc_estimation}} for joint mean covariance
//'         model fitting based on HPC.
//'@export
// [[Rcpp::export]]
Rcpp::List acd_estimation(arma::vec m, arma::vec Y, arma::mat X, arma::mat Z,
                          arma::mat W, arma::vec start, bool trace = false,
                          bool profile = true, bool errormsg = false) {
  int debug = 0;
  int debug2 = 0;

  int n_bta = X.n_cols;
  int n_lmd = Z.n_cols;
  int n_gma = W.n_cols;

  jmcm::ACD acd(m, Y, X, Z, W);
  pan::BFGS<jmcm::ACD> bfgs;
  pan::LineSearch<jmcm::ACD> linesearch;
  linesearch.set_message(errormsg);
  arma::vec x = start;

  double f_min = 0.0;
  int n_iters = 0;

  if (profile) {
    bfgs.set_trace(trace);
    bfgs.set_message(errormsg);

    if (debug) {
      Rcpp::Rcout << "Start profile opt ..." << std::endl;
      x.print("start value: ");
    }

    // Maximum number of iterations
    const int kIterMax = 200;

    // Machine precision
    const double kEpsilon = std::numeric_limits<double>::epsilon();

    // Convergence criterion on x values
    const double kTolX = 4 * kEpsilon;

    // Scaled maximum step length allowed in line searches
    const double kScaStepMax = 100;

    const double grad_tol = 1e-6;

    const int n_pars = x.n_rows;  // number of parameters

    double f = acd(x);
    arma::vec grad;
    acd.Gradient(x, grad);

    // Initialize the inverse Hessian to a unit matrix
    arma::mat hess_inv = arma::eye<arma::mat>(n_pars, n_pars);

    // Initialize Newton Step
    arma::vec p = -hess_inv * grad;

    // Calculate the maximum step length
    double sum = sqrt(arma::dot(x, x));
    const double kStepMax = kScaStepMax * std::max(sum, double(n_pars));

    if (debug) Rcpp::Rcout << "Before for loop" << std::endl;

    // Main loop over the iterations
    for (int iter = 0; iter != kIterMax; ++iter) {
      if (debug) Rcpp::Rcout << "iter " << iter << ":" << std::endl;

      n_iters = iter;

      arma::vec x2 = x;  // Save the old point

      linesearch.GetStep(acd, x, p, kStepMax);

      f = acd(x);  // Update function value
      p = x - x2;  // Update line direction
      x2 = x;
      f_min = f;

      if (trace) {
        Rcpp::Rcout << std::setw(5) << iter << ": " << std::setw(10) << acd(x)
                    << ": ";
        x.t().print();
      }

      if (debug) Rcpp::Rcout << "Checking convergence..." << std::endl;
      // Test for convergence on Delta x
      double test = 0.0;
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(p(i)) / std::max(std::abs(x(i)), 1.0);
        if (temp > test) test = temp;
      }
      if (debug2) Rcpp::Rcout << "test1 = " << test << std::endl;
      if (test < kTolX) {
        if (debug)
          Rcpp::Rcout << "Test for convergence on Delta x: converged."
                      << std::endl;
        break;
      }

      arma::vec grad2 = grad;  // Save the old gradient
      acd.Gradient(x, grad);   // Get the new gradient

      // Test for convergence on zero gradient
      test = 0.0;
      double den = std::max(f, 1.0);
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(grad(i)) * std::max(std::abs(x(i)), 1.0) / den;
        if (temp > test) test = temp;
      }
      if (debug2) Rcpp::Rcout << "test2 = " << test << std::endl;
      if (test < grad_tol) {
        if (debug)
          Rcpp::Rcout << "Test for convergence on zero gradient: converged."
                      << std::endl;
        break;
      }

      if (debug) Rcpp::Rcout << "Update beta..." << std::endl;
      acd.UpdateBeta();

      if (debug) Rcpp::Rcout << "Update lambda and gamma..." << std::endl;
      arma::vec lmdgma = x.rows(n_bta, n_bta + n_lmd + n_gma - 1);

      if (trace) {
        Rcpp::Rcout << "--------------------------------------------------"
                    << "\n Updating Innovation Variance Parameters"
                    << " and Moving Average Parameters..." << std::endl;
      }
      acd.set_free_param(2);
      bfgs.Optimize(acd, lmdgma);
      acd.set_free_param(0);
      if (trace) {
        Rcpp::Rcout << "--------------------------------------------------"
                    << std::endl;
      }

      acd.UpdateLambdaGamma(lmdgma);

      if (debug) Rcpp::Rcout << "Update theta..." << std::endl;
      arma::vec xnew = acd.get_theta();

      p = xnew - x;
    }
  } else {
    bfgs.set_trace(trace);
    bfgs.set_message(errormsg);
    bfgs.Optimize(acd, x);
    f_min = bfgs.f_min();
    n_iters = bfgs.n_iters();
  }

  arma::vec beta = x.rows(0, n_bta - 1);
  arma::vec lambda = x.rows(n_bta, n_bta + n_lmd - 1);
  arma::vec gamma = x.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1);

  int n_par = n_bta + n_lmd + n_gma;
  int n_sub = m.n_rows;

  return Rcpp::List::create(
      Rcpp::Named("par") = x, Rcpp::Named("beta") = beta,
      Rcpp::Named("lambda") = lambda, Rcpp::Named("gamma") = gamma,
      Rcpp::Named("loglik") = -f_min / 2,
      Rcpp::Named("BIC") =
          f_min / n_sub + n_par * log(static_cast<double>(n_sub)) / n_sub,
      Rcpp::Named("iter") = n_iters);
}

//'@title Fit Joint Mean-Covariance Models based on HPC
//'@description Fit joint mean-covariance models based on HPC.
//'@param m an integer vector of numbers of measurements for subject.
//'@param Y a vector of responses for all subjects.
//'@param X model matrix for the mean structure model.
//'@param Z model matrix for the diagonal matrix.
//'@param W model matrix for the lower triangular matrix.
//'@param start starting values for the parameters in the model.
//'@param trace the values of the objective function and the parameters are
//'       printed for all the trace'th iterations.
//'@param profile whether parameters should be estimated sequentially using the
//'       idea of profile likelihood or not.
//'@param errormsg whether or not the error message should be print.
//'@seealso \code{\link{mcd_estimation}} for joint mean covariance model fitting
//'         based on MCD, \code{\link{acd_estimation}} for joint mean covariance
//'         model fitting based on ACD.
//'@export
// [[Rcpp::export]]
Rcpp::List hpc_estimation(arma::vec m, arma::vec Y, arma::mat X, arma::mat Z,
                          arma::mat W, arma::vec start, bool trace = false,
                          bool profile = true, bool errormsg = false) {
  int debug = 0;
  int debug2 = 0;

  int n_bta = X.n_cols;
  int n_lmd = Z.n_cols;
  int n_gma = W.n_cols;

  jmcm::HPC hpc(m, Y, X, Z, W);
  pan::BFGS<jmcm::HPC> bfgs;
  pan::LineSearch<jmcm::HPC> linesearch;
  linesearch.set_message(errormsg);
  arma::vec x = start;

  double f_min = 0.0;
  int n_iters = 0;
  if (profile) {
    bfgs.set_trace(trace);
    bfgs.set_message(errormsg);

    if (debug) {
      Rcpp::Rcout << "Start profile opt ..." << std::endl;
      x.print("start value: ");
    }

    // Maximum number of iterations
    const int kIterMax = 200;

    // Machine precision
    const double kEpsilon = std::numeric_limits<double>::epsilon();

    // Convergence criterion on x values
    const double kTolX = 4 * kEpsilon;

    // Scaled maximum step length allowed in line searches
    const double kScaStepMax = 100;

    const double grad_tol = 1e-6;

    const int n_pars = x.n_rows;  // number of parameters

    double f = hpc(x);
    arma::vec grad;
    hpc.Gradient(x, grad);

    // Initialize the inverse Hessian to a unit matrix
    arma::mat hess_inv = arma::eye<arma::mat>(n_pars, n_pars);

    // Initialize Newton Step
    arma::vec p = -hess_inv * grad;

    // Calculate the maximum step length
    double sum = sqrt(arma::dot(x, x));
    const double kStepMax = kScaStepMax * std::max(sum, double(n_pars));

    if (debug) Rcpp::Rcout << "Before for loop" << std::endl;

    // Main loop over the iterations
    for (int iter = 0; iter != kIterMax; ++iter) {
      if (debug) Rcpp::Rcout << "iter " << iter << ":" << std::endl;

      n_iters = iter;

      arma::vec x2 = x;  // Save the old point

      linesearch.GetStep(hpc, x, p, kStepMax);

      f = hpc(x);  // Update function value
      p = x - x2;  // Update line direction
      x2 = x;
      f_min = f;

      if (trace) {
        Rcpp::Rcout << std::setw(5) << iter << ": " << std::setw(10) << hpc(x)
                    << ": ";
        x.t().print();
      }

      if (debug) Rcpp::Rcout << "Checking convergence..." << std::endl;
      // Test for convergence on Delta x
      double test = 0.0;
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(p(i)) / std::max(std::abs(x(i)), 1.0);
        if (temp > test) test = temp;
      }
      if (debug2) Rcpp::Rcout << "test1 = " << test << std::endl;
      if (test < kTolX) {
        if (debug)
          Rcpp::Rcout << "Test for convergence on Delta x: converged."
                      << std::endl;
        break;
      }

      arma::vec grad2 = grad;  // Save the old gradient
      hpc.Gradient(x, grad);   // Get the new gradient

      // Test for convergence on zero gradient
      test = 0.0;
      double den = std::max(f, 1.0);
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(grad(i)) * std::max(std::abs(x(i)), 1.0) / den;
        if (temp > test) test = temp;
      }
      if (debug2) Rcpp::Rcout << "test2 = " << test << std::endl;
      if (test < grad_tol) {
        if (debug)
          Rcpp::Rcout << "Test for convergence on zero gradient: converged."
                      << std::endl;
        break;
      }

      if (debug) Rcpp::Rcout << "Update beta..." << std::endl;
      hpc.UpdateBeta();

      if (debug) Rcpp::Rcout << "Update lambda and gamma..." << std::endl;
      arma::vec lmdgma = x.rows(n_bta, n_bta + n_lmd + n_gma - 1);

      if (trace) {
        Rcpp::Rcout << "--------------------------------------------------"
                    << "\n Updating Variance Parameters"
                    << " and Angle Parameters..." << std::endl;
      }
      hpc.set_free_param(2);
      bfgs.Optimize(hpc, lmdgma);
      hpc.set_free_param(0);
      if (trace) {
        Rcpp::Rcout << "--------------------------------------------------"
                    << std::endl;
      }

      hpc.UpdateLambdaGamma(lmdgma);

      if (debug) Rcpp::Rcout << "Update theta..." << std::endl;
      arma::vec xnew = hpc.get_theta();

      p = xnew - x;
    }
  } else {
    bfgs.set_trace(trace);
    bfgs.set_message(errormsg);
    bfgs.Optimize(hpc, x);
    f_min = bfgs.f_min();
    n_iters = bfgs.n_iters();
  }

  arma::vec beta = x.rows(0, n_bta - 1);
  arma::vec lambda = x.rows(n_bta, n_bta + n_lmd - 1);
  arma::vec gamma = x.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1);

  int n_par = X.n_cols + Z.n_cols + W.n_cols;
  int n_sub = m.n_rows;

  return Rcpp::List::create(
      Rcpp::Named("par") = x, Rcpp::Named("beta") = beta,
      Rcpp::Named("lambda") = lambda, Rcpp::Named("gamma") = gamma,
      Rcpp::Named("loglik") = -f_min / 2,
      Rcpp::Named("BIC") =
          f_min / n_sub + n_par * log(static_cast<double>(n_sub)) / n_sub,
      Rcpp::Named("iter") = n_iters);
}

RcppExport SEXP MCD__new(SEXP m_, SEXP Y_, SEXP X_, SEXP Z_, SEXP W_) {
  arma::vec m = Rcpp::as<arma::vec>(m_);
  arma::vec Y = Rcpp::as<arma::vec>(Y_);
  arma::mat X = Rcpp::as<arma::mat>(X_);
  arma::mat Z = Rcpp::as<arma::mat>(Z_);
  arma::mat W = Rcpp::as<arma::mat>(W_);

  Rcpp::XPtr<jmcm::MCD> ptr(new jmcm::MCD(m, Y, X, Z, W), true);

  return ptr;
}

RcppExport SEXP MCD__get_m(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_m(i));
}

RcppExport SEXP MCD__get_Y(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_Y(i));
}

RcppExport SEXP MCD__get_X(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_X(i));
}

RcppExport SEXP MCD__get_Z(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_Z(i));
}

RcppExport SEXP MCD__get_W(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_W(i));
}

RcppExport SEXP MCD__get_D(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  ptr->UpdateMCD(x);

  return Rcpp::wrap(ptr->get_D(i));
}

RcppExport SEXP MCD__get_T(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  ptr->UpdateMCD(x);

  return Rcpp::wrap(ptr->get_T(i));
}

RcppExport SEXP MCD__get_Sigma(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  arma::mat Sigmai;

  ptr->UpdateMCD(x);

  return Rcpp::wrap(ptr->get_Sigma(i));
}

RcppExport SEXP MCD__n2loglik(SEXP xp, SEXP x_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);

  double result = ptr->operator()(x);

  return Rcpp::wrap(result);
}

RcppExport SEXP MCD__grad(SEXP xp, SEXP x_) {
  Rcpp::XPtr<jmcm::MCD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);

  arma::vec grad;

  ptr->Gradient(x, grad);

  return Rcpp::wrap(grad);
}

RcppExport SEXP ACD__new(SEXP m_, SEXP Y_, SEXP X_, SEXP Z_, SEXP W_) {
  arma::vec m = Rcpp::as<arma::vec>(m_);
  arma::vec Y = Rcpp::as<arma::vec>(Y_);
  arma::mat X = Rcpp::as<arma::mat>(X_);
  arma::mat Z = Rcpp::as<arma::mat>(Z_);
  arma::mat W = Rcpp::as<arma::mat>(W_);

  Rcpp::XPtr<jmcm::ACD> ptr(new jmcm::ACD(m, Y, X, Z, W), true);

  return ptr;
}

RcppExport SEXP ACD__get_m(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_m(i));
}

RcppExport SEXP ACD__get_Y(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_Y(i));
}

RcppExport SEXP ACD__get_X(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_X(i));
}

RcppExport SEXP ACD__get_Z(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_Z(i));
}

RcppExport SEXP ACD__get_W(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_W(i));
}

RcppExport SEXP ACD__get_D(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  ptr->UpdateACD(x);

  return Rcpp::wrap(ptr->get_D(i));
}

RcppExport SEXP ACD__get_T(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  ptr->UpdateACD(x);

  return Rcpp::wrap(ptr->get_T(i));
}

RcppExport SEXP ACD__get_Sigma(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  arma::mat Sigmai;

  ptr->UpdateACD(x);

  return Rcpp::wrap(ptr->get_Sigma(i));
}

RcppExport SEXP ACD__n2loglik(SEXP xp, SEXP x_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);

  double result = ptr->operator()(x);

  return Rcpp::wrap(result);
}

RcppExport SEXP ACD__grad(SEXP xp, SEXP x_) {
  Rcpp::XPtr<jmcm::ACD> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);

  arma::vec grad;

  ptr->Gradient(x, grad);

  return Rcpp::wrap(grad);
}

RcppExport SEXP HPC__new(SEXP m_, SEXP Y_, SEXP X_, SEXP Z_, SEXP W_) {
  arma::vec m = Rcpp::as<arma::vec>(m_);
  arma::vec Y = Rcpp::as<arma::vec>(Y_);
  arma::mat X = Rcpp::as<arma::mat>(X_);
  arma::mat Z = Rcpp::as<arma::mat>(Z_);
  arma::mat W = Rcpp::as<arma::mat>(W_);

  Rcpp::XPtr<jmcm::HPC> ptr(new jmcm::HPC(m, Y, X, Z, W), true);

  return ptr;
}

RcppExport SEXP HPC__get_m(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_m(i));
}

RcppExport SEXP HPC__get_Y(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_Y(i));
}

RcppExport SEXP HPC__get_X(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_X(i));
}

RcppExport SEXP HPC__get_Z(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_Z(i));
}

RcppExport SEXP HPC__get_W(SEXP xp, SEXP i_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_W(i));
}

RcppExport SEXP HPC__get_D(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  ptr->UpdateHPC(x);

  return Rcpp::wrap(ptr->get_D(i));
}

RcppExport SEXP HPC__get_T(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  ptr->UpdateHPC(x);

  return Rcpp::wrap(ptr->get_T(i));
}

RcppExport SEXP HPC__get_Sigma(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  arma::mat Sigmai;

  ptr->UpdateHPC(x);

  return Rcpp::wrap(ptr->get_Sigma(i));
}

RcppExport SEXP HPC__n2loglik(SEXP xp, SEXP x_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);

  double result = ptr->operator()(x);

  return Rcpp::wrap(result);
}

RcppExport SEXP HPC__grad(SEXP xp, SEXP x_) {
  Rcpp::XPtr<jmcm::HPC> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);

  arma::vec grad;

  ptr->Gradient(x, grad);

  return Rcpp::wrap(grad);
}
