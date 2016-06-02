// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// hpc.h: implementation of joint mean-covariance models based on standard
//        Cholesky decomposition of the correlation matrix R and the
//        hyperspherical parametrization(HPC) of its Cholesky factor
//
// Copyright (C) 2015 Yi Pan and Jianxin Pan
//
// This file is part of jmcm.

#include <cmath>
#include "hpc.h"

namespace jmcm {

HPC::HPC(arma::vec &m, arma::vec &Y, arma::mat &X, arma::mat &Z, arma::mat &W)
    : m_(m), Y_(Y), X_(X), Z_(Z), W_(W) {
  int debug = 0;

  if (debug) Rcpp::Rcout << "Creating HPC object" << std::endl;

  int N = Y_.n_rows;
  int n_bta = X_.n_cols;
  int n_lmd = Z_.n_cols;
  int n_gma = W_.n_cols;

  theta_ = arma::zeros<arma::vec>(n_bta + n_lmd + n_gma);
  beta_ = arma::zeros<arma::vec>(n_bta);
  lambda_ = arma::zeros<arma::vec>(n_lmd);
  gamma_ = arma::zeros<arma::vec>(n_gma);
  lmdgma_ = arma::zeros<arma::vec>(n_lmd + n_gma);

  Xbta_ = arma::zeros<arma::vec>(N);
  Zlmd_ = arma::zeros<arma::vec>(N);
  Wgma_ = arma::zeros<arma::vec>(W_.n_rows);
  Telem_ = arma::zeros<arma::vec>(W_.n_rows + arma::sum(m_));
  invTelem_ = arma::zeros<arma::vec>(W_.n_rows + arma::sum(m_));
  Resid_ = arma::zeros<arma::vec>(N);

  TDResid_ = arma::zeros<arma::vec>(N);
  TDResid2_ = arma::zeros<arma::vec>(N);

  free_param_ = 0;

  if (debug) Rcpp::Rcout << "HPC object created" << std::endl;
}

HPC::~HPC() {}

double HPC::operator()(const arma::vec &x) {
  UpdateHPC(x);

  int i, n_sub = m_.n_elem;
  double result = 0.0;

  //#pragma omp parallel for reduction(+:result)
  for (i = 0; i < n_sub; ++i) {
    // arma::mat Ti = get_T(i);
    arma::mat Ti;
    get_T(i, Ti);
    // arma::vec ri = get_Resid(i);
    arma::vec ri;
    get_Resid(i, ri);
    // arma::mat Sigmai_inv = get_Sigma_inv(i);
    arma::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    result += 2 * arma::sum(arma::log(Ti.diag())) +
              arma::as_scalar(ri.t() * Sigmai_inv * ri);

    // if(result < 0 && debug && 2*arma::sum(arma::log(Ti.diag())) < -20) {
    //     Rcpp::Rcout << "iter " << i << ": "
    //                 << " |T| " << 2*arma::sum(arma::log(Ti.diag()))
    //                 << " Q "<<arma::as_scalar(ri.t()*Sigmai_inv*ri)
    //                 << " result: " << result <<  std::endl;
    //     if (i == 0) {
    //         arma::mat Phii;  get_Phi(0, Phii);
    //         Ti.print("Ti = ");
    //         Phii.print("Phii = ");
    //     }

    // }
  }

  result += 2 * arma::sum(arma::log(arma::exp(Zlmd_ / 2)));
  // if(debug) {
  //         Rcpp::Rcout << "operator() : "
  //                     << result <<  std::endl;
  //         if(result < 0) {
  //             Rcpp::Rcout << "STOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
  //                         << "STOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
  //                         << "STOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
  //                         << std::endl;
  //         }
  // }
  // if (debug) {
  //     Rcpp::Rcout << "Entering HPC::operator()()" << std::endl;
  // }

  // int n_sub = m_.n_rows;
  // int n_bta = X_.n_cols;
  // int n_lmd = Z_.n_cols;
  // int n_gma = W_.n_cols;

  // if (debug) {
  //     Rcpp::Rcout << "Initialing parameters" << std::endl;
  // }

  // arma::vec beta = theta.rows(0, n_bta-1);
  // arma::vec lambda = theta.rows(n_bta, n_bta+n_lmd-1);
  // arma::vec gamma = theta.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1);

  // int W_rowindex = 0;
  // double result = 0.;
  // for(int i = 0; i != n_sub; ++i) {
  //     if (debug) {
  //         Rcpp::Rcout << "i = " << i << std::endl;
  //         Rcpp::Rcout << "Initializing Yi,Xi,Zi" << std::endl;
  //     }

  //     arma::mat Yi, Xi, Zi;

  //     if(i==0) {
  // 	Yi = Y_.rows(0,m_(0)-1);
  // 	Xi = X_.rows(0,m_(0)-1);
  // 	Zi = Z_.rows(0,m_(0)-1);
  //     } else {
  // 	int index = arma::sum(m_.rows(0,i-1));

  // 	Yi = Y_.rows(index, index+m_(i)-1);
  // 	Xi = X_.rows(index, index+m_(i)-1);
  // 	Zi = Z_.rows(index, index+m_(i)-1);
  //     }

  //     if (debug) {
  //         Rcpp::Rcout << "Updating Ti" << std::endl;
  //     }

  //     arma::mat Phii = arma::zeros<arma::mat>(m_(i),m_(i));
  //     arma::mat Ti = arma::zeros<arma::mat>(m_(i),m_(i)); Ti(0,0) = 1;
  //     for (int j = 1; j != m_(i); ++j) {
  //         for (int k = 0; k != j; ++k) {
  //             Phii(j,k) = arma::as_scalar(W_.row(W_rowindex) * gamma);
  //             ++W_rowindex;
  //         }

  //         Ti(j,0) = std::cos(Phii(j,0));
  //         Ti(j,j) =
  //         arma::prod(arma::prod(arma::sin(Phii.submat(j,0,j,j-1))));

  //         for(int l = 1; l != j; ++l) {
  //             Ti(j,l) = std::cos(Phii(j,l))
  //                 *
  //                 arma::prod(arma::prod(arma::sin(Phii.submat(j,0,j,l-1))));
  //         }

  //     }

  //     arma::mat Ti_inv = arma::pinv(Ti);

  //     if (debug) {
  //         Rcpp::Rcout << "Updating Di" << std::endl;
  //     }

  //     arma::vec di = arma::exp(Zi * lambda / 2);
  //     arma::mat Di = arma::diagmat(di);
  //     arma::mat Di_inv = arma::diagmat(pow(di, -1));

  //     if (debug) {
  //         Rcpp::Rcout << "Updating -2loglik function" << std::endl;
  //     }

  //     arma::vec ri = Yi - Xi * beta;
  //     double logdet = 2 * arma::sum(arma::log(di))
  //         + 2 * arma::sum(arma::log(Ti.diag()));
  //     result += logdet
  //         + arma::as_scalar(ri.t()*Di_inv*Ti_inv.t()*Ti_inv*Di_inv*ri);
  // }

  // if (debug) {
  //     Rcpp::Rcout << "Leaving HPC::operator()()" << std::endl;
  // }

  return result;
}

void HPC::Gradient(const arma::vec &x, arma::vec &grad) {
  UpdateHPC(x);

  int n_bta = X_.n_cols, n_lmd = Z_.n_cols, n_gma = W_.n_cols;

  arma::vec grad1, grad2, grad3;

  switch (free_param_) {
    case 0:

      Grad1(grad1);
      Grad2(grad2);

      grad = arma::zeros<arma::vec>(theta_.n_rows);
      grad.subvec(0, n_bta - 1) = grad1;
      grad.subvec(n_bta, n_bta + n_lmd + n_gma - 1) = grad2;

      break;

    case 1:
      Grad1(grad);
      break;

    case 2:
      Grad2(grad);
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
  // int debug = 0;

  // if (debug) {
  //     Rcpp::Rcout << "Entering HPC::Gradient()" << std::endl;
  // }

  // int n_sub = m_.n_rows;
  // int n_bta = X_.n_cols;
  // int n_lmd = Z_.n_cols;
  // int n_gma = W_.n_cols;

  // if (debug) {
  //     Rcpp::Rcout << "Initialing parameters" << std::endl;
  // }

  // arma::vec beta   = theta.rows(0,           n_bta-1);
  // arma::vec lambda = theta.rows(n_bta,       n_bta+n_lmd-1);
  // arma::vec gamma  = theta.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1);

  // grad = arma::zeros<arma::vec>(theta.n_rows);

  // arma::vec grad1 = arma::zeros<arma::vec>(n_bta);
  // arma::vec grad2 = arma::zeros<arma::vec>(n_lmd);
  // arma::vec grad3 = arma::zeros<arma::vec>(n_gma);

  // int W_rowindex = 0;
  // for(int i = 0; i != n_sub; ++i) {
  //     if (debug) {
  //         Rcpp::Rcout << "i = " << i << std::endl;
  //         Rcpp::Rcout << "Initializing Yi,Xi,Zi" << std::endl;
  //     }

  //     arma::mat Yi, Xi, Zi;

  //     if(i==0) {
  //         Yi = Y_.rows(0,m_(0)-1);
  //         Xi = X_.rows(0,m_(0)-1);
  //         Zi = Z_.rows(0,m_(0)-1);
  //     } else {
  //         int index = arma::sum(m_.rows(0,i-1));
  //         Yi = Y_.rows(index, index+m_(i)-1);
  //         Xi = X_.rows(index, index+m_(i)-1);
  //         Zi = Z_.rows(index, index+m_(i)-1);
  //     }

  //     if (debug) {
  //         Rcpp::Rcout << "Updating Ti" << std::endl;
  //     }

  //     arma::vec ri = Yi - Xi * beta;

  //     arma::mat Phii = arma::zeros<arma::mat>(m_(i),m_(i));
  //     arma::mat Ti = arma::zeros<arma::mat>(m_(i),m_(i));
  //     Ti(0,0) = 1;
  //     for (int j = 1; j != m_(i); ++j) {
  //         for (int k = 0; k != j; ++k) {
  //             Phii(j,k) = arma::as_scalar(W_.row(W_rowindex) * gamma);
  //             ++W_rowindex;
  //         }

  //         Ti(j,0) = std::cos(Phii(j,0));
  //         Ti(j,j) =
  //         arma::prod(arma::prod(arma::sin(Phii.submat(j,0,j,j-1))));
  //         for(int l = 1; l != j; ++l) {
  //             Ti(j,l) = std::cos(Phii(j,l))
  //                 *
  //                 arma::prod(arma::prod(arma::sin(Phii.submat(j,0,j,l-1))));
  //         }
  //     }

  //     arma::mat Ti_inv = arma::pinv(Ti);

  //     if (debug) {
  //         Rcpp::Rcout << "Updating Di" << std::endl;
  //     }

  //     arma::vec di = arma::exp(Zi * lambda / 2);
  //     arma::mat Di = arma::diagmat(di);
  //     arma::mat Di_inv = arma::diagmat(pow(di,-1));
  //     arma::mat Sigmai_inv = Di_inv * Ti_inv.t() *  Ti_inv * Di_inv;

  //     arma::vec ei = Ti_inv * Di_inv * ri;
  //     arma::vec one = arma::ones<arma::vec>(m_(i));

  //     if (debug) {
  //         Rcpp::Rcout << "Updating gradient" << std::endl;
  //     }

  //     grad1 += Xi.t() * Sigmai_inv * ri;

  //     arma::vec hi = arma::diagvec(Ti_inv.t()*Ti_inv
  //                                  *Di_inv*ri*ri.t()*Di_inv);
  //     grad2 += 0.5 * Zi.t() * (hi - one);

  //     arma::mat Ti_trans_deriv = CalcTransTiDeriv(i,Phii,Ti);
  //     for (int j = 0; j != m_(i); ++j) {
  //         grad3 += -1/Ti(j,j) * CalcTijkDeriv(i,j,j,Phii,Ti);
  //     }
  //     grad3 += arma::kron(ei.t(),arma::eye(n_gma,n_gma))
  //         * Ti_trans_deriv * Ti_inv.t() * ei;
  // }

  // grad1 *= -2;
  // grad2 *= -2;
  // grad3 *= -2;

  // grad.rows(0, n_bta-1) = grad1;
  // grad.rows(n_bta, n_bta+n_lmd-1) = grad2;
  // grad.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1) = grad3;

  // if (debug) {
  //     Rcpp::Rcout << "Leaving HPC::Gradient()" << std::endl;
  // }
}

void HPC::Grad1(arma::vec &grad1) {
  int debug = 0;

  int i, n_sub = m_.n_elem, n_bta = X_.n_cols;
  grad1 = arma::zeros<arma::vec>(n_bta);

  if (debug) Rcpp::Rcout << "Update grad1" << std::endl;

  for (i = 0; i < n_sub; ++i) {
    // arma::mat Xi = get_X(i);
    arma::mat Xi;
    get_X(i, Xi);
    // arma::vec ri = get_Resid(i);
    arma::vec ri;
    get_Resid(i, ri);
    // arma::mat Sigmai_inv = get_Sigma_inv(i);
    arma::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    grad1 += Xi.t() * Sigmai_inv * ri;
  }

  grad1 *= -2;
}

void HPC::Grad2(arma::vec &grad2) {
  int debug = 0;

  int i, n_sub = m_.n_elem, n_lmd = Z_.n_cols, n_gma = W_.n_cols;
  grad2 = arma::zeros<arma::vec>(n_lmd + n_gma);
  arma::vec grad2_lmd = arma::zeros<arma::vec>(n_lmd);
  arma::vec grad2_gma = arma::zeros<arma::vec>(n_gma);

  if (debug) Rcpp::Rcout << "Update grad2" << std::endl;

  for (i = 0; i < n_sub; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    // arma::mat Zi = get_Z(i);
    arma::mat Zi;
    get_Z(i, Zi);
    // arma::vec hi = get_TDResid2(i);
    arma::vec hi;
    get_TDResid2(i, hi);

    grad2_lmd += 0.5 * Zi.t() * (hi - one);

    // arma::mat Phii = get_Phi(i);
    arma::mat Phii;
    get_Phi(i, Phii);
    // arma::mat Ti = get_T(i);
    arma::mat Ti;
    get_T(i, Ti);
    // arma::mat Ti_inv = arma::pinv(Ti);
    // arma::mat Ti_inv = Ti.i();
    arma::mat Ti_inv;
    get_invT(i, Ti_inv);
    // arma::vec ei = get_TDResid(i);
    arma::vec ei;
    get_TDResid(i, ei);

    arma::mat Ti_trans_deriv = CalcTransTiDeriv(i, Phii, Ti);
    for (int j = 0; j != m_(i); ++j) {
      grad2_gma += -1 / Ti(j, j) * CalcTijkDeriv(i, j, j, Phii, Ti);
    }
    grad2_gma += arma::kron(ei.t(), arma::eye(n_gma, n_gma)) * Ti_trans_deriv *
                 Ti_inv.t() * ei;
  }
  grad2.subvec(0, n_lmd - 1) = grad2_lmd;
  grad2.subvec(n_lmd, n_lmd + n_gma - 1) = grad2_gma;

  grad2 *= -2;
}

void HPC::UpdateHPC(const arma::vec &x) {
  int debug = 0;
  bool update = true;

  switch (free_param_) {
    case 0:
      if (arma::min(x == theta_) == 1) update = false;
      break;

    case 1:
      if (arma::min(x == beta_) == 1) update = false;
      break;

    case 2:
      if (arma::min(x == lmdgma_) == 1) update = false;
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }

  if (update) {
    if (debug) Rcpp::Rcout << "UpdateParam..." << std::endl;
    UpdateParam(x);
    if (debug) Rcpp::Rcout << "UpdateModel..." << std::endl;
    UpdateModel();
  } else {
    if (debug) Rcpp::Rcout << "Hey, I did save some time!:)" << std::endl;
  }
}

void HPC::UpdateParam(const arma::vec &x) {
  int n_bta = X_.n_cols;
  int n_lmd = Z_.n_cols;
  int n_gma = W_.n_cols;

  switch (free_param_) {
    case 0:
      theta_ = x;
      beta_ = x.rows(0, n_bta - 1);
      lambda_ = x.rows(n_bta, n_bta + n_lmd - 1);
      gamma_ = x.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1);
      lmdgma_ = x.rows(n_bta, n_bta + n_lmd + n_gma - 1);
      break;

    case 1:
      theta_.rows(0, n_bta - 1) = x;
      beta_ = x;
      break;

    case 2:
      theta_.rows(n_bta, n_bta + n_lmd + n_gma - 1) = x;
      lambda_ = x.rows(0, n_lmd - 1);
      gamma_ = x.rows(n_lmd, n_lmd + n_gma - 1);
      lmdgma_ = x;
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

void HPC::UpdateModel() {
  int debug = 0;

  if (debug) Rcpp::Rcout << "update Xbta Zlmd Wgam r" << std::endl;

  switch (free_param_) {
    case 0:
      Xbta_ = X_ * beta_;
      Zlmd_ = Z_ * lambda_;
      Wgma_ = W_ * gamma_;
      Resid_ = Y_ - Xbta_;

      UpdateTelem();
      UpdateTDResid();

      break;

    case 1:
      Xbta_ = X_ * beta_;
      Resid_ = Y_ - Xbta_;
      UpdateTDResid();

      break;

    case 2:
      Zlmd_ = Z_ * lambda_;
      Wgma_ = W_ * gamma_;

      UpdateTelem();
      UpdateTDResid();

      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

void HPC::UpdateBeta() {
  int i, n_sub = m_.n_elem, n_bta = X_.n_cols;
  arma::mat XSX = arma::zeros<arma::mat>(n_bta, n_bta);
  arma::vec XSY = arma::zeros<arma::vec>(n_bta);

  for (i = 0; i < n_sub; ++i) {
    // arma::mat Xi = get_X(i);
    // arma::vec Yi = get_Y(i);
    // arma::mat Sigmai_inv = get_Sigma_inv(i);
    arma::mat Xi;
    get_X(i, Xi);
    arma::vec Yi;
    get_Y(i, Yi);
    arma::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);

    XSX += Xi.t() * Sigmai_inv * Xi;
    XSY += Xi.t() * Sigmai_inv * Yi;
  }

  arma::vec beta = XSX.i() * XSY;

  set_beta(beta);
}

void HPC::UpdateLambdaGamma(const arma::vec &x) { set_lmdgma(x); }

// void HPC::CalcMeanCovmati(const arma::vec& theta, int i,
//     		      arma::vec& mui, arma::mat& Sigmai)
// {
//     int debug = 0;

//     int n_bta = X_.n_cols;
//     int n_lmd = Z_.n_cols;
//     int n_gma = W_.n_cols;

//     if (debug) {
//         Rcpp::Rcout << "Initialing parameters" << std::endl;
//     }

//     arma::vec beta = theta.rows(0, n_bta-1);
//     arma::vec lambda = theta.rows(n_bta, n_bta+n_lmd-1);
//     arma::vec gamma = theta.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1);

//     arma::mat Yi, Xi, Zi;

//     if(i==0) {
//         Yi = Y_.rows(0,m_(0)-1);
//         Xi = X_.rows(0,m_(0)-1);
//         Zi = Z_.rows(0,m_(0)-1);
//     } else {
//         int index = arma::sum(m_.rows(0,i-1));

//         Yi = Y_.rows(index, index+m_(i)-1);
//         Xi = X_.rows(index, index+m_(i)-1);
//         Zi = Z_.rows(index, index+m_(i)-1);
//     }

//     if (debug) {
//         Rcpp::Rcout << "Updating Ti" << std::endl;
//     }

//     arma::vec ri = Yi - Xi * beta;

//     arma::mat Phii = arma::zeros<arma::mat>(m_(i),m_(i));
//     arma::mat Ti = arma::zeros<arma::mat>(m_(i),m_(i));
//     Ti(0,0) = 1;

//     int W_rowindex = 0;
//     for (int index = 0; index != i+1; ++ index){
//         for (int j = 1; j != m_(index); ++j) {
//             for (int k = 0; k != j; ++k) {
//     	    if (index == i)
//     		Phii(j,k) = arma::as_scalar(W_.row(W_rowindex) * gamma);
//                 ++W_rowindex;
//             }
//     	if (index == i) {
//     	    Ti(j,0) = std::cos(Phii(j,0));
//     	    Ti(j,j) = arma::prod(arma::prod(arma::sin(Phii.submat(j,0,j,j-1))));
//     	    for(int l = 1; l != j; ++l) {
//     		Ti(j,l) = std::cos(Phii(j,l))
//     		    * arma::prod(arma::prod(arma::sin(Phii.submat(j,0,j,l-1))));
//     	    }
//     	}
//         }
//     }

//     if (debug) {
//         Rcpp::Rcout << "Updating Di" << std::endl;
//     }

//     arma::vec di = arma::exp(Zi * lambda / 2);
//     arma::mat Di = arma::diagmat(di);

//     mui = Xi * beta;
//     Sigmai = Di * Ti * Ti.t() * Di;
// }

// void HPC::SimResp(int n, const arma::vec& theta, arma::mat& Resp)
// {
//     int debug = 0;

//     if (debug) {
//         Rcpp::Rcout << "Entering HPC::SimResp()" << std::endl;
//     }

//     Resp = arma::zeros<arma::mat>(n, Y_.n_rows);

//     int n_sub = m_.n_rows;
//     for(int i = 0; i != n_sub; ++i) {

//         if (debug) {
//     	Rcpp::Rcout << "i = " << i << std::endl;
//     	Rcpp::Rcout << "Calculating mu_i and Sigma_i" << std::endl;
//         }

//         arma::vec mui;
//         arma::mat Sigmai;
//         CalcMeanCovmati(theta, i, mui, Sigmai);

//         int ncols = Sigmai.n_cols;
//         arma::mat Y = arma::randn(n, ncols);

//         int first_col = 0;
//         int last_col = 0;
//         if (i == 0) {
//     	first_col = 0;
//     	last_col = m_(0) - 1;
//         } else {
//     	first_col = arma::sum(m_.rows(0, i-1));
//     	last_col = first_col + m_(i) - 1;
//         }

//         Resp.cols(first_col, last_col)
//     	= arma::repmat(mui, 1, n).t() + Y * arma::chol(Sigmai);

//         if(debug) {
//     	Resp.print("Resp = ");
//         }
//     }
// }

void HPC::UpdateTelem() {
  int i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    // arma::mat Phii = get_Phi(i);
    arma::mat Phii;
    get_Phi(i, Phii);
    arma::mat Ti = arma::eye(m_(i), m_(i));

    Ti(0, 0) = 1;
    for (int j = 1; j != m_(i); ++j) {
      Ti(j, 0) = std::cos(Phii(j, 0));
      Ti(j, j) = arma::prod(arma::prod(arma::sin(Phii.submat(j, 0, j, j - 1))));
      for (int l = 1; l != j; ++l) {
        Ti(j, l) =
            std::cos(Phii(j, l)) *
            arma::prod(arma::prod(arma::sin(Phii.submat(j, 0, j, l - 1))));
      }
    }

    // Ti_inv = Ti.i();
    arma::mat Ti_inv;
    // bool is_Ti_pd = arma::inv(Ti_inv, Ti);
    // if (!is_Ti_pd) Ti_inv = arma::pinv(Ti);
    if (!arma::inv(Ti_inv, Ti)) Ti_inv = arma::pinv(Ti);

    if (i == 0) {
      int first_index = 0;
      int last_index = m_(0) * (m_(0) + 1) / 2 - 1;

      Telem_.subvec(first_index, last_index) = pan::lvectorise(Ti, true);
      invTelem_.subvec(first_index, last_index) = pan::lvectorise(Ti_inv, true);
    } else {
      int first_index = 0;
      for (int idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) + 1) / 2;
      }
      int last_index = first_index + m_(i) * (m_(i) + 1) / 2 - 1;

      Telem_.subvec(first_index, last_index) = pan::lvectorise(Ti, true);
      invTelem_.subvec(first_index, last_index) = pan::lvectorise(Ti_inv, true);
    }
  }
}

void HPC::UpdateTDResid() {
  int i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    // arma::vec ri = get_Resid(i);
    arma::vec ri;
    get_Resid(i, ri);

    // arma::mat Ti = get_T(i);
    arma::mat Ti;
    get_T(i, Ti);
    //  arma::mat Ti_inv = arma::pinv(Ti);
    //           arma::mat Ti_inv = Ti.i();
    arma::mat Ti_inv;
    get_invT(i, Ti_inv);

    // arma::mat Di = get_D(i);
    arma::mat Di;
    get_D(i, Di);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    arma::vec TiDiri = Ti_inv * Di_inv * ri;
    arma::vec TiDiri2 = arma::diagvec(Ti_inv.t() * Ti_inv * Di_inv * ri *
                                      ri.t() * Di_inv);  // hi

    if (i == 0) {
      TDResid_.subvec(0, m_(0) - 1) = TiDiri;
      TDResid2_.subvec(0, m_(0) - 1) = TiDiri2;
    } else {
      int index = arma::sum(m_.subvec(0, i - 1));
      TDResid_.subvec(index, index + m_(i) - 1) = TiDiri;
      TDResid2_.subvec(index, index + m_(i) - 1) = TiDiri2;
    }
  }
}

arma::vec HPC::Wijk(const int i, const int j, const int k) {
  int n_sub = m_.n_rows;
  int n_gma = W_.n_cols;

  int W_rowindex = 0;
  bool indexfound = false;
  arma::vec result = arma::zeros<arma::vec>(n_gma);
  for (int ii = 0; ii != n_sub && !indexfound; ++ii) {
    for (int jj = 0; jj != m_(ii) && !indexfound; ++jj) {
      for (int kk = 0; kk != jj && !indexfound; ++kk) {
        if (ii == i && jj == j && kk == k) {
          indexfound = true;
          result = W_.row(W_rowindex).t();
        }
        ++W_rowindex;
      }
    }
  }
  return result;
}

arma::vec HPC::CalcTijkDeriv(const int i, const int j, const int k,
                             const arma::mat &Phii, const arma::mat &Ti) {
  int n_gma = W_.n_cols;

  arma::vec result = arma::zeros<arma::vec>(n_gma);
  if (k < j) {
    result = Ti(j, k) * (-std::tan(Phii(j, k)) * Wijk(i, j, k));
    for (int l = 0; l != k; ++l) {
      result += Ti(j, k) * Wijk(i, j, l) / std::tan(Phii(j, l));
    }
    return result;
  } else if (k == j) {
    for (int l = 0; l != k; ++l) {
      result += Ti(j, k) * Wijk(i, j, l) / std::tan(Phii(j, l));
    }
    return result;
  }

  return result;
}

arma::mat HPC::CalcTransTiDeriv(const int i, const arma::mat &Phii,
                                const arma::mat &Ti) {
  int n_gma = W_.n_cols;

  arma::mat result = arma::zeros<arma::mat>(n_gma * m_(i), m_(i));
  for (int k = 1; k != m_(i); ++k) {
    for (int j = 0; j <= k; ++j) {
      result.submat(j * n_gma, k, (j * n_gma + n_gma - 1), k) =
          CalcTijkDeriv(i, k, j, Phii, Ti);
    }
  }

  return result;
}

}  // namespace jmcm
