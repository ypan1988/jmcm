// mcd.h: implementation of joint mean-covariance models based on
//        modified Cholesky decomposition(M.CD) of the covariance matrix
//
// Copyright (C) 2015-2016 The University of Manchester
//
// Written by Yi Pan - ypan1988@gmail.com

//#include <omp.h>
#include "mcd.h"

namespace jmcm {

MCD::MCD(const arma::vec& m, const arma::vec& Y, const arma::mat& X,
         const arma::mat& Z, const arma::mat& W)
    : JmcmBase(m, Y, X, Z, W, 0) {
  int debug = 0;

  if (debug) Rcpp::Rcout << "Creating MCD object" << std::endl;
  // m_.print("m = ");

  int N = Y_.n_rows;
  int n_bta = X_.n_cols;
  int n_lmd = Z_.n_cols;
  int n_gma = W_.n_cols;

  G_ = arma::zeros<arma::mat>(N, n_gma);
  TResid_ = arma::zeros<arma::vec>(N);

  free_param_ = 0;

  cov_only_ = false;
  mean_ = Y_;

  if (debug) Rcpp::Rcout << "MCD object created" << std::endl;
}

MCD::~MCD() {}

double MCD::operator()(const arma::vec& x) {
  int debug = 0;

  if (debug) Rcpp::Rcout << "UpdateJmcm(x)" << std::endl;
  UpdateJmcm(x);

  int i, n_sub = m_.n_elem;
  double result = 0.0;

  // arma::wall_clock timer;
  // timer.tic();

  if (debug) Rcpp::Rcout << "before for loop" << std::endl;
  //	#pragma omp parallel for reduction(+:result)
  for (i = 0; i < n_sub; ++i) {
    // arma::vec ri = get_Resid(i);
    arma::vec ri;
    get_Resid(i, ri);
    // arma::mat Sigmai_inv = get_Sigma_inv(i);
    arma::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    result += arma::as_scalar(ri.t() * Sigmai_inv * ri);
  }

  if (debug) Rcpp::Rcout << "after for loop" << std::endl;

  result += arma::sum(arma::log(arma::exp(Zlmd_)));

  // double n = timer.toc();
  // Rcpp::Rcout << "number of seconds: " << n << std::endl;

  return result;

  // int debug = 0;
  // int debug1 = 0;

  // if (debug) {
  //     Rcpp::Rcout << "Entering MCD::operator()()" << std::endl;
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
  // 	Rcpp::Rcout << "i = " << i << std::endl;
  // 	Rcpp::Rcout << "Initializing Yi,Xi,Zi" << std::endl;
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

  //     if (debug1 && i == 82) {
  //         Yi.print("Yi = ");
  //         Xi.print("Xi = ");
  //         Zi.print("Zi = ");
  //     }

  //     if (debug) {
  //         Rcpp::Rcout << "Updating Ti" << std::endl;
  //     }

  //     arma::mat Ti = arma::eye(m_(i),m_(i));
  //     for (int j = 1; j != m_(i); ++j) {
  //         for (int k = 0; k != j; ++k) {

  // 	    if (debug) {
  // 		Rcpp::Rcout << "j = " << j
  // 			    << " k = " << k
  // 			    << " index = " << W_rowindex
  // 			    << std::endl;
  // 	    }
  //  	    if (debug1 && i==82 && j==1 && k == 0) {
  //                 W_.row(W_rowindex).print("Wijk = ");
  //                 gamma.print("gamma = ");
  //             }

  //             Ti(j,k) = -arma::as_scalar(W_.row(W_rowindex)*gamma);
  //             ++W_rowindex;
  //         }
  //     }

  //     if (debug1 && i == 82) {
  //         Ti.print("Ti = ");
  //     }

  //     if (debug) {
  //         Rcpp::Rcout << "Updating Di" << std::endl;
  //     }

  //     arma::vec di = arma::exp(Zi * lambda);
  //     arma::mat Di = arma::diagmat(di);
  //     arma::mat Di_inv = arma::diagmat(pow(di, -1));

  //     if (debug1 && i == 82) {
  //         Di.print("Di = ");
  //         Di_inv.print("Di_inv = ");
  //     }

  //     if (debug) {
  //         Rcpp::Rcout << "Updating -2loglik function" << std::endl;
  //     }

  //     arma::vec ri = Yi - Xi * beta;
  //     double logdet = arma::sum(arma::log(di));
  //     result += logdet
  //         + arma::as_scalar(ri.t()*Ti.t()*Di_inv*Ti*ri);

  //     if (debug1 && i >= 70 && i <= 100) {
  //       if(i ==82) {
  //         ri.print("ri = ");
  //         Rcpp::Rcout << "logdet = " << logdet << std::endl;
  //         arma::mat tmp = ri.t()*Ti.t();
  //         tmp.print("tmp = ");
  //         tmp *= Di_inv;
  //         tmp.print("tmp = ");
  //         tmp *= ri;
  //         tmp.print("tmp = ");
  //       }
  //         Rcpp::Rcout << i+1 << ": n2ll: " << result << std::endl;
  //     }

  // }

  //  if (debug) {
  //     Rcpp::Rcout << "Leaving MCD::operator()()" << std::endl;
  // }

  // return result;
}

void MCD::Gradient(const arma::vec& x, arma::vec& grad) {
  UpdateJmcm(x);

  int n_bta = X_.n_cols, n_lmd = Z_.n_cols, n_gma = W_.n_cols;

  arma::vec grad1, grad2, grad3;

  switch (free_param_) {
    case 0:

      Grad1(grad1);
      Grad2(grad2);
      Grad3(grad3);

      grad = arma::zeros<arma::vec>(theta_.n_rows);
      grad.subvec(0, n_bta - 1) = grad1;
      grad.subvec(n_bta, n_bta + n_lmd - 1) = grad2;
      grad.subvec(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1) = grad3;

      break;

    case 1:
      Grad1(grad);
      break;

    case 2:
      Grad2(grad);
      break;

    case 3:
      Grad3(grad);
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }

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

  //     arma::mat Ti = arma::eye(m_(i),m_(i));
  //     arma::vec ri = Yi - Xi * beta;
  //     arma::mat Gi = arma::zeros<arma::mat>(ri.n_rows, W_.n_cols);
  //     for (int j = 1; j != m_(i); ++j) {
  //         for (int k = 0; k != j; ++k) {
  //             Ti(j,k) = -arma::as_scalar(W_.row(W_rowindex)*gamma);

  //             Gi.row(j) += W_.row(W_rowindex) * ri(k);
  //             ++W_rowindex;
  //         }
  //     }

  //     if (debug) {
  //         Rcpp::Rcout << "Updating Di" << std::endl;
  //     }

  //     arma::vec di = arma::exp(Zi * lambda);
  //     arma::mat Di = arma::diagmat(di);
  //     arma::mat Di_inv = arma::diagmat(pow(di,-1));
  //     arma::mat Sigmai_inv = Ti.t() * Di_inv * Ti;

  //     arma::vec ei = arma::pow(ri-Gi*gamma, 2);
  //     arma::vec one = arma::ones<arma::vec>(m_(i));

  //     if (debug) {
  //         Rcpp::Rcout << "Updating gradient" << std::endl;
  //     }

  //     grad1 += Xi.t() * Sigmai_inv * ri * (-2);
  //     grad2 += 0.5 * Zi.t() * (Di_inv * ei - one) * (-2);
  //     grad3 += Gi.t() * Di_inv * (ri - Gi * gamma) * (-2);
  // }

  // grad.rows(0, n_bta-1) = grad1;
  // grad.rows(n_bta, n_bta+n_lmd-1) = grad2;
  // grad.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1) = grad3;
}

void MCD::Grad1(arma::vec& grad1) {
  int debug = 0;

  int i, n_sub = m_.n_elem, n_bta = X_.n_cols;
  grad1 = arma::zeros<arma::vec>(n_bta);

  if (debug) Rcpp::Rcout << "Update grad1" << std::endl;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Xi = get_X(i);
    arma::vec ri;
    get_Resid(i, ri);
    arma::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    grad1 += Xi.t() * Sigmai_inv * ri;
  }

  grad1 *= -2;
}

void MCD::Grad2(arma::vec& grad2) {
  int debug = 0;

  int i, n_sub = m_.n_elem, n_lmd = Z_.n_cols;
  grad2 = arma::zeros<arma::vec>(n_lmd);

  if (debug) Rcpp::Rcout << "Update grad2" << std::endl;

  for (i = 0; i < n_sub; ++i) {
    arma::vec one = arma::ones<arma::vec>(m_(i));
    arma::mat Zi = get_Z(i);
    //	    arma::mat Gi = get_G(i);

    arma::mat Di;
    get_D(i, Di);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    //	    arma::vec ri = get_Resid(i);
    //      arma::vec ei = arma::pow(ri - Gi * gamma_, 2);
    arma::vec ei = arma::pow(get_TResid(i), 2);

    grad2 += 0.5 * Zi.t() * (Di_inv * ei - one);
  }

  grad2 *= -2;
}

void MCD::Grad3(arma::vec& grad3) {
  int debug = 0;

  int i, n_sub = m_.n_elem, n_gma = W_.n_cols;
  grad3 = arma::zeros<arma::vec>(n_gma);

  if (debug) Rcpp::Rcout << "Update grad3" << std::endl;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi;
    get_G(i, Gi);

    arma::mat Di;
    get_D(i, Di);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    // arma::vec ri = get_Resid(i);

    // grad3 += Gi.t() * Di_inv * (ri - Gi * gamma_);

    arma::vec Tiri;
    get_TResid(i, Tiri);

    grad3 += Gi.t() * Di_inv * Tiri;
  }

  grad3 *= -2;
}

void MCD::UpdateJmcm(const arma::vec& x) {
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
      if (arma::min(x == lambda_) == 1) update = false;

      break;

    case 3:
      if (arma::min(x == gamma_) == 1) update = false;

      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }

  if (update) {
    UpdateParam(x);
    UpdateModel();

  } else {
    if (debug) Rcpp::Rcout << "Hey, I did save some time!:)" << std::endl;
  }
}

void MCD::UpdateParam(const arma::vec& x) {
  int n_bta = X_.n_cols;
  int n_lmd = Z_.n_cols;
  int n_gma = W_.n_cols;

  switch (free_param_) {
    case 0:
      theta_ = x;
      beta_ = x.rows(0, n_bta - 1);
      lambda_ = x.rows(n_bta, n_bta + n_lmd - 1);
      gamma_ = x.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1);
      break;

    case 1:
      theta_.rows(0, n_bta - 1) = x;
      beta_ = x;
      break;

    case 2:
      theta_.rows(n_bta, n_bta + n_lmd - 1) = x;
      lambda_ = x;
      break;

    case 3:
      theta_.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1) = x;
      gamma_ = x;
      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

void MCD::UpdateModel() {
  int debug = 0;

  if (debug) Rcpp::Rcout << "update Xbta Zlmd Wgam r" << std::endl;

  switch (free_param_) {
    case 0:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = X_ * beta_;
      Zlmd_ = Z_ * lambda_;
      Wgma_ = W_ * gamma_;
      Resid_ = Y_ - Xbta_;

      if (debug) Rcpp::Rcout << "UpdateG(x)" << std::endl;
      UpdateG();
      if (debug) Rcpp::Rcout << "UpdateTResid(x)" << std::endl;
      UpdateTResid();
      if (debug) Rcpp::Rcout << "Update Finished.." << std::endl;

      break;

    case 1:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = X_ * beta_;
      Resid_ = Y_ - Xbta_;

      UpdateG();
      UpdateTResid();

      break;

    case 2:
      Zlmd_ = Z_ * lambda_;

      break;

    case 3:
      Wgma_ = W_ * gamma_;

      UpdateTResid();

      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

void MCD::UpdateLambda(const arma::vec& x) { set_lambda(x); }

void MCD::UpdateGamma() {
  int i, n_sub = m_.n_elem, n_gma = W_.n_cols;
  arma::mat GDG = arma::zeros<arma::mat>(n_gma, n_gma);
  arma::vec GDr = arma::zeros<arma::vec>(n_gma);

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi;
    get_G(i, Gi);
    arma::vec ri;
    get_Resid(i, ri);
    arma::mat Di;
    get_D(i, Di);
    arma::mat Di_inv = arma::diagmat(arma::pow(Di.diag(), -1));

    GDG += Gi.t() * Di_inv * Gi;
    GDr += Gi.t() * Di_inv * ri;
  }

  arma::vec gamma = GDG.i() * GDr;

  set_gamma(gamma);
}

void MCD::UpdateG() {
  int i, j, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::mat Gi = arma::zeros<arma::mat>(m_(i), W_.n_cols);

    arma::mat Wi = get_W(i);
    arma::vec ri;
    get_Resid(i, ri);
    for (j = 1; j != m_(i); ++j) {
      int index = 0;
      if (j == 1)
        index = 0;
      else {
        for (int idx = 1; idx < j; ++idx) index += idx;
      }
      Gi.row(j) = ri.subvec(0, j - 1).t() * Wi.rows(index, index + j - 1);
    }
    if (i == 0)
      G_.rows(0, m_(0) - 1) = Gi;
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      G_.rows(index, index + m_(i) - 1) = Gi;
    }
  }
}

void MCD::UpdateTResid() {
  int i, n_sub = m_.n_elem;

  for (i = 0; i < n_sub; ++i) {
    arma::vec ri;
    get_Resid(i, ri);
    arma::mat Ti;
    get_T(i, Ti);
    arma::mat Tiri = Ti * ri;
    if (i == 0)
      TResid_.subvec(0, m_(0) - 1) = Tiri;
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      TResid_.subvec(index, index + m_(i) - 1) = Tiri;
    }
  }
}

// void MCD::CalcMeanCovmati(const arma::vec& theta, int i, arma::vec& mui,
// arma::mat& Sigmai)
// {
//     int debug = 0;

//    	int n_bta = X_.n_cols;
//     int n_lmd = Z_.n_cols;
//     int n_gma = W_.n_cols;

//     if (debug) {
//         Rcpp::Rcout << "Initialing parameters" << std::endl;
//     }

//     arma::vec beta   = theta.rows(0,           n_bta-1);
//     arma::vec lambda = theta.rows(n_bta,       n_bta+n_lmd-1);
//     arma::vec gamma  = theta.rows(n_bta+n_lmd, n_bta+n_lmd+n_gma-1);

//     if (debug) {
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

//     arma::mat Ti = arma::eye(m_(i),m_(i));

//     int W_rowindex = 0;
//     for (int index = 0; index != i+1; ++index) {
//         for (int j = 1; j != m_(index); ++j) {
//             for (int k = 0; k != j; ++k) {
//                 if (index == i)
//                     Ti(j,k) = -arma::as_scalar(W_.row(W_rowindex)*gamma);
//                 ++W_rowindex;
//             }
//         }
//     }

//     arma::mat Ti_inv = Ti.i();

//     if (debug) {
//         Rcpp::Rcout << "Updating Di" << std::endl;
//     }

//     arma::vec di = arma::exp(Zi * lambda);
//     arma::mat Di = arma::diagmat(di);

//     mui = Xi * beta;
//     Sigmai = Ti_inv * Di * Ti_inv.t();
// }

// void MCD::SimResp(int n, const arma::vec& theta, arma::mat& Resp)
// {
//     int debug = 0;

//     if (debug) {
//         Rcpp::Rcout << "Entering MCD::SimResp()" << std::endl;
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

}  // namespace jmcm
