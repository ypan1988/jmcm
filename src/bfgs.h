#ifndef BFGS_H_
#define BFGS_H_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>

namespace pan {

template <typename T>
class BFGS {
 public:
  BFGS()
      : kIterMax_(200),
        p3(1.0e-4),
        kEpsilon_(std::numeric_limits<double>::epsilon()),
        kTolX_(4 * kEpsilon_),
        kScaStepMax_(100) {}
  ~BFGS() = default;

  const int kIterMax_;     // Maximum number of iterations
  const double p3;         // Ensure sufficient decrease in function value
  const double kEpsilon_;  // Machine precision
  const double kTolX_;     // Convergence criterion on x values
  const double
      kScaStepMax_;  // Scaled maximum step length allowed in line searches
  const double grad_tol_ = 1e-6;

  double linesearch(T &fun, const arma::vec &xx, arma::vec &h, double F,
                    const arma::vec &g);
  void minimize(T &fun, arma::vec &x, const double grad_tol = 1e-6);

  void set_message(bool message) { message_ = message; }
  void set_trace(bool trace) { trace_ = trace; }
  int n_iters() const { return n_iters_; }
  double f_min() const { return f_min_; }

  bool test_grad(const arma::vec &x, double F, const arma::vec &g) const {
    arma::vec xtmp = x;
    xtmp.for_each([](arma::vec::elem_type &val) { val = std::max(val, 1.0); });
    return arma::max(arma::abs(g) % xtmp / std::max(F, 1.0)) < grad_tol_;
  }

  bool test_diff_x(const arma::vec &x, const arma::vec &h) const {
    arma::vec xtmp = x;
    xtmp.for_each([](arma::vec::elem_type &val) { val = std::max(val, 1.0); });
    return arma::max(arma::abs(h) / xtmp) < kTolX_;
  }

 private:
  bool message_, trace_;
  int n_iters_;
  double f_min_;

  bool IsInfOrNaN(double x) {
    return (x == std::numeric_limits<double>::infinity() ||
            x == -std::numeric_limits<double>::infinity() || x != x);
  }
};  // class BFGS

template <typename T>
double BFGS<T>::linesearch(T &fun, const arma::vec &xx, arma::vec &h, double F,
                           const arma::vec &g) {
  arma::vec x = xx;

  const arma::vec xold = xx;
  const double Fold = F;

  double dphi0 = dot(h, g);
  if (dphi0 >= 0.0 && message_)
    Rcpp::Rcerr << "Roundoff problem in linesearch." << std::endl;

  // Calculate the minimum step length
  arma::vec xtmp = x;
  xtmp.for_each([](arma::vec::elem_type &val) { val = std::max(val, 1.0); });
  double test = arma::max(arma::abs(h) / xtmp);
  double step_min = kEpsilon_ / test;

  double alpha, alpha2, alpha_tmp, F2;
  alpha = 1.0;  // Always try full Newton step first
  alpha2 = alpha_tmp = F = F2 = 0.0;
  for (int iter = 0; iter != kIterMax_; ++iter) {
    // x is too close to xold, ignored
    if (alpha < step_min) return 0.0;

    // Start of iteration loop
    x = xold + alpha * h;
    F = fun(x);

    if (IsInfOrNaN(F)) {
      // f is INF or NAN
      while (!IsInfOrNaN(alpha) && IsInfOrNaN(F)) {
        alpha *= 0.5;
        x = xold + alpha * h;
        F = fun(x);
      }
      alpha_tmp = 0.5 * alpha;
    } else if (F <= Fold + p3 * alpha * dphi0) {
      // Sufficient function decrease
      return alpha;
    } else {
      // Begin Backtrack
      if (alpha == 1.0) {
        // fisrt backtrack:
        // model phi(alpha) as a quadratic
        // phi(alpha) =
        //    (phi(1) - phi(0) - dphi0) * alpha^2 + dphi0 * alpha + phi(0)
        alpha_tmp = -dphi0 / (2.0 * (F - Fold - dphi0));
      } else {
        // second and subsequent backtracks
        // model phi(alpha) as a cubic
        // phi(alpha) =
        //    a * alpha^3 + b * alpha^2 + dphi0 * alpha + phi(0)
        double val1 = 1 / alpha / alpha;
        double val2 = 1 / alpha2 / alpha2;
        arma::vec ab =
            1 / (alpha - alpha2) *
            arma::mat({{val1, -val2}, {-alpha2 * val1, alpha * val2}}) *
            arma::vec({F - dphi0 * alpha - Fold, F2 - dphi0 * alpha2 - Fold});
        double a = ab(0), b = ab(1);

        if (IsInfOrNaN(a) || IsInfOrNaN(b)) {
          alpha_tmp = 0.5 * alpha;
        } else if (a == 0.0) {
          alpha_tmp = -dphi0 / (2.0 * b);
        } else {
          double disc = b * b - 3.0 * a * dphi0;
          if (disc < 0.0) {
            alpha_tmp = 0.5 * alpha;
          } else if (b <= 0.0) {
            alpha_tmp = (-b + std::sqrt(disc)) / (3.0 * a);
          } else {
            alpha_tmp = -dphi0 / (b + std::sqrt(disc));
          }
        }
        if (alpha_tmp > 0.5 * alpha || IsInfOrNaN(alpha_tmp)) {
          alpha_tmp = 0.5 * alpha;  // alpha_tmp <= 0.5alpha
        }
        // End Backtrack
      }
    }
    alpha2 = alpha;
    F2 = F;
    alpha = std::max(alpha_tmp, 0.1 * alpha);  // alpha_tmp >= 0.1alpha
  }                                            // for loop

  return alpha;
}

template <typename T>
void BFGS<T>::minimize(T &fun, arma::vec &x, const double grad_tol) {
  const int n_pars = x.n_rows;  // number of parameters

  // Calculate starting function value and gradient
  double F = fun(x);
  arma::vec g;
  fun.Gradient(x, g);

  // Initialize the inverse Hessian to a unit matrix
  arma::mat D = arma::eye<arma::mat>(n_pars, n_pars);

  // Calculate the maximum step length
  double delta = kScaStepMax_ * std::max(arma::norm(x, 2), double(n_pars));

  // Main loop over the iterations
  n_iters_ = 0;
  do {
    arma::vec h = -D * g;
    double h_norm = arma::norm(h, 2);
    if (h_norm > delta) h *= delta / h_norm;

    double alpha = linesearch(fun, x, h, F, g);
    arma::vec x2 = x;  // Save the old point
    x += alpha * h;

    if (trace_) {
      Rcpp::Rcout << std::setw(5) << n_iters_ << ": " << std::setw(10) << F
                  << ": ";
      x.t().print();
    }

    h = x - x2;  // Update line direction
    x2 = x;      // Update the current point
    F = fun(x);  // Update function value
    f_min_ = F;
    arma::vec grad2 = g;  // Save the old gradient
    fun.Gradient(x, g);   // Get the new gradient

    // BFGS Updating
    arma::vec y = g - grad2;
    double rho = 1 / arma::dot(h, y);
    arma::vec u = D * y;
    arma::vec v = 0.5 * (1 + rho * arma::dot(u, y)) * h - u;

    if (rho * std::sqrt(kEpsilon_ * arma::dot(y, y) * arma::dot(h, h)) < 1) {
      D += rho * (h * v.t() + v * h.t());
    }

    // Test for convergence on small gradient and small x - xp
    if (test_grad(x, F, g) || test_diff_x(x, h)) return;
  } while (++n_iters_ != kIterMax_);
  if (message_) {
    Rcpp::Rcerr << "too many iterations in bfgs" << std::endl;
  }
}

}  // namespace pan

#endif  // BFGS_H_
