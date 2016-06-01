// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 8 -*-
//
// Copyright (C) 2015 Yi Pan and Jianxin Pan
//
// This file is part of jmcm.

#ifndef JMCM_BFGS_H_
#define JMCM_BFGS_H_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

#include <RcppArmadillo.h>
#include "linesearch.h"

namespace pan {

    template <typename T>
    class BFGS : public LineSearch<T> {
    public:
	BFGS();
	~BFGS();

        void set_trace(bool trace) { trace_ = trace; }
        //        void set_message(bool message) { message_ = message; } 
	void Optimize(T &func, arma::vec& x, const double grad_tol = 1e-6);
	int n_iters() const;
	double f_min() const;
      
    private:
        bool trace_;
        bool message_;
	int n_iters_;
	double f_min_;
    }; // class BFGS

#include "bfgs-impl.h"

} // namespace pan

#endif // JMCM_BFGS_H_
