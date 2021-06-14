//  jmcm_config.h: configuration for code of joint mean-covariance models
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

#ifndef _JMCM_CONFIG_H_
#define _JMCM_CONFIG_H_

#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>

#include "bfgs.h"
#include "roptim.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#endif