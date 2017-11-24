#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP MCD__new(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP MCD__get_m(SEXP, SEXP);
extern SEXP MCD__get_Y(SEXP, SEXP);
extern SEXP MCD__get_X(SEXP, SEXP);
extern SEXP MCD__get_Z(SEXP, SEXP);
extern SEXP MCD__get_W(SEXP, SEXP);
extern SEXP MCD__get_D(SEXP, SEXP, SEXP);
extern SEXP MCD__get_T(SEXP, SEXP, SEXP);
extern SEXP MCD__get_mu(SEXP, SEXP, SEXP);
extern SEXP MCD__get_Sigma(SEXP, SEXP, SEXP);
extern SEXP MCD__n2loglik(SEXP, SEXP);
extern SEXP MCD__grad(SEXP, SEXP);
extern SEXP ACD__new(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP ACD__get_m(SEXP, SEXP);
extern SEXP ACD__get_Y(SEXP, SEXP);
extern SEXP ACD__get_X(SEXP, SEXP);
extern SEXP ACD__get_Z(SEXP, SEXP);
extern SEXP ACD__get_W(SEXP, SEXP);
extern SEXP ACD__get_D(SEXP, SEXP, SEXP);
extern SEXP ACD__get_T(SEXP, SEXP, SEXP);
extern SEXP ACD__get_mu(SEXP, SEXP, SEXP);
extern SEXP ACD__get_Sigma(SEXP, SEXP, SEXP);
extern SEXP ACD__n2loglik(SEXP, SEXP);
extern SEXP ACD__grad(SEXP, SEXP);
extern SEXP HPC__new(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP HPC__get_m(SEXP, SEXP);
extern SEXP HPC__get_Y(SEXP, SEXP);
extern SEXP HPC__get_X(SEXP, SEXP);
extern SEXP HPC__get_Z(SEXP, SEXP);
extern SEXP HPC__get_W(SEXP, SEXP);
extern SEXP HPC__get_D(SEXP, SEXP, SEXP);
extern SEXP HPC__get_T(SEXP, SEXP, SEXP);
extern SEXP HPC__get_mu(SEXP, SEXP, SEXP);
extern SEXP HPC__get_Sigma(SEXP, SEXP, SEXP);
extern SEXP HPC__n2loglik(SEXP, SEXP);
extern SEXP HPC__grad(SEXP, SEXP);
extern SEXP _jmcm_mcd_estimation(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _jmcm_acd_estimation(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _jmcm_hpc_estimation(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"MCD__new",             (DL_FUNC) &MCD__new,              5},
    {"MCD__get_m",           (DL_FUNC) &MCD__get_m,            2},
    {"MCD__get_Y",           (DL_FUNC) &MCD__get_Y,            2},
    {"MCD__get_X",           (DL_FUNC) &MCD__get_X,            2},
    {"MCD__get_Z",           (DL_FUNC) &MCD__get_Z,            2},
    {"MCD__get_W",           (DL_FUNC) &MCD__get_W,            2},
    {"MCD__get_D",           (DL_FUNC) &MCD__get_D,            3},
    {"MCD__get_T",           (DL_FUNC) &MCD__get_T,            3},
    {"MCD__get_mu",          (DL_FUNC) &MCD__get_mu,           3},
    {"MCD__get_Sigma",       (DL_FUNC) &MCD__get_Sigma,        3},
    {"MCD__n2loglik",        (DL_FUNC) &MCD__n2loglik,         2},
    {"MCD__grad",            (DL_FUNC) &MCD__grad,             2},
    {"ACD__new",             (DL_FUNC) &ACD__new,              5},
    {"ACD__get_m",           (DL_FUNC) &ACD__get_m,            2},
    {"ACD__get_Y",           (DL_FUNC) &ACD__get_Y,            2},
    {"ACD__get_X",           (DL_FUNC) &ACD__get_X,            2},
    {"ACD__get_Z",           (DL_FUNC) &ACD__get_Z,            2},
    {"ACD__get_W",           (DL_FUNC) &ACD__get_W,            2},
    {"ACD__get_D",           (DL_FUNC) &ACD__get_D,            3},
    {"ACD__get_T",           (DL_FUNC) &ACD__get_T,            3},
    {"ACD__get_mu",          (DL_FUNC) &ACD__get_mu,           3},
    {"ACD__get_Sigma",       (DL_FUNC) &ACD__get_Sigma,        3},
    {"ACD__n2loglik",        (DL_FUNC) &ACD__n2loglik,         2},
    {"ACD__grad",            (DL_FUNC) &ACD__grad,             2},
    {"HPC__new",             (DL_FUNC) &HPC__new,              5},
    {"HPC__get_m",           (DL_FUNC) &HPC__get_m,            2},
    {"HPC__get_Y",           (DL_FUNC) &HPC__get_Y,            2},
    {"HPC__get_X",           (DL_FUNC) &HPC__get_X,            2},
    {"HPC__get_Z",           (DL_FUNC) &HPC__get_Z,            2},
    {"HPC__get_W",           (DL_FUNC) &HPC__get_W,            2},
    {"HPC__get_D",           (DL_FUNC) &HPC__get_D,            3},
    {"HPC__get_T",           (DL_FUNC) &HPC__get_T,            3},
    {"HPC__get_mu",          (DL_FUNC) &HPC__get_mu,           3},
    {"HPC__get_Sigma",       (DL_FUNC) &HPC__get_Sigma,        3},
    {"HPC__n2loglik",        (DL_FUNC) &HPC__n2loglik,         2},
    {"HPC__grad",            (DL_FUNC) &HPC__grad,             2},
    {"_jmcm_mcd_estimation", (DL_FUNC) &_jmcm_mcd_estimation, 11},
    {"_jmcm_acd_estimation", (DL_FUNC) &_jmcm_acd_estimation, 11},
    {"_jmcm_hpc_estimation", (DL_FUNC) &_jmcm_hpc_estimation, 11},
    {NULL, NULL, 0}
};

void R_init_jmcm(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
