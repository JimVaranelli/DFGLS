import sys
import os
import time
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from numpy.testing import assert_equal, assert_almost_equal


def dfgls(x, regression='c', maxlag=None, autolag='AIC'):
    """
    DFGLS unit-root test

    The Dickey-Fuller Generalized Least Squares (DFGLS) test of Elliot et al
    (1996) can be used to test for a unit root in a univariate process.

    Parameters
    ----------
    x : array_like
        data series
    regression : {'c', 'ct'}
        constant and trend order to include in regression
        * 'c'  : constant only (default)
        * 'ct' : constant and trend
    maxlag : {int, None}
        maximum number of lags, default is from Schwert (1989) and
        specified as 12*(nobs/100)^(0.25)
    autolag : {'AIC', 'BIC', 't-stat', None}
        * 'AIC'    : Akaike information criterion (default)
        * 'BIC'    : Bayesian information criterion
        * 't-stat' : last lag is significant at the 5% level
        * None     : lags set to maxlags

    Returns
    -------
    stat : float
        test statistic
    pvalue : float
        based on MacKinnon (1994, 2010) regression surface model
    lags : int
        number of lags used in regression
    nobs : int
        number of observations used in regression
    cvdict : dict
        critical values for the test statistic at 1%, 5%, 10%
    icbest : float
        maximum information criterion value if autolag is used

    Notes
    -----
    H0 = series has a unit root (non-stationary)

    Basic process is to create a demeaned/detrended series from the original
    input series on which a zero-mean ADF tau test is performed to determine
    unit root existence. The original series is transformed via generalized
    least squares regression. Elliot et al (1996) show that the DFGLS test has
    increased power and efficiency over the ADF test especially with respect
    to near-stationary processes in finite samples.

    The function adfuller() from statsmodels (2010) package is used for
    calculating the ADF tau test on the demeaned/detrended series. Critical
    values are obtained from statsmodels implementation of MacKinnon's (1994,
    2010) regression surface model.

    References
    ----------
    Dickey, D.A., and Fuller, W.A. (1979). Distribution of the estimators for
    autoregressive time series with a unit root. Journal of the American
    Statistical Association, 74: 427-431.

    Dickey, D.A., and Fuller, W.A. (1981). Likelihood ratio statistics for
    autoregressive time series with a unit root. Econometrica, 49: 1057-1072.

    Elliot, G., Rothenberg, T.J., and Stock, J.H. (1996). Efficient tests for
    an autoregressive unit root. Econometrica, 64: 813-836.

    MacKinnon, J.G. (1994). Approximate asymptotic distribution functions for
    unit-root and cointegration tests. Journal of Business and Economic
    Statistics, 12: 167-176.

    MacKinnon, J.G. (2010). Critical values for cointegration tests. Working
    Paper 1227, Queen's University, Department of Economics. Retrieved from
    URL: https://www.econ.queensu.ca/research/working-papers.

    Ng, S., and Perron, P., (2001). Lag length selection and the construction
    of unit root tests with good size and power. Econometrica, 69: 1519–1554.

    Schwert, G.W. (1987). Effects of model specification on tests for unit
    roots in macroeconomic data. Journal of Monetary Economics, 20: 73-103.

    Schwert, G.W. (1989). Tests for unit roots: a Monte Carlo investigation.
    Journal of Business and Economic Statistics, 2: 147-159.

    Seabold, S., and Perktold, J. (2010). Statsmodels: econometric and
    statistical modeling with python. In S. van der Walt and J. Millman
    (Eds.), Proceedings of the 9th Python in Science Conference (pp. 57-61).
    """
    if regression not in ['c', 'ct']:
        raise ValueError(
            'DFGLS: regression option \'{}\' not understood'.format(
                regression))
    if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1):
        raise ValueError(
            'DFGLS: x must be a 1d array or a 2d array with a single column')
    x = np.reshape(x, (-1, 1))
    # set alpha according to regression type
    # and initialize exog matrix
    if regression == 'c':
        alpha = 1 - (7 / x.shape[0])
        exog = np.ones(shape=(x.shape[0], 1))
    else:
        alpha = 1 - (13.5 / x.shape[0])
        exog = np.ones(shape=(x.shape[0], 2))
    # set up endog vector for auxiliary regression
    endog = np.copy(x)
    endog[1:] -= alpha * endog[:-1]
    # set up exog matrix for auxiliary regression
    exog[1:, 0] -= alpha
    if regression == 'ct':
        exog[1:, 1] = np.arange(2, x.shape[0] + 1)
        exog[1:, 1] -= alpha * exog[:-1, 1]
    # run auxiliary regression
    arc = OLS(endog, exog).fit().params
    # demean/detrend original series converted
    # to 1d for statsmodels adfuller() input
    dtx = np.copy(x.reshape(x.shape[0],)) - arc[0]
    if regression == 'ct':
        dtx -= arc[1] * np.arange(1, x.shape[0] + 1)
    # run ADF on demeaned/detrended series
    # setting regression type to 'nc'
    res = adfuller(dtx, regression='nc', maxlag=maxlag, autolag=autolag)
    # need to correct the critical/p-values for 'ct'
    if regression == 'ct':
        cvs = mackinnoncrit(N=1, regression='c', nobs=len(dtx))
        res = [res[0], mackinnonp(res[0], regression='c', N=1), res[2], res[3],
               {"1%": cvs[0], "5%": cvs[1], "10%": cvs[2]}]
    return res


# output results
def _print_res(res, st):
    print("  dfgls-stat =", "{0:0.5f}".format(res[0]), " pval =",
          "{0:0.5f}".format(res[1]), " arlags = {}".format(res[2]),
          " nobs = {}".format(res[3]))
    print("    cvdict = \'1%\': {0:0.5f}".format(res[4]["1%"]),
          " \'5%\': {0:0.5f}".format(res[4]["5%"]),
          " \'10%\': {0:0.5f}".format(res[4]["10%"]))
    print("    time =", "{0:0.5f}".format(time.time() - st))


# unit tests taken from Schwert (1987) and verified
# against SAS 9.4 and R package urca 1.3-0
def main():
    print("DFGLS unit-root test...")
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    run_dir = os.path.join(cur_dir, "results")
    files = ['BAA.csv', 'DBAA.csv', 'SP500.csv', 'DSP500.csv', 'UN.csv',
             'DUN.csv']
    for file in files:
        print(" test file =", file)
        mdl_file = os.path.join(run_dir, file)
        mdl = np.asarray(pd.read_csv(mdl_file))
        st = time.time()
        if file == 'BAA.csv':
            res = dfgls(mdl, maxlag=3, autolag=None)
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], 0.32491, decimal=5)
            assert_almost_equal(res[1], 0.78154, decimal=5)
            assert_equal(res[2], 3)
            st = time.time()
            res = dfgls(mdl, regression='ct')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -2.33832, decimal=5)
            assert_almost_equal(res[1], 0.01866, decimal=5)
            assert_equal(res[2], 17)
        elif file == 'DBAA.csv':
            res = dfgls(mdl)
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -4.07340, decimal=5)
            assert_almost_equal(res[1], 0.00005, decimal=5)
            assert_equal(res[2], 16)
            st = time.time()
            res = dfgls(mdl, regression='ct', maxlag=8, autolag='t-stat')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -4.43873, decimal=5)
            assert_almost_equal(res[1], 0.00001, decimal=5)
            assert_equal(res[2], 8)
        elif file == 'SP500.csv':
            res = dfgls(mdl)
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], 2.45932, decimal=5)
            assert_almost_equal(res[1], 0.99781, decimal=5)
            assert_equal(res[2], 8)
            st = time.time()
            res = dfgls(mdl, regression='ct')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -2.02347, decimal=3)
            assert_almost_equal(res[1], 0.04118, decimal=3)
            assert_equal(res[2], 5)
        elif file == 'DSP500.csv':
            res = dfgls(mdl)
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -6.83435, decimal=5)
            assert_almost_equal(res[1], 0.00000, decimal=5)
            assert_equal(res[2], 4)
            st = time.time()
            res = dfgls(mdl, regression='ct', autolag='BIC')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -6.31870, decimal=5)
            assert_almost_equal(res[1], 0.00000, decimal=5)
            assert_equal(res[2], 4)
        elif file == 'UN.csv':
            res = dfgls(mdl, maxlag=8)
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -1.66359, decimal=5)
            assert_almost_equal(res[1], 0.09096, decimal=5)
            assert_equal(res[2], 7)
            st = time.time()
            res = dfgls(mdl, regression='ct', autolag='t-stat')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -2.98420, decimal=5)
            assert_almost_equal(res[1], 0.00280, decimal=5)
            assert_equal(res[2], 12)
        elif file == 'DUN.csv':
            res = dfgls(mdl, maxlag=8, autolag='t-stat')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -2.26414, decimal=5)
            assert_almost_equal(res[1], 0.02267, decimal=5)
            assert_equal(res[2], 4)
            st = time.time()
            res = dfgls(mdl, regression='ct')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], -2.67000, decimal=5)
            assert_almost_equal(res[1], 0.00737, decimal=5)
            assert_equal(res[2], 14)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
