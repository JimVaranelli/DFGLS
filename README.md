# DFGLS
Python implementation of Dickey-Fuller Generalized Least Squares (DFGLS) test
of Elliot et al (1996) that can be used to test for a unit root in a univariate
process.

## Parameters
x : array_like, 1d \
&nbsp;&nbsp;&nbsp;&nbsp;data series \
regression : {'c','ct'} \
&nbsp;&nbsp;&nbsp;&nbsp;Constant and trend order to include in regression \
&nbsp;&nbsp;&nbsp;&nbsp;* 'c'  : constant only (default) \
&nbsp;&nbsp;&nbsp;&nbsp;* 'ct' : constant and trend \
maxlag : {int, None} \
&nbsp;&nbsp;&nbsp;&nbsp;maximum number of lags, default=12\*(nobs/100)^0.25
(Schwert, 1989) \
autolag : {'AIC','BIC', 't-stat', None} \
&nbsp;&nbsp;&nbsp;&nbsp;automatic lag length selection criterion \
&nbsp;&nbsp;&nbsp;&nbsp;* 'AIC'    : Akaike information criterion \
&nbsp;&nbsp;&nbsp;&nbsp;* 'BIC'    : Bayesian information criterion \
&nbsp;&nbsp;&nbsp;&nbsp;* 't-stat' : last lag is significant at 5% level (Ng
and Perron, 2001)\
&nbsp;&nbsp;&nbsp;&nbsp;* None     : lags set to maxlag

## Returns
stat : float \
&nbsp;&nbsp;&nbsp;&nbsp;test statistic \
pvalue : float \
&nbsp;&nbsp;&nbsp;&nbsp;based on MacKinnon (1994, 2010) regression surface \
lags : int \
&nbsp;&nbsp;&nbsp;&nbsp;number of lags used in regression \
cvdict : dict \
&nbsp;&nbsp;&nbsp;&nbsp;critical values for the test statistic at the 1%, 5%,
and 10% levels \
icbest : int \
&nbsp;&nbsp;&nbsp;&nbsp;maximum information criterion value if autolag is used

## Notes
H0 = series contains a unit-root (i.e., non-stationary)

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

## References
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

Ng, S., and Perron, P., (2001). Lag length selection and the construction of
unit root tests with good size and power. Econometrica, 69: 1519–1554.

Schwert, G.W. (1987). Effects of model specification on tests for unit
roots in macroeconomic data. Journal of Monetary Economics, 20: 73-103.

Schwert, G.W. (1989). Tests for unit roots: a Monte Carlo investigation.
Journal of Business and Economic Statistics, 2: 147-159.

Seabold, S., and Perktold, J. (2010). Statsmodels: econometric and
statistical modeling with python. In S. van der Walt and J. Millman
(Eds.), Proceedings of the 9th Python in Science Conference (pp. 57-61).

## Requirements
Python 3.7 \
Numpy 1.18.1 \
Statsmodels 0.11.0 \
Pandas 1.0.1

## Running
There are no parameters. The program is set up to access test files in the
.\results directory. This path can be modified in the source file.

## Additional Info
Please see comments in the source file for additional info including referenced
output for the test files.
