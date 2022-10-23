# Factor-exposure-adjustments-using-autocorrelations

This study shows that even by using a simple adjustment formula, applying rolling factor autocorrelations to factor exposures can increase the portfolio Sharpe Ratio.

The code is using public available Fama-French factors from the Ken-French library and is applied to the 6-factor model. The study is performed using starting factor exposures equal to the S&P500 factor exposures and using the S&P500 as a benchmark.

The essence of the study is that if autocorrelations are positive and return is positive, the next return is most likely to be positive as well, signaling a buy signal. If autocorrelation are negative and return is positive, the positive return is most likely to be followed by a negative return, leading to a sell signal. If the autocorrelation is positive and return is negative, it is most likely to be a sell signal etc.

The study is using monthly observations, transac cost

In order to cater for empirical limitations, factor exposure boundaries have been set from -0.50 to +0.50. These boundaries can easily be accessed through the variables: min_b, max_b. The boundary for market exposure have been set to 0.7 - 1.3.
