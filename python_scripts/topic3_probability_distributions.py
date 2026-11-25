# Topic 3: Probability Distributions (Python Implementation)
# Libraries: scipy.stats, numpy, pandas, matplotlib
# Real Dataset: New York Daily Air Quality Measurements (Wind Speed mph & Ozone ppb)

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("../data/topic3_airquality_wind_ozone_real.csv")
wind = df["Wind"].dropna().values
ozone = df["Ozone"].dropna().values

# Maximum Likelihood Estimation (MLE) for Wind Speed (mph)
norm_mu, norm_sd = stats.norm.fit(wind)
gamma_shape, gamma_loc, gamma_scale = stats.gamma.fit(wind, floc=0)
weib_shape, weib_loc, weib_scale = stats.weibull_min.fit(wind, floc=0)

# Maximum Likelihood Estimation (MLE) for Ground-Level Ozone (ppb)
oz_norm_mu, oz_norm_sd = stats.norm.fit(ozone)
oz_gamma_shape, oz_gamma_loc, oz_gamma_scale = stats.gamma.fit(ozone, floc=0)
oz_logn_s, oz_logn_loc, oz_logn_scale = stats.lognorm.fit(ozone, floc=0)

# Goodness-of-fit Kolmogorov-Smirnov test and Log-Likelihood comparison
ks_norm = stats.kstest(wind, "norm", args=(norm_mu, norm_sd))
ks_gamma = stats.kstest(wind, "gamma", args=(gamma_shape, 0, gamma_scale))
ks_weib = stats.kstest(wind, "weibull_min", args=(weib_shape, 0, weib_scale))

# Tail exceedance probabilities
p_wind_gt_12 = 1.0 - stats.weibull_min.cdf(12.0, weib_shape, loc=0, scale=weib_scale)
p_ozone_gt_60 = 1.0 - stats.gamma.cdf(60.0, oz_gamma_shape, loc=0, scale=oz_gamma_scale)

print(f"Wind Normal Fit: mu = {norm_mu:.3f}, sigma = {norm_sd:.3f}, KS p = {ks_norm.pvalue:.4f}")
print(f"Wind Gamma Fit: shape = {gamma_shape:.3f}, scale = {gamma_scale:.3f}, KS p = {ks_gamma.pvalue:.4f}")
print(f"Wind Weibull Fit: shape = {weib_shape:.3f}, scale = {weib_scale:.3f}, KS p = {ks_weib.pvalue:.4f}")
print(f"Probability P(Wind > 12 mph) = {p_wind_gt_12:.4f}")
print(f"Probability P(Ozone > 60 ppb) = {p_ozone_gt_60:.4f}")
