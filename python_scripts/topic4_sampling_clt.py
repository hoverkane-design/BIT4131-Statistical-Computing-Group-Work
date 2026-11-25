# Topic 4: Sampling Distributions and Central Limit Theorem (Python Implementation)
# Libraries: numpy, scipy, pandas, matplotlib
# Real Dataset: Retail Gem-Quality Diamond Prices (N = 53,940 Diamonds, USD)

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("../data/topic4_diamonds_prices_clt_real.csv")
prices = df["price"].dropna().values.astype(float)

pop_mean = np.mean(prices)
pop_sd = np.std(prices, ddof=0)
pop_skew = stats.skew(prices)
print(f"Population N = {len(prices)}, Mean = ${pop_mean:.2f}, SD = ${pop_sd:.2f}, Skew = {pop_skew:.3f}")

# Draw B = 2,000 repeated random samples for sample sizes n in [5, 15, 30, 100]
np.random.seed(4131)
B = 2000
for n in [5, 15, 30, 100]:
    idx = np.random.randint(0, len(prices), size=(B, n))
    sample_means = np.mean(prices[idx], axis=1)
    emp_mean = np.mean(sample_means)
    emp_se = np.std(sample_means, ddof=1)
    theo_se = pop_sd / np.sqrt(n)
    skew_val = stats.skew(sample_means)
    print(f"n = {n:3d} | Mean(X_bar) = ${emp_mean:.2f} | Emp SE = ${emp_se:.2f} | Theo SE = ${theo_se:.2f} | Skew = {skew_val:.3f}")
