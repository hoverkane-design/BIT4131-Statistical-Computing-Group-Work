# Topic 8: Linear Regression and Least Squares (Python Implementation)
# Libraries: numpy, scipy.stats, scikit-learn, pandas
# Real Dataset: Babikian et al. (2002) Commercial Aircraft & Jet Engine Efficiency (1959-2002)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

df_tsfc = pd.read_csv("../data/topic8_jet_engine_tsfc_real.csv")
df_eu = pd.read_csv("../data/topic8_aircraft_energy_use_real.csv")
df_eu_models = df_eu[df_eu["Aircraft_Type"].isin(["Large Aircraft", "Regional Aircraft"])].copy()


def fit_ols(X: np.ndarray, y: np.ndarray, names):
    n, k = X.shape
    df_r = n - k
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - (X @ beta)
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_r
    s2 = rss / df_r
    se = np.sqrt(np.diag(s2 * xtx_inv))
    t_v = beta / se
    p_v = 2.0 * (1.0 - stats.t.cdf(np.abs(t_v), df=df_r))
    print(f"R-squared = {r2:.4f} | Adjusted R-squared = {adj_r2:.4f} | N = {n}")
    for nm, b, s, tv, pv in zip(names, beta, se, t_v, p_v):
        print(f"  {nm:18s}: coef = {b:10.4f} | SE = {s:8.4f} | t = {tv:8.3f} | p = {pv:.4e}")


# Model 1: Simple OLS on Jet Engine Cruise TSFC [mg/(N*s)] vs Year (N = 62)
print("Model 1: Simple OLS — Jet Engine Cruise TSFC vs. Year")
X1 = np.column_stack([np.ones(len(df_tsfc)), df_tsfc["Year"].values.astype(float)])
y1 = df_tsfc["TSFC_Cruise_mg_per_Ns"].values.astype(float)
fit_ols(X1, y1, ["Intercept", "Year"])

# Model 2: Multiple OLS controlling for Engine Class (Large Jets, Regional Jets, Turboprops)
print("\nModel 2: Multiple OLS — Jet Engine Cruise TSFC vs. Year + Engine Class")
is_reg = (df_tsfc["Aircraft_Type"] == "Regional Jets").astype(float).values
is_tp = (df_tsfc["Aircraft_Type"] == "Turboprops").astype(float).values
X2 = np.column_stack([np.ones(len(df_tsfc)), df_tsfc["Year"].values.astype(float), is_reg, is_tp])
fit_ols(X2, y1, ["Intercept", "Year", "Regional_Jet", "Turboprop"])

# Model 3: Simple OLS on Total Aircraft Energy Usage EU [MJ/ASK] vs Year
print("\nModel 3: Simple OLS — Commercial Aircraft Energy Usage (MJ/ASK) vs. Year")
X3 = np.column_stack([np.ones(len(df_eu_models)), df_eu_models["Year"].values.astype(float)])
y3 = df_eu_models["EU_MJ_per_ASK"].values.astype(float)
fit_ols(X3, y3, ["Intercept", "Year"])

sk_reg = LinearRegression().fit(df_tsfc[["Year"]], df_tsfc["TSFC_Cruise_mg_per_Ns"])
print(f"\nScikit-Learn Verification: TSFC = {sk_reg.intercept_:.4f} + ({sk_reg.coef_[0]:.5f}) * Year")
