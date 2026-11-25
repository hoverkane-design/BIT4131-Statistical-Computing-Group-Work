# Topic 6: Hypothesis Testing (Python Implementation)
# Libraries: scipy.stats, pandas, numpy
# Real Dataset: US EPA Automotive Engine Fuel Economy Dataset (N = 398 Vehicles)

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("../data/topic6_auto_mpg_engines_real.csv")
mpg_all = df["mpg"].dropna().values.astype(float)
mpg_4 = df[df["cylinders"] == 4]["mpg"].dropna().values.astype(float)
mpg_68 = df[df["cylinders"].isin([6, 8])]["mpg"].dropna().values.astype(float)
mpg_usa = df[df["origin"] == "usa"]["mpg"].dropna().values.astype(float)
mpg_jpn = df[df["origin"] == "japan"]["mpg"].dropna().values.astype(float)
mpg_eur = df[df["origin"] == "europe"]["mpg"].dropna().values.astype(float)

# 1. One-Sample t-Test against CAFE regulatory benchmark mu_0 = 22.0 MPG
t1_stat, t1_p_two = stats.ttest_1samp(mpg_all, popmean=22.0)
print(f"One-Sample t-Test (H0: mu=22.0 vs H1: mu>22.0): t = {t1_stat:.4f}, one-sided p = {t1_p_two/2:.4e}")

# 2. Welch Two-Sample Independent t-Test (4-Cylinder vs 6/8-Cylinder Engines)
lev_stat, lev_p = stats.levene(mpg_4, mpg_68)
t2_stat, t2_p = stats.ttest_ind(mpg_4, mpg_68, equal_var=False)
print(f"Levene Variance Test: W = {lev_stat:.4f}, p = {lev_p:.4e}")
print(f"Welch Two-Sample t-Test (4-Cyl vs 6/8-Cyl): t = {t2_stat:.4f}, p = {t2_p:.4e}")

# 3. One-Way ANOVA across Manufacturing Region (USA, Japan, Europe)
f_stat, f_p = stats.f_oneway(mpg_usa, mpg_jpn, mpg_eur)
print(f"One-Way ANOVA (USA vs Japan vs Europe): F = {f_stat:.4f}, p = {f_p:.4e}")
