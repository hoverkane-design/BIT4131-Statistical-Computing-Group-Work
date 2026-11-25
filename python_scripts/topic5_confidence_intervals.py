# Topic 5: Confidence Intervals (Python Implementation)
# Libraries: scipy.stats, numpy, pandas
# Real Dataset: Baystate Medical Center Clinical Infant Birthweight Dataset (N = 189)

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("../data/topic5_birthwt_clinical_real.csv")

for label, sub in [
    ("All Newborns (N=189)", df["bwt"].values.astype(float)),
    ("Non-Smoking Mothers (n=115)", df[df["smoke"] == 0]["bwt"].values.astype(float)),
    ("Smoking Mothers (n=74)", df[df["smoke"] == 1]["bwt"].values.astype(float)),
]:
    n = len(sub)
    mean_v = np.mean(sub)
    se_v = stats.sem(sub)
    ci90 = stats.t.interval(0.90, df=n - 1, loc=mean_v, scale=se_v)
    ci95 = stats.t.interval(0.95, df=n - 1, loc=mean_v, scale=se_v)
    ci99 = stats.t.interval(0.99, df=n - 1, loc=mean_v, scale=se_v)

    rng = np.random.default_rng(4131)
    boot_means = np.mean(sub[rng.integers(0, n, size=(10000, n))], axis=1)
    boot95 = np.percentile(boot_means, [2.5, 97.5])

    print(f"{label}: Point Estimate = {mean_v:.2f} g (SE = {se_v:.2f} g)")
    print(f"  90% Student-t CI: [{ci90[0]:.2f}, {ci90[1]:.2f}] g")
    print(f"  95% Student-t CI: [{ci95[0]:.2f}, {ci95[1]:.2f}] g")
    print(f"  99% Student-t CI: [{ci99[0]:.2f}, {ci99[1]:.2f}] g")
    print(f"  95% Bootstrap CI: [{boot95[0]:.2f}, {boot95[1]:.2f}] g")
