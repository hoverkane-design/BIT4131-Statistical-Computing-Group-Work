# Topic 3: Probability Distributions (R Implementation)
# Packages: stats, MASS, fitdistrplus
# Real Dataset: New York Daily Air Quality Measurements (Wind Speed mph & Ozone ppb)

library(stats)
library(MASS)
library(fitdistrplus)

df <- read.csv("../data/topic3_airquality_wind_ozone_real.csv")
wind <- na.omit(df$Wind)
ozone <- na.omit(df$Ozone)

# Fit Normal, Gamma, and Weibull distributions to Wind Speed via MLE
fit_norm_w <- fitdist(wind, "norm")
fit_gamma_w <- fitdist(wind, "gamma")
fit_weib_w <- fitdist(wind, "weibull")

summary(fit_norm_w)
summary(fit_gamma_w)
summary(fit_weib_w)

# Goodness-of-fit statistics (AIC, BIC, Kolmogorov-Smirnov)
gofstat(list(fit_norm_w, fit_gamma_w, fit_weib_w),
        fitnames = c("Normal", "Gamma", "Weibull"))

# Fit Normal, Gamma, and Lognormal distributions to right-skewed Ozone (ppb)
fit_norm_oz <- fitdist(ozone, "norm")
fit_gamma_oz <- fitdist(ozone, "gamma")
fit_lnorm_oz <- fitdist(ozone, "lnorm")
gofstat(list(fit_norm_oz, fit_gamma_oz, fit_lnorm_oz),
        fitnames = c("Normal", "Gamma", "Lognormal"))

# Plot histogram with fitted density curves
denscomp(list(fit_norm_w, fit_gamma_w, fit_weib_w),
         legendtext = c("Normal", "Gamma", "Weibull"),
         main = "Fitted Probability Distributions on Real Wind Speed (mph)")
