# Topic 5: Confidence Intervals (R Implementation)
# Packages: stats, DescTools, boot
# Real Dataset: Baystate Medical Center Clinical Infant Birthweight Dataset (N = 189)

library(stats)
library(DescTools)
library(boot)

df <- read.csv("../data/topic5_birthwt_clinical_real.csv")
bwt <- df$bwt

# Parametric Student-t Confidence Intervals (90%, 95%, 99%)
MeanCI(bwt, conf.level = 0.90)
MeanCI(bwt, conf.level = 0.95)
MeanCI(bwt, conf.level = 0.99)

# Stratified 95% Confidence Intervals by Maternal Smoking Status
tapply(df$bwt, df$smoke, function(x) MeanCI(x, conf.level = 0.95))

# Non-parametric Bootstrap 95% Confidence Interval (R = 10,000 resamples)
set.seed(4131)
mean_fn <- function(data, indices) mean(data[indices])
boot_res <- boot(data = bwt, statistic = mean_fn, R = 10000)
boot.ci(boot_res, conf = 0.95, type = c("perc", "bca"))
