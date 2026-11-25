# Topic 6: Hypothesis Testing (R Implementation)
# Packages: stats, car, rstatix
# Real Dataset: US EPA Automotive Engine Fuel Economy Dataset (N = 398 Vehicles)

library(stats)
library(car)
library(rstatix)

df <- read.csv("../data/topic6_auto_mpg_engines_real.csv")

# 1. One-Sample t-Test against CAFE benchmark mu_0 = 22.0 MPG
t.test(df$mpg, mu = 22.0, alternative = "greater")

# 2. Levene Test for Homogeneity of Variance and Welch Two-Sample t-Test
df_sub <- subset(df, cylinders %in% c(4, 6, 8))
df_sub$cyl_group <- ifelse(df_sub$cylinders == 4, "4_Cyl", "6_8_Cyl")
leveneTest(mpg ~ factor(cyl_group), data = df_sub)
t.test(mpg ~ cyl_group, data = df_sub, var.equal = FALSE)

# 3. One-Way ANOVA and Tukey Post-Hoc Test across Manufacturing Origin
fit_aov <- aov(mpg ~ factor(origin), data = df)
summary(fit_aov)
TukeyHSD(fit_aov)
