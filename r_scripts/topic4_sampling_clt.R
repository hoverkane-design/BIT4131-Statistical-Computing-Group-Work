# Topic 4: Sampling Distributions and Central Limit Theorem (R Implementation)
# Packages: stats, sampling, ggplot2
# Real Dataset: Retail Gem-Quality Diamond Prices (N = 53,940 Diamonds, USD)

library(stats)
library(sampling)
library(ggplot2)

df <- read.csv("../data/topic4_diamonds_prices_clt_real.csv")
prices <- na.omit(df$price)

pop_mean <- mean(prices)
pop_sd <- sd(prices)
cat(sprintf("Population N = %d | Mean = %.2f | SD = %.2f\n", length(prices), pop_mean, pop_sd))

set.seed(4131)
B <- 2000
for (n in c(5, 15, 30, 100)) {
  s_means <- replicate(B, mean(sample(prices, size = n, replace = TRUE)))
  emp_se <- sd(s_means)
  theo_se <- pop_sd / sqrt(n)
  cat(sprintf("n = %3d | Mean(X_bar) = %.2f | Empirical SE = %.2f | Theoretical SE = %.2f\n",
              n, mean(s_means), emp_se, theo_se))
}
