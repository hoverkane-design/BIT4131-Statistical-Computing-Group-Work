# Topic 8: Linear Regression and Least Squares (R Implementation)
# Packages: stats, car, broom
# Real Dataset: Babikian et al. (2002) Commercial Aircraft & Jet Engine Efficiency (1959-2002)

library(stats)
library(car)
library(broom)

df_tsfc <- read.csv("../data/topic8_jet_engine_tsfc_real.csv")
df_eu <- read.csv("../data/topic8_aircraft_energy_use_real.csv")
df_eu_models <- subset(df_eu, Aircraft_Type %in% c("Large Aircraft", "Regional Aircraft"))

# Model 1: Simple OLS Regression of Jet Engine Cruise TSFC vs Year (N = 62)
lm_tsfc <- lm(TSFC_Cruise_mg_per_Ns ~ Year, data = df_tsfc)
summary(lm_tsfc)
tidy(lm_tsfc, conf.int = TRUE)
glance(lm_tsfc)

# Model 2: Multiple OLS Regression controlling for Propulsion Class
lm_tsfc_multi <- lm(TSFC_Cruise_mg_per_Ns ~ Year + factor(Aircraft_Type), data = df_tsfc)
summary(lm_tsfc_multi)
Anova(lm_tsfc_multi, type = "II")

# Model 3: Simple OLS Regression of Aircraft Energy Usage EU (MJ/ASK) vs Year (N = 82)
lm_eu <- lm(EU_MJ_per_ASK ~ Year, data = df_eu_models)
summary(lm_eu)
tidy(lm_eu, conf.int = TRUE)
