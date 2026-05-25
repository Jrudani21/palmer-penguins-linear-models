# ============================================================
# Palmer Penguins: Linear Models & ANCOVA
# STAT 4000 Portfolio Project 1  R Implementation
# Author: Jrudani21
# ============================================================

# ── 0. Install & load packages ───────────────────────────────
# install.packages(c("palmerpenguins", "tidyverse", "broom", "car", "emmeans", "GGally"))
library(palmerpenguins)
library(tidyverse)
library(broom)
library(car)
library(emmeans)
library(GGally)

data(penguins)

# ── 1. Quick data overview ────────────────────────────────────
glimpse(penguins)
summary(penguins)

# Drop rows with any NA (11 rows)
penguins_clean <- penguins |> drop_na()
cat("Clean rows:", nrow(penguins_clean), "\n")

# ── 2. EDA ────────────────────────────────────────────────────

# 2a. Pairs plot coloured by species
ggpairs(
  penguins_clean,
  columns  = c("bill_length_mm", "bill_depth_mm",
               "flipper_length_mm", "body_mass_g"),
  aes(colour = species, alpha = 0.6)
) +
  labs(title = "Penguin Measurements  Pairs Plot by Species") +
  theme_minimal()
ggsave("figures/01_pairs_plot.png", width = 10, height = 8)

# 2b. Body mass distribution by species
ggplot(penguins_clean, aes(x = species, y = body_mass_g,
                            fill = species)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.2) +
  labs(
    title   = "Body Mass by Species",
    x       = "Species",
    y       = "Body Mass (g)",
    caption = "Source: palmerpenguins package (Gorman et al. 2014)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("figures/02_boxplot_mass_species.png", width = 7, height = 5)

# ── 3. Linear Regression: body mass ~ flipper length ─────────

lm_flip <- lm(body_mass_g ~ flipper_length_mm, data = penguins_clean)
summary(lm_flip)

# Least-squares line
cat("\nRegression line:\n")
cat(sprintf("body_mass_g_hat = %.3f + %.3f * flipper_length_mm\n",
            coef(lm_flip)[1], coef(lm_flip)[2]))

# Plot with confidence band
ggplot(penguins_clean, aes(x = flipper_length_mm, y = body_mass_g,
                            colour = species)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(aes(group = 1), method = "lm",
              colour = "black", se = TRUE, linewidth = 1) +
  labs(
    title   = "Body Mass vs Flipper Length with Regression Line",
    x       = "Flipper Length (mm)",
    y       = "Body Mass (g)"
  ) +
  theme_minimal()
ggsave("figures/03_regression_flipper_mass.png", width = 8, height = 5)

# ── 4. Model Diagnostics ──────────────────────────────────────
png("figures/04_diagnostics_lm.png", width = 900, height = 700)
par(mfrow = c(2, 2))
plot(lm_flip)
dev.off()

# ── 5. One-Way ANOVA: bill length ~ species ───────────────────

anova_bill <- aov(bill_length_mm ~ species, data = penguins_clean)
summary(anova_bill)

# Normality check on residuals
shapiro_result <- shapiro.test(resid(anova_bill))
cat("\nShapiro-Wilk p-value:", shapiro_result$p.value, "\n")

# Tukey post-hoc
tukey_result <- TukeyHSD(anova_bill)
print(tukey_result)

# ── 6. Two-Way ANOVA: body mass ~ sex * species ───────────────

anova_2way <- aov(body_mass_g ~ sex * species, data = penguins_clean)
summary(anova_2way)

# Interaction plot
with(penguins_clean,
     interaction.plot(species, sex, body_mass_g,
                      col   = c("tomato", "steelblue"),
                      lwd   = 2,
                      xlab  = "Species",
                      ylab  = "Mean Body Mass (g)",
                      main  = "Interaction Plot: Species × Sex"))
# Save via png device
png("figures/05_interaction_plot.png", width = 700, height = 500)
with(penguins_clean,
     interaction.plot(species, sex, body_mass_g,
                      col  = c("tomato", "steelblue"),
                      lwd  = 2, xlab = "Species",
                      ylab = "Mean Body Mass (g)",
                      main = "Interaction Plot: Species × Sex"))
dev.off()

# ── 7. ANCOVA: body mass ~ species + flipper length ───────────

ancova_model <- lm(body_mass_g ~ species + flipper_length_mm,
                   data = penguins_clean)
Anova(ancova_model, type = "III")
summary(ancova_model)

# Adjusted means per species (controlling for flipper length)
emm <- emmeans(ancova_model, ~ species)
print(emm)

# ANCOVA plot: fitted lines per species (same slope, different intercepts)
penguins_clean$fitted_ancova <- fitted(ancova_model)

ggplot(penguins_clean,
       aes(x = flipper_length_mm, y = body_mass_g, colour = species)) +
  geom_point(alpha = 0.4, size = 2) +
  geom_line(aes(y = fitted_ancova), linewidth = 1) +
  labs(
    title   = "ANCOVA: Body Mass ~ Species + Flipper Length",
    subtitle = "Parallel regression lines = no species × flipper interaction",
    x       = "Flipper Length (mm)",
    y       = "Body Mass (g)"
  ) +
  theme_minimal()
ggsave("figures/06_ancova_fitted.png", width = 8, height = 5)

# ── 8. Summary table ─────────────────────────────────────────
cat("\n===== MODEL SUMMARY TABLE =====\n")
cat(sprintf("%-30s %s\n", "Model", "Key Result"))
cat(strrep("-", 60), "\n")
cat(sprintf("%-30s R² = %.3f, slope = %.1f g/mm\n",
            "Linear Reg (flip→mass)",
            summary(lm_flip)$r.squared,
            coef(lm_flip)["flipper_length_mm"]))
cat(sprintf("%-30s F(%d,%d)=%.2f, p<.001\n",
            "One-Way ANOVA (bill~species)",
            summary(anova_bill)[[1]]$Df[1],
            summary(anova_bill)[[1]]$Df[2],
            summary(anova_bill)[[1]]$`F value`[1]))
cat(sprintf("%-30s Interaction p=%.3f\n",
            "Two-Way ANOVA (mass~sex*spp)",
            summary(anova_2way)[[1]]$`Pr(>F)`[3]))
cat(sprintf("%-30s Adj-R² = %.3f\n",
            "ANCOVA (mass~species+flip)",
            summary(ancova_model)$adj.r.squared))
