# palmer-penguins-linear-models
# 🐧 Penguins of the Palmer Archipelago: A Linear Modeling Field Guide

> **Do flipper length and species predict a penguin's body mass?**  
> Can we detect differences in bill length across species after controlling for body size?

[ANCOVA fitted lines](figures/py_06_ancova.png)

---

## 📋 Table of Contents
- [Problem](#problem)
- [Data](#data)
- [Methods](#methods)
- [Key Results](#key-results)
- [How to Reproduce](#how-to-reproduce)
- [What I'd Do Next](#what-id-do-next)

---

## Problem

This project uses the **Palmer Penguins** dataset to practice four classical statistical techniques:

1. **Simple linear regression**  how well does flipper length predict body mass?
2. **One-way ANOVA**  do the three penguin species differ in bill length?
3. **Two-way ANOVA**  is there an interaction between species and sex on body mass?
4. **ANCOVA**  after controlling for flipper length, do species-level differences in body mass remain?

The same analysis is implemented in **both R and Python** to demonstrate cross-language fluency.

---

## Data

| Property | Detail |
|---|---|
| **Source** | [palmerpenguins R package](https://allisonhorst.github.io/palmerpenguins/) / `seaborn.load_dataset("penguins")` |
| **Collector** | Dr. Kristen Gorman, Palmer Station LTER, Antarctica |
| **Size** | 344 penguins → 333 after dropping 11 rows with NAs |
| **Species** | Adélie (n=146), Chinstrap (n=68), Gentoo (n=119) |
| **Key variables** | `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `species`, `sex`, `island`, `year` |
| **License** | CC-0 (public domain) |

---

## Methods

### R Implementation (`R/penguins_analysis.R`)
| Step | Function | Package |
|---|---|---|
| EDA  pairs plot | `ggpairs()` | GGally |
| Linear regression | `lm()` | base R |
| Model diagnostics | `plot(model)` | base R |
| One-way ANOVA | `aov()` + `TukeyHSD()` | base R |
| Two-way ANOVA | `aov(y ~ A * B)` | base R |
| ANCOVA | `lm()` + `Anova(type=III)` | car |
| Adjusted means | `emmeans()` | emmeans |

### Python Implementation (`Python/penguins_analysis.py`)
| Step | Function | Package |
|---|---|---|
| EDA  pairs plot | `pairplot()` | seaborn |
| Linear regression | `smf.ols().fit()` | statsmodels |
| ANOVA | `anova_lm(model, typ=2)` | statsmodels |
| Tukey post-hoc | `pairwise_tukeyhsd()` | statsmodels |
| ANCOVA | `smf.ols("y ~ C(A) + x")` | statsmodels |
| Diagnostics | `qqplot()` | statsmodels |

---

## Key Results

### 1. Linear Regression
- **R² = 0.759**  flipper length alone explains ~76% of the variance in body mass
- Slope: each additional mm of flipper length → ~50 g more body mass
- Residual plots confirm linearity and constant variance ✅

### 2. One-Way ANOVA
- **F(2, 330) = 410.6, p < 0.001**  strong evidence of species differences in bill length
- Tukey post-hoc: all three pairwise comparisons (Adélie vs Chinstrap, Adélie vs Gentoo, Chinstrap vs Gentoo) are significant

### 3. Two-Way ANOVA
- Both `sex` (p < 0.001) and `species` (p < 0.001) have significant main effects on body mass
- **Interaction term p = 0.057**  marginal; the sex gap is similar across species

### 4. ANCOVA
- After controlling for flipper length, species differences in body mass **remain significant** (p < 0.001)
- Adjusted R² = **0.869**  adding species after flipper length improves the model substantially
- Adjusted means at mean flipper length (200.9 mm): Adélie ≈ 3706 g, Chinstrap ≈ 3734 g, Gentoo ≈ 5075 g

---

## How to Reproduce

### R
```r
# Install once
install.packages(c("palmerpenguins","tidyverse","broom","car","emmeans","GGally"))

# Run
source("R/penguins_analysis.R")
```

### Python
```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy
python Python/penguins_analysis.py
```

---

## What I'd Do Next
- Extend to a **linear mixed-effects model** (`lme4::lmer`) with `island` as a random effect
- Fit a **classification model** (logistic regression / LDA) to predict species from measurements
- Add a **Shiny app** (R) or **Streamlit dashboard** (Python) for interactive exploration

---

*Data: Gorman KB, Williams TD, Fraser WR (2014). PLoS ONE. | Package: Horst AM, Hill AP, Gorman KB (2020).*
