# ============================================================
# Palmer Penguins: Linear Models & ANCOVA
# STAT 4000 Portfolio Project 1  Python Implementation
# Author: Jrudani21
# ============================================================

# %% [markdown]
# ## Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Load dataset (seaborn ships it built-in)
penguins = sns.load_dataset("penguins").dropna()
print(f"Dataset shape: {penguins.shape}")
print(penguins.dtypes)
penguins.head()

# %% [markdown]
# ## 1. Exploratory Data Analysis

# %%
# Pairs plot coloured by species
fig = sns.pairplot(
    penguins,
    vars=["bill_length_mm", "bill_depth_mm",
          "flipper_length_mm", "body_mass_g"],
    hue="species",
    diag_kind="kde",
    plot_kws={"alpha": 0.5}
)
fig.fig.suptitle("Penguin Measurements  Pairs Plot by Species", y=1.02)
plt.savefig("figures/py_01_pairs_plot.png", bbox_inches="tight", dpi=150)
plt.show()

# %%
# Boxplot: body mass by species
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=penguins, x="species", y="body_mass_g",
            palette="Set2", width=0.5, ax=ax)
sns.stripplot(data=penguins, x="species", y="body_mass_g",
              color="black", alpha=0.25, size=3, ax=ax)
ax.set_title("Body Mass by Species")
ax.set_xlabel("Species")
ax.set_ylabel("Body Mass (g)")
plt.tight_layout()
plt.savefig("figures/py_02_boxplot_mass.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Linear Regression: Body Mass ~ Flipper Length

# %%
lm_flip = smf.ols("body_mass_g ~ flipper_length_mm", data=penguins).fit()
print(lm_flip.summary())

b0, b1 = lm_flip.params
print(f"\nRegression line:")
print(f"body_mass_g_hat = {b0:.3f} + {b1:.3f} * flipper_length_mm")
print(f"R² = {lm_flip.rsquared:.4f}")

# %%
# Scatter + regression line coloured by species
palette = {"Adelie": "#FF8C00", "Chinstrap": "#9932CC", "Gentoo": "#008B8B"}

fig, ax = plt.subplots(figsize=(8, 5))
for sp, grp in penguins.groupby("species"):
    ax.scatter(grp["flipper_length_mm"], grp["body_mass_g"],
               label=sp, alpha=0.6, color=palette[sp], s=40)

# Overall regression line
x_range = np.linspace(penguins["flipper_length_mm"].min(),
                      penguins["flipper_length_mm"].max(), 200)
ax.plot(x_range, b0 + b1 * x_range, color="black",
        lw=2, label=f"OLS line (R²={lm_flip.rsquared:.3f})")

ax.set_title("Body Mass vs Flipper Length with Regression Line")
ax.set_xlabel("Flipper Length (mm)")
ax.set_ylabel("Body Mass (g)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/py_03_regression.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Model Diagnostics

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residual plot
fitted = lm_flip.fittedvalues
resids = lm_flip.resid
axes[0].scatter(fitted, resids, alpha=0.4, color="steelblue", s=30)
axes[0].axhline(0, color="red", lw=1.5, linestyle="--")
axes[0].set_title("Residual Plot")
axes[0].set_xlabel("Fitted Values")
axes[0].set_ylabel("Residuals")

# Normal Q-Q plot
sm.qqplot(resids, line="s", ax=axes[1], alpha=0.4)
axes[1].set_title("Normal Q-Q Plot")

plt.tight_layout()
plt.savefig("figures/py_04_diagnostics.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. One-Way ANOVA: Bill Length ~ Species

# %%
anova_bill = smf.ols("bill_length_mm ~ C(species)", data=penguins).fit()
anova_table = anova_lm(anova_bill, typ=1)
print("One-Way ANOVA  Bill Length ~ Species")
print(anova_table)

# Normality of residuals
stat, p = stats.shapiro(anova_bill.resid)
print(f"\nShapiro-Wilk: W={stat:.4f}, p={p:.4f}")

# Tukey HSD post-hoc
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(penguins["bill_length_mm"],
                           penguins["species"], alpha=0.05)
print("\nTukey HSD:")
print(tukey)

# %% [markdown]
# ## 5. Two-Way ANOVA: Body Mass ~ Sex * Species

# %%
anova_2way = smf.ols("body_mass_g ~ C(sex) * C(species)",
                      data=penguins).fit()
print("Two-Way ANOVA  Body Mass ~ Sex * Species")
print(anova_lm(anova_2way, typ=2))

# Interaction plot
fig, ax = plt.subplots(figsize=(7, 5))
means = (penguins.groupby(["species", "sex"])["body_mass_g"]
                 .mean().reset_index())
for sex, grp in means.groupby("sex"):
    ax.plot(grp["species"], grp["body_mass_g"],
            marker="o", lw=2, label=sex)
ax.set_title("Interaction Plot: Species × Sex")
ax.set_xlabel("Species")
ax.set_ylabel("Mean Body Mass (g)")
ax.legend(title="Sex")
plt.tight_layout()
plt.savefig("figures/py_05_interaction.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. ANCOVA: Body Mass ~ Species + Flipper Length

# %%
ancova = smf.ols(
    "body_mass_g ~ C(species) + flipper_length_mm",
    data=penguins
).fit()
print(ancova.summary())
print("\nType III ANOVA Table:")
print(anova_lm(ancova, typ=3))

# Adjusted means (marginal means at mean flipper length)
mean_flip = penguins["flipper_length_mm"].mean()
adj_means = {}
for sp in penguins["species"].unique():
    pred = ancova.predict(pd.DataFrame({
        "species": [sp], "flipper_length_mm": [mean_flip]
    }))
    adj_means[sp] = pred.values[0]

print(f"\nAdjusted means at flipper_length = {mean_flip:.1f} mm:")
for sp, val in adj_means.items():
    print(f"  {sp}: {val:.1f} g")

# %%
# ANCOVA fitted lines (parallel slopes)
fig, ax = plt.subplots(figsize=(8, 5))
x_vals = np.linspace(penguins["flipper_length_mm"].min(),
                     penguins["flipper_length_mm"].max(), 200)

for sp, color in palette.items():
    subset = penguins[penguins["species"] == sp]
    ax.scatter(subset["flipper_length_mm"], subset["body_mass_g"],
               alpha=0.4, color=color, s=30)
    y_line = ancova.predict(pd.DataFrame({
        "species": [sp] * 200,
        "flipper_length_mm": x_vals
    }))
    ax.plot(x_vals, y_line, color=color, lw=2, label=sp)

ax.set_title("ANCOVA: Body Mass ~ Species + Flipper Length\n"
             "(Parallel lines = no interaction assumed)")
ax.set_xlabel("Flipper Length (mm)")
ax.set_ylabel("Body Mass (g)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/py_06_ancova.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Results Summary

# %%
print("=" * 55)
print(f"{'Model':<35} {'Key Result'}")
print("-" * 55)
print(f"{'Linear Reg (flip→mass)':<35} R²={lm_flip.rsquared:.3f}")
print(f"{'One-Way ANOVA (bill~species)':<35} p<0.001")
print(f"{'Two-Way ANOVA (mass~sex*spp)':<35} interaction p={anova_lm(anova_2way,typ=2).iloc[2,3]:.3f}")
print(f"{'ANCOVA (mass~species+flip)':<35} Adj-R²={ancova.rsquared_adj:.3f}")
print("=" * 55)
