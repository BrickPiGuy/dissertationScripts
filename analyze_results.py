# === Required Libraries ===
# pip install statsmodels scipy pingouin

import pandas as pd
from statsmodels.stats.anova import AnovaRM
from itertools import combinations
from scipy.stats import ttest_rel, shapiro
from statsmodels.stats.multitest import multipletests
import pingouin as pg

# === Step 1: Load data ===
data = pd.read_csv("results/run_log.csv")

# Check if required columns are present
required_columns = {'trial_number', 'token_count', 'parameter_efficiency_loss'}
if not required_columns.issubset(data.columns):
    raise ValueError(f"Missing required columns: {required_columns - set(data.columns)}")

# === Step 2: Run Repeated-Measures ANOVA for parameter_efficiency_loss ===
anova_loss = AnovaRM(data, depvar='parameter_efficiency_loss', subject='trial_number', within=['token_count']).fit()
print("\nRepeated-Measures ANOVA on Parameter Efficiency Loss:")
print(anova_loss)

# === Check ANOVA p-value significance ===
p_value_anova = anova_loss.anova_table['Pr > F'][0]
alpha = 0.05

if p_value_anova < alpha:
    print(f"\nResult: Significant main effect detected (p = {p_value_anova:.4f} < {alpha}). Proceed with post-hoc tests.")
else:
    print(f"\nResult: No significant main effect (p = {p_value_anova:.4f} ≥ {alpha}). Post-hoc tests may not be necessary.")

# === Step 3: Descriptive Statistics ===
print("\nDescriptive Statistics by Token Count:")
summary_stats = data.groupby('token_count')['parameter_efficiency_loss'].agg(['mean', 'std', 'min', 'max'])
print(summary_stats)

# === Step 4: Post-hoc pairwise t-tests (Bonferroni corrected) ===
df_wide = data.pivot(index='trial_number', columns='token_count', values='parameter_efficiency_loss')
pairs = list(combinations(df_wide.columns, 2))
p_values = []
results = []

for a, b in pairs:
    t_stat, p = ttest_rel(df_wide[a], df_wide[b])
    results.append((a, b, t_stat, p))
    p_values.append(p)

# Apply Bonferroni correction
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

print("\nBonferroni-Corrected Pairwise Comparisons:")
for i, (a, b, t_stat, raw_p) in enumerate(results):
    print(f"Comparison: {a} vs {b}")
    print(f"  t = {t_stat:.3f}, raw p = {raw_p:.4f}, corrected p = {pvals_corrected[i]:.4f}, reject H0 = {reject[i]}")

# === Step 5: Normality Check (Shapiro-Wilk) ===
print("\nNormality Check (Shapiro-Wilk):")
for token in df_wide.columns:
    stat, p = shapiro(df_wide[token])
    print(f"  {token}: W={stat:.3f}, p={p:.4f} -> {'Normal' if p > 0.05 else 'Non-normal'}")

# === Step 6: Sphericity Check (Pingouin) ===
df_long = pd.melt(df_wide.reset_index(), id_vars='trial_number', var_name='token_count', value_name='efficiency')
aov = pg.rm_anova(dv='efficiency', within='token_count', subject='trial_number', data=df_long, detailed=True)
sphericity_test = pg.sphericity(data=df_long, dv='efficiency', subject='trial_number', within='token_count')

print("\nDetailed Repeated-Measures ANOVA (Pingouin):")
print(aov)

print("\nMauchly’s Test for Sphericity:")
print(sphericity_test)
