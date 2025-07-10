
import sys
import os
# Add parent directory to path to import pycic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pydoublelasso import DoublePostLasso

###########################
# Generate synthetic data #
###########################

np.random.seed(1988)
n, p = 100, 20

# Generate covariates as DataFrame
X = np.random.randn(n, p)
X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(p)])

# Generate treatment (binary)
D = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5 > 0).astype(int)
D_series = pd.Series(D, name='treatment')

# Generate outcome
Y = 2 * D + X[:, 2] + np.random.randn(n) * 0.5
Y_series = pd.Series(Y, name='outcome')

# Create full DataFrame
df = pd.DataFrame({
    'outcome': Y_series,
    'treatment': D_series
})
df = pd.concat([df, X_df], axis=1)

#########################
# Summary Statistics #
#########################

print("Data shape:", df.shape)
print("First few rows:")
print(df.head())
print(df.describe())

# Extract features and target variables
X_features = df.drop(columns=['outcome', 'treatment'])
y_target = df['outcome']
d_treatment = df['treatment']

#########################
# Fit Double Post-Lasso #
#########################

# Example with no bootstrap (faster)
print(f"\n{'='*20} CV TUNING (NO BOOTSTRAP) {'='*20}")
model = DoublePostLasso(tuning_method='cv', bootstrap=False)
model.fit(X_features, d_treatment, y_target)
model.print_ols_results() 

print(f"\n{'='*20} ADAPTIVE LASSO {'='*20}")
model_adaptive = DoublePostLasso(tuning_method='adaptive', bootstrap=False)
model_adaptive.fit(X_features, d_treatment, y_target)
model_adaptive.print_ols_results()

print(f"\n{'='*20} PLUGIN FORMULA {'='*20}")
model_plugin = DoublePostLasso(tuning_method='plugin', bootstrap=False)
model_plugin.fit(X_features, d_treatment, y_target)
model_plugin.print_ols_results()

################
# Bootstrap CI #
################

print(f"\n{'='*20} BOOTSTRAP CONFIDENCE INTERVALS {'='*20}")

# Test with bootstrap enabled (smaller sample for speed)
print("\n--- Bootstrap with CV tuning ---")
model_bootstrap = DoublePostLasso(tuning_method='cv', bootstrap=True, n_bootstrap=500, bootstrap_seed=1988)
model_bootstrap.fit(X_features, d_treatment, y_target)
model_bootstrap.print_ols_results()

# Get detailed bootstrap results
bootstrap_ci = model_bootstrap.get_bootstrap_ci(confidence_level=0.95)
print(f"\nBootstrap Treatment Effect Results (95% CI):")
print(f"  Mean: {bootstrap_ci['treatment_effect']['mean']:.6f}")
print(f"  Std: {bootstrap_ci['treatment_effect']['std']:.6f}")
print(f"  Lower bound: {bootstrap_ci['treatment_effect']['lower']:.6f}")
print(f"  Upper bound: {bootstrap_ci['treatment_effect']['upper']:.6f}")
