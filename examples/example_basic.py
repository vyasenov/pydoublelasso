import numpy as np
from pydoublelasso import DoublePostLasso

# Generate synthetic data
np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)
D = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5  # Treatment depends on X0, X1
Y = 2 * D + X[:, 2] + np.random.randn(n) * 0.5  # Outcome depends on D and X2

# Fit Double Post-Lasso
model = DoublePostLasso()
model.fit(X, D, Y)

# Predict
Y_pred = model.predict(X)

print("Selected variables (union):", model.selected_vars_)
print("First 5 predictions:", Y_pred[:5]) 