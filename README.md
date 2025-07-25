# pydoublelasso

![](https://img.shields.io/badge/license-MIT-green)

A Python package for estimating treatment effects using the Double Post-Lasso procedure from Belloni, Chernozhukov, and Hansen (2014). This method is designed for valid inference in the presence of many covariates, using Lasso for model selection followed by OLS for estimation.

## Installation

You can install the package using pip:

```bash
pip install pydoublelasso
````

## Features

* High-dimensional treatment effect estimation with many covariates
* Double selection Lasso for robust control variable selection
* Post-selection inference with valid confidence intervals
* Supports binary or continuous treatment variables
* Bootstrap and asymptotic confidence intervals
* Easy integration with `pandas` and `scikit-learn` pipelines

## Quick Start

```python
import pandas as pd
import numpy as np
from pydoublelasso import DoublePostLasso

# Generate synthetic data
np.random.seed(1988)
n_obs, n_features = 1000, 50

# Generate covariates
X = np.random.randn(n, p)
D = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5  # Treatment depends on X0, X1
Y = 2 * D + X[:, 2] + np.random.randn(n) * 0.5  # Outcome depends on D and X2

# Run Double Post-Lasso
model = DoublePostLasso()
model.fit(X, D, Y)

# Get selected variables
print("Selected variables:", model.selected_vars_)

# Make predictions
y_pred = model.predict(X.values)
print("First 5 predictions:", y_pred[:5])
```

## Examples

See the `examples/` directory for use cases including:

## Background

### Why Double Lasso?

When estimating a treatment effect, including too many irrelevant controls inflates variance, while omitting important confounders introduces bias. In high-dimensional settings, Lasso helps by selecting a sparse set of relevant covariates. However, two problems arise: (1) standard confidence intervals after Lasso are invalid due to model selection, and (2) Lasso estimates are biased toward zero because of regularization.

Double Post-Lasso, proposed by Belloni, Chernozhukov, and Hansen (2014), addresses this by performing variable selection in both the outcome and treatment equations. This approach ensures that the model controls for variables that influence either the treatment or the outcome, yielding valid estimates and confidence intervals for the treatment effect.

---

### Notation

Let's establish the following notation:

* $Y$: outcome variable
* $D$: treatment variable
* $X = (X_1, \dots, X_p)$: high-dimensional controls variables

---

### Estimation

The goal is to estimate the partial effect of $D$ on $Y$, denoted $\alpha$, in the partially linear model:

$$
Y_i = \alpha D_i + f(X_i) + \varepsilon_i
$$

The Double Post-Lasso procedure follows:

1. Fit Lasso of $Y \sim X$, selecting variables $\hat{S}_Y$
2. Fit Lasso of $D \sim X$, selecting variables $\hat{S}_D$
3. Define selected set $\hat{S} = \hat{S}_Y \cup \hat{S}_D$
4. Estimate $\alpha$ by OLS on:

$$
Y_i = \alpha D_i + X_{i,\hat{S}}^\top \beta + \varepsilon_i
$$

Belloni et al. (2014) show this final regression delivers a consistent and asymptotically normal estimator of $\alpha$.

---

### Assumptions

The method relies on the following key assumptions:

* Sparsity: The true regression functions depend only on a small subset of covariates
* Exogeneity: $D$ is exogenous after controlling for $X$
* Approximate linearity: The relationships $Y \sim X$ and $D \sim X$ can be well-approximated linearly
* Regularization: Lasso is appropriately tuned for consistent variable selection

---

### Confidence Intervals

The final post-Lasso OLS regression produces valid asymptotic standard errors, even though variable selection was performed. Additionally, the package supports bootstrap confidence intervals which account for randomness in both the selection and estimation stages.

---

## References

* Belloni, A., Chernozhukov, V., & Hansen, C. (2014). *Inference on treatment effects after selection among high-dimensional controls*. *The Review of Economic Studies*, 81(2), 608–650.
* Tibshirani, R. (1996). *Regression shrinkage and selection via the Lasso*. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

To cite this package in publications, use the following BibTeX entry:

```bibtex
@misc{yasenov2025pydoublelasso,
  author       = {Vasco Yasenov},
  title        = {pydoublelasso: Python Implementation of the Double Post-Lasso Estimator},
  year         = {2025},
  howpublished = {\url{https://github.com/vyasenov/pydoublelasso}},
  note         = {Version 0.1.0}
}
```