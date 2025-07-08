
# pydoublelasso

A Python package for estimating treatment effects using the Double Post-Lasso procedure from Belloni, Chernozhukov, and Hansen (2014): *"Inference on Treatment Effects after Selection among High-Dimensional Controls."* This method is designed for valid inference in the presence of many covariates, using Lasso for model selection followed by OLS for estimation.

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
* Easy integration with pandas and scikit-learn pipelines

## Quick Start

```python
import pandas as pd
from pydoublelasso import DoubleLasso

# Simulated data
df = pd.read_csv("your_data.csv")

# Columns:
# 'Y' = outcome
# 'D' = treatment (binary or continuous)
# other columns = potential controls

# Set up model
X = df.drop(columns=['Y', 'D'])
y = df['Y']
d = df['D']

# Run Double Post-Lasso
dlasso = DoubleLasso()
results = dlasso.fit(y, d, X)

# Summary output
print(results.summary())

# Confidence intervals
ci = results.confint()
print(ci)
```

## Examples

See the `examples/` directory for use cases including:

* Treatment effect estimation with high-dimensional data
* Comparison with OLS and single-selection Lasso
* Binary and continuous treatments
* Bootstrap confidence intervals

## Background

### Why Double Lasso?

When estimating a treatment effect, including too many irrelevant controls inflates variance, while omitting important ones introduces bias. In high-dimensional settings, Lasso helps by selecting the most relevant controls. But naive Lasso inference is not valid due to post-selection bias.

Double Post-Lasso solves this by running Lasso twice:

1. Regress outcome $Y$ on covariates $X$ to find predictors of $Y$
2. Regress treatment $D$ on $X$ to find predictors of $D$
3. Take the union of selected variables, and use them in a final OLS regression of $Y$ on $D$ and selected $X$

This ensures that both confounders of the outcome and predictors of treatment are controlled for, yielding valid inference.

---

### Notation and Key Equations

Let:

* $Y$: outcome
* $D$: treatment (scalar)
* $X = (X_1, \dots, X_p)$: high-dimensional controls

The goal is to estimate the partial effect of $D$ on $Y$, denoted $\alpha$, in the partially linear model:

$$
Y_i = \alpha D_i + X_i^\top \beta + \varepsilon_i
$$

The Double Post-Lasso procedure follows:

1. Fit Lasso of $Y \sim X$, selecting variables $\hat{S}_Y$
2. Fit Lasso of $D \sim X$, selecting variables $\hat{S}_D$
3. Define selected set $\hat{S} = \hat{S}_Y \cup \hat{S}_D$
4. Estimate $\alpha$ by OLS on:

$$
Y_i = \alpha D_i + X_{i,\hat{S}}^\top \beta + \varepsilon_i
$$

This final regression yields an unbiased estimate of $\alpha$ with valid standard errors.

---

### Assumptions

* Sparsity: The true regression functions depend only on a small subset of covariates
* Exogeneity: $D$ is exogenous after controlling for $X$
* Approximate linearity: The relationships $Y \sim X$ and $D \sim X$ can be well-approximated linearly
* Regularization: Lasso is appropriately tuned for consistent variable selection

---

### Confidence Intervals

The final post-Lasso OLS regression produces valid asymptotic standard errors, even though variable selection was performed.

Additionally, the package supports bootstrap confidence intervals:

```python
results.bootstrap(n_bootstrap=500, ci_level=0.95)
```

These intervals account for randomness in both the selection and estimation stages.

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