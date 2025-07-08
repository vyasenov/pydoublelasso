# Deprecated: Use from pydoublelasso.core import DoublePostLasso
from .core import DoublePostLasso
import numpy as np
from sklearn.linear_model import LassoCV

class DoublePostLasso:
    """
    Minimal implementation of the Double Post-Lasso procedure (Belloni, Chernozhukov, et al.).
    """
    def __init__(self, alpha='auto', random_state=None):
        self.alpha = alpha
        self.random_state = random_state
        self.selected_vars_ = None
        self.final_model_ = None

    def fit(self, X, D, Y):
        """
        Run the double post-lasso procedure.
        Parameters:
            X: np.ndarray, shape (n_samples, n_features)
            D: np.ndarray, shape (n_samples,) or (n_samples, 1)
            Y: np.ndarray, shape (n_samples,) or (n_samples, 1)
        """
        # Step 1: Lasso of D on X
        lasso_D = LassoCV(cv=5, random_state=self.random_state) if self.alpha == 'auto' else LassoCV(alphas=[self.alpha], cv=5, random_state=self.random_state)
        lasso_D.fit(X, D.ravel())
        active_D = np.where(lasso_D.coef_ != 0)[0]

        # Step 2: Lasso of Y on X
        lasso_Y = LassoCV(cv=5, random_state=self.random_state) if self.alpha == 'auto' else LassoCV(alphas=[self.alpha], cv=5, random_state=self.random_state)
        lasso_Y.fit(X, Y.ravel())
        active_Y = np.where(lasso_Y.coef_ != 0)[0]

        # Step 3: Union of selected variables
        active_union = np.union1d(active_D, active_Y)
        self.selected_vars_ = active_union

        # Step 4: OLS (or Lasso) of Y on X[:, active_union]
        # For minimalism, use Lasso with very small alpha (almost OLS)
        if len(active_union) == 0:
            raise ValueError("No variables selected by either lasso.")
        X_selected = X[:, active_union]
        final_lasso = LassoCV(alphas=[1e-8], cv=5, random_state=self.random_state)
        final_lasso.fit(X_selected, Y.ravel())
        self.final_model_ = final_lasso
        return self

    def predict(self, X):
        if self.selected_vars_ is None or self.final_model_ is None:
            raise RuntimeError("Model not fitted yet.")
        X_selected = X[:, self.selected_vars_]
        return self.final_model_.predict(X_selected) 