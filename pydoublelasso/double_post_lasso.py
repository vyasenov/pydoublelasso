import numpy as np
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
import statsmodels.api as sm
import pandas as pd

class DoublePostLasso:
    """
    Implementation of the Double Post-Lasso procedure (Belloni, Chernozhukov, et al.)
    with multiple lambda tuning options and bootstrap variance estimation.
    """
    def __init__(self, tuning_method='plugin', alpha=None, random_state=None, 
                 bootstrap=False, n_bootstrap=1000, bootstrap_seed=None):
        """
        Initialize Double Post-Lasso.
        
        Parameters:
            tuning_method: str, one of ['cv', 'adaptive', 'plugin']
                - 'cv': Cross-validation
                - 'adaptive': Adaptive Lasso
                - 'plugin': Optimal plugin formula (default)
            alpha: float, optional. If provided, overrides tuning_method
            random_state: int, random state for reproducibility
            bootstrap: bool, whether to use bootstrap for variance estimation
            n_bootstrap: int, number of bootstrap samples (default: 1000)
            bootstrap_seed: int, random seed for bootstrap (default: random_state)
        """
        self.tuning_method = tuning_method
        self.alpha = alpha
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seed = bootstrap_seed if bootstrap_seed is not None else random_state
        
        self.selected_vars_ = None
        self.ols_results_ = None
        self.treatment_coef_ = None
        self.control_coefs_ = None
        self.intercept_ = None
        self.lambda_D_ = None
        self.lambda_Y_ = None
        
        # Bootstrap results
        self.bootstrap_treatment_coefs_ = None
        self.bootstrap_control_coefs_ = None
        self.bootstrap_intercepts_ = None
        
        # Store original data types and column names
        self.X_columns_ = None
        self.X_is_dataframe_ = False

    def _get_lambda(self, X, y, method):
        """Get lambda using specified tuning method."""
        if method == 'cv':
            # Cross-validation
            lasso_cv = LassoCV(cv=5, random_state=self.random_state)
            lasso_cv.fit(X, y.ravel())
            return lasso_cv.alpha_
        
        elif method == 'adaptive':
            # Adaptive Lasso approach
            # First stage: OLS to get initial weights
            ols = LinearRegression()
            ols.fit(X, y.ravel())
            initial_coefs = ols.coef_
            
            # Avoid division by zero
            weights = 1.0 / (np.abs(initial_coefs) + 1e-8)
            
            # Second stage: weighted Lasso
            # Use cross-validation to find optimal lambda for weighted problem
            lasso_cv = LassoCV(cv=5, random_state=self.random_state)
            lasso_cv.fit(X * weights, y.ravel())
            return lasso_cv.alpha_
        
        elif method == 'plugin':
            # Optimal plugin formula
            n, p = X.shape
            
            # Belloni et al. (2014) plugin formula
            # λ = 2c * sqrt(log(p)/n) where c is typically 1.1
            c = 1.1
            lambda_plugin = 2 * c * np.sqrt(np.log(p) / n)
            
            return lambda_plugin
        
        else:
            raise ValueError(f"Unknown tuning method: {method}")

    def _fit_lasso_with_tuning(self, X, y, step_name):
        """Fit Lasso with specified tuning method."""
        if self.alpha is not None:
            # Use provided alpha
            lambda_val = self.alpha
        else:
            # Use specified tuning method
            lambda_val = self._get_lambda(X, y, self.tuning_method)
        
        # Store lambda value
        if step_name == 'D':
            self.lambda_D_ = lambda_val
        else:
            self.lambda_Y_ = lambda_val
        
        # Fit Lasso with selected lambda
        lasso = Lasso(alpha=lambda_val, random_state=self.random_state)
        lasso.fit(X, y.ravel())
        
        return lasso

    def _bootstrap_sample(self, X, D, Y):
        """Generate bootstrap samples."""
        n = X.shape[0]
        np.random.seed(self.bootstrap_seed)
        
        bootstrap_treatment_coefs = []
        bootstrap_control_coefs = []
        bootstrap_intercepts = []
        
        for i in range(self.n_bootstrap):
            # Generate bootstrap indices
            indices = np.random.choice(n, size=n, replace=True)
            
            # Bootstrap samples
            X_boot = X[indices]
            D_boot = D[indices]
            Y_boot = Y[indices]
            
            try:
                # Fit Double Post-Lasso on bootstrap sample
                # Step 1: Lasso of D on X
                lasso_D = self._fit_lasso_with_tuning(X_boot, D_boot, 'D')
                active_D = np.where(lasso_D.coef_ != 0)[0]

                # Step 2: Lasso of Y on X
                lasso_Y = self._fit_lasso_with_tuning(X_boot, Y_boot, 'Y')
                active_Y = np.where(lasso_Y.coef_ != 0)[0]

                # Step 3: Union of selected variables
                active_union = np.union1d(active_D, active_Y)
                
                if len(active_union) > 0:
                    # Step 4: OLS of Y on D and X[:, active_union]
                    X_selected_boot = X_boot[:, active_union]
                    X_with_D_boot = np.column_stack([D_boot.ravel(), X_selected_boot])
                    X_with_constant_boot = sm.add_constant(X_with_D_boot)
                    
                    ols_model_boot = sm.OLS(Y_boot.ravel(), X_with_constant_boot)
                    ols_results_boot = ols_model_boot.fit()
                    
                    # Store coefficients
                    bootstrap_intercepts.append(ols_results_boot.params[0])
                    bootstrap_treatment_coefs.append(ols_results_boot.params[1])
                    bootstrap_control_coefs.append(ols_results_boot.params[2:])
                else:
                    # If no variables selected, use zeros
                    bootstrap_intercepts.append(0)
                    bootstrap_treatment_coefs.append(0)
                    bootstrap_control_coefs.append(np.array([]))
                    
            except:
                # If fitting fails, use zeros
                bootstrap_intercepts.append(0)
                bootstrap_treatment_coefs.append(0)
                bootstrap_control_coefs.append(np.array([]))
        
        return (np.array(bootstrap_intercepts), 
                np.array(bootstrap_treatment_coefs), 
                bootstrap_control_coefs)

    def fit(self, X, D, Y):
        """
        Run the double post-lasso procedure.
        Parameters:
            X: np.ndarray or pd.DataFrame, shape (n_samples, n_features) - control variables
            D: np.ndarray or pd.Series, shape (n_samples,) - treatment variable
            Y: np.ndarray or pd.Series, shape (n_samples,) - outcome variable
        """
        # Convert to numpy arrays and store metadata
        X_original = X
        X = np.asarray(X)
        D = np.asarray(D)
        Y = np.asarray(Y)
        
        # Store column names if X was a DataFrame
        if isinstance(X_original, pd.DataFrame):
            self.X_columns_ = X_original.columns.tolist()
            self.X_is_dataframe_ = True
        else:
            self.X_columns_ = None
            self.X_is_dataframe_ = False
        
        # Step 1: Lasso of D on X
        lasso_D = self._fit_lasso_with_tuning(X, D, 'D')
        active_D = np.where(lasso_D.coef_ != 0)[0]

        # Step 2: Lasso of Y on X
        lasso_Y = self._fit_lasso_with_tuning(X, Y, 'Y')
        active_Y = np.where(lasso_Y.coef_ != 0)[0]

        # Step 3: Union of selected variables
        active_union = np.union1d(active_D, active_Y)
        self.selected_vars_ = active_union

        # Step 4: OLS of Y on D and X[:, active_union]
        if len(active_union) == 0:
            raise ValueError("No variables selected by either lasso.")
        
        X_selected = X[:, active_union]
        
        # Include treatment variable D in the final regression
        # Y = αD + X_selected*β + ε
        X_with_D = np.column_stack([D.ravel(), X_selected])
        X_with_constant = sm.add_constant(X_with_D)
        
        # Create variable names for better output
        var_names = ['const', 'D']
        if self.X_columns_ is not None:
            # Use actual column names
            selected_names = [self.X_columns_[i] for i in active_union]
            var_names.extend(selected_names)
        else:
            # Use generic names
            var_names.extend([f'X{i}' for i in active_union])
        
        # Create DataFrame with proper column names to preserve variable names
        df_for_ols = pd.DataFrame(X_with_constant, columns=var_names)
        ols_model = sm.OLS(Y.ravel(), df_for_ols)
        self.ols_results_ = ols_model.fit(cov_type='HC1')
        
        # Store coefficients for prediction
        self.intercept_ = self.ols_results_.params.iloc[0]  # constant
        self.treatment_coef_ = self.ols_results_.params.iloc[1]  # treatment effect
        self.control_coefs_ = self.ols_results_.params.iloc[2:].values  # control variables
        
        # Run bootstrap if requested
        if self.bootstrap:
            print(f"Running bootstrap with {self.n_bootstrap} samples...")
            (self.bootstrap_intercepts_, 
             self.bootstrap_treatment_coefs_, 
             self.bootstrap_control_coefs_) = self._bootstrap_sample(X, D, Y)
        
        return self

    def predict(self, X, D=None):
        """
        Predict using the fitted model.
        Parameters:
            X: np.ndarray or pd.DataFrame, shape (n_samples, n_features) - control variables
            D: np.ndarray or pd.Series, shape (n_samples,) - treatment variable (optional)
        """
        if self.selected_vars_ is None or self.ols_results_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        # Convert to numpy arrays
        X = np.asarray(X)
        if D is not None:
            D = np.asarray(D)
        
        X_selected = X[:, self.selected_vars_]
        
        # If D is provided, use the full model: Y = intercept + α*D + X_selected*β
        if D is not None:
            predictions = (self.intercept_ + 
                         self.treatment_coef_ * D.ravel() + 
                         np.dot(X_selected, self.control_coefs_))
        else:
            # If D is not provided, assume D=0 (control group prediction)
            predictions = (self.intercept_ + 
                         np.dot(X_selected, self.control_coefs_))
        
        return predictions
    
    def get_selected_variable_names(self):
        """
        Get names of selected variables if X was a pandas DataFrame.
        
        Returns:
            list: Names of selected variables, or None if X was not a DataFrame
        """
        if self.X_columns_ is not None and self.selected_vars_ is not None:
            return [self.X_columns_[i] for i in self.selected_vars_]
        return None
    
    def get_bootstrap_ci(self, confidence_level=0.95):
        """
        Get bootstrap confidence intervals.
        
        Parameters:
            confidence_level: float, confidence level (default: 0.95)
            
        Returns:
            dict: Confidence intervals for treatment effect and control coefficients
        """
        if not self.bootstrap or self.bootstrap_treatment_coefs_ is None:
            raise RuntimeError("Bootstrap not run. Set bootstrap=True in __init__.")
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Treatment effect confidence interval
        treatment_ci = np.percentile(self.bootstrap_treatment_coefs_, 
                                   [lower_percentile, upper_percentile])
        
        return {
            'treatment_effect': {
                'lower': treatment_ci[0],
                'upper': treatment_ci[1],
                'mean': np.mean(self.bootstrap_treatment_coefs_),
                'std': np.std(self.bootstrap_treatment_coefs_)
            }
        }
    
    def print_ols_results(self):
        """Print the OLS regression results from Step 4."""
        if self.ols_results_ is None:
            raise RuntimeError("Model not fitted yet.")
        print("=" * 60)
        print("DOUBLE POST-LASSO RESULTS")
        print("=" * 60)
        print(f"Tuning method: {self.tuning_method}")
        if self.lambda_D_ is not None:
            print(f"Lambda for D regression: {self.lambda_D_:.6f}")
        if self.lambda_Y_ is not None:
            print(f"Lambda for Y regression: {self.lambda_Y_:.6f}")
        
        # Show selected variables with names if available
        if self.X_columns_ is not None:
            selected_names = self.get_selected_variable_names()
            print(f"Selected control variables: {selected_names}")
        else:
            print(f"Selected control variables (indices): {self.selected_vars_}")
        
        print(f"Number of selected control variables: {len(self.selected_vars_)}")
        
        if self.bootstrap and self.bootstrap_treatment_coefs_ is not None:
            print(f"Bootstrap samples: {self.n_bootstrap}")
            bootstrap_ci = self.get_bootstrap_ci()
            print(f"Bootstrap treatment effect CI (95%): [{bootstrap_ci['treatment_effect']['lower']:.4f}, {bootstrap_ci['treatment_effect']['upper']:.4f}]")
            print(f"Bootstrap treatment effect std: {bootstrap_ci['treatment_effect']['std']:.4f}")
        
        print("\n" + "=" * 60)
        print("FINAL OLS REGRESSION (Step 4)")
        print("=" * 60)
        print("Model: Y = α*D + X_selected*β + ε")
        print("=" * 60)
        
        # Show treatment effect prominently
        print(f"TREATMENT EFFECT (α): {self.treatment_coef_:.6f}")
        treatment_se = self.ols_results_.bse.iloc[1]  # Standard error of treatment effect
        treatment_t = self.ols_results_.tvalues.iloc[1]  # t-statistic
        treatment_p = self.ols_results_.pvalues.iloc[1]  # p-value
        print(f"Treatment effect std error: {treatment_se:.6f}")
        print(f"Treatment effect t-statistic: {treatment_t:.6f}")
        print(f"Treatment effect p-value: {treatment_p:.6f}")
        
        print("\n" + "=" * 60)
        print("FULL REGRESSION RESULTS:")
        print("=" * 60)
        print(self.ols_results_.summary())
        return self.ols_results_ 