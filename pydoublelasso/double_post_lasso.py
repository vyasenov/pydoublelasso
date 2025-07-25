import numpy as np
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
import statsmodels.api as sm
import pandas as pd

class DoublePostLasso:
    """
    Implementation of the Double Post-Lasso procedure (Belloni, Chernozhukov, et al. 2014)
    with multiple lambda tuning options and bootstrap variance estimation.
    """
    def __init__(self, tuning_method='plugin', lambda_val=None, random_state=None, 
                 bootstrap=False, n_bootstrap=1000):
        """
        Initialize Double Post-Lasso.
        
        Parameters:
            tuning_method: str, one of ['cv', 'adaptive', 'plugin']
                - 'cv': Cross-validation
                - 'adaptive': Adaptive Lasso
                - 'plugin': Optimal plugin formula (default)
            lambda_val: float, optional. If provided, overrides tuning_method
            random_state: int, random state for reproducibility
            bootstrap: bool, whether to use bootstrap for variance estimation
            n_bootstrap: int, number of bootstrap samples (default: 1000)
        """
        self.tuning_method = tuning_method
        self.lambda_val = lambda_val
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        
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

    def _validate_data(self, X, D, Y):
        """
        Validate input data for quality and consistency.
        
        Parameters:
            X: np.ndarray or pd.DataFrame, control variables
            D: np.ndarray or pd.Series, treatment variable  
            Y: np.ndarray or pd.Series, outcome variable
            
        Raises:
            ValueError: If data validation fails
        """
        # Convert to numpy arrays for validation
        X_arr = np.asarray(X)
        D_arr = np.asarray(D)
        Y_arr = np.asarray(Y)
        
        # Check for missing values
        if np.any(np.isnan(X_arr)):
            raise ValueError("Control variables (X) contain missing values")
        if np.any(np.isnan(D_arr)):
            raise ValueError("Treatment variable (D) contains missing values")
        if np.any(np.isnan(Y_arr)):
            raise ValueError("Outcome variable (Y) contains missing values")

        # Check data types (should be numeric)
        if not np.issubdtype(X_arr.dtype, np.number):
            raise ValueError("Control variables (X) must be numeric")
        if not np.issubdtype(D_arr.dtype, np.number):
            raise ValueError("Treatment variable (D) must be numeric")
        if not np.issubdtype(Y_arr.dtype, np.number):
            raise ValueError("Outcome variable (Y) must be numeric")
        
        # Check dimensions
        n_samples_X = X_arr.shape[0]
        n_samples_D = D_arr.shape[0]
        n_samples_Y = Y_arr.shape[0]
        
        if not (n_samples_X == n_samples_D == n_samples_Y):
            raise ValueError(f"All variables must have the same number of samples. "
                           f"X: {n_samples_X}, D: {n_samples_D}, Y: {n_samples_Y}")
        
        # Check for sufficient sample size
        if n_samples_X < 10:
            raise ValueError(f"Sample size too small: {n_samples_X}. Need at least 10 observations.")
        
        # Check for sufficient features
        if X_arr.ndim == 1:
            n_features = 1
        else:
            n_features = X_arr.shape[1]
        
        if n_features == 0:
            raise ValueError("Control variables (X) must have at least one feature")
        
        # Check for constant variables (which can cause issues)
        for i in range(n_features):
            if np.std(X_arr[:, i]) == 0:
                raise ValueError(f"Control variable X[:, {i}] is constant")
        
        if np.std(D_arr) == 0:
            raise ValueError("Treatment variable (D) is constant")
        
        if np.std(Y_arr) == 0:
            raise ValueError("Outcome variable (Y) is constant")
        
        return X_arr, D_arr, Y_arr

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

    def _fit_lasso(self, X, y, step_name):
        """Fit Lasso with specified tuning method."""
        if self.lambda_val is not None:
            # Use provided lambda
            lambda_val = self.lambda_val
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
        np.random.seed(self.random_state)
        
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
                lasso_D = self._fit_lasso(X_boot, D_boot, 'D')
                active_D = np.where(lasso_D.coef_ != 0)[0]

                # Step 2: Lasso of Y on X
                lasso_Y = self._fit_lasso(X_boot, Y_boot, 'Y')
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

    def _get_bootstrap_ci(self):
        """
        Get bootstrap confidence intervals.
        
        Returns:
            dict: Confidence intervals for treatment effect and control coefficients
        """
        if not self.bootstrap or self.bootstrap_treatment_coefs_ is None:
            raise RuntimeError("Bootstrap not run. Set bootstrap=True in __init__.")
        
        alpha = 1 - 0.95 # Confidence level is 95%
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
        
    def fit(self, X, D, Y):
        """
        Run the double post-lasso procedure.
        Parameters:
            X: np.ndarray or pd.DataFrame, shape (n_samples, n_features) - control variables
            D: np.ndarray or pd.Series, shape (n_samples,) - treatment variable
            Y: np.ndarray or pd.Series, shape (n_samples,) - outcome variable
        """
        # Validate and convert data
        X_original = X
        X, D, Y = self._validate_data(X, D, Y)
        
        # Store column names if X was a DataFrame
        if isinstance(X_original, pd.DataFrame):
            self.X_columns_ = X_original.columns.tolist()
            self.X_is_dataframe_ = True
        else:
            self.X_columns_ = None
            self.X_is_dataframe_ = False
        
        # Step 1: Lasso of D on X
        lasso_D = self._fit_lasso(X, D, 'D')
        active_D = np.where(lasso_D.coef_ != 0)[0]

        # Step 2: Lasso of Y on X
        lasso_Y = self._fit_lasso(X, Y, 'Y')
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
        Predict outcomes using the fitted model.
        Parameters:
            X: np.ndarray or pd.DataFrame, shape (n_samples, n_features) - control variables
            D: np.ndarray or pd.Series, shape (n_samples,) - treatment variable (optional)
            
        Returns:
            np.ndarray: Predictions
        """
        if self.selected_vars_ is None or self.ols_results_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        # Store original input for pandas handling
        X_original = X
        
        # Simple validation for X
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        
        # Check for missing/infinite values in X
        if np.any(np.isnan(X_arr)) or np.any(np.isinf(X_arr)):
            raise ValueError("Control variables (X) contain missing or infinite values")
        
        # Validate D if provided
        if D is not None:
            D_arr = np.asarray(D)
            if np.any(np.isnan(D_arr)) or np.any(np.isinf(D_arr)):
                raise ValueError("Treatment variable (D) contains missing or infinite values")
        else:
            D_arr = None
        
        X_selected = X_arr[:, self.selected_vars_]
        
        # If D is provided, use the full model: Y = intercept + α*D + X_selected*β
        if D_arr is not None:
            predictions = (self.intercept_ + 
                         self.treatment_coef_ * D_arr.ravel() + 
                         np.dot(X_selected, self.control_coefs_))
        else:
            # If D is not provided, assume D=0 (control group prediction)
            predictions = (self.intercept_ + 
                         np.dot(X_selected, self.control_coefs_))
        
        # Return as pandas Series if input was a DataFrame and we have an index
        if isinstance(X_original, pd.DataFrame) and hasattr(X_original, 'index'):
            return pd.Series(predictions, index=X_original.index, name='predictions')
        
        return predictions
    
    def summary(self):
        """
        Display comprehensive results of the Double Post-Lasso procedure.
        
        Shows tuning parameters, selected variables, treatment effect with statistics,
        and bootstrap confidence intervals (if bootstrap=True was used).
        
        Returns:
            statsmodels.regression.linear_model.RegressionResultsWrapper: 
            The underlying OLS results object
        """
        if self.ols_results_ is None:
            raise RuntimeError("Model not fitted yet.")

        print("\n")
        print(f"Tuning method: {self.tuning_method}")
        if self.lambda_D_ is not None:
            print(f"Lambda for D regression: {self.lambda_D_:.6f}")
        if self.lambda_Y_ is not None:
            print(f"Lambda for Y regression: {self.lambda_Y_:.6f}")
        
        # Show selected variables with names if available
        names = ['intercept', 'treatment']
        if self.X_columns_ is not None and self.selected_vars_ is not None:
            names.extend([self.X_columns_[i] for i in self.selected_vars_])
        else:
            names.extend([f'X{i}' for i in self.selected_vars_])
        selected_names = names[2:]  # Skip 'intercept' and 'treatment'
        if selected_names is not None:
            print(f"Selected control variables: {selected_names}")
        else:
            print(f"Selected control variables (indices): {self.selected_vars_}")
        
        print(f"Number of selected control variables: {len(self.selected_vars_)}")
        
        # Always show bootstrap CI if available
        if self.bootstrap and self.bootstrap_treatment_coefs_ is not None:
            bootstrap_ci = self._get_bootstrap_ci()
            print(f"Bootstrap treatment effect CI (95%): [{bootstrap_ci['treatment_effect']['lower']:.4f}, {bootstrap_ci['treatment_effect']['upper']:.4f}]")
            print(f"Bootstrap treatment effect std: {bootstrap_ci['treatment_effect']['std']:.4f}")
        
        print("\n")
        print("Model: Y = α*D + X_selected*β + ε")
        # Show treatment effect prominently
        print(f"TREATMENT EFFECT (α): {self.treatment_coef_:.6f}")
        treatment_se = self.ols_results_.bse.iloc[1]  # Standard error of treatment effect
        treatment_t = self.ols_results_.tvalues.iloc[1]  # t-statistic
        treatment_p = self.ols_results_.pvalues.iloc[1]  # p-value
        print(f"Treatment effect std error: {treatment_se:.6f}")
        print(f"Treatment effect t-statistic: {treatment_t:.6f}")
        print(f"Treatment effect p-value: {treatment_p:.6f}")
        
        print("\n")
        print(self.ols_results_.summary())
        return self.ols_results_ 
