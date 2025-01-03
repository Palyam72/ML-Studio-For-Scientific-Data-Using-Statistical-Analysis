import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.metrics import (
    d2_absolute_error_score, d2_pinball_score, d2_tweedie_score,
    explained_variance_score, max_error, mean_absolute_error,
    mean_absolute_percentage_error, mean_gamma_deviance,
    mean_pinball_loss, mean_poisson_deviance, mean_squared_error,
    mean_squared_log_error, mean_tweedie_deviance, median_absolute_error,
    r2_score
)

class Regression:
    def __init__(self, dataset):
        self.dataset = dataset
        self.col1, self.col2, self.col3 = st.columns([1, 1, 1])
        self.xTrain, self.xTest, self.yTrain, self.yTest = None, None, None, None
        self.model = None
    def display(self):
        self.train_test_split()
        with self.col2:
            st.subheader("Classical Linear Model",divider='blue')
            models = [
                "LinearRegression", "Ridge", "RidgeCV", "SGDRegressor",
                "ElasticNet", "ElasticNetCV", "Lars", "LarsCV", "Lasso",
                "LassoCV", "LassoLars", "LassoLarsCV", "LassoLarsIC",
                "OrthogonalMatchingPursuit", "OrthogonalMatchingPursuitCV",
                "ARDRegression", "BayesianRidge",
                "MultiTaskElasticNet", "MultiTaskElasticNetCV",
                "MultiTaskLasso", "MultiTaskLassoCV",
                "HuberRegressor", "QuantileRegressor", 
                "RANSACRegressor", "TheilSenRegressor",
                "GammaRegressor", "PoissonRegressor", "TweedieRegressor"
            ]
            Model=st.pills("Select the regressor that you want",models)
            # Define the actions based on the selected model
            if Model == "LinearRegression":
                self.linear_regression()
            elif Model == "Ridge":
                self.ridge_regression()
            elif Model == "RidgeCV":
                self.ridge_cv()
            elif Model == "SGDRegressor":
                self.sgd_regressor()
            elif Model == "ElasticNet":
                self.elasticNet()
            elif Model == "ElasticNetCV":
                self.elasticNetCV()
            elif Model == "Lars":
                self.lars()
            elif Model == "LarsCV":
                self.larscv()
            elif Model == "Lasso":
                self.lasso()
            elif Model == "LassoCV":
                self.lasso_cv()
            elif Model == "LassoLars":
                self.lassolars()
            elif Model == "LassoLarsCV":
                self.lasso_lars_cv()
            elif Model == "LassoLarsIC":
                self.lasso_lars_ic()
            elif Model == "OrthogonalMatchingPursuit":
                self.omp()
            elif Model == "OrthogonalMatchingPursuitCV":
                self.omp_cv()
            elif Model == "ARDRegression":
                self.ard_regression()
            elif Model == "BayesianRidge":
                self.bayesian_ridge()
            elif Model == "MultiTaskElasticNet":
                self.multi_task_elastic_net()
            elif Model == "MultiTaskElasticNetCV":
                self.multi_task_elastic_net_cv()
            elif Model == "MultiTaskLasso":
                self.multi_task_lasso()
            elif Model == "MultiTaskLassoCV":
                self.multi_task_lasso_cv()
            elif Model == "HuberRegressor":
                self.huber_regressor()
            elif Model == "QuantileRegressor":
                self.quantile_regressor()
            elif Model == "RANSACRegressor":
                self.ransac_regressor()
            elif Model == "TheilSenRegressor":
                self.theil_sen_regressor()
            elif Model == "GammaRegressor":
                self.gamma_regressor()
            elif Model == "PoissonRegressor":
                self.poisson_regressor()
            elif Model == "TweedieRegressor":
                self.tweedie_regressor()
                
                
                                       
    def train_test_split(self):
        with self.col1:
            st.subheader("Train-Test Split Configuration",divider='blue')
            target_column = st.selectbox("Select the target column", self.dataset.columns)

            if not target_column:
                st.warning("Please select a target column to proceed.")
                return None, None

            x_data = self.dataset.drop(columns=[target_column])
            y_data = self.dataset[target_column]

            test_size = st.slider("Test size (as a proportion)", 0.1, 0.9, 0.2, 0.05)
            shuffle = st.checkbox("Shuffle the data before splitting", value=True)
            if st.checkbox("Confirm to apply the train test split"):
                self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
                    x_data, y_data, test_size=test_size, shuffle=shuffle
                )
    
                st.markdown("### Train-Test Split Completed!")
                st.write("Training data shape:", self.xTrain.shape, self.yTrain.shape)
                st.write("Testing data shape:", self.xTest.shape, self.yTest.shape)

    def linear_regression(self):
        with self.col2:
            st.subheader("Linear Regression Configuration",divider='blue')
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            positive = st.checkbox("Force Positive Coefficients", value=False)

            if st.checkbox("Train Linear Regression Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return

                self.model = LinearRegression(fit_intercept=fit_intercept, positive=positive)
                self.model.fit(self.xTrain, self.yTrain)

                st.success("Linear Regression Model Trained Successfully!")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                self.regression_metrics()
    def ridge_regression(self):
        with self.col2:
            st.subheader("Ridge Regression Configuration", divider='blue')
            
            # Ridge-specific hyperparameters
            alpha = st.slider("Alpha (Regularization Strength)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            positive = st.checkbox("Force Positive Coefficients", value=False)
            solver = st.selectbox("Solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"])
            
            if st.checkbox("Train Ridge Regression Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Ridge Regression model
                self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, positive=positive, solver=solver)
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Ridge Regression Model Trained Successfully!")
                
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                
                # Call regression metrics
                self.regression_metrics()
    def ridge_cv(self):
        with self.col2:
            st.subheader("RidgeCV Regression Configuration", divider='blue')
    
            # RidgeCV-specific hyperparameters
            alphas = st.text_input("Alphas (Regularization Strength)", value="(0.1, 1.0, 10.0)")
            alphas = eval(alphas)  # Convert input string to tuple of floats
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            alpha_per_target = st.checkbox("Alpha Per Target", value=False)
            scoring = st.selectbox("Scoring Method", ["None", "neg_mean_squared_error", "r2", "neg_mean_absolute_error"])
            cv = st.number_input("Cross-validation Folds", min_value=2, max_value=10, value=5)
            
            if st.checkbox("Train RidgeCV Regression Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the RidgeCV model
                self.model = RidgeCV(alphas=alphas, fit_intercept=fit_intercept, alpha_per_target=alpha_per_target,
                                     scoring=scoring if scoring != "None" else None, cv=cv)
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("RidgeCV Regression Model Trained Successfully!")
                
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Best Alpha:** {self.model.alpha_}")
                
                # Call regression metrics
                self.regression_metrics()
    def sgd_regressor(self):
        with self.col2:
            st.subheader("SGD Regressor Configuration", divider='blue')
    
            # SGD Regressor-specific hyperparameters
            loss = st.selectbox("Loss Function", ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"], index=0)
            penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "None"], index=0)
            alpha = st.number_input("Alpha (Regularization Strength)", min_value=0.0, value=0.0001)
            l1_ratio = st.slider("L1 Ratio (Elastic Net)", min_value=0.0, max_value=1.0, value=0.15)
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            max_iter = st.number_input("Max Iterations", min_value=1, max_value=10000, value=1000)
            tol = st.number_input("Tolerance", min_value=0.0, value=0.001)
            shuffle = st.checkbox("Shuffle Data", value=True)
            verbose = st.number_input("Verbose Level", min_value=0, value=0)
            epsilon = st.number_input("Epsilon (Huber Loss)", min_value=0.0, value=0.1)
            learning_rate = st.selectbox("Learning Rate", ["constant", "optimal", "invscaling", "adaptive"], index=2)
            eta0 = st.number_input("Initial Learning Rate", min_value=0.0, value=0.01)
            power_t = st.number_input("Power T for InvScaling", min_value=0.0, value=0.25)
            early_stopping = st.checkbox("Early Stopping", value=False)
            validation_fraction = st.number_input("Validation Fraction (for Early Stopping)", min_value=0.0, max_value=1.0, value=0.1)
            n_iter_no_change = st.number_input("Number of Iterations with No Change", min_value=1, value=5)
            warm_start = st.checkbox("Warm Start", value=False)
            average = st.checkbox("Use Averaging for SGD", value=False)
    
            if st.checkbox("Train SGD Regressor Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the SGDRegressor model
                self.model = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
                                          fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
                                          shuffle=shuffle, verbose=verbose, epsilon=epsilon,
                                          learning_rate=learning_rate, eta0=eta0, power_t=power_t,
                                          early_stopping=early_stopping, validation_fraction=validation_fraction,
                                          n_iter_no_change=n_iter_no_change, warm_start=warm_start,
                                          average=average)
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("SGD Regressor Model Trained Successfully!")
    
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Total Updates:** {self.model.t_}")
    
                # Call regression metrics
                self.regression_metrics()
    def elasticNet(self):
        with self.col2:
            st.subheader("ElasticNet Configuration", divider='blue')
    
            # ElasticNet-specific hyperparameters
            alpha = st.number_input("Alpha (Regularization Strength)", min_value=0.0, value=1.0)
            l1_ratio = st.slider("L1 Ratio (Elastic Net Mixing)", min_value=0.0, max_value=1.0, value=0.5)
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            precompute = st.checkbox("Use Precomputed Gram Matrix", value=False)
            max_iter = st.number_input("Max Iterations", min_value=1, max_value=10000, value=1000)
            tol = st.number_input("Tolerance", min_value=0.0, value=0.0001)
            warm_start = st.checkbox("Warm Start", value=False)
            positive = st.checkbox("Force Positive Coefficients", value=False)
            random_state = st.number_input("Random State (Set for Reproducibility)", min_value=0, value=0)
            selection = st.selectbox("Feature Selection Strategy", ["cyclic", "random"], index=0)
    
            if st.checkbox("Train ElasticNet Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the ElasticNet model
                self.model = ElasticNet(
                    alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
                    precompute=precompute, max_iter=max_iter, tol=tol, warm_start=warm_start,
                    positive=positive, random_state=random_state if random_state != 0 else None,
                    selection=selection
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("ElasticNet Model Trained Successfully!")
    
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
    
                # Call regression metrics
                self.regression_metrics()
    def elasticNetCV(self):
        with self.col2:
            st.subheader("ElasticNetCV Configuration", divider="blue")
            
            # ElasticNetCV-specific hyperparameters
            l1_ratio = st.multiselect(
                "L1 Ratio (Elastic Net Mixing Parameter)",
                options=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                default=[0.5]
            )
            eps = st.number_input("Eps (Length of Path)", min_value=0.0001, value=0.001)
            n_alphas = st.number_input("Number of Alphas", min_value=1, value=100)
            max_iter = st.number_input("Max Iterations", min_value=1, max_value=10000, value=1000)
            tol = st.number_input("Tolerance", min_value=0.0, value=0.0001)
            cv = st.number_input("Number of Folds for Cross-Validation", min_value=2, value=5)
            positive = st.checkbox("Force Positive Coefficients", value=False)
            selection = st.selectbox("Feature Selection Method", ["cyclic", "random"], index=0)
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            random_state = st.number_input("Random State", min_value=0, value=0, step=1)
            verbose = st.number_input("Verbose Level", min_value=0, value=0)
            
            if st.checkbox("Train ElasticNetCV Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the ElasticNetCV model
                self.model = ElasticNetCV(
                    l1_ratio=l1_ratio,
                    eps=eps,
                    n_alphas=n_alphas,
                    max_iter=max_iter,
                    tol=tol,
                    cv=cv,
                    positive=positive,
                    selection=selection,
                    fit_intercept=fit_intercept,
                    random_state=random_state,
                    verbose=verbose,
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("ElasticNetCV Model Trained Successfully!")
                
                st.markdown("### Model Attributes")
                st.write(f"**Alpha (Best Regularization):** {self.model.alpha_}")
                st.write(f"**L1 Ratio (Best Mix):** {self.model.l1_ratio_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
    
                # Call regression metrics
                self.regression_metrics()
    def lars(self):
        with self.col2:
            st.subheader("Lars Configuration", divider="blue")
            
            # Lars-specific hyperparameters
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            verbose = st.checkbox("Enable Verbose Output", value=False)
            precompute = st.selectbox(
                "Precompute Gram Matrix",
                options=["auto", True, False],
                index=0,
            )
            n_nonzero_coefs = st.number_input(
                "Number of Non-Zero Coefficients", min_value=1, value=500
            )
            eps = st.number_input(
                "Machine Precision Regularization (eps)", 
                min_value=0.0, 
                value=float(np.finfo(float).eps),
            )
            copy_X = st.checkbox("Copy X (Avoid Overwriting Data)", value=True)
            fit_path = st.checkbox("Store Full Path in `coef_path_`", value=True)
            jitter = st.number_input(
                "Jitter (Upper Bound of Noise)", 
                min_value=0.0, 
                value=0.0,
                help="Add noise to improve stability; leave as 0 if not needed.",
            )
            random_state = st.number_input(
                "Random State", 
                min_value=0, 
                value=0, 
                step=1,
                help="Set for reproducibility; ignored if jitter is None.",
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train Lars Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Lars model
                self.model = Lars(
                    fit_intercept=fit_intercept,
                    verbose=verbose,
                    precompute=precompute,
                    n_nonzero_coefs=n_nonzero_coefs,
                    eps=eps,
                    copy_X=copy_X,
                    fit_path=fit_path,
                    jitter=None if jitter == 0 else jitter,
                    random_state=random_state if jitter != 0 else None,
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Lars Model Trained Successfully!")
                
                st.markdown("### Model Attributes")
                st.write(f"**Alpha Values:** {self.model.alphas_}")
                st.write(f"**Active Variables:** {self.model.active_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
    
                # Call regression metrics
                self.regression_metrics()
    def lars_cv(self):
        with self.col2:
            st.subheader("LarsCV Configuration", divider="blue")
    
            # LarsCV-specific hyperparameters
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            verbose = st.checkbox("Enable Verbose Output", value=False)
            max_iter = st.number_input(
                "Maximum Number of Iterations", min_value=1, value=500
            )
            precompute = st.selectbox(
                "Precompute Gram Matrix",
                options=["auto", True, False],
                index=0,
            )
            cv = st.selectbox(
                "Cross-Validation Strategy",
                options=[
                    "None (default 5-fold CV)",
                    "Integer (Specify number of folds)",
                    "Custom CV Splitter",
                ],
            )
            if cv == "Integer (Specify number of folds)":
                cv_folds = st.number_input(
                    "Number of Folds for Cross-Validation", min_value=2, value=5
                )
            elif cv == "Custom CV Splitter":
                st.warning(
                    "Custom CV splitters are not directly supported in this app. You need to implement it in your code."
                )
            else:
                cv_folds = None
    
            max_n_alphas = st.number_input(
                "Maximum Number of Alpha Points", min_value=1, value=1000
            )
            n_jobs = st.number_input(
                "Number of Jobs (-1 for all processors)", value=1, step=1
            )
            eps = st.number_input(
                "Machine Precision Regularization (eps)",
                min_value=0.0,
                value=float(np.finfo(float).eps),
            )
            copy_X = st.checkbox("Copy X (Avoid Overwriting Data)", value=True)
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train LarsCV Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Determine CV parameter
                cv_param = None
                if cv == "Integer (Specify number of folds)":
                    cv_param = int(cv_folds)
                elif cv == "None (default 5-fold CV)":
                    cv_param = None
    
                # Initialize and train the LarsCV model
                self.model = LarsCV(
                    fit_intercept=fit_intercept,
                    verbose=verbose,
                    max_iter=max_iter,
                    precompute=precompute,
                    cv=cv_param,
                    max_n_alphas=max_n_alphas,
                    n_jobs=n_jobs if n_jobs != 1 else None,
                    eps=eps,
                    copy_X=copy_X,
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("LarsCV Model Trained Successfully!")
    
                st.markdown("### Model Attributes")
                st.write(f"**Alpha Values:** {self.model.alphas_}")
                st.write(f"**Optimal Alpha:** {self.model.alpha_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Mean Squared Error Path:** {self.model.mse_path_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
    
                # Call regression metrics
                self.regression_metrics()
    def lasso(self):
        with self.col2:
            st.subheader("Lasso Configuration", divider="blue")
    
            # Lasso-specific hyperparameters
            alpha = st.number_input(
                "Regularization Strength (alpha)",
                min_value=0.0,
                value=1.0,
                help="Controls the regularization strength. Must be a non-negative float.",
            )
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=True,
                help="Whether to calculate the intercept for this model.",
            )
            precompute = st.selectbox(
                "Precompute Gram Matrix",
                options=["False", "True"],
                index=0,
                help="Whether to use a precomputed Gram matrix to speed up calculations.",
            )
            max_iter = st.number_input(
                "Maximum Number of Iterations",
                min_value=1,
                value=1000,
                help="The maximum number of iterations.",
            )
            tol = st.number_input(
                "Tolerance for Optimization (tol)",
                min_value=0.0,
                value=1e-4,
                format="%.1e",
                help="Convergence tolerance. Optimization stops if updates are smaller than this value.",
            )
            warm_start = st.checkbox(
                "Warm Start",
                value=False,
                help="Reuse the solution of the previous call to fit as initialization.",
            )
            positive = st.checkbox(
                "Force Positive Coefficients",
                value=False,
                help="If set to True, forces the coefficients to be positive.",
            )
            selection = st.selectbox(
                "Feature Selection Method",
                options=["cyclic", "random"],
                index=0,
                help="Choose 'cyclic' for sequential feature updates or 'random' for faster convergence.",
            )
            random_state = st.number_input(
                "Random State (Optional)",
                min_value=0,
                value=0,
                step=1,
                help="Seed for reproducibility when using random selection. Leave 0 for default.",
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train Lasso Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Lasso model
                self.model = Lasso(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    precompute=precompute == "True",
                    max_iter=max_iter,
                    tol=tol,
                    warm_start=warm_start,
                    positive=positive,
                    random_state=random_state if random_state != 0 else None,
                    selection=selection,
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Lasso Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
    
                # Call regression metrics
                self.regression_metrics()
    def lassocv(self):
        with self.col2:
            st.subheader("LassoCV Configuration", divider="blue")
    
            # LassoCV-specific hyperparameters
            eps = st.number_input(
                "Path Length (eps)",
                min_value=1e-6,
                value=1e-3,
                step=1e-3,
                format="%.1e",
                help="Length of the path; controls the ratio of alpha_min / alpha_max.",
            )
            n_alphas = st.number_input(
                "Number of Alphas",
                min_value=1,
                value=100,
                help="Number of alphas along the regularization path.",
            )
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=True,
                help="Whether to calculate the intercept for this model.",
            )
            precompute = st.selectbox(
                "Precompute Gram Matrix",
                options=["auto", "False", "True"],
                index=0,
                help="Whether to use a precomputed Gram matrix to speed up calculations.",
            )
            max_iter = st.number_input(
                "Maximum Number of Iterations",
                min_value=1,
                value=1000,
                help="The maximum number of iterations.",
            )
            tol = st.number_input(
                "Tolerance for Optimization (tol)",
                min_value=0.0,
                value=1e-4,
                format="%.1e",
                help="Convergence tolerance. Optimization stops if updates are smaller than this value.",
            )
            cv = st.number_input(
                "Cross-Validation Folds (cv)",
                min_value=2,
                value=5,
                help="Number of folds for cross-validation. Default is 5-fold.",
            )
            positive = st.checkbox(
                "Force Positive Coefficients",
                value=False,
                help="If set to True, restrict regression coefficients to be positive.",
            )
            random_state = st.number_input(
                "Random State (Optional)",
                min_value=0,
                value=0,
                step=1,
                help="Seed for reproducibility when using random selection. Leave 0 for default.",
            )
            selection = st.selectbox(
                "Feature Selection Method",
                options=["cyclic", "random"],
                index=0,
                help="Choose 'cyclic' for sequential feature updates or 'random' for faster convergence.",
            )
            n_jobs = st.selectbox(
                "Number of Jobs",
                options=[None, -1, 1, 2, 4],
                index=1,
                help="Number of CPUs to use during cross-validation. -1 uses all processors.",
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train LassoCV Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the LassoCV model
                self.model = LassoCV(
                    eps=eps,
                    n_alphas=n_alphas,
                    fit_intercept=fit_intercept,
                    precompute=precompute if precompute != "auto" else "auto",
                    max_iter=max_iter,
                    tol=tol,
                    cv=cv,
                    positive=positive,
                    random_state=random_state if random_state != 0 else None,
                    selection=selection,
                    n_jobs=n_jobs,
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("LassoCV Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Optimal Alpha:** {self.model.alpha_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Alphas Grid:** {self.model.alphas_}")
    
                # Plot the MSE path
                st.markdown("### Mean Squared Error Path")
                for i, mse_path in enumerate(self.model.mse_path_):
                    st.line_chart(mse_path, height=200, width=700, caption=f"Fold {i+1}")
    
                # Call regression metrics
                self.regression_metrics()
    def lassolars(self):
        with self.col2:
            st.subheader("LassoLars Configuration", divider="blue")
    
            # LassoLars-specific hyperparameters
            alpha = st.number_input(
                "Alpha",
                min_value=0.0,
                value=1.0,
                step=0.1,
                help="Constant that multiplies the penalty term. Alpha = 0 is equivalent to ordinary least squares.",
            )
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=True,
                help="Whether to calculate the intercept for this model.",
            )
            verbose = st.checkbox(
                "Verbose",
                value=False,
                help="Sets the verbosity amount.",
            )
            precompute = st.selectbox(
                "Precompute Gram Matrix",
                options=["auto", "False", "True"],
                index=0,
                help="Whether to use a precomputed Gram matrix to speed up calculations.",
            )
            max_iter = st.number_input(
                "Maximum Number of Iterations",
                min_value=1,
                value=500,
                help="The maximum number of iterations.",
            )
            eps = st.number_input(
                "Machine Precision Regularization (eps)",
                min_value=1e-16,
                value=1e-3,
                step=1e-3,
                format="%.1e",
                help="Machine-precision regularization in Cholesky diagonal factor computation.",
            )
            copy_X = st.checkbox(
                "Copy X",
                value=True,
                help="If True, X will be copied; else, it may be overwritten.",
            )
            fit_path = st.checkbox(
                "Fit Path",
                value=True,
                help="If True, the full path will be stored in coef_path_.",
            )
            positive = st.checkbox(
                "Force Positive Coefficients",
                value=False,
                help="If True, restrict regression coefficients to be positive.",
            )
            jitter = st.number_input(
                "Jitter",
                min_value=0.0,
                value=0.0,
                step=0.1,
                help="Upper bound on a uniform noise parameter added to y values for stability.",
            )
            random_state = st.number_input(
                "Random State (Optional)",
                min_value=0,
                value=0,
                step=1,
                help="Seed for reproducibility when using random selection.",
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train LassoLars Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the LassoLars model
                self.model = LassoLars(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    verbose=verbose,
                    precompute=precompute if precompute != "auto" else "auto",
                    max_iter=max_iter,
                    eps=eps,
                    copy_X=copy_X,
                    fit_path=fit_path,
                    positive=positive,
                    jitter=jitter,
                    random_state=random_state if random_state != 0 else None,
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("LassoLars Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Alpha:** {self.model.alpha_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Active Variables:** {self.model.active_}")
                st.write(f"**Alphas Grid:** {self.model.alphas_}")
    
                # Plot the MSE path (if available)
                if hasattr(self.model, 'mse_path_'):
                    st.markdown("### Mean Squared Error Path")
                    for i, mse_path in enumerate(self.model.mse_path_):
                        st.line_chart(mse_path, height=200, width=700, caption=f"Fold {i+1}")
    
                # Call regression metrics (if any)
                self.regression_metrics()

    def lasso_lars_cv(self):
        with self.col2:
            st.subheader("LassoLarsCV Configuration", divider="blue")
    
            # LassoLarsCV-specific hyperparameters
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=True,
                help="Whether to calculate the intercept for this model.",
            )
            verbose = st.checkbox(
                "Verbose",
                value=False,
                help="Set verbosity level for detailed output.",
            )
            max_iter = st.number_input(
                "Maximum Number of Iterations",
                min_value=1,
                value=500,
                help="The maximum number of iterations for the algorithm.",
            )
            precompute = st.selectbox(
                "Precompute Gram Matrix",
                options=["auto", "False", "True"],
                index=0,
                help="Whether to use a precomputed Gram matrix.",
            )
            cv = st.number_input(
                "Cross-Validation Folds (cv)",
                min_value=2,
                value=5,
                help="Number of folds for cross-validation.",
            )
            max_n_alphas = st.number_input(
                "Maximum Number of Alphas",
                min_value=1,
                value=1000,
                help="The maximum number of points on the path to compute residuals.",
            )
            n_jobs = st.selectbox(
                "Number of Jobs",
                options=[None, -1, 1, 2, 4],
                index=1,
                help="Number of CPUs to use during cross-validation. -1 uses all processors.",
            )
            eps = st.number_input(
                "Epsilon (eps)",
                min_value=1e-6,
                value=2.220446049250313e-16,
                step=1e-6,
                format="%.1e",
                help="Machine precision regularization.",
            )
            copy_X = st.checkbox(
                "Copy X",
                value=True,
                help="Whether to copy the input matrix X.",
            )
            positive = st.checkbox(
                "Force Positive Coefficients",
                value=False,
                help="Restrict coefficients to be greater than or equal to 0.",
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train LassoLarsCV Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the LassoLarsCV model
                self.model = LassoLarsCV(
                    fit_intercept=fit_intercept,
                    verbose=verbose,
                    max_iter=max_iter,
                    precompute=precompute if precompute != "auto" else "auto",
                    cv=cv,
                    max_n_alphas=max_n_alphas,
                    n_jobs=n_jobs,
                    eps=eps,
                    copy_X=copy_X,
                    positive=positive,
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("LassoLarsCV Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Optimal Alpha:** {self.model.alpha_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Alphas Grid:** {self.model.alphas_}")
    
                # Plot the MSE path
                st.markdown("### Mean Squared Error Path")
                for i, mse_path in enumerate(self.model.mse_path_):
                    st.line_chart(mse_path, height=200, width=700, caption=f"Fold {i+1}")
    
                # Call regression metrics
                self.regression_metrics()
    def lasso_lars_ic(self):
        with self.col2:
            st.subheader("LassoLarsIC Configuration", divider="blue")
    
            # LassoLarsIC-specific hyperparameters
            criterion = st.selectbox(
                "Criterion",
                options=["aic", "bic"],
                index=0,
                help="The criterion to use for model selection: AIC or BIC.",
            )
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=True,
                help="Whether to calculate the intercept for this model.",
            )
            verbose = st.checkbox(
                "Verbose",
                value=False,
                help="Set verbosity level for detailed output.",
            )
            precompute = st.selectbox(
                "Precompute Gram Matrix",
                options=["auto", "False", "True"],
                index=0,
                help="Whether to use a precomputed Gram matrix.",
            )
            max_iter = st.number_input(
                "Maximum Number of Iterations",
                min_value=1,
                value=500,
                help="The maximum number of iterations for the algorithm.",
            )
            eps = st.number_input(
                "Epsilon (eps)",
                min_value=1e-6,
                value=2.220446049250313e-16,
                step=1e-6,
                format="%.1e",
                help="Machine precision regularization.",
            )
            copy_X = st.checkbox(
                "Copy X",
                value=True,
                help="Whether to copy the input matrix X.",
            )
            positive = st.checkbox(
                "Force Positive Coefficients",
                value=False,
                help="Restrict coefficients to be greater than or equal to 0.",
            )
            noise_variance = st.number_input(
                "Noise Variance",
                min_value=0.0,
                value=0.0,
                step=1e-6,
                format="%.1e",
                help="Estimated noise variance. Set to None for automatic estimation.",
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train LassoLarsIC Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the LassoLarsIC model
                self.model = LassoLarsIC(
                    criterion=criterion,
                    fit_intercept=fit_intercept,
                    verbose=verbose,
                    precompute=precompute if precompute != "auto" else "auto",
                    max_iter=max_iter,
                    eps=eps,
                    copy_X=copy_X,
                    positive=positive,
                    noise_variance=noise_variance if noise_variance != 0 else None,
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("LassoLarsIC Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Optimal Alpha:** {self.model.alpha_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Criterion Values (AIC/BIC):** {self.model.criterion_}")
                st.write(f"**Noise Variance:** {self.model.noise_variance_}")
    
                # Plot the information criterion values across alphas
                st.markdown("### Information Criterion Path")
                st.line_chart(self.model.criterion_, height=200, width=700, caption="AIC/BIC Path")
    
                # Call regression metrics
                self.regression_metrics()
    def orthogonal_matching_pursuit(self):
        with self.col2:
            st.subheader("Orthogonal Matching Pursuit Configuration", divider="blue")
    
            # Orthogonal Matching Pursuit-specific hyperparameters
            n_nonzero_coefs = st.number_input(
                "Desired Number of Non-zero Coefficients",
                min_value=1,
                value=10,
                help="The desired number of non-zero coefficients in the solution. Ignored if tol is set.",
            )
            tol = st.number_input(
                "Tolerance (tol)",
                min_value=0.0,
                value=0.0,
                step=1e-6,
                format="%.1e",
                help="Maximum squared norm of the residual. Overrides n_nonzero_coefs.",
            )
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=True,
                help="Whether to calculate the intercept for this model.",
            )
            precompute = st.selectbox(
                "Precompute",
                options=["auto", "True", "False"],
                index=0,
                help="Whether to use a precomputed Gram and Xy matrix to speed up calculations.",
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train OrthogonalMatchingPursuit Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the OrthogonalMatchingPursuit model
                self.model = OrthogonalMatchingPursuit(
                    n_nonzero_coefs=n_nonzero_coefs if tol is None else None,
                    tol=tol if tol is not None else None,
                    fit_intercept=fit_intercept,
                    precompute=precompute if precompute != "auto" else "auto",
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("OrthogonalMatchingPursuit Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Number of Active Features:** {self.model.n_iter_}")
                st.write(f"**Number of Non-zero Coefficients:** {self.model.n_nonzero_coefs_}")
                
                # Call regression metrics
                self.regression_metrics()
    def orthogonal_matching_pursuit_cv(self):
        with self.col2:
            st.subheader("Orthogonal Matching Pursuit with Cross-Validation", divider="blue")
    
            # Cross-validation and OMP-specific hyperparameters
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to calculate the intercept for this model."
            )
            max_iter = st.number_input(
                "Max Iterations (max_iter)", 
                min_value=1, 
                value=10, 
                help="Maximum number of iterations to perform."
            )
            cv = st.number_input(
                "Cross-Validation Folds (cv)", 
                min_value=2, 
                value=5, 
                help="Number of folds in cross-validation."
            )
            n_jobs = st.number_input(
                "Number of Jobs (n_jobs)", 
                min_value=-1, 
                value=-1, 
                help="Number of CPUs to use during cross-validation."
            )
            verbose = st.checkbox(
                "Verbose", 
                value=False, 
                help="Sets the verbosity amount."
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train Orthogonal Matching Pursuit CV Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the OrthogonalMatchingPursuitCV model
                self.model = OrthogonalMatchingPursuitCV(
                    fit_intercept=fit_intercept,
                    max_iter=max_iter,
                    cv=cv,
                    n_jobs=n_jobs,
                    verbose=verbose,
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("OrthogonalMatchingPursuitCV Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Estimated Non-zero Coefficients:** {self.model.n_nonzero_coefs_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
    
                # Call regression metrics
                self.regression_metrics()

    def ard_regression(self):
        with self.col2:
            st.subheader("Bayesian ARD Regression", divider="blue")
    
            # Hyperparameters for ARD regression
            max_iter = st.number_input(
                "Max Iterations (max_iter)", 
                min_value=1, 
                value=300, 
                help="Maximum number of iterations for convergence."
            )
            tol = st.number_input(
                "Tolerance (tol)", 
                min_value=0.0001, 
                value=0.001, 
                help="Stop the algorithm if the weights converge."
            )
            alpha_1 = st.number_input(
                "Alpha 1 (alpha_1)", 
                min_value=0.0, 
                value=1e-6, 
                help="Shape parameter for the Gamma distribution prior over alpha."
            )
            alpha_2 = st.number_input(
                "Alpha 2 (alpha_2)", 
                min_value=0.0, 
                value=1e-6, 
                help="Inverse scale parameter for the Gamma distribution prior over alpha."
            )
            lambda_1 = st.number_input(
                "Lambda 1 (lambda_1)", 
                min_value=0.0, 
                value=1e-6, 
                help="Shape parameter for the Gamma distribution prior over lambda."
            )
            lambda_2 = st.number_input(
                "Lambda 2 (lambda_2)", 
                min_value=0.0, 
                value=1e-6, 
                help="Inverse scale parameter for the Gamma distribution prior over lambda."
            )
            threshold_lambda = st.number_input(
                "Threshold Lambda (threshold_lambda)", 
                min_value=1.0, 
                value=10000.0, 
                help="Threshold for pruning weights with high precision."
            )
            compute_score = st.checkbox(
                "Compute Objective Function", 
                value=False, 
                help="If True, compute the objective function at each step."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to calculate the intercept for this model."
            )
            verbose = st.checkbox(
                "Verbose", 
                value=False, 
                help="Set to True for verbose output during fitting."
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train ARD Regression Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the ARD regression model
                self.model = ARDRegression(
                    max_iter=max_iter,
                    tol=tol,
                    alpha_1=alpha_1,
                    alpha_2=alpha_2,
                    lambda_1=lambda_1,
                    lambda_2=lambda_2,
                    threshold_lambda=threshold_lambda,
                    compute_score=compute_score,
                    fit_intercept=fit_intercept,
                    verbose=verbose
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("ARD Regression Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Alpha (Noise Precision):** {self.model.alpha_}")
                st.write(f"**Lambda (Weight Precision):** {self.model.lambda_}")
                st.write(f"**Sigma (Variance-Covariance Matrix):** {self.model.sigma_}")
                if compute_score:
                    st.write(f"**Scores (Objective Function):** {self.model.scores_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
    
                # Call regression metrics
                self.regression_metrics()
    def bayesian_ridge(self):
        with self.col2:
            st.subheader("Bayesian Ridge Regression", divider="blue")
    
            # Hyperparameters for Bayesian Ridge regression
            max_iter = st.number_input(
                "Max Iterations (max_iter)", 
                min_value=1, 
                value=300, 
                help="Maximum number of iterations for convergence."
            )
            tol = st.number_input(
                "Tolerance (tol)", 
                min_value=0.0001, 
                value=0.001, 
                help="Stop the algorithm if the weights converge."
            )
            alpha_1 = st.number_input(
                "Alpha 1 (alpha_1)", 
                min_value=0.0, 
                value=1e-6, 
                help="Shape parameter for the Gamma distribution prior over alpha."
            )
            alpha_2 = st.number_input(
                "Alpha 2 (alpha_2)", 
                min_value=0.0, 
                value=1e-6, 
                help="Inverse scale parameter for the Gamma distribution prior over alpha."
            )
            lambda_1 = st.number_input(
                "Lambda 1 (lambda_1)", 
                min_value=0.0, 
                value=1e-6, 
                help="Shape parameter for the Gamma distribution prior over lambda."
            )
            lambda_2 = st.number_input(
                "Lambda 2 (lambda_2)", 
                min_value=0.0, 
                value=1e-6, 
                help="Inverse scale parameter for the Gamma distribution prior over lambda."
            )
            alpha_init = st.number_input(
                "Initial Alpha (alpha_init)", 
                min_value=0.0, 
                value=None, 
                help="Initial value for alpha (precision of the noise)."
            )
            lambda_init = st.number_input(
                "Initial Lambda (lambda_init)", 
                min_value=0.0, 
                value=None, 
                help="Initial value for lambda (precision of the weights)."
            )
            compute_score = st.checkbox(
                "Compute Log Marginal Likelihood", 
                value=False, 
                help="If True, compute the log marginal likelihood at each iteration."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to calculate the intercept for this model."
            )
            verbose = st.checkbox(
                "Verbose", 
                value=False, 
                help="Set to True for verbose output during fitting."
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train Bayesian Ridge Regression Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Bayesian Ridge regression model
                self.model = BayesianRidge(
                    max_iter=max_iter,
                    tol=tol,
                    alpha_1=alpha_1,
                    alpha_2=alpha_2,
                    lambda_1=lambda_1,
                    lambda_2=lambda_2,
                    alpha_init=alpha_init,
                    lambda_init=lambda_init,
                    compute_score=compute_score,
                    fit_intercept=fit_intercept,
                    verbose=verbose
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Bayesian Ridge Regression Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Alpha (Noise Precision):** {self.model.alpha_}")
                st.write(f"**Lambda (Weight Precision):** {self.model.lambda_}")
                st.write(f"**Sigma (Variance-Covariance Matrix):** {self.model.sigma_}")
                if compute_score:
                    st.write(f"**Scores (Log Marginal Likelihood):** {self.model.scores_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
    
                # Call regression metrics
                self.regression_metrics()
    def multi_task_elastic_net(self):
        with self.col2:
            st.subheader("Multi-task ElasticNet Regression", divider="blue")
    
            # Hyperparameters for MultiTaskElasticNet regression
            alpha = st.number_input(
                "Alpha", 
                min_value=0.0, 
                value=1.0, 
                help="Constant that multiplies the L1/L2 term. Defaults to 1.0."
            )
            l1_ratio = st.number_input(
                "L1 Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                help="The ElasticNet mixing parameter, with 0 < l1_ratio <= 1."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to calculate the intercept for this model."
            )
            max_iter = st.number_input(
                "Max Iterations", 
                min_value=1, 
                value=1000, 
                help="The maximum number of iterations."
            )
            tol = st.number_input(
                "Tolerance", 
                min_value=1e-5, 
                value=1e-4, 
                help="The tolerance for optimization."
            )
            warm_start = st.checkbox(
                "Warm Start", 
                value=False, 
                help="Reuse the solution of the previous call to fit."
            )
            random_state = st.number_input(
                "Random State", 
                value=None, 
                help="The seed of the pseudo-random number generator."
            )
            selection = st.selectbox(
                "Selection", 
                ["cyclic", "random"], 
                help="If set to 'random', a random coefficient is updated every iteration."
            )
    
            # Train the model if the checkbox is selected
            if st.checkbox("Train Multi-task ElasticNet Regression Model"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Multi-task ElasticNet regression model
                self.model = MultiTaskElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=fit_intercept,
                    max_iter=max_iter,
                    tol=tol,
                    warm_start=warm_start,
                    random_state=random_state,
                    selection=selection
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Multi-task ElasticNet Regression Model Trained Successfully!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Intercept:** {self.model.intercept_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Dual Gap:** {self.model.dual_gap_}")
                st.write(f"**Iterations:** {self.model.n_iter_}")
                st.write(f"**Tolerance Scaled (eps):** {self.model.eps_}")
    
                # Call regression metrics
                self.regression_metrics()
    def multi_task_elastic_net_cv(self):
        with self.col2:
            st.subheader("Multi-task ElasticNet with Cross-Validation", divider="blue")
    
            # Hyperparameters for MultiTaskElasticNetCV
            l1_ratio = st.slider(
                "L1 Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                help="The ElasticNet mixing parameter. Value between 0 and 1."
            )
            eps = st.number_input(
                "Epsilon", 
                min_value=1e-6, 
                value=1e-3, 
                help="Length of the regularization path."
            )
            n_alphas = st.number_input(
                "Number of Alphas", 
                min_value=1, 
                value=100, 
                help="The number of alphas along the regularization path."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to calculate the intercept for this model."
            )
            max_iter = st.number_input(
                "Max Iterations", 
                min_value=1, 
                value=1000, 
                help="Maximum number of iterations."
            )
            tol = st.number_input(
                "Tolerance", 
                min_value=1e-5, 
                value=1e-4, 
                help="The tolerance for optimization."
            )
            cv = st.number_input(
                "Cross-validation Folds", 
                min_value=2, 
                value=5, 
                help="The number of folds for cross-validation."
            )
            verbose = st.slider(
                "Verbosity", 
                min_value=0, 
                max_value=3, 
                value=0, 
                help="Amount of verbosity for the fitting process."
            )
            n_jobs = st.number_input(
                "Number of Jobs", 
                min_value=-1, 
                value=None, 
                help="Number of CPUs to use during cross-validation. Use -1 for all processors."
            )
            random_state = st.number_input(
                "Random State", 
                value=None, 
                help="The seed for random number generation."
            )
            selection = st.selectbox(
                "Selection", 
                ["cyclic", "random"], 
                help="Choose whether to update coefficients sequentially or randomly."
            )
    
            # Train the model with cross-validation if selected
            if st.checkbox("Train Multi-task ElasticNet with Cross-validation"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Multi-task ElasticNetCV model
                self.model = MultiTaskElasticNetCV(
                    l1_ratio=l1_ratio,
                    eps=eps,
                    n_alphas=n_alphas,
                    fit_intercept=fit_intercept,
                    max_iter=max_iter,
                    tol=tol,
                    cv=cv,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    selection=selection
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Multi-task ElasticNetCV Model Trained Successfully with Cross-validation!")
    
                # Display the optimal parameters
                st.markdown("### Model Attributes")
                st.write(f"**Optimal alpha:** {self.model.alpha_}")
                st.write(f"**Best L1 Ratio:** {self.model.l1_ratio_}")
                st.write(f"**MSE Path:** {self.model.mse_path_}")
                st.write(f"**Alpha Path:** {self.model.alphas_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Dual Gap:** {self.model.dual_gap_}")
    
                # Call regression metrics
                self.regression_metrics()
    def multi_task_lasso(self):
        with self.col2:
            st.subheader("Multi-task Lasso", divider="blue")
    
            # Hyperparameters for MultiTaskLasso
            alpha = st.number_input(
                "Alpha", 
                min_value=0.01, 
                value=1.0, 
                step=0.01, 
                help="Constant that multiplies the L1/L2 term."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to calculate the intercept for this model."
            )
            max_iter = st.number_input(
                "Max Iterations", 
                min_value=1, 
                value=1000, 
                help="Maximum number of iterations."
            )
            tol = st.number_input(
                "Tolerance", 
                min_value=1e-5, 
                value=1e-4, 
                help="Tolerance for optimization."
            )
            warm_start = st.checkbox(
                "Warm Start", 
                value=False, 
                help="Reuse the solution of the previous call to fit as initialization."
            )
            selection = st.selectbox(
                "Selection", 
                ["cyclic", "random"], 
                help="Choose the feature selection method."
            )
    
            # Train the model with user input
            if st.checkbox("Train Multi-task Lasso"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Multi-task Lasso model
                self.model = MultiTaskLasso(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    max_iter=max_iter,
                    tol=tol,
                    warm_start=warm_start,
                    selection=selection
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Multi-task Lasso Model Trained Successfully!")
    
                # Display the trained model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:**\n{self.model.coef_}")
                st.write(f"**Intercepts:**\n{self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Dual Gaps:** {self.model.dual_gap_}")
                st.write(f"**Feature Names:** {self.model.feature_names_in_}")
    
                # Call regression metrics (Assuming self.regression_metrics() is defined)
                self.regression_metrics()
    def multi_task_lasso_cv(self):
        with self.col2:
            st.subheader("Multi-task LassoCV", divider="blue")
    
            # Hyperparameters for MultiTaskLassoCV
            eps = st.number_input(
                "Epsilon", 
                min_value=1e-5, 
                value=1e-3, 
                step=1e-5, 
                help="Length of the path. Determines alpha_min/alpha_max."
            )
            n_alphas = st.number_input(
                "Number of Alphas", 
                min_value=10, 
                value=100, 
                help="Number of alphas along the regularization path."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to calculate the intercept for this model."
            )
            max_iter = st.number_input(
                "Max Iterations", 
                min_value=1, 
                value=1000, 
                help="Maximum number of iterations."
            )
            tol = st.number_input(
                "Tolerance", 
                min_value=1e-5, 
                value=1e-4, 
                help="Tolerance for optimization."
            )
            cv = st.number_input(
                "Cross-validation Folds", 
                min_value=2, 
                value=5, 
                help="Number of cross-validation folds."
            )
            verbose = st.checkbox(
                "Verbose", 
                value=False, 
                help="Print detailed output during model fitting."
            )
            n_jobs = st.number_input(
                "Number of Jobs", 
                min_value=-1, 
                value=None, 
                help="Number of CPUs to use during cross-validation."
            )
            selection = st.selectbox(
                "Selection", 
                ["cyclic", "random"], 
                help="Choose the feature selection method."
            )
    
            # Train the model with user input
            if st.checkbox("Train Multi-task LassoCV"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Multi-task LassoCV model
                self.model = MultiTaskLassoCV(
                    eps=eps,
                    n_alphas=n_alphas,
                    fit_intercept=fit_intercept,
                    max_iter=max_iter,
                    tol=tol,
                    cv=cv,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    selection=selection
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Multi-task LassoCV Model Trained Successfully!")
    
                # Display the trained model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:**\n{self.model.coef_}")
                st.write(f"**Intercepts:**\n{self.model.intercept_}")
                st.write(f"**Alpha Chosen by CV:** {self.model.alpha_}")
                st.write(f"**Mean Squared Error Path:**\n{self.model.mse_path_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Dual Gap:** {self.model.dual_gap_}")
                st.write(f"**Feature Names:** {self.model.feature_names_in_}")
    
                # Call regression metrics (Assuming self.regression_metrics() is defined)
                self.regression_metrics()
    def huber_regressor(self):
        with self.col2:
            st.subheader("Huber Regressor", divider="blue")
    
            # Hyperparameters for Huber Regressor
            epsilon = st.number_input(
                "Epsilon", 
                min_value=1.0, 
                value=1.35, 
                step=0.05, 
                help="Epsilon controls the robustness to outliers. Smaller epsilon makes the model more robust."
            )
            alpha = st.number_input(
                "Alpha (Regularization)", 
                min_value=0.0, 
                value=0.0001, 
                step=1e-5, 
                help="Strength of the squared L2 regularization."
            )
            max_iter = st.number_input(
                "Max Iterations", 
                min_value=10, 
                value=100, 
                help="Maximum number of iterations for fitting the model."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to calculate the intercept for this model."
            )
            tol = st.number_input(
                "Tolerance", 
                min_value=1e-6, 
                value=1e-5, 
                help="Tolerance for optimization."
            )
            warm_start = st.checkbox(
                "Warm Start", 
                value=False, 
                help="Whether to reuse the solution of the previous fit."
            )
    
            # Train the model with user input
            if st.checkbox("Train Huber Regressor"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Huber Regressor
                self.model = HuberRegressor(
                    epsilon=epsilon,
                    alpha=alpha,
                    max_iter=max_iter,
                    fit_intercept=fit_intercept,
                    tol=tol,
                    warm_start=warm_start
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Huber Regressor Model Trained Successfully!")
    
                # Display the trained model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:**\n{self.model.coef_}")
                st.write(f"**Intercept:**\n{self.model.intercept_}")
                st.write(f"**Scale:**\n{self.model.scale_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Outliers Identified:**\n{self.model.outliers_}")
                st.write(f"**Feature Names:** {self.model.feature_names_in_}")
    
                # Call regression metrics (Assuming self.regression_metrics() is defined)
                self.regression_metrics()
    def quantile_regressor(self):
        with self.col2:
            st.subheader("Quantile Regressor", divider="blue")
    
            # Hyperparameters for Quantile Regressor
            quantile = st.number_input(
                "Quantile", 
                min_value=0.01, 
                max_value=0.99, 
                value=0.5, 
                step=0.01, 
                help="The quantile the model predicts (default is the 50% quantile, i.e., the median)."
            )
            alpha = st.number_input(
                "Alpha (Regularization)", 
                min_value=0.0, 
                value=1.0, 
                step=0.1, 
                help="Regularization constant for the L1 penalty."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether or not to fit the intercept."
            )
            solver = st.selectbox(
                "Solver", 
                options=['highs-ds', 'highs-ipm', 'highs', 'interior-point', 'revised simplex'], 
                index=2,
                help="Method used to solve the linear programming formulation."
            )
            solver_options = st.text_input(
                "Solver Options", 
                value='',
                help="Additional solver parameters in dictionary format (optional)."
            )
    
            # Train the model with user input
            if st.checkbox("Train Quantile Regressor"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the Quantile Regressor
                self.model = QuantileRegressor(
                    quantile=quantile,
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    solver=solver,
                    solver_options=eval(solver_options) if solver_options else None
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Quantile Regressor Model Trained Successfully!")
    
                # Display the trained model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:**\n{self.model.coef_}")
                st.write(f"**Intercept:**\n{self.model.intercept_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Feature Names:** {self.model.feature_names_in_}")
    
                # Call regression metrics (Assuming self.regression_metrics() is defined)
                self.regression_metrics()
    def ransac_regressor(self):
        with self.col2:
            st.subheader("RANSAC Regressor", divider="blue")
    
            # Hyperparameters for RANSAC Regressor
            min_samples = st.number_input(
                "Min Samples", 
                min_value=1, 
                max_value=self.xTrain.shape[0], 
                value=None, 
                help="Minimum number of samples to be randomly selected."
            )
            residual_threshold = st.number_input(
                "Residual Threshold", 
                min_value=0.0, 
                value=None, 
                step=0.1, 
                help="Maximum residual for a sample to be classified as an inlier."
            )
            max_trials = st.number_input(
                "Max Trials", 
                min_value=1, 
                value=100, 
                help="Maximum number of iterations for random sample selection."
            )
            stop_n_inliers = st.number_input(
                "Stop if N Inliers", 
                min_value=1, 
                value=None, 
                help="Stop if at least this many inliers are found."
            )
            stop_score = st.number_input(
                "Stop Score", 
                min_value=0.0, 
                value=None, 
                help="Stop if the score exceeds this value."
            )
            stop_probability = st.slider(
                "Stop Probability", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.99, 
                step=0.01, 
                help="Stop if the probability of outlier-free data exceeds this threshold."
            )
            loss = st.selectbox(
                "Loss Function", 
                options=['absolute_error', 'squared_error'], 
                index=0, 
                help="Choose loss function for RANSAC."
            )
    
            # Train the model with user input
            if st.checkbox("Train RANSAC Regressor"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train the RANSAC Regressor
                self.model = RANSACRegressor(
                    min_samples=min_samples,
                    residual_threshold=residual_threshold,
                    max_trials=max_trials,
                    stop_n_inliers=stop_n_inliers,
                    stop_score=stop_score,
                    stop_probability=stop_probability,
                    loss=loss
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("RANSAC Regressor Model Trained Successfully!")
    
                # Display the trained model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Number of Trials:** {self.model.n_trials_}")
                st.write(f"**Number of Inliers:** {sum(self.model.inlier_mask_)}")
                st.write(f"**Inlier Mask:**\n{self.model.inlier_mask_}")
                st.write(f"**Final Model Coefficients:**\n{self.model.estimator_.coef_}")
                st.write(f"**Final Model Intercept:**\n{self.model.estimator_.intercept_}")
    
                # Call regression metrics (Assuming self.regression_metrics() is defined)
                self.regression_metrics()
    def theil_sen_regressor(self):
        with self.col2:
            st.subheader("Theil-Sen Regressor", divider="blue")
    
            # Hyperparameters for Theil-Sen Regressor
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to calculate the intercept for this model."
            )
            max_subpopulation = st.number_input(
                "Max Subpopulation", 
                min_value=1, 
                value=10000, 
                step=1000, 
                help="Maximum number of subsets to consider when calculating least square solutions."
            )
            n_subsamples = st.number_input(
                "Number of Subsamples", 
                min_value=self.xTrain.shape[1] + 1, 
                max_value=self.xTrain.shape[0], 
                value=None, 
                help="Number of samples to calculate parameters. Default is the minimum for maximal robustness."
            )
            max_iter = st.number_input(
                "Max Iterations", 
                min_value=1, 
                value=300, 
                help="Maximum number of iterations for the spatial median calculation."
            )
            tol = st.number_input(
                "Tolerance", 
                min_value=0.0, 
                value=1e-3, 
                step=0.001, 
                help="Tolerance when calculating the spatial median."
            )
            n_jobs = st.number_input(
                "Number of Jobs", 
                min_value=-1, 
                value=None, 
                help="Number of CPUs to use during the cross-validation. -1 uses all processors."
            )
            verbose = st.checkbox(
                "Verbose Mode", 
                value=False, 
                help="Enable verbose mode during fitting."
            )
    
            # Train the model with user input
            if st.checkbox("Train Theil-Sen Regressor"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train Theil-Sen Regressor
                self.model = TheilSenRegressor(
                    fit_intercept=fit_intercept,
                    max_subpopulation=max_subpopulation,
                    n_subsamples=n_subsamples,
                    max_iter=max_iter,
                    tol=tol,
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Theil-Sen Regressor Model Trained Successfully!")
    
                # Display the trained model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Breakdown Point:** {self.model.breakdown_}")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
    
                # Call regression metrics (Assuming self.regression_metrics() is defined)
                self.regression_metrics()
    def gamma_regressor(self):
        with self.col2:
            st.subheader("Gamma Regressor", divider="blue")
    
            # Hyperparameters for Gamma Regressor
            alpha = st.number_input(
                "Alpha (Regularization Strength)", 
                min_value=0.0, 
                value=1.0, 
                step=0.1, 
                help="Constant that multiplies the L2 penalty term."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Whether to add an intercept term to the model."
            )
            solver = st.selectbox(
                "Solver", 
                options=['lbfgs', 'newton-cholesky'], 
                index=0, 
                help="Algorithm used to solve the optimization problem."
            )
            max_iter = st.number_input(
                "Max Iterations", 
                min_value=1, 
                value=100, 
                help="Maximum number of iterations for the solver."
            )
            tol = st.number_input(
                "Tolerance", 
                min_value=0.0, 
                value=1e-4, 
                step=0.0001, 
                help="Stopping criterion for the solver."
            )
            warm_start = st.checkbox(
                "Warm Start", 
                value=False, 
                help="Reuse the solution of the previous call to fit."
            )
            verbose = st.number_input(
                "Verbose", 
                min_value=0, 
                value=0, 
                help="Verbosity for the solver."
            )
    
            # Train the model with user input
            if st.checkbox("Train Gamma Regressor"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Train-test split not performed. Please perform it first.")
                    return
    
                # Initialize and train Gamma Regressor
                self.model = GammaRegressor(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    solver=solver,
                    max_iter=max_iter,
                    tol=tol,
                    warm_start=warm_start,
                    verbose=verbose
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Gamma Regressor Model Trained Successfully!")
    
                # Display the trained model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
    
                # Call regression metrics (Assuming self.regression_metrics() is defined)
                self.regression_metrics()
    def poisson_regressor(self):
        with self.col2:
            st.subheader("Poisson Regressor", divider="blue")
    
            # Input fields for Poisson Regressor hyperparameters
            alpha = st.number_input(
                "Alpha (Regularization Strength)", 
                min_value=0.0, 
                value=1.0, 
                step=0.1, 
                help="Constant that multiplies the L2 penalty term."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Indicates whether to add an intercept term to the model."
            )
            solver = st.selectbox(
                "Solver", 
                options=['lbfgs', 'newton-cholesky'], 
                index=0, 
                help="Choose the algorithm for solving the optimization problem."
            )
            max_iter = st.number_input(
                "Max Iterations", 
                min_value=1, 
                value=100, 
                help="Define the maximum number of iterations for the solver."
            )
            tol = st.number_input(
                "Tolerance", 
                min_value=0.0, 
                value=1e-4, 
                step=0.0001, 
                help="Set the stopping criterion for the solver."
            )
            warm_start = st.checkbox(
                "Warm Start", 
                value=False, 
                help="Re-use the solution of the previous fit as the starting point."
            )
            verbose = st.number_input(
                "Verbose", 
                min_value=0, 
                value=0, 
                help="Set the verbosity level for the solver."
            )
    
            # Button to train the Poisson Regressor model
            if st.checkbox("Train Poisson Regressor"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Please perform the train-test split first.")
                    return
    
                # Initialize and train the Poisson Regressor model
                self.model = PoissonRegressor(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    solver=solver,
                    max_iter=max_iter,
                    tol=tol,
                    warm_start=warm_start,
                    verbose=verbose
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Poisson Regressor model has been successfully trained!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
    
                # Call the regression metrics method (assuming it's defined)
                self.regression_metrics()
    def tweedie_regressor(self):
        with self.col2:
            st.subheader("Tweedie Regressor", divider="blue")
    
            # Input fields for Tweedie Regressor hyperparameters
            power = st.number_input(
                "Power", 
                min_value=0.0, 
                value=0.0, 
                step=0.1, 
                help="The power determines the target distribution (e.g., Poisson, Gamma, Inverse Gaussian)."
            )
            alpha = st.number_input(
                "Alpha (Regularization Strength)", 
                min_value=0.0, 
                value=1.0, 
                step=0.1, 
                help="Constant that multiplies the L2 penalty term."
            )
            fit_intercept = st.checkbox(
                "Fit Intercept", 
                value=True, 
                help="Indicates whether to add an intercept term to the model."
            )
            link = st.selectbox(
                "Link Function", 
                options=['auto', 'identity', 'log'], 
                index=0, 
                help="Select the link function for the GLM."
            )
            solver = st.selectbox(
                "Solver", 
                options=['lbfgs', 'newton-cholesky'], 
                index=0, 
                help="Choose the algorithm for solving the optimization problem."
            )
            max_iter = st.number_input(
                "Max Iterations", 
                min_value=1, 
                value=100, 
                help="Maximum number of iterations for the solver."
            )
            tol = st.number_input(
                "Tolerance", 
                min_value=0.0, 
                value=1e-4, 
                step=0.0001, 
                help="Stopping criterion for the solver."
            )
            warm_start = st.checkbox(
                "Warm Start", 
                value=False, 
                help="Re-use the solution of the previous fit as the starting point."
            )
            verbose = st.number_input(
                "Verbose", 
                min_value=0, 
                value=0, 
                help="Set the verbosity level for the solver."
            )
    
            # Button to train the Tweedie Regressor model
            if st.checkbox("Train Tweedie Regressor"):
                if self.xTrain is None or self.yTrain is None:
                    st.error("Please perform the train-test split first.")
                    return
    
                # Initialize and train the Tweedie Regressor model
                self.model = TweedieRegressor(
                    power=power,
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    link=link,
                    solver=solver,
                    max_iter=max_iter,
                    tol=tol,
                    warm_start=warm_start,
                    verbose=verbose
                )
                self.model.fit(self.xTrain, self.yTrain)
    
                st.success("Tweedie Regressor model has been successfully trained!")
    
                # Display model attributes
                st.markdown("### Model Attributes")
                st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                st.write(f"**Coefficients:** {self.model.coef_}")
                st.write(f"**Intercept:** {self.model.intercept_}")
    
                # Call the regression metrics method (assuming it's defined)
                self.regression_metrics()
    def regression_metrics(self):
        with self.col3:
            st.markdown("### Evaluate Regression Metrics")
    
            # D2 Absolute Error
            try:
                y_pred = self.model.predict(self.xTest)
                sample_weight = None
                multioutput = "uniform_average"
                d2_abs_error_score = d2_absolute_error_score(self.yTest, y_pred, sample_weight=sample_weight, multioutput=multioutput)
                st.write(f"**D2 Absolute Error Score:** {d2_abs_error_score}")
            except Exception as e:
                st.error(f"An error occurred while calculating D2 Absolute Error: {str(e)}")
    
            # D2 Pinball Loss
            try:
                alpha = 0.5
                d2_pinball_loss = d2_pinball_score(self.yTest, y_pred, sample_weight=None, alpha=alpha, multioutput="uniform_average")
                st.write(f"**D2 Pinball Loss Score:** {d2_pinball_loss}")
            except Exception as e:
                st.error(f"An error occurred while calculating D2 Pinball Loss: {str(e)}")
    
            # D2 Tweedie Score
            try:
                power = 0.0
                d2_tweedie_score = d2_tweedie_score(self.yTest, y_pred, sample_weight=None, power=power)
                st.write(f"**D2 Tweedie Score:** {d2_tweedie_score}")
            except Exception as e:
                st.error(f"An error occurred while calculating D2 Tweedie Score: {str(e)}")
    
            # Explained Variance
            try:
                force_finite = True
                explained_variance = explained_variance_score(self.yTest, y_pred, sample_weight=None, multioutput="uniform_average", force_finite=force_finite)
                st.write(f"**Explained Variance Score:** {explained_variance}")
            except Exception as e:
                st.error(f"An error occurred while calculating Explained Variance: {str(e)}")
    
            # Max Error
            try:
                max_error_value = max_error(self.yTest, y_pred)
                st.write(f"**Max Error Score:** {max_error_value}")
            except Exception as e:
                st.error(f"An error occurred while calculating Max Error: {str(e)}")
    
            # Mean Absolute Error
            try:
                mae = mean_absolute_error(self.yTest, y_pred)
                st.write(f"**Mean Absolute Error (MAE):** {mae}")
            except Exception as e:
                st.error(f"An error occurred while calculating Mean Absolute Error: {str(e)}")
    
            # Mean Absolute Percentage Error
            try:
                mape = mean_absolute_percentage_error(self.yTest, y_pred)
                st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape}")
            except Exception as e:
                st.error(f"An error occurred while calculating Mean Absolute Percentage Error: {str(e)}")
    
            # Mean Gamma Deviance
            try:
                gamma_deviance = mean_gamma_deviance(self.yTest, y_pred)
                st.write(f"**Mean Gamma Deviance:** {gamma_deviance}")
            except Exception as e:
                st.error(f"An error occurred while calculating Mean Gamma Deviance: {str(e)}")
    
            # Mean Pinball Loss
            try:
                pinball_loss = mean_pinball_loss(self.yTest, y_pred)
                st.write(f"**Mean Pinball Loss:** {pinball_loss}")
            except Exception as e:
                st.error(f"An error occurred while calculating Mean Pinball Loss: {str(e)}")
    
            # Mean Poisson Deviance
            try:
                poisson_deviance = mean_poisson_deviance(self.yTest, y_pred)
                st.write(f"**Mean Poisson Deviance:** {poisson_deviance}")
            except Exception as e:
                st.error(f"An error occurred while calculating Mean Poisson Deviance: {str(e)}")
    
            # Mean Squared Error
            try:
                mse = mean_squared_error(self.yTest, y_pred)
                st.write(f"**Mean Squared Error (MSE):** {mse}")
            except Exception as e:
                st.error(f"An error occurred while calculating Mean Squared Error: {str(e)}")
    
            # Mean Squared Log Error
            try:
                msle = mean_squared_log_error(self.yTest, y_pred)
                st.write(f"**Mean Squared Logarithmic Error (MSLE):** {msle}")
            except Exception as e:
                st.error(f"An error occurred while calculating Mean Squared Logarithmic Error: {str(e)}")
    
            # Mean Tweedie Deviance
            try:
                tweedie_deviance = mean_tweedie_deviance(self.yTest, y_pred)
                st.write(f"**Mean Tweedie Deviance:** {tweedie_deviance}")
            except Exception as e:
                st.error(f"An error occurred while calculating Mean Tweedie Deviance: {str(e)}")
    
            # Median Absolute Error
            try:
                median_abs_error = median_absolute_error(self.yTest, y_pred)
                st.write(f"**Median Absolute Error:** {median_abs_error}")
            except Exception as e:
                st.error(f"An error occurred while calculating Median Absolute Error: {str(e)}")
    
            # R2 Score
            try:
                r2 = r2_score(self.yTest, y_pred)
                st.write(f"**R2 Score:** {r2}")
            except Exception as e:
                st.error(f"An error occurred while calculating R2 Score: {str(e)}")
