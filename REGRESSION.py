import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.metrics import *
import numpy as np

session_variables=['regressionLinear','regressionRidge','regressionRidgeCV',
                   'regressionSGD','regressionElasticNet','regressionElasticNetCV','regressionLars','regressionLarsCV',
                   'regressionLasso','regressionLassoCV',"regressionLassoLars",'regressionLassoLarsCV','regressionLassoLarsIC','regressionOMP',
                  'regressionOMPCV','regressionARD','regressionBayesianRidge','regressionMTElasticNet','regressionMTLasso','regressionMTElasticNetCV',
                   'regressionHuber','regressionQuantile','regressionRANSAC','regressionPoisson','regressionTheilSen','regressionTweedie']
for i in session_variables:
    if i not in st.session_state:
        st.session_state[i]=None
class Regression:
    def __init__(self, dataset):
        self.dataset = dataset
        self.col1, self.col2, self.col3 = st.columns([1, 1, 1],border=True)
        self.xTrain, self.xTest, self.yTrain, self.yTest = None, None, None, None
        self.model = None
    def display(self):
        self.train_test_split()
        with self.col2:
            st.subheader("Classical Linear Model", divider='blue')
            model_map = {"LinearRegression": self.linear_regression, "Ridge": self.ridge_regression, "RidgeCV": self.ridge_cv, "SGDRegressor": self.sgd_regressor, "ElasticNet": self.elasticNet, "ElasticNetCV": self.elasticNetCV, "Lars": self.lars, "LarsCV": self.lars_cv, "Lasso": self.lasso, "LassoCV": self.lassocv, "LassoLars": self.lassolars, "LassoLarsCV": self.lasso_lars_cv, "LassoLarsIC": self.lasso_lars_ic, "OrthogonalMatchingPursuit": self.orthogonal_matching_pursuit, "OrthogonalMatchingPursuitCV": self.orthogonal_matching_pursuit_cv, "ARDRegression": self.ard_regression, "BayesianRidge": self.bayesian_ridge, "MultiTaskElasticNet": self.multi_task_elastic_net, "MultiTaskElasticNetCV": self.multi_task_elastic_net_cv, "MultiTaskLasso": self.multi_task_lasso, "MultiTaskLassoCV": self.multi_task_lasso_cv, "HuberRegressor": self.huber_regressor, "QuantileRegressor": self.quantile_regressor, "RANSACRegressor": self.ransac_regressor, "TheilSenRegressor": self.theil_sen_regressor, "GammaRegressor": self.gamma_regressor, "PoissonRegressor": self.poisson_regressor, "TweedieRegressor": self.tweedie_regressor}
            model = st.selectbox("Select the regressor that you want", model_map.keys())
            model_map[model]()
                             
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
        if st.session_state['regressionLinear'] == None:
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
                    st.session_state['regressionLinear']=self.model
                    st.success("Linear Regression Model Trained Successfully!")
                    st.markdown("### Model Attributes")
                    st.write(f"**Coefficients:** {self.model.coef_}")
                    st.write(f"**Intercept:** {self.model.intercept_}")
                    self.regression_metrics(st.session_state['regressionLinear'])
        else:
            with self.col2:
                st.success("Linear Regression Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {st.session_state['regressionLinear'].coef_}")
                st.write(f"**Intercept:** {st.session_state['regressionLinear'].intercept_}")
                self.regression_metrics(st.session_state['regressionLinear'])
                if st.button("Re Train The Model",use_container_width=True,type='primary'):
                    st.session_state['regressionLinear']=None
    def ridge_regression(self):
        if st.session_state.get('regressionRidge') is None:
            with self.col2:
                st.subheader("Ridge Regression Configuration", divider='blue')
                alpha = st.slider("Alpha (Regularization Strength)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                fit_intercept = st.checkbox("Fit Intercept", value=True)
                positive = st.checkbox("Force Positive Coefficients", value=False)
                solver = st.selectbox("Solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"])
                if st.checkbox("Train Ridge Regression Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, positive=positive, solver=solver)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionRidge'] = self.model
                        st.success("Ridge Regression Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        self.regression_metrics(st.session_state.get('regressionRidge'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Ridge Regression Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {st.session_state.get('regressionRidge').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionRidge').intercept_}")
                self.regression_metrics(st.session_state.get('regressionRidge'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionRidge'] = None
    def ridge_cv(self):
        if st.session_state.get('regressionRidgeCV') is None:
            with self.col2:
                st.subheader("RidgeCV Regression Configuration", divider='blue')
                alphas = st.text_input("Alphas (Regularization Strength)", value="(0.1, 1.0, 10.0)")
                alphas = eval(alphas)
                fit_intercept = st.checkbox("Fit Intercept", value=True)
                alpha_per_target = st.checkbox("Alpha Per Target", value=False)
                scoring = st.selectbox("Scoring Method", ["None", "neg_mean_squared_error", "r2", "neg_mean_absolute_error"])
                cv = st.number_input("Cross-validation Folds", min_value=2, max_value=10, value=5)
                if st.checkbox("Train RidgeCV Regression Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = RidgeCV(alphas=alphas, fit_intercept=fit_intercept, alpha_per_target=alpha_per_target, scoring=scoring if scoring != "None" else None, cv=cv)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionRidgeCV'] = self.model
                        st.success("RidgeCV Regression Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Best Alpha:** {self.model.alpha_}")
                        self.regression_metrics(st.session_state.get('regressionRidgeCV'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("RidgeCV Regression Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {st.session_state.get('regressionRidgeCV').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionRidgeCV').intercept_}")
                st.write(f"**Best Alpha:** {st.session_state.get('regressionRidgeCV').alpha_}")
                self.regression_metrics(st.session_state.get('regressionRidgeCV'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionRidgeCV'] = None
    def sgd_regressor(self):
        if st.session_state.get('regressionSGD') is None:
            with self.col2:
                st.subheader("SGD Regressor Configuration", divider='blue')
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
                    try:
                        self.model = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=epsilon, learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, warm_start=warm_start, average=average)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionSGD'] = self.model
                        st.success("SGD Regressor Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Total Updates:** {self.model.t_}")
                        self.regression_metrics(st.session_state.get('regressionSGD'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("SGD Regressor Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {st.session_state.get('regressionSGD').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionSGD').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionSGD').n_iter_}")
                st.write(f"**Total Updates:** {st.session_state.get('regressionSGD').t_}")
                self.regression_metrics(st.session_state.get('regressionSGD'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionSGD'] = None
    def elasticNet(self):
        if st.session_state.get('regressionElasticNet') is None:
            with self.col2:
                st.subheader("ElasticNet Configuration", divider='blue')
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
                    try:
                        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, precompute=precompute, max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state if random_state != 0 else None, selection=selection)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionElasticNet'] = self.model
                        st.success("ElasticNet Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        self.regression_metrics(st.session_state.get('regressionElasticNet'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("ElasticNet Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {st.session_state.get('regressionElasticNet').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionElasticNet').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionElasticNet').n_iter_}")
                self.regression_metrics(st.session_state.get('regressionElasticNet'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionElasticNet'] = None
    def elasticNetCV(self):
        if st.session_state.get('regressionElasticNetCV') is None:
            with self.col2:
                st.subheader("ElasticNetCV Configuration", divider="blue")
                l1_ratio = st.multiselect("L1 Ratio (Elastic Net Mixing Parameter)", options=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], default=[0.5])
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
                    try:
                        self.model = ElasticNetCV(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, max_iter=max_iter, tol=tol, cv=cv, positive=positive, selection=selection, fit_intercept=fit_intercept, random_state=random_state if random_state != 0 else None, verbose=verbose)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionElasticNetCV'] = self.model
                        st.success("ElasticNetCV Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Alpha (Best Regularization):** {self.model.alpha_}")
                        st.write(f"**L1 Ratio (Best Mix):** {self.model.l1_ratio_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        self.regression_metrics(st.session_state.get('regressionElasticNetCV'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("ElasticNetCV Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Alpha (Best Regularization):** {st.session_state.get('regressionElasticNetCV').alpha_}")
                st.write(f"**L1 Ratio (Best Mix):** {st.session_state.get('regressionElasticNetCV').l1_ratio_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionElasticNetCV').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionElasticNetCV').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionElasticNetCV').n_iter_}")
                self.regression_metrics(st.session_state.get('regressionElasticNetCV'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionElasticNetCV'] = None
    def lars(self):
        if st.session_state.get('regressionLars') is None:
            with self.col2:
                st.subheader("Lars Configuration", divider="blue")
                fit_intercept = st.checkbox("Fit Intercept", value=True)
                verbose = st.checkbox("Enable Verbose Output", value=False)
                precompute = st.selectbox("Precompute Gram Matrix", options=["auto", True, False], index=0)
                n_nonzero_coefs = st.number_input("Number of Non-Zero Coefficients", min_value=1, value=500)
                eps = st.number_input("Machine Precision Regularization (eps)", min_value=0.0, value=float(np.finfo(float).eps))
                copy_X = st.checkbox("Copy X (Avoid Overwriting Data)", value=True)
                fit_path = st.checkbox("Store Full Path in `coef_path_`", value=True)
                jitter = st.number_input("Jitter (Upper Bound of Noise)", min_value=0.0, value=0.0, help="Add noise to improve stability; leave as 0 if not needed.")
                random_state = st.number_input("Random State", min_value=0, value=0, step=1, help="Set for reproducibility; ignored if jitter is None.")
                if st.checkbox("Train Lars Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = Lars(fit_intercept=fit_intercept, verbose=verbose, precompute=precompute, n_nonzero_coefs=n_nonzero_coefs, eps=eps, copy_X=copy_X, fit_path=fit_path, jitter=None if jitter == 0 else jitter, random_state=random_state if jitter != 0 else None)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionLars'] = self.model
                        st.success("Lars Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Alpha Values:** {self.model.alphas_}")
                        st.write(f"**Active Variables:** {self.model.active_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        self.regression_metrics(st.session_state.get('regressionLars'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Lars Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Alpha Values:** {st.session_state.get('regressionLars').alphas_}")
                st.write(f"**Active Variables:** {st.session_state.get('regressionLars').active_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionLars').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionLars').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionLars').n_iter_}")
                self.regression_metrics(st.session_state.get('regressionLars'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionLars'] = None
    def lars_cv(self):
        if st.session_state.get('regressionLarsCV') is None:
            with self.col2:
                st.subheader("LarsCV Configuration", divider="blue")
                fit_intercept = st.checkbox("Fit Intercept", value=True)
                verbose = st.checkbox("Enable Verbose Output", value=False)
                max_iter = st.number_input("Maximum Number of Iterations", min_value=1, value=500)
                precompute = st.selectbox("Precompute Gram Matrix", options=["auto", True, False], index=0)
                cv = st.selectbox("Cross-Validation Strategy", options=["None (default 5-fold CV)", "Integer (Specify number of folds)", "Custom CV Splitter"])
                cv_folds = st.number_input("Number of Folds for Cross-Validation", min_value=2, value=5) if cv == "Integer (Specify number of folds)" else None
                if cv == "Custom CV Splitter": st.warning("Custom CV splitters are not directly supported in this app. You need to implement it in your code.")
                max_n_alphas = st.number_input("Maximum Number of Alpha Points", min_value=1, value=1000)
                n_jobs = st.number_input("Number of Jobs (-1 for all processors)", value=1, step=1)
                eps = st.number_input("Machine Precision Regularization (eps)", min_value=0.0, value=float(np.finfo(float).eps))
                copy_X = st.checkbox("Copy X (Avoid Overwriting Data)", value=True)
                if st.checkbox("Train LarsCV Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        cv_param = int(cv_folds) if cv == "Integer (Specify number of folds)" else None
                        self.model = LarsCV(fit_intercept=fit_intercept, verbose=verbose, max_iter=max_iter, precompute=precompute, cv=cv_param, max_n_alphas=max_n_alphas, n_jobs=n_jobs if n_jobs != 1 else None, eps=eps, copy_X=copy_X)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionLarsCV'] = self.model
                        st.success("LarsCV Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Alpha Values:** {self.model.alphas_}")
                        st.write(f"**Optimal Alpha:** {self.model.alpha_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Mean Squared Error Path:** {self.model.mse_path_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        self.regression_metrics(st.session_state.get('regressionLarsCV'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("LarsCV Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Alpha Values:** {st.session_state.get('regressionLarsCV').alphas_}")
                st.write(f"**Optimal Alpha:** {st.session_state.get('regressionLarsCV').alpha_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionLarsCV').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionLarsCV').intercept_}")
                st.write(f"**Mean Squared Error Path:** {st.session_state.get('regressionLarsCV').mse_path_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionLarsCV').n_iter_}")
                self.regression_metrics(st.session_state.get('regressionLarsCV'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionLarsCV'] = None
    def lasso(self):
        if st.session_state.get('regressionLasso') is None:
            with self.col2:
                st.subheader("Lasso Configuration", divider="blue")
                alpha = st.number_input("Regularization Strength (alpha)", min_value=0.0, value=1.0, help="Controls the regularization strength. Must be a non-negative float.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                precompute = st.selectbox("Precompute Gram Matrix", options=["False", "True"], index=0, help="Whether to use a precomputed Gram matrix to speed up calculations.")
                max_iter = st.number_input("Maximum Number of Iterations", min_value=1, value=1000, help="The maximum number of iterations.")
                tol = st.number_input("Tolerance for Optimization (tol)", min_value=0.0, value=1e-4, format="%.1e", help="Convergence tolerance. Optimization stops if updates are smaller than this value.")
                warm_start = st.checkbox("Warm Start", value=False, help="Reuse the solution of the previous call to fit as initialization.")
                positive = st.checkbox("Force Positive Coefficients", value=False, help="If set to True, forces the coefficients to be positive.")
                selection = st.selectbox("Feature Selection Method", options=["cyclic", "random"], index=0, help="Choose 'cyclic' for sequential feature updates or 'random' for faster convergence.")
                random_state = st.number_input("Random State (Optional)", min_value=0, value=0, step=1, help="Seed for reproducibility when using random selection. Leave 0 for default.")
                if st.checkbox("Train Lasso Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = Lasso(alpha=alpha, fit_intercept=fit_intercept, precompute=precompute == "True", max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state if random_state != 0 else None, selection=selection)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionLasso'] = self.model
                        st.success("Lasso Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        self.regression_metrics(st.session_state.get('regressionLasso'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Lasso Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {st.session_state.get('regressionLasso').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionLasso').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionLasso').n_iter_}")
                self.regression_metrics(st.session_state.get('regressionLasso'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionLasso'] = None
    def lassocv(self):
        if st.session_state.get('regressionLassoCV') is None:
            with self.col2:
                st.subheader("LassoCV Configuration", divider="blue")
                eps = st.number_input("Path Length (eps)", min_value=1e-6, value=1e-3, step=1e-3, format="%.1e", help="Length of the path; controls the ratio of alpha_min / alpha_max.")
                n_alphas = st.number_input("Number of Alphas", min_value=1, value=100, help="Number of alphas along the regularization path.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                precompute = st.selectbox("Precompute Gram Matrix", options=["auto", "False", "True"], index=0, help="Whether to use a precomputed Gram matrix to speed up calculations.")
                max_iter = st.number_input("Maximum Number of Iterations", min_value=1, value=1000, help="The maximum number of iterations.")
                tol = st.number_input("Tolerance for Optimization (tol)", min_value=0.0, value=1e-4, format="%.1e", help="Convergence tolerance. Optimization stops if updates are smaller than this value.")
                cv = st.number_input("Cross-Validation Folds (cv)", min_value=2, value=5, help="Number of folds for cross-validation. Default is 5-fold.")
                positive = st.checkbox("Force Positive Coefficients", value=False, help="If set to True, restrict regression coefficients to be positive.")
                random_state = st.number_input("Random State (Optional)", min_value=0, value=0, step=1, help="Seed for reproducibility when using random selection. Leave 0 for default.")
                selection = st.selectbox("Feature Selection Method", options=["cyclic", "random"], index=0, help="Choose 'cyclic' for sequential feature updates or 'random' for faster convergence.")
                n_jobs = st.selectbox("Number of Jobs", options=[None, -1, 1, 2, 4], index=1, help="Number of CPUs to use during cross-validation. -1 uses all processors.")
                if st.checkbox("Train LassoCV Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = LassoCV(eps=eps, n_alphas=n_alphas, fit_intercept=fit_intercept, precompute=precompute if precompute != "auto" else "auto", max_iter=max_iter, tol=tol, cv=cv, positive=positive, random_state=random_state if random_state != 0 else None, selection=selection, n_jobs=n_jobs)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionLassoCV'] = self.model
                        st.success("LassoCV Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Optimal Alpha:** {self.model.alpha_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Alphas Grid:** {self.model.alphas_}")
                        st.markdown("### Mean Squared Error Path")
                        for i, mse_path in enumerate(self.model.mse_path_):
                            st.line_chart(mse_path, height=200, width=700)
                        self.regression_metrics(st.session_state.get('regressionLassoCV'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("LassoCV Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Optimal Alpha:** {st.session_state.get('regressionLassoCV').alpha_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionLassoCV').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionLassoCV').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionLassoCV').n_iter_}")
                st.write(f"**Alphas Grid:** {st.session_state.get('regressionLassoCV').alphas_}")
                st.markdown("### Mean Squared Error Path")
                for i, mse_path in enumerate(st.session_state.get('regressionLassoCV').mse_path_):
                    st.line_chart(mse_path, height=200, width=700)
                self.regression_metrics(st.session_state.get('regressionLassoCV'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionLassoCV'] = None
    def lassolars(self):
        if st.session_state.get('regressionLassoLars') is None:
            with self.col2:
                st.subheader("LassoLars Configuration", divider="blue")
                alpha = st.number_input("Alpha", min_value=0.0, value=1.0, step=0.1, help="Constant that multiplies the penalty term. Alpha = 0 is equivalent to ordinary least squares.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                verbose = st.checkbox("Verbose", value=False, help="Sets the verbosity amount.")
                precompute = st.selectbox("Precompute Gram Matrix", options=["auto", "False", "True"], index=0, help="Whether to use a precomputed Gram matrix to speed up calculations.")
                max_iter = st.number_input("Maximum Number of Iterations", min_value=1, value=500, help="The maximum number of iterations.")
                eps = st.number_input("Machine Precision Regularization (eps)", min_value=1e-16, value=1e-3, step=1e-3, format="%.1e", help="Machine-precision regularization in Cholesky diagonal factor computation.")
                copy_X = st.checkbox("Copy X", value=True, help="If True, X will be copied; else, it may be overwritten.")
                fit_path = st.checkbox("Fit Path", value=True, help="If True, the full path will be stored in coef_path_.")
                positive = st.checkbox("Force Positive Coefficients", value=False, help="If True, restrict regression coefficients to be positive.")
                jitter = st.number_input("Jitter", min_value=0.0, value=0.0, step=0.1, help="Upper bound on a uniform noise parameter added to y values for stability.")
                random_state = st.number_input("Random State (Optional)", min_value=0, value=0, step=1, help="Seed for reproducibility when using random selection.")
                if st.checkbox("Train LassoLars Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = LassoLars(alpha=alpha, fit_intercept=fit_intercept, verbose=verbose, precompute=precompute if precompute != "auto" else "auto", max_iter=max_iter, eps=eps, copy_X=copy_X, fit_path=fit_path, positive=positive, jitter=jitter if jitter != 0 else None, random_state=random_state if random_state != 0 else None)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionLassoLars'] = self.model
                        st.success("LassoLars Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Alpha:** {self.model.alphas_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Active Variables:** {self.model.active_}")
                        st.write(f"**Alphas Grid:** {self.model.alphas_}")
                        if hasattr(self.model, 'mse_path_'):
                            st.markdown("### Mean Squared Error Path")
                            for i, mse_path in enumerate(self.model.mse_path_):
                                st.line_chart(mse_path, height=200, width=700, caption=f"Fold {i+1}")
                        self.regression_metrics()
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("LassoLars Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Alpha:** {st.session_state.get('regressionLassoLars').alphas_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionLassoLars').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionLassoLars').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionLassoLars').n_iter_}")
                st.write(f"**Active Variables:** {st.session_state.get('regressionLassoLars').active_}")
                st.write(f"**Alphas Grid:** {st.session_state.get('regressionLassoLars').alphas_}")
                if hasattr(st.session_state.get('regressionLassoLars'), 'mse_path_'):
                    st.markdown("### Mean Squared Error Path")
                    for i, mse_path in enumerate(st.session_state.get('regressionLassoLars').mse_path_):
                        st.line_chart(mse_path, height=200, width=700, caption=f"Fold {i+1}")
                self.regression_metrics(st.session_state.get('regressionLassoLars'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionLassoLars'] = None
    def lasso_lars_cv(self):
        if st.session_state.get('regressionLassoLarsCV') is None:
            with self.col2:
                st.subheader("LassoLarsCV Configuration", divider="blue")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                verbose = st.checkbox("Verbose", value=False, help="Set verbosity level for detailed output.")
                max_iter = st.number_input("Maximum Number of Iterations", min_value=1, value=500, help="The maximum number of iterations for the algorithm.")
                precompute = st.selectbox("Precompute Gram Matrix", options=["auto", "False", "True"], index=0, help="Whether to use a precomputed Gram matrix.")
                cv = st.number_input("Cross-Validation Folds (cv)", min_value=2, value=5, help="Number of folds for cross-validation.")
                max_n_alphas = st.number_input("Maximum Number of Alphas", min_value=1, value=1000, help="The maximum number of points on the path to compute residuals.")
                n_jobs = st.selectbox("Number of Jobs", options=[None, -1, 1, 2, 4], index=1, help="Number of CPUs to use during cross-validation. -1 uses all processors.")
                eps = st.number_input("Epsilon (eps)", min_value=1e-6, value=1e-6, step=1e-6, format="%.1e", help="Machine precision regularization.")
                copy_X = st.checkbox("Copy X", value=True, help="Whether to copy the input matrix X.")
                positive = st.checkbox("Force Positive Coefficients", value=False, help="Restrict coefficients to be greater than or equal to 0.")
                if st.checkbox("Train LassoLarsCV Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = LassoLarsCV(fit_intercept=fit_intercept, verbose=verbose, max_iter=max_iter, precompute=precompute if precompute != "auto" else "auto", cv=cv, max_n_alphas=max_n_alphas, n_jobs=n_jobs, eps=eps, copy_X=copy_X, positive=positive)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionLassoLarsCV'] = self.model
                        st.success("LassoLarsCV Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Optimal Alpha:** {self.model.alpha_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Alphas Grid:** {self.model.alphas_}")
                        st.markdown("### Mean Squared Error Path")
                        for i, mse_path in enumerate(self.model.mse_path_):
                            st.line_chart(mse_path, height=200, width=700)
                        self.regression_metrics(st.session_state.get('regressionLassoLarsCV'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("LassoLarsCV Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Optimal Alpha:** {st.session_state.get('regressionLassoLarsCV').alpha_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionLassoLarsCV').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionLassoLarsCV').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionLassoLarsCV').n_iter_}")
                st.write(f"**Alphas Grid:** {st.session_state.get('regressionLassoLarsCV').alphas_}")
                st.markdown("### Mean Squared Error Path")
                for i, mse_path in enumerate(st.session_state.get('regressionLassoLarsCV').mse_path_):
                    st.line_chart(mse_path, height=200, width=700)
                self.regression_metrics(st.session_state.get('regressionLassoLarsCV'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionLassoLarsCV'] = None
    def lasso_lars_ic(self):
        if st.session_state.get('regressionLassoLarsIC') is None:
            with self.col2:
                st.subheader("LassoLarsIC Configuration", divider="blue")
                criterion = st.selectbox("Criterion", options=["aic", "bic"], index=0, help="The criterion to use for model selection: AIC or BIC.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                verbose = st.checkbox("Verbose", value=False, help="Set verbosity level for detailed output.")
                precompute = st.selectbox("Precompute Gram Matrix", options=["auto", "False", "True"], index=0, help="Whether to use a precomputed Gram matrix.")
                max_iter = st.number_input("Maximum Number of Iterations", min_value=1, value=500, help="The maximum number of iterations for the algorithm.")
                eps = st.number_input("Epsilon (eps)", min_value=1e-6, value=1e-6, step=1e-6, format="%.1e", help="Machine precision regularization.")
                copy_X = st.checkbox("Copy X", value=True, help="Whether to copy the input matrix X.")
                positive = st.checkbox("Force Positive Coefficients", value=False, help="Restrict coefficients to be greater than or equal to 0.")
                noise_variance = st.number_input("Noise Variance", min_value=0.0, value=0.0, step=1e-6, format="%.1e", help="Estimated noise variance. Set to None for automatic estimation.")
                if st.checkbox("Train LassoLarsIC Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = LassoLarsIC(criterion=criterion, fit_intercept=fit_intercept, verbose=verbose, precompute=precompute if precompute != "auto" else "auto", max_iter=max_iter, eps=eps, copy_X=copy_X, positive=positive, noise_variance=noise_variance if noise_variance != 0 else None)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionLassoLarsIC'] = self.model
                        st.success("LassoLarsIC Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Optimal Alpha:** {self.model.alpha_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Criterion Values (AIC/BIC):** {self.model.criterion_}")
                        st.write(f"**Noise Variance:** {self.model.noise_variance_}")
                        st.markdown("### Information Criterion Path")
                        st.line_chart(self.model.criterion_, height=200, width=700)
                        self.regression_metrics()
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("LassoLarsIC Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Optimal Alpha:** {st.session_state.get('regressionLassoLarsIC').alpha_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionLassoLarsIC').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionLassoLarsIC').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionLassoLarsIC').n_iter_}")
                st.write(f"**Criterion Values (AIC/BIC):** {st.session_state.get('regressionLassoLarsIC').criterion_}")
                st.write(f"**Noise Variance:** {st.session_state.get('regressionLassoLarsIC').noise_variance_}")
                st.markdown("### Information Criterion Path")
                st.line_chart(st.session_state.get('regressionLassoLarsIC').criterion_, height=200, width=700)
                self.regression_metrics(st.session_state.get('regressionLassoLarsIC'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionLassoLarsIC'] = None
    def orthogonal_matching_pursuit(self):
        if st.session_state.get('regressionOMP') is None:
            with self.col2:
                st.subheader("Orthogonal Matching Pursuit Configuration", divider="blue")
                n_nonzero_coefs = st.number_input("Desired Number of Non-zero Coefficients", min_value=1, value=10, help="The desired number of non-zero coefficients in the solution. Ignored if tol is set.")
                tol = st.number_input("Tolerance (tol)", min_value=0.0, value=0.0, step=1e-6, format="%.1e", help="Maximum squared norm of the residual. Overrides n_nonzero_coefs.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                precompute = st.selectbox("Precompute", options=["auto", "True", "False"], index=0, help="Whether to use a precomputed Gram and Xy matrix to speed up calculations.")
                if st.checkbox("Train OrthogonalMatchingPursuit Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs if tol == 0 else None, tol=tol if tol != 0 else None, fit_intercept=fit_intercept, precompute=precompute if precompute != "auto" else "auto")
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionOMP'] = self.model
                        st.success("OrthogonalMatchingPursuit Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Number of Active Features:** {self.model.n_iter_}")
                        st.write(f"**Number of Non-zero Coefficients:** {self.model.n_nonzero_coefs_}")
                        self.regression_metrics()
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("OrthogonalMatchingPursuit Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:** {st.session_state.get('regressionOMP').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionOMP').intercept_}")
                st.write(f"**Number of Active Features:** {st.session_state.get('regressionOMP').n_iter_}")
                st.write(f"**Number of Non-zero Coefficients:** {st.session_state.get('regressionOMP').n_nonzero_coefs_}")
                self.regression_metrics(st.session_state.get('regressionOMP'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionOMP'] = None
    def orthogonal_matching_pursuit_cv(self):
        if st.session_state.get('regressionOMPCV') is None:
            with self.col2:
                st.subheader("Orthogonal Matching Pursuit with Cross-Validation", divider="blue")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                max_iter = st.number_input("Max Iterations (max_iter)", min_value=1, value=10, help="Maximum number of iterations to perform.")
                cv = st.number_input("Cross-Validation Folds (cv)", min_value=2, value=5, help="Number of folds in cross-validation.")
                n_jobs = st.number_input("Number of Jobs (n_jobs)", min_value=-1, value=-1, help="Number of CPUs to use during cross-validation.")
                verbose = st.checkbox("Verbose", value=False, help="Sets the verbosity amount.")
                if st.checkbox("Train Orthogonal Matching Pursuit CV Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = OrthogonalMatchingPursuitCV(fit_intercept=fit_intercept, max_iter=max_iter, cv=cv, n_jobs=n_jobs if n_jobs != -1 else None, verbose=verbose)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionOMPCV'] = self.model
                        st.success("OrthogonalMatchingPursuitCV Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Estimated Non-zero Coefficients:** {self.model.n_nonzero_coefs_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        self.regression_metrics()
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("OrthogonalMatchingPursuitCV Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Intercept:** {st.session_state.get('regressionOMP').intercept_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionOMP').coef_}")
                st.write(f"**Estimated Non-zero Coefficients:** {st.session_state.get('regressionOMP').n_nonzero_coefs_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionOMP').n_iter_}")
                self.regression_metrics(st.session_state.get('regressionOMP'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionOMPCV'] = None

    def ard_regression(self):
        if st.session_state.get('regressionARD') is None:
            with self.col2:
                st.subheader("Bayesian ARD Regression", divider="blue")
                max_iter = st.number_input("Max Iterations (max_iter)", min_value=1, value=300, help="Maximum number of iterations for convergence.")
                tol = st.number_input("Tolerance (tol)", min_value=0.0001, value=0.001, help="Stop the algorithm if the weights converge.")
                alpha_1 = st.number_input("Alpha 1 (alpha_1)", min_value=0.0, value=1e-6, help="Shape parameter for the Gamma distribution prior over alpha.")
                alpha_2 = st.number_input("Alpha 2 (alpha_2)", min_value=0.0, value=1e-6, help="Inverse scale parameter for the Gamma distribution prior over alpha.")
                lambda_1 = st.number_input("Lambda 1 (lambda_1)", min_value=0.0, value=1e-6, help="Shape parameter for the Gamma distribution prior over lambda.")
                lambda_2 = st.number_input("Lambda 2 (lambda_2)", min_value=0.0, value=1e-6, help="Inverse scale parameter for the Gamma distribution prior over lambda.")
                threshold_lambda = st.number_input("Threshold Lambda (threshold_lambda)", min_value=1.0, value=10000.0, help="Threshold for pruning weights with high precision.")
                compute_score = st.checkbox("Compute Objective Function", value=False, help="If True, compute the objective function at each step.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                verbose = st.checkbox("Verbose", value=False, help="Set to True for verbose output during fitting.")
                if st.checkbox("Train ARD Regression Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = ARDRegression(max_iter=max_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2, threshold_lambda=threshold_lambda, compute_score=compute_score, fit_intercept=fit_intercept, verbose=verbose)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionARD'] = self.model
                        st.success("ARD Regression Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Alpha (Noise Precision):** {self.model.alpha_}")
                        st.write(f"**Lambda (Weight Precision):** {self.model.lambda_}")
                        st.write(f"**Sigma (Variance-Covariance Matrix):** {self.model.sigma_}")
                        if compute_score:
                            st.write(f"**Scores (Objective Function):** {self.model.scores_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        self.regression_metrics()
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("ARD Regression Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Intercept:** {st.session_state.get('regressionOMP').intercept_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionOMP').coef_}")
                st.write(f"**Alpha (Noise Precision):** {st.session_state.get('regressionOMP').alpha_}")
                st.write(f"**Lambda (Weight Precision):** {st.session_state.get('regressionOMP').lambda_}")
                st.write(f"**Sigma (Variance-Covariance Matrix):** {st.session_state.get('regressionOMP').sigma_}")
                if hasattr(st.session_state.get('regressionOMP'), 'scores_'):
                    st.write(f"**Scores (Objective Function):** {st.session_state.get('regressionOMP').scores_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionOMP').n_iter_}")
                self.regression_metrics(st.session_state.get('regressionOMP'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionARD'] = None
    def bayesian_ridge(self):
        if st.session_state.get('regressionBayesianRidge') is None:
            with self.col2:
                st.subheader("Bayesian Ridge Regression", divider="blue")
                max_iter = st.number_input("Max Iterations (max_iter)", min_value=1, value=300, help="Maximum number of iterations for convergence.")
                tol = st.number_input("Tolerance (tol)", min_value=0.0001, value=0.001, help="Stop the algorithm if the weights converge.")
                alpha_1 = st.number_input("Alpha 1 (alpha_1)", min_value=0.0, value=1e-6, help="Shape parameter for the Gamma distribution prior over alpha.")
                alpha_2 = st.number_input("Alpha 2 (alpha_2)", min_value=0.0, value=1e-6, help="Inverse scale parameter for the Gamma distribution prior over alpha.")
                lambda_1 = st.number_input("Lambda 1 (lambda_1)", min_value=0.0, value=1e-6, help="Shape parameter for the Gamma distribution prior over lambda.")
                lambda_2 = st.number_input("Lambda 2 (lambda_2)", min_value=0.0, value=1e-6, help="Inverse scale parameter for the Gamma distribution prior over lambda.")
                alpha_init = st.number_input("Initial Alpha (alpha_init)", min_value=0.0, value=None, help="Initial value for alpha (precision of the noise).")
                lambda_init = st.number_input("Initial Lambda (lambda_init)", min_value=0.0, value=None, help="Initial value for lambda (precision of the weights).")
                compute_score = st.checkbox("Compute Log Marginal Likelihood", value=False, help="If True, compute the log marginal likelihood at each iteration.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                verbose = st.checkbox("Verbose", value=False, help="Set to True for verbose output during fitting.")
                if st.checkbox("Train Bayesian Ridge Regression Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = BayesianRidge(max_iter=max_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2, alpha_init=alpha_init if alpha_init is not None else None, lambda_init=lambda_init if lambda_init is not None else None, compute_score=compute_score, fit_intercept=fit_intercept, verbose=verbose)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionBayesianRidge'] = self.model
                        st.success("Bayesian Ridge Regression Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Alpha (Noise Precision):** {self.model.alpha_}")
                        st.write(f"**Lambda (Weight Precision):** {self.model.lambda_}")
                        st.write(f"**Sigma (Variance-Covariance Matrix):** {self.model.sigma_}")
                        if compute_score:
                            st.write(f"**Scores (Log Marginal Likelihood):** {self.model.scores_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        self.regression_metrics(st.session_state.get('regressionBayesianRidge'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Bayesian Ridge Regression Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Intercept:** {st.session_state.get('regressionBayesianRidge').intercept_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionBayesianRidge').coef_}")
                st.write(f"**Alpha (Noise Precision):** {st.session_state.get('regressionBayesianRidge').alpha_}")
                st.write(f"**Lambda (Weight Precision):** {st.session_state.get('regressionBayesianRidge').lambda_}")
                st.write(f"**Sigma (Variance-Covariance Matrix):** {st.session_state.get('regressionBayesianRidge').sigma_}")
                if hasattr(st.session_state.get('regressionBayesianRidge'), 'scores_'):
                    st.write(f"**Scores (Log Marginal Likelihood):** {st.session_state.get('regressionBayesianRidge').scores_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionBayesianRidge').n_iter_}")
                self.regression_metrics(st.session_state.get('regressionBayesianRidge'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionBayesianRidge'] = None
    def multi_task_elastic_net(self):
        if st.session_state.get('regressionMTElasticNet') is None:
            with self.col2:
                st.subheader("Multi-task ElasticNet Regression", divider="blue")
                alpha = st.number_input("Alpha", min_value=0.0, value=1.0, help="Constant that multiplies the L1/L2 term. Defaults to 1.0.")
                l1_ratio = st.number_input("L1 Ratio", min_value=0.0, max_value=1.0, value=0.5, help="The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                max_iter = st.number_input("Max Iterations", min_value=1, value=1000, help="The maximum number of iterations.")
                tol = st.number_input("Tolerance", min_value=1e-5, value=1e-4, help="The tolerance for optimization.")
                warm_start = st.checkbox("Warm Start", value=False, help="Reuse the solution of the previous call to fit.")
                random_state = st.number_input("Random State", value=None, help="The seed of the pseudo-random number generator.")
                selection = st.selectbox("Selection", ["cyclic", "random"], help="If set to 'random', a random coefficient is updated every iteration.")
                if st.checkbox("Train Multi-task ElasticNet Regression Model"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, warm_start=warm_start, random_state=random_state if random_state is not None else None, selection=selection)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionMTElasticNet'] = self.model
                        st.success("Multi-task ElasticNet Regression Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Dual Gap:** {self.model.dual_gap_}")
                        st.write(f"**Iterations:** {self.model.n_iter_}")
                        st.write(f"**Tolerance Scaled (eps):** {self.model.eps_}")
                        self.regression_metrics(st.session_state.get('regressionMTElasticNet'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Multi-task ElasticNet Regression Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Intercept:** {st.session_state.get('regressionMTElasticNet').intercept_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionMTElasticNet').coef_}")
                st.write(f"**Dual Gap:** {st.session_state.get('regressionMTElasticNet').dual_gap_}")
                st.write(f"**Iterations:** {st.session_state.get('regressionMTElasticNet').n_iter_}")
                st.write(f"**Tolerance Scaled (eps):** {st.session_state.get('regressionMTElasticNet').eps_}")
                self.regression_metrics(st.session_state.get('regressionMTElasticNet'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionMTElasticNet'] = None
    def multi_task_elastic_net_cv(self):
        if st.session_state.get('regressionMTElasticNetCV') is None:
            with self.col2:
                st.subheader("Multi-task ElasticNet with Cross-Validation", divider="blue")
                l1_ratio = st.slider("L1 Ratio", min_value=0.0, max_value=1.0, value=0.5, help="The ElasticNet mixing parameter. Value between 0 and 1.")
                eps = st.number_input("Epsilon", min_value=1e-6, value=1e-3, help="Length of the regularization path.")
                n_alphas = st.number_input("Number of Alphas", min_value=1, value=100, help="The number of alphas along the regularization path.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                max_iter = st.number_input("Max Iterations", min_value=1, value=1000, help="Maximum number of iterations.")
                tol = st.number_input("Tolerance", min_value=1e-5, value=1e-4, help="The tolerance for optimization.")
                cv = st.number_input("Cross-validation Folds", min_value=2, value=5, help="The number of folds for cross-validation.")
                verbose = st.slider("Verbosity", min_value=0, max_value=3, value=0, help="Amount of verbosity for the fitting process.")
                n_jobs = st.number_input("Number of Jobs", min_value=-1, value=None, help="Number of CPUs to use during cross-validation. Use -1 for all processors.")
                random_state = st.number_input("Random State", value=None, help="The seed for random number generation.")
                selection = st.selectbox("Selection", ["cyclic", "random"], help="Choose whether to update coefficients sequentially or randomly.")
                if st.checkbox("Train Multi-task ElasticNet with Cross-validation"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = MultiTaskElasticNetCV(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, cv=cv, verbose=verbose, n_jobs=n_jobs if n_jobs != -1 else None, random_state=random_state, selection=selection)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionMTElasticNetCV'] = self.model
                        st.success("Multi-task ElasticNetCV Model Trained Successfully with Cross-validation!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Optimal alpha:** {self.model.alpha_}")
                        st.write(f"**Best L1 Ratio:** {self.model.l1_ratio_}")
                        st.write(f"**MSE Path:** {self.model.mse_path_}")
                        st.write(f"**Alpha Path:** {self.model.alphas_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Dual Gap:** {self.model.dual_gap_}")
                        self.regression_metrics(st.session_state.get('regressionMTElasticNetCV'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Multi-task ElasticNetCV Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Optimal alpha:** {st.session_state.get('regressionMTElasticNetCV').alpha_}")
                st.write(f"**Best L1 Ratio:** {st.session_state.get('regressionMTElasticNetCV').l1_ratio_}")
                st.write(f"**MSE Path:** {st.session_state.get('regressionMTElasticNetCV').mse_path_}")
                st.write(f"**Alpha Path:** {st.session_state.get('regressionMTElasticNetCV').alphas_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionMTElasticNetCV').n_iter_}")
                st.write(f"**Dual Gap:** {st.session_state.get('regressionMTElasticNetCV').dual_gap_}")
                self.regression_metrics(st.session_state.get('regressionMTElasticNetCV'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionMTElasticNetCV'] = None
    def multi_task_lasso(self):
        if st.session_state.get('regressionMTLasso') is None:
            with self.col2:
                st.subheader("Multi-task Lasso", divider="blue")
                alpha = st.number_input("Alpha", min_value=0.01, value=1.0, step=0.01, help="Constant that multiplies the L1/L2 term.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                max_iter = st.number_input("Max Iterations", min_value=1, value=1000, help="Maximum number of iterations.")
                tol = st.number_input("Tolerance", min_value=1e-5, value=1e-4, help="Tolerance for optimization.")
                warm_start = st.checkbox("Warm Start", value=False, help="Reuse the solution of the previous call to fit as initialization.")
                selection = st.selectbox("Selection", ["cyclic", "random"], help="Choose the feature selection method.")
                if st.checkbox("Train Multi-task Lasso"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = MultiTaskLasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, warm_start=warm_start, selection=selection)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionMTLasso'] = self.model
                        st.success("Multi-task Lasso Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:**\n{self.model.coef_}")
                        st.write(f"**Intercepts:**\n{self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Dual Gaps:** {self.model.dual_gap_}")
                        if hasattr(self.model, 'feature_names_in_'):
                            st.write(f"**Feature Names:** {self.model.feature_names_in_}")
                        self.regression_metrics(st.session_state.get('regressionMTLasso'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Multi-task Lasso Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:**\n{st.session_state.get('regressionMTLasso').coef_}")
                st.write(f"**Intercepts:**\n{st.session_state.get('regressionMTLasso').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionMTLasso').n_iter_}")
                st.write(f"**Dual Gaps:** {st.session_state.get('regressionMTLasso').dual_gap_}")
                if hasattr(self.model, 'feature_names_in_'):
                    st.write(f"**Feature Names:** {st.session_state.get('regressionMTLasso').feature_names_in_}")
                self.regression_metrics(st.session_state.get('regressionMTLasso'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionMTLasso'] = None
    def multi_task_lasso_cv(self):
        if st.session_state.get('regressionMTLassoCV') is None:
            with self.col2:
                st.subheader("Multi-task LassoCV", divider="blue")
                eps = st.number_input("Epsilon", min_value=1e-5, value=1e-3, step=1e-5, help="Length of the path. Determines alpha_min/alpha_max.")
                n_alphas = st.number_input("Number of Alphas", min_value=10, value=100, help="Number of alphas along the regularization path.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                max_iter = st.number_input("Max Iterations", min_value=1, value=1000, help="Maximum number of iterations.")
                tol = st.number_input("Tolerance", min_value=1e-5, value=1e-4, help="Tolerance for optimization.")
                cv = st.number_input("Cross-validation Folds", min_value=2, value=5, help="Number of cross-validation folds.")
                verbose = st.checkbox("Verbose", value=False, help="Print detailed output during model fitting.")
                n_jobs = st.number_input("Number of Jobs", min_value=-1, value=None, help="Number of CPUs to use during cross-validation.")
                selection = st.selectbox("Selection", ["cyclic", "random"], help="Choose the feature selection method.")
                if st.checkbox("Train Multi-task LassoCV"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = MultiTaskLassoCV(eps=eps, n_alphas=n_alphas, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, cv=cv, verbose=verbose, n_jobs=n_jobs if n_jobs != -1 else None, selection=selection)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionMTLassoCV'] = self.model
                        st.success("Multi-task LassoCV Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:**\n{self.model.coef_}")
                        st.write(f"**Intercepts:**\n{self.model.intercept_}")
                        st.write(f"**Alpha Chosen by CV:** {self.model.alpha_}")
                        st.write(f"**Mean Squared Error Path:**\n{self.model.mse_path_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Dual Gap:** {self.model.dual_gap_}")
                        if hasattr(self.model, 'feature_names_in_'):
                            st.write(f"**Feature Names:** {self.model.feature_names_in_}")
                        self.regression_metrics(st.session_state.get('regressionMTLassoCV'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Multi-task LassoCV Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:**\n{st.session_state.get('regressionMTLassoCV').coef_}")
                st.write(f"**Intercepts:**\n{st.session_state.get('regressionMTLassoCV').intercept_}")
                st.write(f"**Alpha Chosen by CV:** {st.session_state.get('regressionMTLassoCV').alpha_}")
                st.write(f"**Mean Squared Error Path:**\n{st.session_state.get('regressionMTLassoCV').mse_path_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionMTLassoCV').n_iter_}")
                st.write(f"**Dual Gap:** {st.session_state.get('regressionMTLassoCV').dual_gap_}")
                if hasattr(st.session_state.get('regressionMTLassoCV'), 'feature_names_in_'):
                    st.write(f"**Feature Names:** {st.session_state.get('regressionMTLassoCV').feature_names_in_}")
                self.regression_metrics(st.session_state.get('regressionMTLassoCV'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionMTLassoCV'] = None
    def huber_regressor(self):
        if st.session_state.get('regressionHuber') is None:
            with self.col2:
                st.subheader("Huber Regressor", divider="blue")
                epsilon = st.number_input("Epsilon", min_value=1.0, value=1.35, step=0.05, help="Epsilon controls the robustness to outliers. Smaller epsilon makes the model more robust.")
                alpha = st.number_input("Alpha (Regularization)", min_value=0.0, value=0.0001, step=1e-5, help="Strength of the squared L2 regularization.")
                max_iter = st.number_input("Max Iterations", min_value=10, value=100, help="Maximum number of iterations for fitting the model.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                tol = st.number_input("Tolerance", min_value=1e-6, value=1e-5, help="Tolerance for optimization.")
                warm_start = st.checkbox("Warm Start", value=False, help="Whether to reuse the solution of the previous fit.")
                if st.checkbox("Train Huber Regressor"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=max_iter, fit_intercept=fit_intercept, tol=tol, warm_start=warm_start)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionHuber'] = self.model
                        st.success("Huber Regressor Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:**\n{self.model.coef_}")
                        st.write(f"**Intercept:**\n{self.model.intercept_}")
                        st.write(f"**Scale:**\n{self.model.scale_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Outliers Identified:**\n{self.model.outliers_}")
                        if hasattr(self.model, 'feature_names_in_'):
                            st.write(f"**Feature Names:** {self.model.feature_names_in_}")
                        self.regression_metrics(st.session_state.get('regressionHuber'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Huber Regressor Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:**\n{st.session_state.get('regressionHuber').coef_}")
                st.write(f"**Intercept:**\n{st.session_state.get('regressionHuber').intercept_}")
                st.write(f"**Scale:**\n{st.session_state.get('regressionHuber').scale_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionHuber').n_iter_}")
                st.write(f"**Outliers Identified:**\n{st.session_state.get('regressionHuber').outliers_}")
                if hasattr(st.session_state.get('regressionHuber'), 'feature_names_in_'):
                    st.write(f"**Feature Names:** {st.session_state.get('regressionHuber').feature_names_in_}")
                self.regression_metrics(st.session_state.get('regressionHuber'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionHuber'] = None
    def quantile_regressor(self):
        if st.session_state.get('regressionQuantile') is None:
            with self.col2:
                st.subheader("Quantile Regressor", divider="blue")
                quantile = st.number_input("Quantile", min_value=0.01, max_value=0.99, value=0.5, step=0.01, help="The quantile the model predicts (default is the 50% quantile, i.e., the median).")
                alpha = st.number_input("Alpha (Regularization)", min_value=0.0, value=1.0, step=0.1, help="Regularization constant for the L1 penalty.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether or not to fit the intercept.")
                solver = st.selectbox("Solver", options=['highs-ds', 'highs-ipm', 'highs', 'interior-point', 'revised simplex'], index=2, help="Method used to solve the linear programming formulation.")
                solver_options = st.text_input("Solver Options", value='', help="Additional solver parameters in dictionary format (optional).")
                if st.checkbox("Train Quantile Regressor"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        solver_opts = eval(solver_options) if solver_options else None
                        self.model = QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=fit_intercept, solver=solver, solver_options=solver_opts)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionQuantile'] = self.model
                        st.success("Quantile Regressor Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Coefficients:**\n{self.model.coef_}")
                        st.write(f"**Intercept:**\n{self.model.intercept_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        if hasattr(self.model, 'feature_names_in_'):
                            st.write(f"**Feature Names:** {self.model.feature_names_in_}")
                        self.regression_metrics(st.session_state.get('regressionQuantile'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Quantile Regressor Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Coefficients:**\n{st.session_state.get('regressionQuantile').coef_}")
                st.write(f"**Intercept:**\n{st.session_state.get('regressionQuantile').intercept_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionQuantile').n_iter_}")
                if hasattr(st.session_state.get('regressionQuantile'), 'feature_names_in_'):
                    st.write(f"**Feature Names:** {st.session_state.get('regressionQuantile').feature_names_in_}")
                self.regression_metrics(st.session_state.get('regressionQuantile'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionQuantile'] = None
    def ransac_regressor(self):
        if st.session_state.get('regressionRANSAC') is None:
            with self.col2:
                st.subheader("RANSAC Regressor", divider="blue")
                min_samples = st.number_input("Min Samples", min_value=1, max_value=self.xTrain.shape[0] if hasattr(self, 'xTrain') and self.xTrain is not None else 100, value=None, help="Minimum number of samples to be randomly selected.")
                residual_threshold = st.number_input("Residual Threshold", min_value=0.0, value=None, step=0.1, help="Maximum residual for a sample to be classified as an inlier.")
                max_trials = st.number_input("Max Trials", min_value=1, value=100, help="Maximum number of iterations for random sample selection.")
                stop_n_inliers = st.number_input("Stop if N Inliers", min_value=1, value=None, help="Stop if at least this many inliers are found.")
                stop_score = st.number_input("Stop Score", min_value=0.0, value=None, help="Stop if the score exceeds this value.")
                stop_probability = st.slider("Stop Probability", min_value=0.0, max_value=1.0, value=0.99, step=0.01, help="Stop if the probability of outlier-free data exceeds this threshold.")
                loss = st.selectbox("Loss Function", options=['absolute_error', 'squared_error'], index=0, help="Choose loss function for RANSAC.")
                if st.checkbox("Train RANSAC Regressor"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, stop_n_inliers=stop_n_inliers, stop_score=stop_score, stop_probability=stop_probability, loss=loss)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionRANSAC'] = self.model
                        st.success("RANSAC Regressor Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Number of Trials:** {self.model.n_trials_}")
                        st.write(f"**Number of Inliers:** {sum(self.model.inlier_mask_)}")
                        st.write(f"**Inlier Mask:**\n{self.model.inlier_mask_}")
                        st.write(f"**Final Model Coefficients:**\n{self.model.estimator_.coef_}")
                        st.write(f"**Final Model Intercept:**\n{self.model.estimator_.intercept_}")
                        self.regression_metrics(st.session_state.get('regressionRANSAC'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("RANSAC Regressor Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Number of Trials:** {st.session_state.get('regressionRANSAC').n_trials_}")
                st.write(f"**Number of Inliers:** {sum(st.session_state.get('regressionRANSAC').inlier_mask_)}")
                st.write(f"**Inlier Mask:**\n{st.session_state.get('regressionRANSAC').inlier_mask_}")
                st.write(f"**Final Model Coefficients:**\n{st.session_state.get('regressionRANSAC').estimator_.coef_}")
                st.write(f"**Final Model Intercept:**\n{st.session_state.get('regressionRANSAC').estimator_.intercept_}")
                self.regression_metrics(st.session_state.get('regressionRANSAC'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionRANSAC'] = None
    def theil_sen_regressor(self):
        if st.session_state.get('regressionTheilSen') is None:
            with self.col2:
                st.subheader("Theil-Sen Regressor", divider="blue")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to calculate the intercept for this model.")
                max_subpopulation = st.number_input("Max Subpopulation", min_value=1, value=10000, step=1000, help="Maximum number of subsets to consider when calculating least square solutions.")
                n_subsamples = st.number_input("Number of Subsamples", min_value=self.xTrain.shape[1] + 1 if hasattr(self, 'xTrain') and self.xTrain is not None else 2, max_value=self.xTrain.shape[0] if hasattr(self, 'xTrain') and self.xTrain is not None else 100, value=None, help="Number of samples to calculate parameters. Default is the minimum for maximal robustness.")
                max_iter = st.number_input("Max Iterations", min_value=1, value=300, help="Maximum number of iterations for the spatial median calculation.")
                tol = st.number_input("Tolerance", min_value=0.0, value=1e-3, step=0.001, help="Tolerance when calculating the spatial median.")
                n_jobs = st.number_input("Number of Jobs", min_value=-1, value=None, help="Number of CPUs to use during the cross-validation. -1 uses all processors.")
                verbose = st.checkbox("Verbose Mode", value=False, help="Enable verbose mode during fitting.")
                if st.checkbox("Train Theil-Sen Regressor"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = TheilSenRegressor(fit_intercept=fit_intercept, max_subpopulation=max_subpopulation, n_subsamples=n_subsamples, max_iter=max_iter, tol=tol, n_jobs=n_jobs if n_jobs != -1 else None, verbose=verbose)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionTheilSen'] = self.model
                        st.success("Theil-Sen Regressor Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Breakdown Point:** {self.model.breakdown_}")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        self.regression_metrics(st.session_state['regressionTheilSen'])
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Theil-Sen Regressor Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Breakdown Point:** {st.session_state.get('regressionTheilSen').breakdown_}")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionTheilSen').n_iter_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionTheilSen').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionTheilSen').intercept_}")
                self.regression_metrics(st.session_state.get('regressionTheilSen'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionTheilSen'] = None
    def gamma_regressor(self):
        if st.session_state.get('regressionGamma') is None:
            with self.col2:
                st.subheader("Gamma Regressor", divider="blue")
                alpha = st.number_input("Alpha (Regularization Strength)", min_value=0.0, value=1.0, step=0.1, help="Constant that multiplies the L2 penalty term.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Whether to add an intercept term to the model.")
                solver = st.selectbox("Solver", options=['lbfgs', 'newton-cholesky'], index=0, help="Algorithm used to solve the optimization problem.")
                max_iter = st.number_input("Max Iterations", min_value=1, value=100, help="Maximum number of iterations for the solver.")
                tol = st.number_input("Tolerance", min_value=0.0, value=1e-4, step=0.0001, help="Stopping criterion for the solver.")
                warm_start = st.checkbox("Warm Start", value=False, help="Reuse the solution of the previous call to fit.")
                verbose = st.number_input("Verbose", min_value=0, value=0, help="Verbosity for the solver.")
                if st.checkbox("Train Gamma Regressor"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Train-test split not performed. Please perform it first.")
                        return
                    try:
                        self.model = GammaRegressor(alpha=alpha, fit_intercept=fit_intercept, solver=solver, max_iter=max_iter, tol=tol, warm_start=warm_start, verbose=verbose)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionGamma'] = self.model
                        st.success("Gamma Regressor Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        self.regression_metrics(st.session_state.get('regressionGamma'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Gamma Regressor Model Is Trained Successfully")
                st.markdown("### Model Attributes")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionGamma').n_iter_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionGamma').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionGamma').intercept_}")
                self.regression_metrics(st.session_state.get('regressionGamma'))
                if st.button("Re Train The Model", use_container_width=True, type='primary'):
                    st.session_state['regressionGamma'] = None
    def poisson_regressor(self):
        if st.session_state.get('regressionPoisson') is None:
            with self.col2:
                st.subheader("Poisson Regressor", divider="blue")
                alpha = st.number_input("Alpha (Regularization Strength)", min_value=0.0, value=1.0, step=0.1, help="Constant that multiplies the L2 penalty term.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Indicates whether to add an intercept term to the model.")
                solver = st.selectbox("Solver", options=['lbfgs', 'newton-cholesky'], index=0, help="Choose the algorithm for solving the optimization problem.")
                max_iter = st.number_input("Max Iterations", min_value=1, value=100, help="Define the maximum number of iterations for the solver.")
                tol = st.number_input("Tolerance", min_value=0.0, value=1e-4, step=0.0001, help="Set the stopping criterion for the solver.")
                warm_start = st.checkbox("Warm Start", value=False, help="Re-use the solution of the previous fit as the starting point.")
                verbose = st.number_input("Verbose", min_value=0, value=0, help="Set the verbosity level for the solver.")
                if st.checkbox("Train Poisson Regressor"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Please perform the train-test split first.")
                        return
                    try:
                        self.model = PoissonRegressor(alpha=alpha, fit_intercept=fit_intercept, solver=solver, max_iter=max_iter, tol=tol, warm_start=warm_start, verbose=verbose)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionPoisson'] = self.model
                        st.success("Poisson Regressor model has been successfully trained!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        self.regression_metrics(st.session_state.get('regressionPoisson'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Poisson Regressor Model Is Already Trained")
                st.markdown("### Model Attributes")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionPoisson').n_iter_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionPoisson').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionPoisson').intercept_}")
                self.regression_metrics()
                if st.button("Retrain Model", use_container_width=True, type='primary'):
                    st.session_state['regressionPoisson'] = None
    def tweedie_regressor(self):
        if st.session_state.get('regressionTweedie') is None:
            with self.col2:
                st.subheader("Tweedie Regressor", divider="blue")
                power = st.number_input("Power", min_value=0.0, value=0.0, step=0.1, help="The power determines the target distribution (e.g., Poisson, Gamma, Inverse Gaussian).")
                alpha = st.number_input("Alpha (Regularization Strength)", min_value=0.0, value=1.0, step=0.1, help="Constant that multiplies the L2 penalty term.")
                fit_intercept = st.checkbox("Fit Intercept", value=True, help="Indicates whether to add an intercept term to the model.")
                link = st.selectbox("Link Function", options=['auto', 'identity', 'log'], index=0, help="Select the link function for the GLM.")
                solver = st.selectbox("Solver", options=['lbfgs', 'newton-cholesky'], index=0, help="Choose the algorithm for solving the optimization problem.")
                max_iter = st.number_input("Max Iterations", min_value=1, value=100, help="Maximum number of iterations for the solver.")
                tol = st.number_input("Tolerance", min_value=0.0, value=1e-4, step=0.0001, help="Stopping criterion for the solver.")
                warm_start = st.checkbox("Warm Start", value=False, help="Re-use the solution of the previous fit as the starting point.")
                verbose = st.number_input("Verbose", min_value=0, value=0, help="Set the verbosity level for the solver.")
                if st.checkbox("Train Tweedie Regressor"):
                    if self.xTrain is None or self.yTrain is None:
                        st.error("Please perform the train-test split first.")
                        return
                    try:
                        self.model = TweedieRegressor(power=power, alpha=alpha, fit_intercept=fit_intercept, link=link, solver=solver, max_iter=max_iter, tol=tol, warm_start=warm_start, verbose=verbose)
                        self.model.fit(self.xTrain, self.yTrain)
                        st.session_state['regressionTweedie'] = self.model
                        st.success("Tweedie Regressor model has been successfully trained!")
                        st.markdown("### Model Attributes")
                        st.write(f"**Number of Iterations:** {self.model.n_iter_}")
                        st.write(f"**Coefficients:** {self.model.coef_}")
                        st.write(f"**Intercept:** {self.model.intercept_}")
                        self.regression_metrics(st.session_state.get('regressionTweedie'))
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            with self.col2:
                st.success("Tweedie Regressor Model Is Already Trained")
                st.markdown("### Model Attributes")
                st.write(f"**Number of Iterations:** {st.session_state.get('regressionTweedie').n_iter_}")
                st.write(f"**Coefficients:** {st.session_state.get('regressionTweedie').coef_}")
                st.write(f"**Intercept:** {st.session_state.get('regressionTweedie').intercept_}")
                self.regression_metrics(st.session_state.get('regressionTweedie'))
                if st.button("Retrain Model", use_container_width=True, type='primary'):
                    st.session_state['regressionTweedie'] = None
    def regression_metrics(self,model):
        with self.col3:
            self.model=model
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
