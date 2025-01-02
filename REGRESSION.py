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
            classicalLinearModel=st.pills("Select the classical linear model",["Linear Regression","Ridge","RidgeCV","SGDRegressor"])
            if classicalLinearModel=="Linear Regression":
                self.linear_regression()
            elif classicalLinearModel=="Ridge":
                self.ridge_regression()
            elif classicalLinearModel=="RidgeCV":
                self.ridge_cv()
            elif classicalLinearModel=="SGDRegressor":
                self.sgd_regressor()
                
                
                                       
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
