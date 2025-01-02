import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    d2_absolute_error_score, d2_pinball_score, d2_tweedie_score,
    explained_variance_score, max_error, mean_absolute_error,
    mean_absolute_percentage_error, mean_gamma_deviance,
    mean_pinball_loss, mean_poisson_deviance, mean_squared_error,
    mean_squared_log_error, mean_tweedie_deviance, median_absolute_error,
    r2_score
)
from streamlit_extras import *
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
                if st.session_state["LinearRegression"]:
                    with self.col1:
                        st.subheader("Train-Test Split Completed!", divider='blue')
                        metric_card("Training Data Shape", f"{self.xTrain.shape}", delta=None)
                        metric_card("Testing Data Shape", f"{self.xTest.shape}", delta=None)
                    
                    with self.col2:
                        st.success("Linear Regression Model Trained Successfully!")
                        st.markdown("### Model Attributes")
                        metric_card("Coefficients", f"{self.model.coef_}", delta=None)
                        metric_card("Intercept", f"{self.model.intercept_}", delta=None)
                else:
                    self.linear_regression()
                                       
    def train_test_split(self):
        with self.col1:
            st.markdown("### Train-Test Split Configuration")
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
            st.markdown("### Linear Regression Configuration")
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

    def regression_metrics(self):
        with self.col3:
            st.markdown("### Evaluate Regression Metrics")
            metrics = [
                "D2 Absolute Error", "D2 Pinball Loss", "D2 Tweedie Score",
                "Explained Variance", "Max Error", "Mean Absolute Error",
                "Mean Absolute Percentage Error", "Mean Gamma Deviance",
                "Mean Pinball Loss", "Mean Poisson Deviance", "Mean Squared Error",
                "Mean Squared Log Error", "Mean Tweedie Deviance", "Median Absolute Error",
                "R2 Score"
            ]
            selectedMetric = st.pills("Select the metric", metrics)
            if selectedMetric == metrics[0]:
                self.d2AbsoluteError()
            elif selectedMetric == metrics[1]:
                self.d2PinballLoss()
            elif selectedMetric == metrics[2]:
                self.d2TweedieScore()
            elif selectedMetric == metrics[3]:
                self.explainedVariance()
            elif selectedMetric == metrics[4]:
                self.maxError()
            elif selectedMetric == metrics[5]:
                self.meanAbsoluteError()
            elif selectedMetric == metrics[6]:
                self.meanAbsolutePercentageError()
            elif selectedMetric == metrics[7]:
                self.meanGammaDeviance()
            elif selectedMetric == metrics[8]:
                self.meanPinballLoss()
            elif selectedMetric == metrics[9]:
                self.meanPoissonDeviance()
            elif selectedMetric == metrics[10]:
                self.meanSquaredError()
            elif selectedMetric == metrics[11]:
                self.meanSquaredLogError()
            elif selectedMetric == metrics[12]:
                self.meanTweedieDeviance()
            elif selectedMetric == metrics[13]:
                self.medianAbsoluteError()
            elif selectedMetric == metrics[14]:
                self.r2Score()

    def d2AbsoluteError(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### D2 Absolute Error")
        sample_weight = st.text_input("Enter sample weights (comma-separated, optional)", value="")
        multioutput = st.selectbox("Select multioutput handling", ["uniform_average", "raw_values"])

        # Parse sample weights if provided
        if sample_weight.strip():
            try:
                sample_weight = [float(x.strip()) for x in sample_weight.split(",")]
            except ValueError:
                st.error("Invalid sample weights format. Please provide a comma-separated list of numbers.")
                return
        else:
            sample_weight = None

        # Perform prediction and calculate the metric
        y_pred = self.model.predict(self.xTest)
        score = d2_absolute_error_score(
            y_true=self.yTest,
            y_pred=y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput
        )

        st.write(f"**D2 Absolute Error Score:** {score}")

    def d2PinballLoss(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### D2 Pinball Loss")
        sample_weight = st.text_input("Enter sample weights (comma-separated, optional)", value="")
        alpha = st.slider("Select alpha (quantile level)", 0.0, 1.0, 0.5, 0.05)
        multioutput = st.selectbox("Select multioutput handling", ["uniform_average", "raw_values"])

        # Parse sample weights if provided
        if sample_weight.strip():
            try:
                sample_weight = [float(x.strip()) for x in sample_weight.split(",")]
            except ValueError:
                st.error("Invalid sample weights format. Please provide a comma-separated list of numbers.")
                return
        else:
            sample_weight = None

        # Perform prediction and calculate the metric
        y_pred = self.model.predict(self.xTest)
        score = d2_pinball_score(
            y_true=self.yTest,
            y_pred=y_pred,
            sample_weight=sample_weight,
            alpha=alpha,
            multioutput=multioutput
        )

        st.write(f"**D2 Pinball Loss Score:** {score}")

    def d2TweedieScore(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### D2 Tweedie Score")
        sample_weight = st.text_input("Enter sample weights (comma-separated, optional)", value="")
        power = st.slider("Select Tweedie Power", -1.0, 3.0, 0.0, 0.1)

        # Parse sample weights if provided
        if sample_weight.strip():
            try:
                sample_weight = [float(x.strip()) for x in sample_weight.split(",")]
            except ValueError:
                st.error("Invalid sample weights format. Please provide a comma-separated list of numbers.")
                return
        else:
            sample_weight = None

        # Perform prediction and calculate the metric
        y_pred = self.model.predict(self.xTest)
        score = d2_tweedie_score(
            y_true=self.yTest,
            y_pred=y_pred,
            sample_weight=sample_weight,
            power=power
        )

        st.write(f"**D2 Tweedie Score:** {score}")

    def explainedVariance(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Explained Variance Score")
        sample_weight = st.text_input("Enter sample weights (comma-separated, optional)", value="")
        multioutput = st.selectbox("Select multioutput handling", ["uniform_average", "raw_values", "variance_weighted"])
        force_finite = st.checkbox("Force Finite (Replace NaN or -Inf with real numbers)", value=True)

        # Parse sample weights if provided
        if sample_weight.strip():
            try:
                sample_weight = [float(x.strip()) for x in sample_weight.split(",")]
            except ValueError:
                st.error("Invalid sample weights format. Please provide a comma-separated list of numbers.")
                return
        else:
            sample_weight = None

        # Perform prediction and calculate the metric
        y_pred = self.model.predict(self.xTest)
        score = explained_variance_score(
            y_true=self.yTest,
            y_pred=y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
            force_finite=force_finite
        )

        st.write(f"**Explained Variance Score:** {score}")

    def maxError(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Max Error Score")
        y_pred = self.model.predict(self.xTest)
        score = max_error(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Max Error Score:** {score}")

    def meanAbsoluteError(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Mean Absolute Error")
        y_pred = self.model.predict(self.xTest)
        score = mean_absolute_error(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Mean Absolute Error (MAE):** {score}")

    def meanAbsolutePercentageError(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Mean Absolute Percentage Error")
        y_pred = self.model.predict(self.xTest)
        score = mean_absolute_percentage_error(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Mean Absolute Percentage Error (MAPE):** {score}")

    def meanGammaDeviance(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Mean Gamma Deviance")
        y_pred = self.model.predict(self.xTest)
        score = mean_gamma_deviance(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Mean Gamma Deviance:** {score}")

    def meanPinballLoss(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Mean Pinball Loss")
        y_pred = self.model.predict(self.xTest)
        score = mean_pinball_loss(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Mean Pinball Loss:** {score}")

    def meanPoissonDeviance(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Mean Poisson Deviance")
        y_pred = self.model.predict(self.xTest)
        score = mean_poisson_deviance(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Mean Poisson Deviance:** {score}")

    def meanSquaredError(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Mean Squared Error")
        y_pred = self.model.predict(self.xTest)
        score = mean_squared_error(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Mean Squared Error (MSE):** {score}")

    def meanSquaredLogError(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Mean Squared Logarithmic Error")
        y_pred = self.model.predict(self.xTest)
        score = mean_squared_log_error(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Mean Squared Logarithmic Error (MSLE):** {score}")

    def meanTweedieDeviance(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Mean Tweedie Deviance")
        y_pred = self.model.predict(self.xTest)
        score = mean_tweedie_deviance(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Mean Tweedie Deviance:** {score}")

    def medianAbsoluteError(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### Median Absolute Error")
        y_pred = self.model.predict(self.xTest)
        score = median_absolute_error(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**Median Absolute Error:** {score}")

    def r2Score(self):
        if self.yTest is None or self.model is None:
            st.error("Model not trained or data split not performed. Please ensure both steps are completed.")
            return

        st.markdown("### R2 Score")
        y_pred = self.model.predict(self.xTest)
        score = r2_score(y_true=self.yTest, y_pred=y_pred)
        st.write(f"**R2 Score:** {score}")
