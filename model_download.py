import streamlit as st
import pickle

class DownloadModel:
    def __init__(self,df):
        self.classification = [
            "Bagging Classifier", "Extra Tree Classifier", "Decision Trees", "Ada Boost Classifier",
            "Hist Gradient Boosting Classifier", "Random Forest Classifier", "Stacking Classifier",
            "Voting Classifier", "LinearSVC", "NuSVC", "OneClassSVM", "KNN", "RadiusNeighbors",
            "BernoulliNB", "CategoricalNB", "ComplementNB", "GaussianNB", "MultinomialNB"
        ]
        self.regression = ['regressionLinear','regressionRidge','regressionRidgeCV',
                   'regressionSGD','regressionElasticNet','regressionElasticNetCV','regressionLars','regressionLarsCV',
                   'regressionLasso','regressionLassoCV',"regressionLassoLars",'regressionLassoLarsCV','regressionLassoLarsIC','regressionOMP',
                  'regressionOMPCV','regressionARD','regressionBayesianRidge','regressionMTElasticNet','regressionMTLasso','regressionMTElasticNetCV',
                   'regressionHuber','regressionQuantile','regressionRANSAC','regressionPoisson','regressionTheilSen','regressionTweedie']
        self.clustering = None

    def download_classification(self):
        selected_model = st.selectbox("Select model to download", self.classification)

        model_obj = st.session_state.get(selected_model, None)
        if model_obj is not None:
            bytes_data = pickle.dumps(model_obj)

            st.download_button(
                label="Confirm To Download",
                data=bytes_data,
                file_name=f"{selected_model.replace(' ', '_')}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                type='primary'
            )
        else:
            st.info("Selected model is not available in session state.")
    def download_regression(self):
        selected_model = st.selectbox("Select model to download", self.regression)

        model_obj = st.session_state.get(selected_model, None)
        if model_obj is not None:
            bytes_data = pickle.dumps(model_obj)

            st.download_button(
                label="Confirm To Download",
                data=bytes_data,
                file_name=f"{selected_model.replace(' ', '_')}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                type='primary'
            )
        else:
            st.info("Selected model is not available in session state.")

    def display(self):
        col1, col2 = st.columns([1, 2], border=True)
        with col1:
            radio_option = st.radio("Select the techniques to download", ["Classification", "Regression", "Clustering"])

        with col2:
            if radio_option == "Classification":
                self.download_classification()
            elif radio_option == "Regression":
                self.download_regression()
            elif radio_option == "Clustering":
                st.info("Clustering download support coming soon.")
