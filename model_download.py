import streamlit as st
import pickle
import os

class DownloadModel:
    def __init__(self,dataset):
        self.classification = [
            "Bagging Classifier", "Extra Tree Classifier", "Decision Trees", "Ada Boost Classifier",
            "Hist Gradient Boosting Classifier", "Random Forest Classifier", "Stacking Classifier",
            "Voting Classifier", "LinearSVC", "NuSVC", "OneClassSVM", "KNN", "RadiusNeighbors",
            "BernoulliNB", "CategoricalNB", "ComplementNB", "GaussianNB", "MultinomialNB"
        ]
        self.regression = None
        self.clustering = None

    def download_classification(self):
        selected_model = st.selectbox("Select model to download", self.classification)

        if selected_model in st.session_state and st.session_state[selected_model] is not None:
            model_path = st.session_state[selected_model]

            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    bytes_data = f.read()

                st.download_button(
                    label="Confirm To Download",
                    data=bytes_data,
                    file_name=os.path.basename(model_path),
                    mime="application/octet-stream",
                    use_container_width=True,
                    type='primary'
                )
            else:
                st.warning("Model file does not exist at the specified path.")
        else:
            st.info("Selected model is not yet available in session state.")

    def display(self):
        col1, col2 = st.columns([1, 2], border=True)
        with col1:
            radio_option = st.radio("Select the techniques to download", ["Classification", "Regression", "Clustering"])

        with col2:
            if radio_option == "Classification":
                self.download_classification()
            elif radio_option == "Regression":
                st.info("Regression download support coming soon.")
            elif radio_option == "Clustering":
                st.info("Clustering download support coming soon.")
