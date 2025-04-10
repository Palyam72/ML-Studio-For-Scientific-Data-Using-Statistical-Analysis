import streamlit as st
import pickel
class DownloadModel:
  def __init__(self):
    self.classification=["Bagging Classifier","Extra Tree Classifier","Decision Trees","Ada Boost Classifier","Hist Gradient Boosting Classifier","Random Forest Classifier",
                        "Stacking Classifier","Voting Classifier","LinearSVC","NuSVC",
                       "OneClassSVM","KNN","RadiusNeighbors","BernoulliNB","CategoricalNB","ComplementNB","GaussianNB","MultinomialNB"]
    self.regression=None
    self.clustering=None
  def download_classification(self):
    selectedModel=st.selectbox("Select models to download",self.classification)
    if st.download_button("Confirm To Download",use_container_width=True,type='primary'):
      if st.session_state[selectedModel] != None:
        with open(st.session_state[selectedModel],'wb'
      pass
  def main_layout(self):
    col1,col2=st.columns([1,2],border=True)
    radio_options=col1.radio("Select the techniques to download",["Classification","Regression","Clustering"])
    if radio_options =="Classification":
      self.download_classification()
