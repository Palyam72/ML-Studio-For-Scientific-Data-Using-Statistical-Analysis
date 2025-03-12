import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import *
import streamlit as st
class clusters:
  def __init__(self,df):
    self.dataset=df
  def display(self):
    tab1,tab2=st.tabs(["Perform Operations","Learn Content"])
    with tab1:
      col1,col2=st.columns([1,2],border=True)
      radio_options=col1.radio("Perform Operations",["Implement Clustering","Implement Metrics"])
      if radio_options=="Implement Clustering":
        clustering_dict={"KMeans":self.Kmeans}
        clustering_options=col2.selectbox("Select clustering type",clustering_dict.keys())
        if clustering_options:
          with col2:
            try:
              model=clustering_dict[clustering_options]()
              self.evaluate(model)
            except Exception as e:
              st.warning(e)
  def Kmeans(self):
    n_clusters=int(st.number_input("The number of clusters to form as well as the number of centroids to generate.",min_value=1,value=8))
    init=st.selectbox("Method For Initialization",["k-means++","random"])
    max_iter=int(st.number_input("Maximum number of iterations of the k-means algorithm for a single run.",min_value=1,value=300))
    random_state=st.text_input("Determines random number generation for centroid initialization. Use an int to make the randomness deterministic (Integer Value)","None")
    if random_state=="None":
      random_state=None
    else:
      random_state=int(random_state)
    algorithm=st.selectbox("K-means algorithm to use.",["lloyd","elkan"])
    if st.button("Apply",use_container_width=True,type='primary'):
      return KMeans(n_clusters=n_clusters,init=init,max_iter=max_iter,random_state=random_state,algorithm=algorithm)
  def evaluate(self,model):
    model=model.fit_transform(self.dataset)
    st.success("Model Fitted & Transformed")
    st.session_state['KMeans']=model
