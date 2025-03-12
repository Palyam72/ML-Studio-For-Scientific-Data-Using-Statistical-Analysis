import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
import streamlit as st

class clusters:
    def __init__(self, df):
        self.dataset = df

    def display(self):
        tab1, tab2 = st.tabs(["Perform Operations", "Learn Content"])
        with tab1:
            col1, col2 = st.columns([1, 2], border=True)
            radio_options = col1.radio("Perform Operations", ["Implement Clustering", "Implement Metrics"])
            
            if radio_options == "Implement Clustering":
                clustering_dict = {"KMeans": self.Kmeans, "DBSCAN": self.dbscan}
                clustering_options = col2.selectbox("Select clustering type", clustering_dict.keys())
                
                if clustering_options:
                    with col2:
                        try:
                            model = clustering_dict[clustering_options]()
                            self.evaluate(model)
                        except Exception as e:
                            st.warning(e)

    def Kmeans(self):
        n_clusters = int(st.number_input("The number of clusters to form as well as the number of centroids to generate.", min_value=1, value=8))
        init = st.selectbox("Method For Initialization", ["k-means++", "random"])
        max_iter = int(st.number_input("Maximum number of iterations of the k-means algorithm for a single run.", min_value=1, value=300))
        random_state = st.text_input("Determines random number generation for centroid initialization. Use an int to make the randomness deterministic (Integer Value)", "None")
        
        if random_state == "None":
            random_state = None
        else:
            random_state = int(random_state)

        algorithm = st.selectbox("K-means algorithm to use.", ["lloyd", "elkan"])
        
        if st.button("Apply", use_container_width=True, type='primary'):
            return KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state, algorithm=algorithm)

    def dbscan(self):
        eps = float(st.number_input("The maximum distance between two samples for one to be considered as in the neighborhood of the other.", min_value=0.1, value=0.5, step=0.1))
        min_samples = int(st.number_input("The number of samples in a neighborhood for a point to be considered a core point.", min_value=1, value=5))
        metric = st.selectbox("Metric to use for distance calculation", ["euclidean", "manhattan", "chebyshev", "minkowski"])
        algorithm = st.selectbox("Algorithm to compute pointwise distances and nearest neighbors", ["auto", "ball_tree", "kd_tree", "brute"])
        leaf_size = int(st.number_input("Leaf size for tree-based algorithms", min_value=1, value=30))
        p = st.text_input("Power of Minkowski metric (float or None)", "None")

        if p == "None":
            p = None
        else:
            p = float(p)

        if st.button("Apply", use_container_width=True, type='primary'):
            return DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size, p=p)

    def evaluate(self, model):
        model = model.fit_predict(self.dataset)
        st.success("Model Fitted & Transformed")
