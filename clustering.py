import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import *
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
                clustering_dict = {
                    "KMeans": self.Kmeans,
                    "DBSCAN": self.dbscan,
                    "AffinityPropagation": self.affinity_propagation,
                    "AgglomerativeClustering": self.agglomerative_clustering,
                    "Birch": self.birch_clustering,
                    "BisectingKMeans": self.bisecting_kmeans,
                    "FeatureAgglomeration": self.feature_agglomeration,
                    "HDBSCAN": self.hdbscan_clustering,
                    "MeanShift": self.mean_shift_clustering
                }

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

    def affinity_propagation(self):
        damping = float(st.number_input("Damping factor (0.5 to 1.0)", min_value=0.5, max_value=0.99, value=0.5, step=0.01))
        max_iter = int(st.number_input("Maximum number of iterations", min_value=1, value=200))
        convergence_iter = int(st.number_input("Number of iterations with no change before convergence", min_value=1, value=15))
        copy = st.checkbox("Make a copy of input data", value=True)
        preference = st.text_input("Preferences for each point (float or 'None')", "None")
        affinity = st.selectbox("Affinity metric", ["euclidean", "precomputed"])
        verbose = st.checkbox("Enable verbose mode", value=False)
        random_state = st.text_input("Random state (integer or 'None')", "None")
    
        if preference == "None":
            preference = None
        else:
            preference = float(preference)
    
        if random_state == "None":
            random_state = None
        else:
            random_state = int(random_state)
    
        if st.button("Apply", use_container_width=True, type='primary'):
            return AffinityPropagation(
                damping=damping,
                max_iter=max_iter,
                convergence_iter=convergence_iter,
                copy=copy,
                preference=preference,
                affinity=affinity,
                verbose=verbose,
                random_state=random_state
            )
    def agglomerative_clustering(self):
        n_clusters = st.text_input("Number of clusters (integer or 'None')", "2")
        metric = st.selectbox("Metric for computing linkage", ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"])
        linkage = st.selectbox("Linkage criterion", ["ward", "complete", "average", "single"])
        compute_full_tree = st.selectbox("Compute full tree", ["auto", True, False])
        distance_threshold = st.text_input("Distance threshold (float or 'None')", "None")
        compute_distances = st.checkbox("Compute distances for dendrogram visualization", value=False)
    
        if n_clusters == "None":
            n_clusters = None
        else:
            n_clusters = int(n_clusters)
    
        if distance_threshold == "None":
            distance_threshold = None
        else:
            distance_threshold = float(distance_threshold)
    
        if st.button("Apply", use_container_width=True, type='primary'):
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                metric=metric,
                linkage=linkage,
                compute_full_tree=compute_full_tree,
                distance_threshold=distance_threshold,
                compute_distances=compute_distances
            )
    def birch_clustering(self):
        threshold = float(st.number_input("Threshold (radius for merging subclusters)", min_value=0.01, value=0.5, step=0.01))
        branching_factor = int(st.number_input("Branching Factor (max CF subclusters per node)", min_value=2, value=50))
        n_clusters = st.text_input("Number of clusters (integer, 'None', or sklearn.cluster model)", "3")
        compute_labels = st.checkbox("Compute labels for each fit", value=True)
    
        if n_clusters.lower() == "none":
            n_clusters = None
        else:
            try:
                n_clusters = int(n_clusters)
            except ValueError:
                st.warning("Invalid input for n_clusters. Use an integer, 'None', or a clustering model instance.")
                return None
    
        if st.button("Apply", use_container_width=True, type='primary'):
            return Birch(
                threshold=threshold,
                branching_factor=branching_factor,
                n_clusters=n_clusters,
                compute_labels=compute_labels
            )
    def bisecting_kmeans(self):
        n_clusters = int(st.number_input("Number of clusters to form", min_value=1, value=8))
        init = st.selectbox("Method for Initialization", ["k-means++", "random"])
        n_init = int(st.number_input("Number of initializations", min_value=1, value=1))
        random_state = st.text_input("Random state (integer or 'None')", "None")
        max_iter = int(st.number_input("Maximum number of iterations", min_value=1, value=300))
        tol = float(st.number_input("Tolerance for convergence", min_value=1e-6, value=1e-4, format="%.6f"))
        copy_x = st.checkbox("Copy X (keep original data unchanged)", value=True)
        algorithm = st.selectbox("K-means Algorithm", ["lloyd", "elkan"])
        bisecting_strategy = st.selectbox("Bisecting Strategy", ["biggest_inertia", "largest_cluster"])
    
        if random_state.lower() == "none":
            random_state = None
        else:
            try:
                random_state = int(random_state)
            except ValueError:
                st.warning("Invalid input for random_state. Use an integer or 'None'.")
                return None
    
        if st.button("Apply", use_container_width=True, type='primary'):
            return BisectingKMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                random_state=random_state,
                max_iter=max_iter,
                tol=tol,
                copy_x=copy_x,
                algorithm=algorithm,
                bisecting_strategy=bisecting_strategy
            )
    def feature_agglomeration(self):
        n_clusters = st.number_input("Number of clusters", min_value=1, value=2)
        metric = st.selectbox("Metric", ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"])
        compute_full_tree = st.selectbox("Compute Full Tree", ["auto", True, False])
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        pooling_func = st.selectbox("Pooling Function", ["mean", "median", "max", "min"])
        distance_threshold = st.text_input("Distance Threshold (float or 'None')", "None")
        compute_distances = st.checkbox("Compute Distances", value=False)
    
        if distance_threshold.lower() == "none":
            distance_threshold = None
        else:
            try:
                distance_threshold = float(distance_threshold)
            except ValueError:
                st.warning("Invalid input for distance_threshold. Use a float or 'None'.")
                return None
    
        if st.button("Apply", use_container_width=True, type='primary'):
            pooling_func_dict = {
                "mean": np.mean,
                "median": np.median,
                "max": np.max,
                "min": np.min
            }
    
            return FeatureAgglomeration(
                n_clusters=n_clusters,
                metric=metric,
                compute_full_tree=compute_full_tree,
                linkage=linkage,
                pooling_func=pooling_func_dict[pooling_func],
                distance_threshold=distance_threshold,
                compute_distances=compute_distances
            )
    def mean_shift_clustering(self):
        bandwidth = st.text_input("Bandwidth (float or 'None')", "None")
        bin_seeding = st.checkbox("Bin Seeding", value=False)
        min_bin_freq = st.number_input("Min Bin Frequency", min_value=1, value=1)
        cluster_all = st.checkbox("Cluster All Points", value=True)
        n_jobs = st.text_input("Number of Jobs (int or 'None')", "None")
        max_iter = st.number_input("Max Iterations", min_value=1, value=300)
    
        # Handling None values
        bandwidth = None if bandwidth.lower() == "none" else float(bandwidth)
        n_jobs = None if n_jobs.lower() == "none" else int(n_jobs)
    
        if st.button("Apply", use_container_width=True, type='primary'):
            return MeanShift(
                bandwidth=bandwidth,
                bin_seeding=bin_seeding,
                min_bin_freq=min_bin_freq,
                cluster_all=cluster_all,
                n_jobs=n_jobs,
                max_iter=max_iter
            )
    def hdbscan_clustering(self):
        min_cluster_size = st.number_input("Min Cluster Size", min_value=1, value=5)
        min_samples = st.text_input("Min Samples (int or 'None')", "None")
        cluster_selection_epsilon = st.number_input("Cluster Selection Epsilon", min_value=0.0, value=0.0)
        max_cluster_size = st.text_input("Max Cluster Size (int or 'None')", "None")
        metric = st.selectbox("Metric", ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"])
        alpha = st.number_input("Alpha", min_value=0.0, value=1.0)
        algorithm = st.selectbox("Algorithm", ["auto", "brute", "kd_tree", "ball_tree"])
        leaf_size = st.number_input("Leaf Size", min_value=1, value=40)
        n_jobs = st.text_input("Number of Jobs (int or 'None')", "None")
        cluster_selection_method = st.selectbox("Cluster Selection Method", ["eom", "leaf"])
        allow_single_cluster = st.checkbox("Allow Single Cluster", value=False)
        store_centers = st.selectbox("Store Centers", ["None", "centroid", "medoid", "both"])
        copy = st.checkbox("Copy Data", value=False)
    
        # Handling None values
        min_samples = None if min_samples.lower() == "none" else int(min_samples)
        max_cluster_size = None if max_cluster_size.lower() == "none" else int(max_cluster_size)
        n_jobs = None if n_jobs.lower() == "none" else int(n_jobs)
        store_centers = None if store_centers == "None" else store_centers
    
        if st.button("Apply", use_container_width=True, type='primary'):
            return HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                max_cluster_size=max_cluster_size,
                metric=metric,
                alpha=alpha,
                algorithm=algorithm,
                leaf_size=leaf_size,
                n_jobs=n_jobs,
                cluster_selection_method=cluster_selection_method,
                allow_single_cluster=allow_single_cluster,
                store_centers=store_centers,
                copy=copy
            )
    def evaluate(self, model):
        model = model.fit_predict(self.dataset)
        st.success("Model Fitted & Transformed")
