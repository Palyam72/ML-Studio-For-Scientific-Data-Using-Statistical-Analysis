import streamlit as st
import pandas as pd
import missingno as mso
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, NuSVC, OneClassSVM, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix

class Classification:
    def __init__(self, dataset):
        self.dataset = dataset

    def display(self):
        # Create three tabs
        tab1, tab2, tab3 = st.tabs(["Perform Operations", "View Operations", "Delete Operations"])

        # Content for Perform Operations tab
        with tab1:
            # Create two columns with a ratio of 1:2
            col1, col2 = st.columns([1, 2],border=True)
            operation = col1.radio("Select Operation", ["Train Test Split", "Classifiers"])
            if operation == "Train Test Split":
                col2.subheader("You Are Going To Iplement Train Test Split",divider=True)
                test_columns=col2.selectbox("Selct the target columnn",self.dataset.columns.tolist())
                if col2.button("Apply Train Test Split",use_container_width=True,type='primary'):
                    test_size = col2.slider("Select Test Size", 0.1, 0.5, 0.2, 0.1)
                    X = self.dataset.drop(columns=['target'])
                    y = self.dataset['target']
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    col2.write(f"Train set size: {self.X_train.shape[0]}")
                    col2.write(f"Test set size: {self.X_test.shape[0]}")
            elif operation == "Classifiers":
                col2.subheader("You Make Model Here", divider='blue')
                option = col2.selectbox("Select the classification model that you want", 
                                        ["Ada Boost Classifier", "Bagging Classifier", "Extra Tree Classifier", 
                                         "Gradient Boosting Classifier", "Hist Gradient Boosting Classifier", 
                                         "Random Forest Classifier", "Stacking Classifier", "Voting Classifier", 
                                         "Decision Tree Classifier", "Linear SVM", "NuSVC", "One Class SVM", "SVC", 
                                         "KNeighbours Classifier", "Radius Neighbours Classifier", "BernoulliNB", 
                                         "CategoricalNB", "ComplementNB", "GaussianNB", "MultinomialNB"])
                if option=="Decision Tree Classifier":
                    self.decision_tree(col2)
        # Content for View Operations tab
        with tab2:
            col1, col2 = st.columns([1, 2], border=True)
            col1.subheader("Select The View Mode", divider='blue')
            options = col1.radio("Options", ["View Data Frame", "View Missing Information"])
            if options == "View Data Frame":
                col2.dataframe(self.dataset)
            elif options == "View Missing Information":
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mso.matrix(self.dataset, ax=ax)
                    st.pyplot(fig)
                    fig1, ax1 = plt.subplots(figsize=(10, 5))
                    mso.heatmap(self.dataset, ax=ax1)
                    st.pyplot(fig1)

        # Content for Delete Operations tab
        with tab3:
            st.write("Delete Operations content goes here")

    def ada_boost_classifier(self, col2):
        pass

    def bagging_classifier(self, col2):
        pass

    def extra_tree_classifier(self, col2):
        pass

    def gradient_boosting_classifier(self, col2):
        pass

    def hist_gradient_boosting_classifier(self, col2):
        pass

    def random_forest_classifier(self, col2):
        pass

    def stacking_classifier(self, col2):
        pass

    def voting_classifier(self, col2):
        pass

    def decision_tree(self, col2):
        st.write("### Decision Tree Classifier Settings")
    
        criterion = col2.selectbox("Select Criterion", ["gini", "entropy", "log_loss"], index=0)
        splitter = col2.selectbox("Select Splitter Strategy", ["best", "random"], index=0)
        max_depth = None if col2.checkbox("Use Default Max Depth") else col2.number_input("Max Depth (None for unlimited)", min_value=1, value=10)
        min_samples_split = col2.number_input("Minimum Samples to Split", min_value=2, value=2)
        min_samples_leaf = col2.number_input("Minimum Samples at a Leaf Node", min_value=1, value=1)
        max_features = col2.selectbox("Max Features", [None, "sqrt", "log2"] + list(range(1, len(self.xtrain.columns) + 1)), index=0)
        max_leaf_nodes = None if col2.checkbox("Use Default Max Leaf Nodes") else col2.number_input("Max Leaf Nodes (None for unlimited)", min_value=1, value=10)
        min_impurity_decrease = col2.slider("Minimum Impurity Decrease", 0.0, 1.0, 0.0, 0.01)
        random_state = col2.number_input("Random State (None for random)", min_value=0, value=42)
    
        if col2.checkbox("Train Decision Tree Model"):
            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf, max_features=max_features if max_features != "None" else None, 
                                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, random_state=random_state)
            col2.subheader("Your Created Model",divider='blue')
            col2.write(model)
            model.fit(self.xtrain, self.ytrain)
            self.metrics(col2, model)

    def metrics(self,col2,model):
        pass
        
    def linear_svm(self, col2):
        pass

    def nu_svc(self, col2):
        pass

    def one_class_svm(self, col2):
        pass

    def svc(self, col2):
        pass

    def k_neighbors_classifier(self, col2):
        pass

    def radius_neighbors_classifier(self, col2):
        pass

    def bernoulli_nb(self, col2):
        pass

    def categorical_nb(self, col2):
        pass

    def complement_nb(self, col2):
        pass

    def gaussian_nb(self, col2):
        pass

    def multinomial_nb(self, col2):
        pass
