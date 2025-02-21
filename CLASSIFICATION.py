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
from sklearn.tree import DecisionTreeClassifier
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
            col1, col2 = st.columns([1, 2], gap="large")
            operation = col1.radio("Select Operation", ["Train Test Split", "Classifiers"])
            if operation == "Train Test Split":
                col2.write("You selected Train Test Split")
                test_size = col2.slider("Select Test Size", 0.1, 0.5, 0.2, 0.1)
                X = self.dataset.drop(columns=['target'])
                y = self.dataset['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                col2.write(f"Train set size: {X_train.shape[0]}")
                col2.write(f"Test set size: {X_test.shape[0]}")
            elif operation == "Classifiers":
                col2.subheader("You Make Model Here", divider='blue')
                option = col2.selectbox("Select the classification model that you want",
                                        ["Ada Boost Classifier", "Bagging Classifier", "Extra Tree Classifier",
                                         "Gradient Boosting Classifier", "Hist Gradient Boosting Classifier",
                                         "Random Forest Classifier", "Stacking Classifier", "Voting Classifier",
                                         "Decision Tree Classifier", "Linear SVM", "NuSVC", "One Class SVM", "SVC",
                                         "KNeighbours Classifier", "Radius Neighbours Classifier", "BernoulliNB",
                                         "CategoricalNB", "ComplementNB", "GaussianNB", "MultinomialNB"])
                if option == "Ada Boost Classifier":
                    self.ada_boost_classifier(col2)
                elif option == "Bagging Classifier":
                    self.bagging_classifier(col2)
                elif option == "Extra Tree Classifier":
                    self.extra_tree_classifier(col2)
                elif option == "Gradient Boosting Classifier":
                    self.gradient_boosting_classifier(col2)
                elif option == "Hist Gradient Boosting Classifier":
                    self.hist_gradient_boosting_classifier(col2)
                elif option == "Random Forest Classifier":
                    self.random_forest_classifier(col2)
                elif option == "Stacking Classifier":
                    self.stacking_classifier(col2)
                elif option == "Voting Classifier":
                    self.voting_classifier(col2)
                elif option == "Decision Tree Classifier":
                    self.decision_tree_classifier(col2)
                elif option == "Linear SVM":
                    self.linear_svm(col2)
                elif option == "NuSVC":
                    self.nu_svc(col2)
                elif option == "One Class SVM":
                    self.one_class_svm(col2)
                elif option == "SVC":
                    self.svc(col2)
                elif option == "KNeighbours Classifier":
                    self.k_neighbors_classifier(col2)
                elif option == "Radius Neighbours Classifier":
                    self.radius_neighbors_classifier(col2)
                elif option == "BernoulliNB":
                    self.bernoulli_nb(col2)
                elif option == "CategoricalNB":
                    self.categorical_nb(col2)
                elif option == "ComplementNB":
                    self.complement_nb(col2)
                elif option == "GaussianNB":
                    self.gaussian_nb(col2)
                elif option == "MultinomialNB":
                    self.multinomial_nb(col2)

        # Content for View Operations tab
        with tab2:
            col1, col2 = st.columns([1, 2], gap="large")
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
        col2.write("You selected Ada Boost Classifier")
        X = self.dataset.drop(columns=['target'])
        y = self.dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = AdaBoostClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        col2.write("Classification Report:")
        col2.write(classification_report(y_test, y_pred))
        col2.write("Confusion Matrix:")
        col2.write(confusion_matrix(y_test, y_pred))

    def bagging_classifier(self, col2):
        col2.write("You selected Bagging Classifier")
        X = self.dataset.drop(columns=['target'])
        y = self.dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = BaggingClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        col2.write("Classification Report:")
        col2.write(classification_report(y_test, y_pred))
        col2.write("Confusion Matrix:")
        col2.write(confusion_matrix(y_test, y_pred))

    def extra_tree_classifier(self, col2):
        col2.write("You selected Extra Tree Classifier")
        X = self.dataset.drop(columns=['target'])
        y = self.dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = ExtraTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        col2.write("Classification Report:")
        col2.write(classification_report(y_test, y_pred))
        col2.write("Confusion Matrix:")
        col2.write(confusion_matrix(y_test, y_pred))

    def gradient_boosting_classifier(self, col2):
        col2.write("You selected Gradient Boosting Classifier")
        X = self.dataset.drop(columns=['target'])
        y = self.dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        col2.write("Classification Report:")
        col2.write(classification_report(y_test, y_pred))
        col2.write("Confusion Matrix:")
        col2.write(confusion_matrix(y_test, y_pred))

    def hist_gradient_boosting_classifier(self, col2):
        col2.write("You selected Hist Gradient Boosting Classifier")
        X = self.dataset.drop(columns=['target'])
        y = self.dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = HistGradientBoostingClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        col2.write("Classification Report:")
        col2.write(classification_report(y_test, y_pred))
        col2.write("Confusion Matrix:")
        col2.write(confusion_matrix(y_test, y_pred))

    def random_forest_classifier(self, col2):
        col2.write("You selected Random Forest Classifier")
        X = self.dataset.drop(columns=['target'])
        y = self.dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        col2.write("Classification Report:")
        col2.write(classification_report(y_test, y_pred))
        col2.write("Confusion Matrix:")
        col2.write(confusion_matrix(y_test, y_pred))

    def stacking_classifier(self, col2):
        col2
