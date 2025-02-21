import streamlit as st
import pandas as pd
import missingno as mso
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import *
from sklearn.svm import SVC, NuSVC, OneClassSVM, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix

if "availableDatasets" not in st.session_state:
    st.session_state["availableDatasets"]={}
class Classification:
    def __init__(self, dataset):
        self.dataset = dataset
        self.xtrain, self.xtest, self.ytrain, self.ytest = None, None, None, None

    def display(self):
        # Create three tabs
        tab1, tab2, tab3 = st.tabs(["Perform Operations", "View Operations", "Delete Operations"])

        with tab1:
            col1, col2 = st.columns([1, 2], border=True)
            operation = col1.radio("Select Operation", ["Train Test Split", "Classifiers"])
            if operation == "Train Test Split":
                col2.subheader("You Are Going To Implement Train Test Split", divider=True)
                test_columns = col2.selectbox("Select the target column", self.dataset.columns.tolist())
                if col2.button("Apply Train Test Split", use_container_width=True, type='primary'):
                    test_size = col2.slider("Select Test Size", 0.1, 0.5, 0.2, 0.1)
                    X = self.dataset.drop(columns=[test_columns])
                    y = self.dataset[test_columns]
                    self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(X, y, test_size=test_size, random_state=42)
                    st.session_state["availableDatasets"]["classification_train_test_split_xtrain"] = self.xtrain
                    st.session_state["availableDatasets"]["classification_train_test_split_xtest"] = self.xtest
                    st.session_state["availableDatasets"]["classification_train_test_split_ytrain"] = self.ytrain
                    st.session_state["availableDatasets"]["classification_train_test_split_ytest"] = self.ytest
                    col2.write(f"Train set size: {self.xtrain.shape[0]}")
                    col2.write(f"Test set size: {self.xtest.shape[0]}")
            elif operation == "Classifiers":
                col2.subheader("You Make Model Here", divider='blue')
                option = col2.selectbox("Select the classification model that you want", 
                                        ["Ada Boost Classifier", "Bagging Classifier", "Extra Tree Classifier", 
                                         "Gradient Boosting Classifier", "Hist Gradient Boosting Classifier", 
                                         "Random Forest Classifier", "Stacking Classifier", "Voting Classifier", 
                                         "Decision Tree Classifier", "Linear SVM", "NuSVC", "One Class SVM", "SVC", 
                                         "KNeighbours Classifier", "Radius Neighbours Classifier", "BernoulliNB", 
                                         "CategoricalNB", "ComplementNB", "GaussianNB", "MultinomialNB"])
                if option == "Decision Tree Classifier":
                    self.decision_tree(col2)

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

        with tab3:
            st.write("Delete Operations content goes here")

    def decision_tree(self, col2):
        col2.subheader("Decision Tree Classifier Settings",divider='blue')
        xtrain_key = col2.selectbox("Select X Train", list(st.session_state["availableDatasets"].keys()))
        ytrain_key = col2.selectbox("Select Y Train", list(st.session_state["availableDatasets"].keys()))

        if "classification_train_test_split_xtrain" not in st.session_state["availableDatasets"] or "classification_train_test_split_ytrain" not in st.session_state["availableDatasets"]:
            col2.error("Error: Train Test Split must be performed first!")
            return

        xtrain = st.session_state["availableDatasets"][xtrain_key]
        ytrain = st.session_state["availableDatasets"][ytrain_key]
    
        criterion = col2.selectbox("Select Criterion", ["gini", "entropy", "log_loss"], index=0)
        splitter = col2.selectbox("Select Splitter Strategy", ["best", "random"], index=0)
        max_depth = None if col2.checkbox("Use Default Max Depth") else col2.number_input("Max Depth (None for unlimited)", min_value=1, value=10)
        min_samples_split = col2.number_input("Minimum Samples to Split", min_value=2, value=2)
        min_samples_leaf = col2.number_input("Minimum Samples at a Leaf Node", min_value=1, value=1)
        max_features = col2.selectbox("Max Features", [None, "sqrt", "log2"] + list(range(1, len(xtrain.columns) + 1)), index=0)
        max_leaf_nodes = None if col2.checkbox("Use Default Max Leaf Nodes") else col2.number_input("Max Leaf Nodes (None for unlimited)", min_value=1, value=10)
        min_impurity_decrease = col2.slider("Minimum Impurity Decrease", 0.0, 1.0, 0.0, 0.01)
        random_state = col2.number_input("Random State (None for random)", min_value=0, value=42)
    
        if col2.checkbox("Train Decision Tree Model"):
            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, 
                                           min_samples_leaf=min_samples_leaf, max_features = None if max_features == "None" else max_features, 
                                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, random_state=random_state)
            col2.subheader("Your Created Model", divider='blue')
            col2.write(model)
            model.fit(xtrain, ytrain)
            col2.subheader("Here are the detailed list of parameters",divider='blue')
            col2.write(f"Trained Model Parameters:\n{model.get_params()}")
            col2.subheader("Decision Tree Model Attributes", divider='blue')
            col2.write(f"Classes: {model.classes_}")
            col2.write(f"Feature Importances: {model.feature_importances_}")
            col2.write(f"Max Features: {model.max_features_}")
            col2.write(f"Number of Classes: {model.n_classes_}")
            col2.write(f"Number of Features: {model.n_features_in_}")
            col2.write(f"Feature Names: {getattr(model, 'feature_names_in_', 'Not Available')}")
            col2.write(f"Number of Outputs: {model.n_outputs_}")
            col2.write(f"Tree Structure: {model.tree_}")
            col2.subheader("Decision Tree Visualization", divider='blue')
            feature_names = list(map(str, xtrain.columns))  # Ensure feature names are strings
            
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(model, filled=True, feature_names=feature_names, class_names=list(map(str, model.classes_)), ax=ax)
            col2.pyplot(fig)
            col2.subheader("Your Model Metrics On Test Data",divider='blue')
            self.metrics(col2)
    def metrics(self, col2):
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        col2.subheader("Select Data for Metrics Calculation", divider="blue")
    
        try:
            # Select datasets
            xtrain_key = col2.selectbox("Select X Train", list(st.session_state["availableDatasets"].keys()))
            xtest_key = col2.selectbox("Select X Test", list(st.session_state["availableDatasets"].keys()))
            ytrain_key = col2.selectbox("Select Y Train", list(st.session_state["availableDatasets"].keys()))
            ytest_key = col2.selectbox("Select Y Test", list(st.session_state["availableDatasets"].keys()))
    
            xtrain = st.session_state["availableDatasets"][xtrain_key]
            xtest = st.session_state["availableDatasets"][xtest_key]
            ytrain = st.session_state["availableDatasets"][ytrain_key]
            ytest = st.session_state["availableDatasets"][ytest_key]
    
            # Ensure model is available
            if not hasattr(self, "model"):
                col2.error("Error: No trained model found. Train a model first!")
                return
    
            model = self.model
    
            # Make predictions
            y_pred = model.predict(xtest)
            y_train_pred = model.predict(xtrain)
            
            # Get prediction probabilities (if model supports it)
            try:
                y_proba = model.predict_proba(xtest)[:, 1]  # Only for binary classification
            except AttributeError:
                y_proba = None
    
            col2.subheader("📊 Model Evaluation Metrics", divider="blue")
    
            # Dictionary to store computed metrics
            successful_metrics = {}
    
            # Compute all metrics with exception handling
            def compute_metric(metric_name, func, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    successful_metrics[metric_name] = result
                except Exception:
                    pass  # Ignore any failing metric
    
            # Standard Metrics
            compute_metric("Accuracy Score", metrics.accuracy_score, ytest, y_pred)
            compute_metric("Balanced Accuracy Score", metrics.balanced_accuracy_score, ytest, y_pred)
            compute_metric("F1 Score", metrics.f1_score, ytest, y_pred, average="weighted")
            compute_metric("F-Beta Score (β=0.5)", metrics.fbeta_score, ytest, y_pred, beta=0.5, average="weighted")
            compute_metric("Precision Score", metrics.precision_score, ytest, y_pred, average="weighted", zero_division=0)
            compute_metric("Recall Score", metrics.recall_score, ytest, y_pred, average="weighted")
            compute_metric("Jaccard Score", metrics.jaccard_score, ytest, y_pred, average="weighted")
            compute_metric("Matthews Correlation Coefficient", metrics.matthews_corrcoef, ytest, y_pred)
            compute_metric("Cohen's Kappa Score", metrics.cohen_kappa_score, ytest, y_pred)
            compute_metric("Hamming Loss", metrics.hamming_loss, ytest, y_pred)
            compute_metric("Zero-One Loss", metrics.zero_one_loss, ytest, y_pred)
    
            # Log Loss & Brier Score (only if y_proba is available)
            if y_proba is not None:
                compute_metric("Log Loss", metrics.log_loss, ytest, y_proba)
                compute_metric("Brier Score Loss", metrics.brier_score_loss, ytest, y_proba)
            
            # Confusion Matrix
            conf_matrix = None
            try:
                conf_matrix = metrics.confusion_matrix(ytest, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
                col2.pyplot(fig)
            except Exception:
                pass  # Ignore if confusion matrix fails
    
            # Multilabel Confusion Matrix
            compute_metric("Multilabel Confusion Matrix", metrics.multilabel_confusion_matrix, ytest, y_pred)
    
            # Classification Report
            try:
                class_report = metrics.classification_report(ytest, y_pred, output_dict=True)
                col2.dataframe(pd.DataFrame(class_report).transpose())
            except Exception:
                pass  # Ignore if classification report fails
    
            # Ranking Metrics
            compute_metric("Discounted Cumulative Gain (DCG)", metrics.dcg_score, [ytest], [y_pred])
            compute_metric("Normalized DCG", metrics.ndcg_score, [ytest], [y_pred])
    
            # ROC & AUC Metrics (only for binary classification)
            if len(set(ytest)) == 2 and y_proba is not None:
                compute_metric("ROC AUC Score", metrics.roc_auc_score, ytest, y_proba)
    
                try:
                    precision, recall, _ = metrics.precision_recall_curve(ytest, y_proba)
                    fig, ax = plt.subplots()
                    ax.plot(recall, precision, marker=".")
                    ax.set_title("Precision-Recall Curve")
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    col2.pyplot(fig)
                except Exception:
                    pass  # Ignore if precision-recall curve fails
    
                try:
                    fpr, tpr, _ = metrics.roc_curve(ytest, y_proba)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, marker=".")
                    ax.set_title("ROC Curve")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    col2.pyplot(fig)
                except Exception:
                    pass  # Ignore if ROC curve fails
    
            # Top-K Accuracy Score
            try:
                y_proba_full = model.predict_proba(xtest) if hasattr(model, "predict_proba") else np.zeros((len(ytest), 1))
                compute_metric("Top-K Accuracy Score (k=3)", metrics.top_k_accuracy_score, ytest, y_proba_full, k=3)
            except Exception:
                pass  # Ignore if Top-K Accuracy Score fails
    
            # Display computed metrics
            col2.subheader("📊 Computed Metrics Summary", divider="blue")
            if successful_metrics:
                metrics_df = pd.DataFrame(successful_metrics.items(), columns=["Metric", "Value"])
                col2.dataframe(metrics_df)
            else:
                col2.warning("⚠️ No metrics were successfully computed.")
    
        except KeyError as e:
            col2.error(f"Key Error: {str(e)} - Please check dataset selections.")
        except Exception as e:
            col2.error(f"Unexpected error: {str(e)}")
    
