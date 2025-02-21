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
from sklearn import metrics
import warnings
from sklearn.linear_model import LogisticRegression

session_models=["Hist Gradient Boosting Classifier","Random Forest Classifier","Stacking Classifier","Voting Classifier"]
for i in session_models:
    if i not in st.session_state:
        st.session_state[i]=None
        

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
                elif option == "Ada Boost Classifier":
                    self.ada_boost(col2)
                elif option=="Extra Tree Classifier":
                    self.extra_trees(col2)
                elif option =="Bagging Classifier":
                    self.bagging_classifier(col2)
                elif option == "Hist Gradient Boosting Classifier":
                    self.hist_gradient_boosting_classifier(col2)
                elif option == "Random Forest Classifier":
                    self.random_forest_classifier(col2)
                elif option == "Stacking Classifier":
                    self.stacking_classifier(col2)
                elif option == "Voting Classifier":
                    self.voting_classifier(col2)
                    
                    
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
            self.metrics(col2,model)
            
    def ada_boost(self, col2):
        col2.subheader("Ada Boost Classifier Settings", divider='blue')
        xtrain_key = col2.selectbox("Select X Train", list(st.session_state["availableDatasets"].keys()))
        ytrain_key = col2.selectbox("Select Y Train", list(st.session_state["availableDatasets"].keys()))
        
        if "classification_train_test_split_xtrain" not in st.session_state["availableDatasets"] or "classification_train_test_split_ytrain" not in st.session_state["availableDatasets"]:
            col2.error("Error: Train Test Split must be performed first!")
            return
        
        xtrain = st.session_state["availableDatasets"][xtrain_key]
        ytrain = st.session_state["availableDatasets"][ytrain_key]
        
        n_estimators = col2.number_input("Number of Estimators", min_value=1, value=50)
        learning_rate = col2.number_input("Learning Rate", min_value=0.01, value=1.0, step=0.01)
        random_state = col2.number_input("Random State (None for random)", min_value=0, value=42)
        
        if col2.checkbox("Train Ada Boost Model"):
            model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
            col2.subheader("Your Created Model", divider='blue')
            col2.write(model)
            model.fit(xtrain, ytrain)
            col2.subheader("Here are the detailed list of parameters", divider='blue')
            col2.write(f"Trained Model Parameters:\n{model.get_params()}")
            col2.subheader("Your Model Metrics On Test Data", divider='blue')
            self.metrics(col2, model)
    def extra_trees(self, col2):
        col2.subheader("Extra Trees Classifier Settings", divider='blue')
        xtrain_key = col2.selectbox("Select X Train", list(st.session_state["availableDatasets"].keys()))
        ytrain_key = col2.selectbox("Select Y Train", list(st.session_state["availableDatasets"].keys()))
        
        if "classification_train_test_split_xtrain" not in st.session_state["availableDatasets"] or "classification_train_test_split_ytrain" not in st.session_state["availableDatasets"]:
            col2.error("Error: Train Test Split must be performed first!")
            return
        
        xtrain = st.session_state["availableDatasets"][xtrain_key]
        ytrain = st.session_state["availableDatasets"][ytrain_key]
        
        n_estimators = col2.number_input("Number of Trees", min_value=1, value=100)
        criterion = col2.selectbox("Select Criterion", ["gini", "entropy", "log_loss"], index=0)
        max_depth = None if col2.checkbox("Use Default Max Depth") else col2.number_input("Max Depth (None for unlimited)", min_value=1, value=10)
        min_samples_split = col2.number_input("Minimum Samples to Split", min_value=2, value=2)
        min_samples_leaf = col2.number_input("Minimum Samples at a Leaf Node", min_value=1, value=1)
        max_features = col2.selectbox("Max Features", [None, "sqrt", "log2"] + list(range(1, len(xtrain.columns) + 1)), index=0)
        random_state = col2.number_input("Random State (None for random)", min_value=0, value=42)
        
        if col2.checkbox("Train Extra Trees Model"):
            model = ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf, max_features=None if max_features == "None" else max_features, 
                                         random_state=random_state)
            col2.subheader("Your Created Model", divider='blue')
            col2.write(model)
            model.fit(xtrain, ytrain)
            col2.subheader("Here are the detailed list of parameters", divider='blue')
            col2.write(f"Trained Model Parameters:\n{model.get_params()}")
            col2.subheader("Feature Importances", divider='blue')
            col2.write(model.feature_importances_)
            col2.subheader("Your Model Metrics On Test Data", divider='blue')
            self.metrics(col2, model)
    def metrics(self, col2, model):
        col2.subheader("Metrics On Predictions", divider='blue')
        
        xtrain = col2.selectbox("Select the xtrain dataset from the available datasets", list(st.session_state["availableDatasets"].keys()))
        ytrain = col2.selectbox("Select the ytrain dataset from the available datasets", list(st.session_state["availableDatasets"].keys()))
        xtest = col2.selectbox("Select the xtest dataset from the available datasets", list(st.session_state["availableDatasets"].keys()))
        ytest = col2.selectbox("Select the ytest dataset from the available datasets", list(st.session_state["availableDatasets"].keys()))
        
        if col2.button("Evaluate Metrics", use_container_width=True):
            try:
                ypred = model.predict(st.session_state["availableDatasets"][xtest])
                col2.success("Predictions Successful, below is the evaluation")
            except Exception as e:
                col2.error(f"Error in prediction: {str(e)}")
                return
            
            ytest_data = st.session_state["availableDatasets"][ytest]
            
            metrics_list = {
                "Accuracy Score": metrics.accuracy_score,
                "AUC Score": metrics.roc_auc_score,
                "Average Precision Score": metrics.average_precision_score,
                "Balanced Accuracy Score": metrics.balanced_accuracy_score,
                "Classification Report": metrics.classification_report,
                "Confusion Matrix": metrics.confusion_matrix,
                "F1 Score": metrics.f1_score,
                "FBeta Score": lambda y_true, y_pred: metrics.fbeta_score(y_true, y_pred, beta=1),
                "Hamming Loss": metrics.hamming_loss,
                "Jaccard Score": metrics.jaccard_score,
                "Log Loss": metrics.log_loss,
                "Matthews Correlation Coefficient": metrics.matthews_corrcoef,
                "Multilabel Confusion Matrix": metrics.multilabel_confusion_matrix,
                "Precision Recall F-score Support": metrics.precision_recall_fscore_support,
                "Precision Score": metrics.precision_score,
                "Recall Score": metrics.recall_score,
                "Zero-One Loss": metrics.zero_one_loss,
            }
            
            for metric_name, metric_func in metrics_list.items():
                try:
                    result = metric_func(ytest_data, ypred)
                    col2.write(f"**{metric_name}:**")
                    col2.code(result)
                except Exception as e:
                    col2.warning(f"{metric_name} could not be calculated: {str(e)}")
    def bagging_classifier(self,col2):
        xtrain_key = col2.selectbox("Select X Train for bagging classifier", list(st.session_state["availableDatasets"].keys()))
        ytrain_key = col2.selectbox("Select Y Train for bagging classifer", list(st.session_state["availableDatasets"].keys()))
        nestimators=int(col2.number_input("Please enter the number of estimators",10))
        max_samples=col2.number_input("Please enetr the max samples",1.0)
        max_features=col2.number_input("Please enter the max features",1.0)
        bootstrap=col2.checkbox("Whether samples are drawn with replacement. If False, sampling without replacement is performed.",True)
        bootstrap_features=col2.checkbox("Whether features are drawn with replacement.",False)
        oob_score=col2.checkbox("Whether to use out-of-bag samples to estimate the generalization error. Only available if bootstrap=True.",False)
        warm_start=col2.checkbox("When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble",False)
        n_jobs=int(col2.number_input("The number of jobs to run in parallel for both fit and predict. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. "))
        if not n_jobs:
            n_jobs=None
        random_state=int(col2.number_input("Controls the random resampling of the original dataset (sample wise and feature wise). If the base estimator accepts a random_state attribute, a different seed is generated for each instance in the ensemble. Pass an int for reproducible output across multiple function calls"))
        if not random_state:
            random_state=None
        verbose=int(col2.number_input("Controls the verbosity when fitting and predicting.",0))
        col2.divider()
        if col2.checkbox("Continue To Fit The Model"):
            model=BaggingClassifier(n_estimators=nestimators, max_samples=max_samples, max_features=max_features,
                                    bootstrap=bootstrap, bootstrap_features=bootstrap_features, oob_score=oob_score, 
                                    warm_start=warm_start, n_jobs=n_jobs, random_state=random_state, verbose=verbose)
            col2.subheader("Your Model",divider='blue')
            col2.write(model.get_params())
            model.fit(st.session_state["availableDatasets"][xtrain_key],st.session_state["availableDatasets"][ytrain_key])
            col2.success("Model Fitted Successfully")
            col2.divider()
            self.metrics(col2,model)
    def hist_gradient_boosting_classifier(self, col2):
        xtrain_key = col2.selectbox("Select X Train for HistGradientBoostingClassifier", list(st.session_state["availableDatasets"].keys()))
        ytrain_key = col2.selectbox("Select Y Train for HistGradientBoostingClassifier", list(st.session_state["availableDatasets"].keys()))
        
        loss = col2.selectbox("Loss function", ["log_loss"], index=0)
        learning_rate = col2.number_input("Learning rate", value=0.1, min_value=0.001, step=0.01)
        max_iter = col2.number_input("Max iterations", value=100, min_value=1, step=1)
        max_leaf_nodes = col2.number_input("Max leaf nodes", value=31, min_value=2, step=1)
        max_depth = col2.number_input("Max depth", value=None, min_value=1, step=1, format="%d")
        min_samples_leaf = col2.number_input("Min samples per leaf", value=20, min_value=1, step=1)
        l2_regularization = col2.number_input("L2 regularization", value=0.0, step=0.01)
        max_features = col2.number_input("Max features", value=1.0, min_value=0.1, max_value=1.0, step=0.1)
        max_bins = col2.number_input("Max bins", value=255, min_value=2, step=1)
        
        warm_start = col2.checkbox("Warm start", False)
        early_stopping = col2.selectbox("Early stopping", ["auto", True, False], index=0)
        scoring = col2.selectbox("Scoring", ["loss", None], index=0)
        validation_fraction = col2.number_input("Validation fraction", value=0.1, min_value=0.0, max_value=1.0, step=0.01)
        n_iter_no_change = col2.number_input("Number of iterations with no change for early stopping", value=10, min_value=1, step=1)
        tol = col2.number_input("Tolerance", value=1e-7, min_value=0.0, step=1e-7, format="%e")
        verbose = col2.number_input("Verbosity level", value=0, min_value=0, step=1)
        random_state = col2.number_input("Random state (leave blank for None)", value=None, format="%d")
        if not random_state:
            random_state = None
        
        class_weight = col2.selectbox("Class weight", [None, "balanced"], index=0)
        
        col2.divider()
        
        if col2.checkbox("Continue To Fit The Model"):
            model = HistGradientBoostingClassifier(
                loss=loss, learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                max_depth=max_depth, min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
                max_features=max_features, max_bins=max_bins, warm_start=warm_start, early_stopping=early_stopping,
                scoring=scoring, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
                tol=tol, verbose=verbose, random_state=random_state, class_weight=class_weight
            )
            
            col2.subheader("Your Model", divider='blue')
            col2.write(model.get_params())
            if st.session_state["Hist Gradient Boosting Classifier"] ==None:
                st.session_state["Hist Gradient Boosting Classifier"]=model.fit(st.session_state["availableDatasets"][xtrain_key], st.session_state["availableDatasets"][ytrain_key])
            else:
                col2.success("Model Created")
                delete=col2.checkbox("DO You Want to recreate model")
                if delete:
                    st.session_state["Hist Gradient Boosting Classifier"]=None
            col2.success("Model Fitted Successfully")
            col2.divider()
            self.metrics(col2, st.session_state["Hist Gradient Boosting Classifier"])
    def random_forest_classifier(self, col2):
        xtrain_key = col2.selectbox("Select X Train for RandomForestClassifier", list(st.session_state["availableDatasets"].keys()))
        ytrain_key = col2.selectbox("Select Y Train for RandomForestClassifier", list(st.session_state["availableDatasets"].keys()))
        
        n_estimators = int(col2.number_input("Number of estimators", value=100, min_value=1, step=1))
        criterion = col2.selectbox("Criterion", ["gini", "entropy", "log_loss"], index=0)
        max_depth = col2.number_input("Max depth", value=None, min_value=1, step=1, format="%d")
        min_samples_split = int(col2.number_input("Min samples split", value=2, min_value=1, step=1))
        min_samples_leaf = int(col2.number_input("Min samples leaf", value=1, min_value=1, step=1))
        min_weight_fraction_leaf = col2.number_input("Min weight fraction leaf", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
        max_features = col2.selectbox("Max features", ["sqrt", "log2", None], index=0)
        max_leaf_nodes = col2.number_input("Max leaf nodes", value=None, min_value=2, step=1, format="%d")
        min_impurity_decrease = col2.number_input("Min impurity decrease", value=0.0, min_value=0.0, step=0.01)
        bootstrap = col2.checkbox("Bootstrap", True)
        oob_score = col2.checkbox("Out-of-bag score", False)
        n_jobs = col2.number_input("Number of jobs (-1 for all CPUs)", value=None, format="%d")
        random_state = col2.number_input("Random state (leave blank for None)", value=None, format="%d")
        warm_start = col2.checkbox("Warm start", False)
        class_weight = col2.selectbox("Class weight", [None, "balanced", "balanced_subsample"], index=0)
        ccp_alpha = col2.number_input("CCP alpha", value=0.0, min_value=0.0, step=0.01)
        max_samples = col2.number_input("Max samples (for bootstrap)", value=None, min_value=0.0, max_value=1.0, step=0.01)
        
        col2.divider()
        
        if col2.checkbox("Continue To Fit The Model"):
            model = RandomForestClassifier(
                n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples
            )
            
            col2.subheader("Your Model", divider='blue')
            col2.write(model.get_params())
            
            if st.session_state.get("Random Forest Classifier") is None:
                st.session_state["Random Forest Classifier"] = model.fit(
                    st.session_state["availableDatasets"][xtrain_key],
                    st.session_state["availableDatasets"][ytrain_key]
                )
            else:
                col2.success("Model Created")
                delete = col2.checkbox("Do you want to recreate the model?")
                if delete:
                    st.session_state["Random Forest Classifier"] = None
            
            col2.success("Model Fitted Successfully")
            col2.divider()
            self.metrics(col2, st.session_state["Random Forest Classifier"])
    def stacking_classifier(self, col2):
        xtrain_key = col2.selectbox("Select X Train for StackingClassifier", list(st.session_state["availableDatasets"].keys()))
        ytrain_key = col2.selectbox("Select Y Train for StackingClassifier", list(st.session_state["availableDatasets"].keys()))
        
        base_estimators = [
            ("RandomForest", RandomForestClassifier(n_estimators=100)),
            ("LogisticRegression", LogisticRegression())
        ]
        final_estimator = LogisticRegression()
        
        estimators_selected = col2.multiselect("Select Base Estimators", ["RandomForest", "LogisticRegression", "SVC", "KNeighbors"], default=["RandomForest", "LogisticRegression"])
        
        selected_estimators = []
        for estimator in estimators_selected:
            if estimator == "RandomForest":
                selected_estimators.append(("RandomForest", RandomForestClassifier(n_estimators=100)))
            elif estimator == "LogisticRegression":
                selected_estimators.append(("LogisticRegression", LogisticRegression()))
            elif estimator == "SVC":
                selected_estimators.append(("SVC", SVC(probability=True)))
            elif estimator == "KNeighbors":
                selected_estimators.append(("KNeighbors", KNeighborsClassifier()))
        
        final_estimator_choice = col2.selectbox("Select Final Estimator", ["LogisticRegression", "SVC", "RandomForest"], index=0)
        
        if final_estimator_choice == "LogisticRegression":
            final_estimator = LogisticRegression()
        elif final_estimator_choice == "SVC":
            final_estimator = SVC(probability=True)
        elif final_estimator_choice == "RandomForest":
            final_estimator = RandomForestClassifier(n_estimators=100)
        
        cv = col2.number_input("Cross-validation folds", value=5, min_value=2, step=1)
        passthrough = col2.checkbox("Pass-through original features", False)
        
        col2.divider()
        
        if col2.checkbox("Continue To Fit The Model"):
            model = StackingClassifier(
                estimators=selected_estimators,
                final_estimator=final_estimator,
                cv=cv,
                passthrough=passthrough
            )
            
            col2.subheader("Your Model", divider='blue')
            col2.write(model.get_params())
            
            if st.session_state.get("Stacking Classifier") is None:
                st.session_state["Stacking Classifier"] = model.fit(
                    st.session_state["availableDatasets"][xtrain_key],
                    st.session_state["availableDatasets"][ytrain_key]
                )
            else:
                col2.success("Model Created")
                delete = col2.checkbox("Do you want to recreate the model?")
                if delete:
                    st.session_state["Stacking Classifier"] = None
            
            col2.success("Model Fitted Successfully")
            col2.divider()
            self.metrics(col2, st.session_state["Stacking Classifier"])
    def voting_classifier(self, col2):
        xtrain_key = col2.selectbox("Select X Train for VotingClassifier", list(st.session_state["availableDatasets"].keys()))
        ytrain_key = col2.selectbox("Select Y Train for VotingClassifier", list(st.session_state["availableDatasets"].keys()))
        
        # Base estimators selection
        estimators_selected = col2.multiselect("Select Base Estimators", ["RandomForest", "LogisticRegression", "SVC", "KNeighbors"], default=["RandomForest", "LogisticRegression"])
        
        selected_estimators = []
        for estimator in estimators_selected:
            if estimator == "RandomForest":
                selected_estimators.append(("RandomForest", RandomForestClassifier(n_estimators=100)))
            elif estimator == "LogisticRegression":
                selected_estimators.append(("LogisticRegression", LogisticRegression()))
            elif estimator == "SVC":
                selected_estimators.append(("SVC", SVC(probability=True)))
            elif estimator == "KNeighbors":
                selected_estimators.append(("KNeighbors", KNeighborsClassifier()))
        
        # Voting type selection
        voting_type = col2.selectbox("Select Voting Type", ["hard", "soft"], index=0)
        
        # Weights input (optional)
        use_weights = col2.checkbox("Assign Weights to Estimators?")
        weights = None
        if use_weights:
            weights = []
            for estimator in estimators_selected:
                weight = col2.number_input(f"Weight for {estimator}", value=1, min_value=1, step=1)
                weights.append(weight)
    
        # Additional parameters
        n_jobs = col2.number_input("Number of jobs (-1 for all CPUs)", value=None, format="%d")
        verbose = col2.checkbox("Verbose", False)
    
        col2.divider()
    
        if col2.checkbox("Continue To Fit The Model"):
            model = VotingClassifier(
                estimators=selected_estimators,
                voting=voting_type,
                weights=weights,
                n_jobs=n_jobs,
                verbose=verbose
            )
    
            col2.subheader("Your Model", divider='blue')
            col2.write(model.get_params())
    
            if st.session_state.get("Voting Classifier") is None:
                st.session_state["Voting Classifier"] = model.fit(
                    st.session_state["availableDatasets"][xtrain_key],
                    st.session_state["availableDatasets"][ytrain_key]
                )
            else:
                col2.success("Model Created")
                delete = col2.checkbox("Do you want to recreate the model?")
                if delete:
                    st.session_state["Voting Classifier"] = None
    
            col2.success("Model Fitted Successfully")
            col2.divider()
            self.metrics(col2, st.session_state["Voting Classifier"])
    
            
