import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.feature_selection import *
import matplotlib.pyplot as plt

class FeatureSelection:
    def __init__(self, dataset):
        self.dataset = dataset

    def pearson(self):
        st.write("### Pearson Correlation Matrix")
        corr_matrix = self.dataset.corr(method='pearson', numeric_only=True)
        st.dataframe(corr_matrix)
        st.write("Heatmap:")
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    def spearman(self):
        st.write("### Spearman Correlation Matrix")
        corr_matrix = self.dataset.corr(method='spearman', numeric_only=True)
        st.dataframe(corr_matrix)
        st.write("Heatmap:")
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    def kendall(self):
        st.write("### Kendall Correlation Matrix")
        corr_matrix = self.dataset.corr(method='kendall', numeric_only=True)
        st.dataframe(corr_matrix)
        st.write("Heatmap:")
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    def point(self):
        st.write("### Point-Biserial Correlation")
        binary_columns = [col for col in self.dataset.columns if len(self.dataset[col].unique()) == 2]
        continuous_columns = [col for col in self.dataset.columns if col not in binary_columns]

        if not binary_columns:
            st.warning("No binary columns found for Point-Biserial Correlation.")
            return

        results = {}
        for bin_col in binary_columns:
            for cont_col in continuous_columns:
                try:
                    corr, _ = pointbiserialr(self.dataset[bin_col], self.dataset[cont_col])
                    results[(bin_col, cont_col)] = corr
                except Exception as e:
                    st.error(f"Error processing {bin_col} and {cont_col}: {e}")

        st.write("Point-Biserial Correlation Results:")
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Correlation'])
        st.dataframe(result_df)
        fig, ax = plt.subplots()
        sns.heatmap(result_df, annot=True, cmap='coolwarm', ax=ax, cbar=False)
        st.pyplot(fig)

    def cramers(self):
        st.write("### Cramér's V (Association Between Categorical Variables)")
        categorical_columns = self.dataset.select_dtypes(include=['object', 'category']).columns

        if len(categorical_columns) < 2:
            st.warning("Not enough categorical columns for Cramér's V calculation.")
            return

        results = {}
        for col1 in categorical_columns:
            for col2 in categorical_columns:
                if col1 != col2:
                    contingency_table = pd.crosstab(self.dataset[col1], self.dataset[col2])
                    chi2, _, _, _ = chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    r, k = contingency_table.shape
                    cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
                    results[(col1, col2)] = cramers_v

        st.write("Cramér's V Results:")
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Cramér\'s V'])
        st.dataframe(result_df)
        fig, ax = plt.subplots()
        sns.heatmap(result_df, annot=True, cmap='coolwarm', ax=ax, cbar=False)
        st.pyplot(fig)

    def variance_threshold(self):
        st.write("### Variance Threshold Method")

        threshold=st.number_input("Threshold value")
        if threshold:
            st.write(f"Variance Threshold: {threshold}")
            
            numeric_data = self.dataset.select_dtypes(include=['number'])
            if numeric_data.empty:
                st.warning("No numeric columns found for Variance Threshold.")
                return
            
            selector = VarianceThreshold(threshold=threshold)
            try:
                selector.fit(numeric_data)
                mask = selector.get_support()
                selected_features = numeric_data.columns[mask]
                st.write("Selected Features:")
                st.write(list(selected_features))
                st.write("Removed Features:")
                st.write(list(numeric_data.columns[~mask]))
            except Exception as e:
                st.error(f"Error during variance threshold selection: {e}")




class StatisticalFunctions:
    def __init__(self, dataset):
        self.dataset = dataset
        self.score_functions = {
            'f_classif': f_classif,
            'f_regression': f_regression,
            'mutual_info_classif': mutual_info_classif,
            'mutual_info_regression': mutual_info_regression,
            'r_regression': r_regression,
            'chi2': chi2
        }

    def generic_univariate_select(self):
        score_func_name = st.selectbox("Select a score function", self.score_functions.keys())
        mode = st.selectbox("Select the mode", ['percentile', 'k_best', 'fpr', 'fdr', 'fwe'])
        param = st.text_input("Parameter for the mode (percentile, k, alpha)", "Enter Here")
        param = float(param) if param != "Enter Here" else 5e-2

        features = st.multiselect("Select feature columns", self.dataset.columns)
        target = st.selectbox("Select target column (optional)", [None] + list(self.dataset.columns))

        if st.checkbox("Confirm to apply Generic Univariate Select"):
            if not features:
                st.error("Please select at least one feature column.")
                return

            x = self.dataset[features]
            y = self.dataset[target] if target else None
            score_func = self.score_functions[score_func_name]

            transformer = GenericUnivariateSelect(score_func=score_func, mode=mode, param=param)
            X_new = transformer.fit_transform(x, y)

            st.header("Transformed Data Frame")
            transformed_df = pd.DataFrame(X_new, columns=[f"Feature_{i}" for i in range(X_new.shape[1])])
            st.dataframe(transformed_df)

            # Calling common_attributes
            self.common_attributes(transformer)
        else:
            st.warning("Please confirm to apply Generic Univariate Select.")

    def common_attributes(self, transformer):
        st.subheader("Attributes for the Current Result")
        if hasattr(transformer, 'scores_'):
            st.info("Scores")
            st.write(transformer.scores_)
        if hasattr(transformer, 'pvalues_'):
            st.info("P-Values")
            st.write(transformer.pvalues_)
        st.info("Number of Features In")
        st.write(transformer.n_features_in_)
        st.info("Feature Names In")
        st.write(transformer.feature_names_in_)

    def select_fdr(self):
        self._apply_selection_method("Select FDR", SelectFdr)

    def select_fpr(self):
        self._apply_selection_method("Select FPR", SelectFpr)

    def select_fwe(self):
        self._apply_selection_method("Select FWE", SelectFwe)

    def select_k_best(self):
        st.header("Select K Best")
        self._apply_selection_method_with_param("k", SelectKBest, default_param=10)

    def select_percentile(self):
        st.header("Select Percentile")
        self._apply_selection_method_with_param("percentile", SelectPercentile, default_param=10)

    def _apply_selection_method(self, header, selector_class):
        st.header(header)
        score_func_name = st.selectbox("Select a score function", self.score_functions.keys())
        alpha = st.slider(f"Select the alpha value for {header}", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

        features = st.multiselect("Select feature columns", self.dataset.columns)
        target = st.selectbox("Select target column (optional)", [None] + list(self.dataset.columns))

        if st.checkbox(f"Confirm to apply {header}"):
            if not features:
                st.error("Please select at least one feature column.")
                return

            x = self.dataset[features]
            y = self.dataset[target] if target else None
            score_func = self.score_functions[score_func_name]

            transformer = selector_class(score_func=score_func, alpha=alpha)
            X_new = transformer.fit_transform(x, y)

            st.header("Transformed Data Frame")
            transformed_df = pd.DataFrame(X_new, columns=[f"Feature_{i}" for i in range(X_new.shape[1])])
            st.dataframe(transformed_df)

            # Calling common_attributes
            self.common_attributes(transformer)

    def _apply_selection_method_with_param(self, param_name, selector_class, default_param):
        param = st.number_input(f"Select the {param_name} value", min_value=1, value=default_param, step=1)

        features = st.multiselect("Select feature columns", self.dataset.columns)
        target = st.selectbox("Select target column (optional)", [None] + list(self.dataset.columns))

        if st.checkbox(f"Confirm to apply {selector_class.__name__}"):
            if not features:
                st.error("Please select at least one feature column.")
                return

            x = self.dataset[features]
            y = self.dataset[target] if target else None
            score_func = self.score_functions['f_classif']  # Default score function for simplicity

            transformer = selector_class(score_func=score_func, **{param_name: param})
            X_new = transformer.fit_transform(x, y)

            st.header("Transformed Data Frame")
            transformed_df = pd.DataFrame(X_new, columns=[f"Feature_{i}" for i in range(X_new.shape[1])])
            st.dataframe(transformed_df)

            # Calling common_attributes
            self.common_attributes(transformer)
