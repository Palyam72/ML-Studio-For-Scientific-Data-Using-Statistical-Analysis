import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.feature_selection import VarianceThreshold
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

    def variance_threshold(self, threshold=0.0):
        st.write("### Variance Threshold Method")
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
