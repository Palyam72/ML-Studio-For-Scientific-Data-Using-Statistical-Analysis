import streamlit as st
import pandas as pd
from feature_engine.discretisation import EqualWidthDiscretiser, EqualFrequencyDiscretiser, DecisionTreeDiscretiser, GeometricWidthDiscretiser

class Descritizers:
    def __init__(self, dataset: pd.DataFrame):
        self.data = dataset

    def equal_width_discretiser(self):
        """EqualWidthDiscretiser with Streamlit widgets"""
        
        st.header('Equal Width Discretiser')
        
        # Select columns for transformation
        variables = st.multiselect('Select variables', self.data.columns.tolist(), default=self.data.select_dtypes(include=['number']).columns.tolist())
        
        # Widgets for input parameters
        bins = st.slider('Select number of bins', min_value=2, max_value=20, value=10, step=1)
        return_object = st.checkbox('Return as object type (categorical)', value=False)
        return_boundaries = st.checkbox('Return interval boundaries', value=False)
        precision = st.slider('Precision for bin labels', min_value=1, max_value=10, value=3, step=1)
        
        # Checkbox to apply transformation
        apply_transformation = st.checkbox('Apply transformation')

        if apply_transformation:
            # Initialize the discretiser and apply fit_transform
            discretiser = EqualWidthDiscretiser(variables=variables, bins=bins, return_object=return_object, 
                                                return_boundaries=return_boundaries, precision=precision)
            transformed_data = discretiser.fit_transform(self.data)
            st.dataframe(transformed_data)
            return transformed_data

    def equal_frequency_discretiser(self):
        """EqualFrequencyDiscretiser with Streamlit widgets"""
        
        st.header('Equal Frequency Discretiser')
        
        # Select columns for transformation
        variables = st.multiselect('Select variables', self.data.columns.tolist(), default=self.data.select_dtypes(include=['number']).columns.tolist())
        
        # Widgets for input parameters
        bins = st.slider('Select number of bins', min_value=2, max_value=20, value=10, step=1)
        return_object = st.checkbox('Return as object type (categorical)', value=False)
        return_boundaries = st.checkbox('Return interval boundaries', value=False)
        precision = st.slider('Precision for bin labels', min_value=1, max_value=10, value=3, step=1)
        
        # Checkbox to apply transformation
        apply_transformation = st.checkbox('Apply transformation')

        if apply_transformation:
            # Initialize the discretiser and apply fit_transform
            discretiser = EqualFrequencyDiscretiser(variables=variables, q=bins, return_object=return_object, 
                                                    return_boundaries=return_boundaries, precision=precision)
            transformed_data = discretiser.fit_transform(self.data)
            st.dataframe(transformed_data)
            return transformed_data

    def decision_tree_discretiser(self):
        """DecisionTreeDiscretiser with Streamlit widgets"""
        
        st.header('Decision Tree Discretiser')
        
        # Select columns for transformation
        variables = st.multiselect('Select variables', self.data.columns.tolist(), default=self.data.select_dtypes(include=['number']).columns.tolist())
        
        # Widgets for input parameters
        bin_output = st.radio('Select bin output', options=["prediction", "bin_number", "boundaries"], index=0)
        precision = st.slider('Precision for bin labels', min_value=1, max_value=10, value=3, step=1)
        cv = st.slider('Cross-validation splits', min_value=2, max_value=10, value=3, step=1)
        scoring = st.selectbox('Select scoring metric', options=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'])
        
        # Select target variable (y) for decision tree
        target_column = st.selectbox('Select target variable (y)', options=self.data.columns.tolist())
        y = self.data[target_column]
        
        # Checkbox to apply transformation
        apply_transformation = st.checkbox('Apply transformation')

        if apply_transformation:
            # Initialize the discretiser and apply fit_transform
            discretiser = DecisionTreeDiscretiser(variables=variables, bin_output=bin_output, precision=precision,
                                                  cv=cv, scoring=scoring)
            transformed_data = discretiser.fit_transform(self.data, y)
            st.dataframe(transformed_data)
            return transformed_data

    def geometric_width_discretiser(self):
        """GeometricWidthDiscretiser with Streamlit widgets"""
        
        st.header('Geometric Width Discretiser')
        
        # Select columns for transformation
        variables = st.multiselect('Select variables', self.data.columns.tolist(), default=self.data.select_dtypes(include=['number']).columns.tolist())
        
        # Widgets for input parameters
        bins = st.slider('Select number of bins', min_value=2, max_value=20, value=10, step=1)
        return_object = st.checkbox('Return as object type (categorical)', value=False)
        return_boundaries = st.checkbox('Return interval boundaries', value=False)
        precision = st.slider('Precision for bin labels', min_value=1, max_value=10, value=7, step=1)
        
        # Checkbox to apply transformation
        apply_transformation = st.checkbox('Apply transformation')

        if apply_transformation:
            # Initialize the discretiser and apply fit_transform
            discretiser = GeometricWidthDiscretiser(variables=variables, bins=bins, return_object=return_object, 
                                                    return_boundaries=return_boundaries, precision=precision)
            transformed_data = discretiser.fit_transform(self.data)
            st.dataframe(transformed_data)
            return transformed_data
