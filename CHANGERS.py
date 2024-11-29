import streamlit as st
import pandas as pd
from feature_engine.discretisation import EqualWidthDiscretiser, EqualFrequencyDiscretiser, DecisionTreeDiscretiser, GeometricWidthDiscretiser
from sklearn.preprocessing import *

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
class PreprocessingMethods:
    def __init__(self, dataset):
        self.dataset = dataset

    def normalizer(self):
        st.subheader("Parameters for Normalizer")
        norm = st.selectbox("The normalization method to use. Can be 'l1', 'l2', or 'max'", ['l1', 'l2', 'max'], key="normalizer_norm")
        axis = st.selectbox("Axis along which to normalize. If axis=0, each feature (column) is normalized. If axis=1, each sample (row) is normalized.", [0, 1], key="normalizer_axis")
        
        if st.checkbox("Confirm to apply normalizer", key="normalizer_checkbox"):
            normalizer = Normalizer(norm=norm, axis=axis)
            transformed_dataset = normalizer.fit_transform(self.dataset)
            return transformed_dataset

    def binarizer(self):
        st.subheader("Parameters for Binarizer")
        threshold = st.number_input("Threshold value for binarization", min_value=0.0, max_value=1.0, step=0.01, value=0.0, key="binarizer_threshold")
        copy = st.checkbox("Confirm to apply binarizer", key="binarizer_copy")
        
        if st.checkbox("Confirm to apply binarizer", key="binarizer_checkbox"):
            binarizer = Binarizer(threshold=threshold, copy=copy)
            transformed_dataset = binarizer.fit_transform(self.dataset)
            return transformed_dataset

    def label_binarizer(self):
        st.subheader("Parameters for LabelBinarizer")
        neg_label = st.number_input("Negative label encoding value", value=0, key="label_binarizer_neg_label")
        pos_label = st.number_input("Positive label encoding value", value=1, key="label_binarizer_pos_label")
        sparse_output = st.checkbox("Output in sparse format", key="label_binarizer_sparse_output")
        
        if st.checkbox("Confirm to apply LabelBinarizer", key="label_binarizer_checkbox"):
            label_binarizer = LabelBinarizer(neg_label=neg_label, pos_label=pos_label, sparse_output=sparse_output)
            transformed_dataset = label_binarizer.fit_transform(self.dataset)
            return transformed_dataset

    def multi_label_binarizer(self):
        st.subheader("Parameters for MultiLabelBinarizer")
        classes_input = st.text_area("Enter the class labels (comma separated)", key="multi_label_binarizer_classes")
        classes = classes_input.split(",") if classes_input else None
        sparse_output = st.checkbox("Output in sparse format", key="multi_label_binarizer_sparse_output")
        
        if st.checkbox("Confirm to apply MultiLabelBinarizer", key="multi_label_binarizer_checkbox"):
            multi_label_binarizer = MultiLabelBinarizer(classes=classes, sparse_output=sparse_output)
            transformed_dataset = multi_label_binarizer.fit_transform(self.dataset)
            return transformed_dataset

        
