import pandas as pd
import streamlit as st
import numpy as np


# Class for methods offered by pandas
class PandasMethods:
    def __init__(self, dataset):
        self.dataset = dataset

    def backward_fill(self):
        axis = st.selectbox("select axis to perform backward fill", ["columns", "index"])
        limit = st.text_input("How many consecutive nan values do you wanted to dill using backwaard fill [enter none to fill all nan values]")
        if st.checkbox("Fix the above settings for backward fill"):
            if limit.lower() == "none":
                return self.dataset.bfill(axis=axis)
            else:
                return self.dataset.bfill(axis=axis, limit=int(limit))
    def forward_fill(self):
        axis = st.selectbox("select axis to perform forward fill", ["columns", "index"])
        limit = st.text_input("How many consecutive nan values do you wanted to dill using forward fill [enter none to fill all nan values]")
        if st.checkbox("Fix the above settings for forward fill"):
            if limit.lower() == "none":
                return self.dataset.ffill(axis=axis)
            else:
                return self.dataset.ffill(axis=axis, limit=int(limit))
    def drop_na(self):
        axis = st.selectbox("Select axis to drop missing values", ["rows", "columns"], index=0)
        choice = st.radio("Select criteria to drop rows/columns", ["how", "threshold"])
        if choice == "how":
            how = st.selectbox("How to drop?", ["any", "all"], index=0)
        if choice=="threshold":
            thresh = st.number_input("Require this many non-NA values", min_value=0, max_value=len(self.dataset), value=None, step=1)

        if st.checkbox("Fix the above settings for dropna"):
            if axis == "rows":
                axis_value = 0
            else:
                axis_value = 1

            if choice=="how":
                return self.dataset.dropna(axis=axis_value, how=how)
            else:
                return self.dataset.dropna(axis=axis_value, thresh=thresh)
    def fill_na(self):
        selected_columns = st.multiselect("select the columns to fill the na values",self.dataset.columns)
        fill_value = st.text_input("Enter the value to fill the corresponding na values with ',' as separator")
        if fill_value:
            fill_value=fill_value.split(",")
        values = self.make_dictionary(selected_columns, fill_value)
        if st.checkbox("Confirm to apply fill values"):
            return self.dataset.fillna(values)
    def interpolate_missing_values(self):
        method = st.selectbox("Select interpolation method", [
            'linear', 'time', 'index', 'pad', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        ])
        if method == 'polynomial':
            order = st.number_input("Specify the order for polynomial interpolation", min_value=1, step=1)

        if st.checkbox("Confirm interpolate missing values operation"):
            dataframe = self.dataset.interpolate(method=method)
            return dataframe


    def make_dictionary(self, selected_columns, values):
        fill_dict = {}
        for i in range(len(selected_columns)):
            if values[i].isdigit():
                fill_dict[selected_columns[i]] = int(values[i])
            elif values[i].isnumeric():
                fill_dict[selected_columns[i]] = float(values[i])
            else:
                fill_dict[selected_columns[i]] = values[i]
        return fill_dict
