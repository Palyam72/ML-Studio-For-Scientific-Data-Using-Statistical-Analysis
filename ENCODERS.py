import streamlit as st
import pandas as pd
import category_encoders as ce

class Encoders:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.transformed_data = None

    def get_user_input(self, param_name, default_value, param_type='selectbox', options=None):
        # Handle the 'selectbox' input type
        if param_type == 'selectbox':
            # If default_value is None, simply return the selectbox without an index
            if default_value is None:
                return st.selectbox(param_name, options)
            # If default_value is provided, find the index and use it
            else:
                try:
                    return st.selectbox(param_name, options, index=options.index(default_value))
                except ValueError:
                    # Handle case where default_value is not in options (e.g., 'None' or unexpected value)
                    st.warning(f"Default value '{default_value}' not found in options, defaulting to first option.")
                    return st.selectbox(param_name, options)

        # Handle other input types
        elif param_type == 'checkbox':
            return st.checkbox(param_name, value=default_value)
        elif param_type == 'slider':
            return st.slider(param_name, min_value=1, max_value=10, value=default_value)
        elif param_type == 'text':
            return st.text_input(param_name, default_value)
        else:
            return default_value

    def apply_encoder(self, encoder_type):
        # Streamlit widgets for user input
        verbose = self.get_user_input("Verbose", 0, 'selectbox', [0, 1])
        cols = self.get_user_input("Columns to Encode", None, 'selectbox', ['All Columns'] + list(self.data.select_dtypes(include=['object']).columns))
        drop_invariant = self.get_user_input("Drop Invariant Columns", False, 'checkbox')
        handle_unknown = self.get_user_input("Handle Unknown", 'value', 'selectbox', ['error', 'return_nan', 'value', 'indicator'])
        handle_missing = self.get_user_input("Handle Missing", 'value', 'selectbox', ['error', 'return_nan', 'value', 'indicator'])

        # If 'All Columns' is selected, encode all object columns; otherwise, encode only the selected column(s)
        if cols == 'All Columns':
            cols = self.data.select_dtypes(include=['object']).columns.tolist()
        else:
            cols = [cols]

        # Apply the encoder based on the chosen type
        if encoder_type == 'BaseNEncoder':
            base = self.get_user_input("Base", 2, 'slider', options=range(2, 11))
            encoder = ce.BaseNEncoder(verbose=verbose, cols=cols, drop_invariant=drop_invariant, 
                                      handle_unknown=handle_unknown, handle_missing=handle_missing, 
                                      base=base, return_df=True)

        elif encoder_type == 'BinaryEncoder':
            base = self.get_user_input("Base", 2, 'slider', options=range(2, 11))
            encoder = ce.BinaryEncoder(verbose=verbose, cols=cols, drop_invariant=drop_invariant, 
                                       handle_unknown=handle_unknown, handle_missing=handle_missing, 
                                       base=base, return_df=True)

        elif encoder_type == 'CatBoostEncoder':
            sigma = self.get_user_input("Sigma (Gaussian Noise)", None, 'slider', options=[None, 0.1, 0.5, 1.0, 2.0])
            a = self.get_user_input("Additive Smoothing (a)", 1.0, 'slider', options=[0.1, 0.5, 1.0, 2.0])
            encoder = ce.CatBoostEncoder(verbose=verbose, cols=cols, drop_invariant=drop_invariant, 
                                         handle_unknown=handle_unknown, handle_missing=handle_missing, 
                                         sigma=sigma, a=a, return_df=True)

        elif encoder_type == 'CountEncoder':
            normalize = self.get_user_input("Normalize Counts", False, 'checkbox')
            min_group_size = self.get_user_input("Minimum Group Size", None, 'text')
            combine_min_nan_groups = self.get_user_input("Combine Small Groups with NaN Groups", True, 'checkbox')
            encoder = ce.CountEncoder(verbose=verbose, cols=cols, drop_invariant=drop_invariant, 
                                      handle_unknown=handle_unknown, handle_missing=handle_missing, 
                                      normalize=normalize, min_group_size=min_group_size, 
                                      combine_min_nan_groups=combine_min_nan_groups, return_df=True)

        elif encoder_type == 'GeneralizedLinearMixedModelEncoder':
            sigma = self.get_user_input("Sigma (Gaussian Noise)", 0.05, 'slider', options=[0.0, 0.05, 0.1, 0.5])
            encoder = ce.GeneralizedLinearMixedModelEncoder(verbose=verbose, cols=cols, drop_invariant=drop_invariant, 
                                                           handle_unknown=handle_unknown, handle_missing=handle_missing, 
                                                           sigma=sigma, return_df=True)

        # Fit and transform data
        self.transformed_data = encoder.fit_transform(self.data)

        # Display the transformed data and encoder parameters
        st.write("Transformed Data:")
        st.dataframe(self.transformed_data)

        st.write("Encoder Parameters:")
        st.write(f"Verbose: {verbose}")
        st.write(f"Columns Encoded: {cols}")
        st.write(f"Drop Invariant: {drop_invariant}")
        st.write(f"Handle Unknown: {handle_unknown}")
        st.write(f"Handle Missing: {handle_missing}")

        return self.transformed_data
