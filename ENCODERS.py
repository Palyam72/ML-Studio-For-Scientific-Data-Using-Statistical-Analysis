import streamlit as st
import pandas as pd
import category_encoders as ce

class Encoders:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.transformed_data = None

    def apply_basen_encoder(self):
        """
        Apply BaseNEncoder transformation based on user inputs.
        """
        verbose = st.selectbox("Verbose", [0, 1])
        cols = st.selectbox("Columns to Encode", ['All Columns'] + list(self.data.select_dtypes(include=['object']).columns))
        drop_invariant = st.checkbox("Drop Invariant Columns", False)
        handle_unknown = st.selectbox("Handle Unknown", ['error', 'return_nan', 'value', 'indicator'])
        handle_missing = st.selectbox("Handle Missing", ['error', 'return_nan', 'value', 'indicator'])
        base = st.slider("Base", 2, 10, 2)
        
        cols = self.data.select_dtypes(include=['object']).columns.tolist() if cols == 'All Columns' else [cols]

        encoder = ce.BaseNEncoder(cols=cols, base=base, verbose=verbose, drop_invariant=drop_invariant,
                                  handle_unknown=handle_unknown, handle_missing=handle_missing, return_df=True)
        
        apply_transformation = st.checkbox("Apply BaseNEncoder transformation")

        if apply_transformation:
            self.transformed_data = encoder.fit_transform(self.data)
            st.write("Transformed Data:")
            st.dataframe(self.transformed_data)

            st.write("Encoder Parameters:")
            st.write(f"Verbose: {verbose}")
            st.write(f"Columns Encoded: {cols}")
            st.write(f"Drop Invariant: {drop_invariant}")
            st.write(f"Handle Unknown: {handle_unknown}")
            st.write(f"Handle Missing: {handle_missing}")
            st.write(f"Base: {base}")

        return self.transformed_data

    def apply_binary_encoder(self):
        """
        Apply BinaryEncoder transformation based on user inputs.
        """
        verbose = st.selectbox("Verbose", [0, 1])
        cols = st.selectbox("Columns to Encode", ['All Columns'] + list(self.data.select_dtypes(include=['object']).columns))
        drop_invariant = st.checkbox("Drop Invariant Columns", False)
        handle_unknown = st.selectbox("Handle Unknown", ['error', 'return_nan', 'value', 'indicator'])
        handle_missing = st.selectbox("Handle Missing", ['error', 'return_nan', 'value', 'indicator'])
        base = st.slider("Base", 2, 10, 2)
        
        cols = self.data.select_dtypes(include=['object']).columns.tolist() if cols == 'All Columns' else [cols]

        encoder = ce.BinaryEncoder(cols=cols, base=base, verbose=verbose, drop_invariant=drop_invariant,
                                   handle_unknown=handle_unknown, handle_missing=handle_missing, return_df=True)
        
        apply_transformation = st.checkbox("Apply BinaryEncoder transformation")

        if apply_transformation:
            self.transformed_data = encoder.fit_transform(self.data)
            st.write("Transformed Data:")
            st.dataframe(self.transformed_data)

            st.write("Encoder Parameters:")
            st.write(f"Verbose: {verbose}")
            st.write(f"Columns Encoded: {cols}")
            st.write(f"Drop Invariant: {drop_invariant}")
            st.write(f"Handle Unknown: {handle_unknown}")
            st.write(f"Handle Missing: {handle_missing}")
            st.write(f"Base: {base}")

        return self.transformed_data

    def apply_catboost_encoder(self):
        """
        Apply CatBoostEncoder transformation based on user inputs.
        """
        verbose = st.selectbox("Verbose", [0, 1])
        cols = st.selectbox("Columns to Encode", ['All Columns'] + list(self.data.select_dtypes(include=['object']).columns))
        drop_invariant = st.checkbox("Drop Invariant Columns", False)
        handle_unknown = st.selectbox("Handle Unknown", ['error', 'return_nan', 'value', 'indicator'])
        handle_missing = st.selectbox("Handle Missing", ['error', 'return_nan', 'value', 'indicator'])
        sigma = st.slider("Sigma (Gaussian Noise)", 0.0, 2.0, 0.1)
        a = st.slider("Additive Smoothing (a)", 0.1, 2.0, 1.0)
        
        cols = self.data.select_dtypes(include=['object']).columns.tolist() if cols == 'All Columns' else [cols]

        encoder = ce.CatBoostEncoder(cols=cols, sigma=sigma, a=a, verbose=verbose, drop_invariant=drop_invariant,
                                     handle_unknown=handle_unknown, handle_missing=handle_missing, return_df=True)
        
        apply_transformation = st.checkbox("Apply CatBoostEncoder transformation")

        if apply_transformation:
            y=st.multiselect("select the target column",[None].extend(self.data.columns))
            y=None if y==None else y
            self.transformed_data = encoder.fit_transform(self.data,y=y)
            st.write("Transformed Data:")
            st.dataframe(self.transformed_data)

            st.write("Encoder Parameters:")
            st.write(f"Verbose: {verbose}")
            st.write(f"Columns Encoded: {cols}")
            st.write(f"Drop Invariant: {drop_invariant}")
            st.write(f"Handle Unknown: {handle_unknown}")
            st.write(f"Handle Missing: {handle_missing}")
            st.write(f"Sigma: {sigma}")
            st.write(f"Additive Smoothing (a): {a}")

        return self.transformed_data

    def apply_count_encoder(self):
        """
        Apply CountEncoder transformation based on user inputs.
        """
        verbose = st.selectbox("Verbose", [0, 1])
        cols = st.selectbox("Columns to Encode", ['All Columns'] + list(self.data.select_dtypes(include=['object']).columns))
        drop_invariant = st.checkbox("Drop Invariant Columns", False)
        handle_unknown = st.selectbox("Handle Unknown", ['error', 'return_nan', 'value', 'indicator'])
        handle_missing = st.selectbox("Handle Missing", ['error', 'return_nan', 'value', 'indicator'])
        normalize = st.checkbox("Normalize Counts", False)
        min_group_size = st.text_input("Minimum Group Size", '1')
        combine_min_nan_groups = st.checkbox("Combine Small Groups with NaN Groups", True)
        
        cols = self.data.select_dtypes(include=['object']).columns.tolist() if cols == 'All Columns' else [cols]

        encoder = ce.CountEncoder(cols=cols, normalize=normalize, min_group_size=int(min_group_size),
                                  combine_min_nan_groups=combine_min_nan_groups, verbose=verbose,
                                  drop_invariant=drop_invariant, handle_unknown=handle_unknown,
                                  handle_missing=handle_missing, return_df=True)
        
        apply_transformation = st.checkbox("Apply CountEncoder transformation")

        if apply_transformation:
            self.transformed_data = encoder.fit_transform(self.data)
            st.write("Transformed Data:")
            st.dataframe(self.transformed_data)

            st.write("Encoder Parameters:")
            st.write(f"Verbose: {verbose}")
            st.write(f"Columns Encoded: {cols}")
            st.write(f"Drop Invariant: {drop_invariant}")
            st.write(f"Handle Unknown: {handle_unknown}")
            st.write(f"Handle Missing: {handle_missing}")
            st.write(f"Normalize Counts: {normalize}")
            st.write(f"Min Group Size: {min_group_size}")
            st.write(f"Combine Small Groups with NaN Groups: {combine_min_nan_groups}")

        return self.transformed_data

# Example usage
# Assuming `data` is a pandas DataFrame loaded earlier
# encoders = Encoders(data)
# encoders.apply_basen_encoder()  # Call the method for BaseNEncoder
