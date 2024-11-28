import streamlit as st
import pandas as pd
import category_encoders as ce
import numpy

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
    def generalized_linear_mixed_model(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
        
        verbose = int(st.text_input("Enter the verbose level (Int)", 0, key="glmm_verbose_level"))
        cols = st.multiselect("Select the columns on which you want to apply transformation",
                              ["All Columns"].extend(self.data.columns), key="glmm_columns_selection")
        y = st.selectbox("Select the target column", [None].extend(self.data.columns), key="glmm_target_column")
        drop_invariant = st.checkbox("Drop the invariant", key="glmm_drop_invariant")
        handle_unknown = st.selectbox("How to handle unknown", ['value', 'error', 'return_nan'], key="glmm_handle_unknown")
        handle_missing = st.selectbox("How to handle missing", ['value', 'error', 'return_nan'], key="glmm_handle_missing")
        randomized = st.checkbox(
            "Adds normal (Gaussian) distribution noise into training data to decrease overfitting (testing data are untouched).",
            key="glmm_randomized")
        if randomized:
            sigma = st.slider("Standard deviation (spread or “width”) of the normal distribution.", 0, 1, 0.05, key="glmm_sigma")
        
        bionomial_target = st.checkbox(
            "If True, the target must be binomial with values {0, 1} and Binomial mixed model is used. "
            "If False, the target must be continuous and Linear mixed model is used.", key="glmm_bionomial_target")
        
        random_state = st.text_input("Specify random state", "None", key="glmm_random_state")
        
        # Setting the values
        cols = None if cols == "All Columns" else cols
        random_state = None if random_state == "None" else int(random_state)
        
        if st.checkbox("Apply Generalized Linear Mixed Model", key="glmm_apply"):
            if randomized:
                glmm = GLMMEncoder(
                    verbose=verbose, cols=cols, drop_invariant=drop_invariant, handle_unknown=handle_unknown,
                    handle_missing=handle_missing, randomized=randomized, sigma=sigma,
                    bionomial_target=bionomial_target, random_state=random_state)
            else:
                glmm = GLMMEncoder(
                    verbose=verbose, cols=cols, drop_invariant=drop_invariant, handle_unknown=handle_unknown,
                    handle_missing=handle_missing, bionomial_target=bionomial_target, random_state=random_state)
            
            self.transformed = glmm.fit_transform(cols, y)
        return self.transformed
    def apply_gray_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
        
        verbose = int(st.text_input("Enter the verbose level (Int)", 0, key="gray_verbose_level"))
        cols = st.multiselect("Select the columns on which you want to apply transformation",
                              ["All Columns"].extend(self.data.columns), key="gray_columns_selection")
        y = st.selectbox("Select the target column", [None].extend(self.data.columns), key="gray_target_column")
        drop_invariant = st.checkbox("Drop the invariant columns", key="gray_drop_invariant")
        handle_unknown = st.selectbox("How to handle unknown values", ['value', 'error', 'return_nan'], key="gray_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values", ['value', 'error', 'return_nan'], key="gray_handle_missing")
        base = st.number_input("Base for Gray encoding (e.g., 2 for binary)", min_value=2, value=2, step=1, key="gray_base")
        
        # Setting the values
        cols = None if cols == "All Columns" else cols
        
        if st.checkbox("Apply Gray Encoder", key="gray_apply"):
            gray_encoder = category_encoders.GrayEncoder(
                verbose=verbose, cols=cols, drop_invariant=drop_invariant,
                handle_unknown=handle_unknown, handle_missing=handle_missing, base=base
            )
            if cols!=None:
                self.transformed = gray_encoder.fit_transform(cols,y)
            else:
                self.transformed=gray_encoder.fit_transform(self.data,y)
        return self.transformed
                
    def apply_hashing_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
        
        verbose = int(st.text_input("Enter the verbose level (Int)", 0, key="hashing_verbose"))
        cols = st.multiselect("Select the columns on which you want to apply transformation",
                              ["All Columns"].extend(self.data.columns), key="hashing_columns")
        y = st.selectbox("Select the target column", [None].extend(self.data.columns), key="hassing_target_column")
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="hashing_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)", key="hashing_return_df")
        hash_method = st.text_input("Enter the hashing method (e.g., 'md5')", "md5", key="hashing_hash_method")
        max_process = st.number_input("Maximum number of processes (Range: 1-64)", min_value=1, max_value=64, value=0, key="hashing_max_process")
        max_sample = st.number_input("Maximum number of samples to encode per process", min_value=0, value=0, key="hashing_max_sample")
        n_components = st.number_input("Number of bits to use to represent the feature", min_value=1, value=8, key="hashing_n_components")
        process_creation_method = st.selectbox(
            "Process creation method",
            ["fork", "spawn", "forkserver"],
            index=0 if not self.data.empty else 1,
            key="hashing_process_creation_method"
        )
        
        # Setting the values
        cols = None if cols == "All Columns" else cols
        
        if st.checkbox("Apply Hashing Encoder", key="hashing_apply"):
            hashing_encoder = category_encoders.HashingEncoder(
                verbose=verbose,
                cols=cols,
                drop_invariant=drop_invariant,
                return_df=return_df,
                hash_method=hash_method,
                max_process=max_process,
                max_sample=max_sample,
                n_components=n_components,
                process_creation_method=process_creation_method
            )
            
            if y is not None and cols is None:
                self.transformed = hashing_encoder.fit_transform(self.data, self.data[y])
            else:
                self.transformed = hashing_encoder.fit_transform(self.data[cols])
            return self.transformed

    def apply_helmert_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
        
        # Input widgets for HelmertEncoder parameters
        verbose = int(st.text_input("Enter the verbose level (Int)", 0, key="helmert_verbose"))
        cols = st.multiselect("Select the columns on which you want to apply transformation",
                              ["All Columns"].extend(self.data.columns), key="helmert_columns")
        y = st.selectbox("Select the target column", [None].extend(self.data.columns), key="helmert_target_column")
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="helmert_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)", key="helmert_return_df")
        handle_unknown = st.selectbox("How to handle unknown values", 
                                       ['value', 'error', 'return_nan', 'indicator'], 
                                       key="helmert_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values", 
                                       ['value', 'error', 'return_nan', 'indicator'], 
                                       key="helmert_handle_missing")
        
        # Setting the values
        cols = None if cols == "All Columns" else cols
        
        if st.checkbox("Apply Helmert Encoder", key="helmert_apply"):
            helmert_encoder = ce.HelmertEncoder(
                verbose=verbose,
                cols=cols,
                drop_invariant=drop_invariant,
                return_df=return_df,
                handle_unknown=handle_unknown,
                handle_missing=handle_missing
            )
            
            # Ensure the y parameter is handled properly
            if y is not None and cols is None:
                self.transformed = helmert_encoder.fit_transform(self.data, self.data[y])
            else:
                self.transformed = helmert_encoder.fit_transform(self.data[cols])
            
            return self.transformed
    def apply_james_stein_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for James-Stein Encoder parameters
        verbose = int(st.number_input("Enter the verbose level (Int)", value=0, min_value=0, key="james_stein_verbose"))
        cols = st.multiselect("Select the columns on which you want to apply transformation", 
                              ["All Columns"] + list(self.data.columns), key="james_stein_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="james_stein_target_column")
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="james_stein_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)", 
                                value=True, key="james_stein_return_df")
        handle_unknown = st.selectbox("How to handle unknown values", 
                                       ['value', 'error', 'return_nan'], 
                                       key="james_stein_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values", 
                                       ['value', 'error', 'return_nan'], 
                                       key="james_stein_handle_missing")
        model = st.selectbox("Select the model type", 
                             ['pooled', 'independent', 'binary', 'beta'], 
                             key="james_stein_model")
        randomized = st.checkbox("Add Gaussian noise to training data (to reduce overfitting)", key="james_stein_randomized")
        sigma = st.number_input("Standard deviation (sigma) for Gaussian noise", 
                                min_value=0.0, value=0.05, step=0.01, key="james_stein_sigma")
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
    
        # Button to apply James-Stein Encoder
        if st.checkbox("Apply James-Stein Encoder", key="james_stein_apply"):
            try:
    
                # Initialize the encoder
                james_stein_encoder = ce.JamesSteinEncoder(
                    verbose=verbose,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing,
                    model=model,
                    randomized=randomized,
                    sigma=sigma
                )
    
                # Fit and transform the data
                self.transformed = james_stein_encoder.fit_transform(self.data, self.data[y])
                
                # Display success message and return transformed data
                st.success("James-Stein Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_leave_one_out_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for Leave-One-Out Encoder parameters
        verbose = int(st.number_input("Enter the verbose level (Int)", value=0, min_value=0, key="leave_one_out_verbose"))
        cols = st.multiselect("Select the columns to encode", 
                              ["All Columns"] + list(self.data.columns), key="leave_one_out_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="leave_one_out_target_column")
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="leave_one_out_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)", 
                                value=True, key="leave_one_out_return_df")
        handle_unknown = st.selectbox("How to handle unknown values", 
                                       ['value', 'error', 'return_nan'], 
                                       key="leave_one_out_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values", 
                                       ['value', 'error', 'return_nan'], 
                                       key="leave_one_out_handle_missing")
        random_state = st.number_input("Random seed (optional)", min_value=0, value=42, key="leave_one_out_random_state")
        sigma = st.number_input("Standard deviation (sigma) for Gaussian noise", 
                                min_value=0.0, value=0.05, step=0.01, key="leave_one_out_sigma")
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
    
        # Button to apply Leave-One-Out Encoder
        if st.checkbox("Apply Leave-One-Out Encoder", key="leave_one_out_apply"):
            try:
    
                # Initialize the encoder
                leave_one_out_encoder = ce.LeaveOneOutEncoder(
                    verbose=verbose,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing,
                    random_state=random_state,
                    sigma=sigma
                )
    
                # Fit and transform the data
                self.transformed = leave_one_out_encoder.fit_transform(self.data, self.data[y])
    
                # Display success message and return transformed data
                st.success("Leave-One-Out Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_m_estimate_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for MEstimateEncoder parameters
        verbose = int(st.number_input("Enter the verbose level (Int)", value=0, min_value=0, key="m_estimate_verbose"))
        cols = st.multiselect("Select the columns to encode",
                              ["All Columns"] + list(self.data.columns), key="m_estimate_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="m_estimate_target_column")
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="m_estimate_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)",
                                value=True, key="m_estimate_return_df")
        handle_unknown = st.selectbox("How to handle unknown values",
                                       ['value', 'error', 'return_nan'],
                                       key="m_estimate_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                       ['value', 'error', 'return_nan'],
                                       key="m_estimate_handle_missing")
        randomized = st.checkbox("Add Gaussian noise for regularization (randomized)", key="m_estimate_randomized")
        sigma = st.number_input("Standard deviation (sigma) for Gaussian noise", min_value=0.0, value=0.05, step=0.01, key="m_estimate_sigma")
        m = st.number_input("The m parameter (higher values mean stronger smoothing)", min_value=0.0, value=1.0, step=0.1, key="m_estimate_m")
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
    
        # Button to apply M-Estimate Encoder
        if st.checkbox("Apply M-Estimate Encoder", key="m_estimate_apply"):
                # Initialize the encoder
                m_estimate_encoder = ce.MEstimateEncoder(
                    verbose=verbose,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing,
                    randomized=randomized,
                    sigma=sigma,
                    m=m
                )
    
                # Fit and transform the data
                if y!=None:
                    self.transformed = m_estimate_encoder.fit_transform(self.data, self.data[y])
                else:
                    self.transformed = m_estimate_encoder.fit_transform(self.data)
                st.success("M-Estimate Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
    def apply_one_hot_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for OneHotEncoder parameters
        verbose = int(st.number_input("Enter the verbose level (Int)", value=0, min_value=0, key="one_hot_verbose"))
        cols = st.multiselect("Select the columns to encode",
                              ["All Columns"] + list(self.data.columns), key="one_hot_columns")
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="one_hot_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)",
                                value=True, key="one_hot_return_df")
        use_cat_names = st.checkbox("Use category values as column names (may cause duplicates)", key="one_hot_use_cat_names")
        handle_unknown = st.selectbox("How to handle unknown values",
                                       ['value', 'error', 'return_nan', 'indicator'],
                                       key="one_hot_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                       ['value', 'error', 'return_nan', 'indicator'],
                                       key="one_hot_handle_missing")
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
    
        # Button to apply OneHot Encoder
        if st.checkbox("Apply OneHot Encoder", key="one_hot_apply"):
            try:
                # Initialize the encoder
                one_hot_encoder = ce.OneHotEncoder(
                    verbose=verbose,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    use_cat_names=use_cat_names,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing
                )
    
                # Fit and transform the data
                self.transformed = one_hot_encoder.fit_transform(self.data)
    
                # Display success message and return transformed data
                st.success("OneHot Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_ordinal_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for OrdinalEncoder parameters
        cols = st.multiselect("Select the columns to encode",
                              ["All Columns"] + list(self.data.columns), key="ordinal_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="ordinal_target_column")
        handle_unknown = st.selectbox("How to handle unknown values",
                                      ['value', 'error', 'return_nan'],
                                      key="ordinal_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                      ['value', 'error', 'return_nan'],
                                      key="ordinal_handle_missing")
        mapping = st.text_area("Provide the ordinal mapping for each column (in JSON format)", 
                               key="ordinal_mapping", height=150)
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
        
        # Button to apply Ordinal Encoder
        if st.checkbox("Apply Ordinal Encoder", key="ordinal_apply"):
            try:
                # Validate if the mapping is provided and is in correct JSON format
                if mapping:
                    import json
                    try:
                        mapping_dict = json.loads(mapping)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format. Please provide a valid JSON mapping.")
                        return
                else:
                    mapping_dict = None  # If no mapping is provided, the encoder will create its own
    
                # Initialize the encoder
                ordinal_encoder = ce.OrdinalEncoder(
                    cols=cols,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing,
                    mapping=mapping_dict
                )
    
                # Fit and transform the data, using y if needed
                if y is not None:
                    self.transformed = ordinal_encoder.fit_transform(self.data, self.data[y])
                else:
                    self.transformed = ordinal_encoder.fit_transform(self.data)
    
                # Display success message and return transformed data
                st.success("Ordinal Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
    
            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_polynomial_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for PolynomialEncoder parameters
        cols = st.multiselect("Select the columns to encode",
                              ["All Columns"] + list(self.data.columns), key="polynomial_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="polynomial_target_column")
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="polynomial_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)",
                                value=True, key="polynomial_return_df")
        handle_unknown = st.selectbox("How to handle unknown values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="polynomial_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="polynomial_handle_missing")
        verbose = int(st.number_input("Enter the verbosity level (Int)", value=0, min_value=0, key="polynomial_verbose"))
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
        
        # Button to apply Polynomial Encoder
        if st.checkbox("Apply Polynomial Encoder", key="polynomial_apply"):
            try:
                # Initialize the encoder
                polynomial_encoder = ce.PolynomialEncoder(
                    verbose=verbose,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing
                )
    
                # Fit and transform the data, using y if provided
                if y is not None:
                    self.transformed = polynomial_encoder.fit_transform(self.data, self.data[y])
                else:
                    self.transformed = polynomial_encoder.fit_transform(self.data)
    
                # Display success message and return transformed data
                st.success("Polynomial Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
    
            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_quantile_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for QuantileEncoder parameters
        cols = st.multiselect("Select the columns to encode",
                              ["All Columns"] + list(self.data.columns), key="quantile_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="quantile_target_column")
        quantile = st.slider("Select the quantile", 0.0, 1.0, 0.5, step=0.01, key="quantile_value")
        m = st.number_input("Enter the m parameter (shrinkage factor)", value=1.0, min_value=0.0, key="quantile_m")
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="quantile_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)",
                                value=True, key="quantile_return_df")
        handle_unknown = st.selectbox("How to handle unknown values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="quantile_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="quantile_handle_missing")
        verbose = int(st.number_input("Enter the verbosity level (Int)", value=0, min_value=0, key="quantile_verbose"))
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
        
        # Button to apply Quantile Encoder
        if st.checkbox("Apply Quantile Encoder", key="quantile_apply"):
            try:
                # Initialize the encoder
                quantile_encoder = ce.QuantileEncoder(
                    verbose=verbose,
                    quantile=quantile,
                    m=m,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing
                )
    
                # Fit and transform the data, using y if provided
                if y is not None:
                    self.transformed = quantile_encoder.fit_transform(self.data, self.data[y])
                else:
                    self.transformed = quantile_encoder.fit_transform(self.data)
    
                # Display success message and return transformed data
                st.success("Quantile Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
    
            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_rankhot_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for RankHotEncoder parameters
        cols = st.multiselect("Select the columns to encode",
                              ["All Columns"] + list(self.data.columns), key="rankhot_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="rankhot_target_column")
        verbose = int(st.number_input("Enter the verbosity level (Int)", value=0, min_value=0, key="rankhot_verbose"))
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="rankhot_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)",
                                value=True, key="rankhot_return_df")
        handle_unknown = st.selectbox("How to handle unknown values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="rankhot_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="rankhot_handle_missing")
        use_cat_names = st.checkbox("Use category names for encoded column names", key="rankhot_use_cat_names")
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
    
        # Button to apply RankHot Encoder
        if st.checkbox("Apply RankHot Encoder", key="rankhot_apply"):
            try:    
                # Initialize the encoder
                rankhot_encoder = ce.RankHotEncoder(
                    verbose=verbose,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing,
                    use_cat_names=use_cat_names
                )
    
                # Fit and transform the data, using y if provided
                if y is not None:
                    self.transformed = rankhot_encoder.fit_transform(self.data, self.data[y])
                else:
                    self.transformed = rankhot_encoder.fit_transform(self.data)
    
                # Display success message and return transformed data
                st.success("RankHot Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
    
            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_sum_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for SumEncoder parameters
        cols = st.multiselect("Select the columns to encode",
                              ["All Columns"] + list(self.data.columns), key="sumenc_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="sumenc_target_column")
        verbose = int(st.number_input("Enter the verbosity level (Int)", value=0, min_value=0, key="sumenc_verbose"))
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="sumenc_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)",
                                value=True, key="sumenc_return_df")
        handle_unknown = st.selectbox("How to handle unknown values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="sumenc_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="sumenc_handle_missing")
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
    
        # Button to apply Sum Encoder
        if st.checkbox("Apply Sum Encoder", key="sumenc_apply"):
            try:
                # Initialize the encoder
                sum_encoder = ce.SumEncoder(
                    verbose=verbose,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing
                )
    
                # Fit and transform the data, using y if provided
                if y is not None:
                    self.transformed = sum_encoder.fit_transform(self.data, self.data[y])
                else:
                    self.transformed = sum_encoder.fit_transform(self.data)
    
                # Display success message and return transformed data
                st.success("Sum Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
    
            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_summary_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for SummaryEncoder parameters
        cols = st.multiselect("Select the columns to encode",
                              ["All Columns"] + list(self.data.columns), key="sumenc_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="sumenc_target_column")
        verbose = int(st.number_input("Enter the verbosity level (Int)", value=0, min_value=0, key="sumenc_verbose"))
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="sumenc_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)",
                                value=True, key="sumenc_return_df")
        handle_unknown = st.selectbox("How to handle unknown values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="sumenc_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="sumenc_handle_missing")
        quantiles = st.text_input("Enter quantiles (e.g., [0.25, 0.75])", value="[0.25, 0.75]", key="sumenc_quantiles")
        m = float(st.number_input("Enter the m value for smoothing", value=1.0, min_value=0.0, key="sumenc_m"))
    
        # Convert quantiles input to a list of floats
        try:
            quantiles = eval(quantiles)  # Convert the input string to a list
        except Exception as e:
            st.error(f"Invalid quantiles format: {e}")
            return
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
    
        # Button to apply Summary Encoder
        if st.checkbox("Apply Summary Encoder", key="sumenc_apply"):
            try:
                summary_encoder = ce.SummaryEncoder(
                    verbose=verbose,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing,
                    quantiles=quantiles,
                    m=m
                )
    
                # Fit and transform the data, using y if provided
                if y is not None:
                    self.transformed = summary_encoder.fit_transform(self.data, self.data[y])
                else:
                    self.transformed = summary_encoder.fit_transform(self.data)
    
                # Display success message and return transformed data
                st.success("Summary Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
    
            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_target_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
            # Input widgets for TargetEncoder parameters
        cols = st.multiselect("Select the columns to encode",
                                  ["All Columns"] + list(self.data.columns), key="tenc_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="tenc_target_column")
        verbose = int(st.number_input("Enter the verbosity level (Int)", value=0, min_value=0, key="tenc_verbose"))
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="tenc_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)",
                                    value=True, key="tenc_return_df")
        handle_unknown = st.selectbox("How to handle unknown values",
                                          ['value', 'error', 'return_nan', 'indicator'],
                                          key="tenc_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                          ['value', 'error', 'return_nan', 'indicator'],
                                          key="tenc_handle_missing")
        min_samples_leaf = int(st.number_input("Enter the min_samples_leaf value", value=20, min_value=1, key="tenc_min_samples_leaf"))
        smoothing = float(st.number_input("Enter the smoothing value", value=10.0, min_value=0.0, key="tenc_smoothing"))
    
            # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
    
            # Button to apply Target Encoder
        if st.checkbox("Apply Target Encoder", key="tenc_apply"):
            try:
                target_encoder = ce.TargetEncoder(
                        verbose=verbose,
                        cols=cols,
                        drop_invariant=drop_invariant,
                        return_df=return_df,
                        handle_unknown=handle_unknown,
                        handle_missing=handle_missing,
                        min_samples_leaf=min_samples_leaf,
                        smoothing=smoothing
                )
    
                    # Fit and transform the data, using y if provided
                if y is not None:
                    self.transformed = target_encoder.fit_transform(self.data, self.data[y])
                else:
                    self.transformed = target_encoder.fit_transform(self.data)
    
                    # Display success message and return transformed data
                st.success("Target Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed
    
            except Exception as e:
                    # Handle any errors that occur
                st.error(f"An error occurred: {e}")
    def apply_woe_encoder(self):
        st.subheader("Enter the parameters and hit the checkbox to apply and store the transformation")
    
        # Input widgets for WOEEncoder parameters
        cols = st.multiselect("Select the columns to encode",
                              ["All Columns"] + list(self.data.columns), key="woe_columns")
        y = st.selectbox("Select the target column", [None] + list(self.data.columns), key="woe_target_column")
        verbose = int(st.number_input("Enter the verbosity level (Int)", value=0, min_value=0, key="woe_verbose"))
        drop_invariant = st.checkbox("Drop columns with 0 variance", key="woe_drop_invariant")
        return_df = st.checkbox("Return a Pandas DataFrame from transform (otherwise returns a NumPy array)",
                                value=True, key="woe_return_df")
        handle_unknown = st.selectbox("How to handle unknown values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="woe_handle_unknown")
        handle_missing = st.selectbox("How to handle missing values",
                                      ['value', 'error', 'return_nan', 'indicator'],
                                      key="woe_handle_missing")
        randomized = st.checkbox("Add Gaussian noise to decrease overfitting", key="woe_randomized")
        sigma = float(st.number_input("Enter the sigma value for Gaussian noise", value=0.05, min_value=0.0, key="woe_sigma"))
        regularization = float(st.number_input("Enter the regularization value", value=1.0, min_value=0.0, key="woe_regularization"))
    
        # Setting the cols parameter to None if "All Columns" is selected
        cols = None if "All Columns" in cols else cols
    
        # Button to apply WOE Encoder
        if st.checkbox("Apply WOE Encoder", key="woe_apply"):
            try:
                # Validate if 'y' is not None when columns are selected
                if y is None and cols is None:
                    st.error("Please select columns for encoding or a target column.")
                    return None

                # Check that the columns exist in the data
                if cols and not all(col in self.data.columns for col in cols):
                    st.error("One or more selected columns do not exist in the data.")
                    return None
                
                # Initialize WOEEncoder with user inputs
                woe_encoder = ce.woe.WOEEncoder(
                    verbose=verbose,
                    cols=cols,
                    drop_invariant=drop_invariant,
                    return_df=return_df,
                    handle_unknown=handle_unknown,
                    handle_missing=handle_missing,
                    randomized=randomized,
                    sigma=sigma,
                    regularization=regularization
                )
                
                # Fit and transform the data, using 'y' if provided
                if y is not None:
                    self.transformed = woe_encoder.fit_transform(self.data, self.data[y])
                else:
                    self.transformed = woe_encoder.fit_transform(self.data)
                
                # Display success message and return transformed data
                st.success("WOE Encoder applied successfully!")
                st.dataframe(self.transformed)  # Display the transformed data for preview
                return self.transformed

            except Exception as e:
                # Handle any errors that occur during the application of the encoder
                st.error(f"An error occurred: {e}")
                return None
