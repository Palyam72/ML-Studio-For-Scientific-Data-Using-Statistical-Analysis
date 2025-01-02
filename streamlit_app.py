import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from DATAREADERS import DataExtractor
import missingno as mso
import matplotlib.pyplot as plt
from DATACLEANERS import PandasMethods, UnivariateImputers, OutliersTreatment  # Import OutliersTreatment
from FEATURE_SELECTION import *
from ENCODERS import *
from CHANGERS import *
from REGRESSION import *
# Initialize DataExtractor object
dataextractor = DataExtractor()

# Initialize session variables if not already present
for key in ["availableDatasets", "selected_dataset"]:
    if key not in st.session_state:
        st.session_state[key] = {}

# Title of the app
st.title("Data Science Workflow Application")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=[
            "Data Upload", "Data Cleaning", "Identify & Select the Most Important Features",
            "Create New Features", "Encode Categorical Data",
            "Normalize or Scale the Features", "Exploratory Data Analysis",
            "Regression", "Classification",
            "Clustering", "Model Download"
        ],
        icons=[
            "cloud-upload", "eraser", "star", "plus-circle", "code",
            "arrow-down-up", "bar-chart", "scissors", "gear", "check-circle", "download"
        ],
        menu_icon="cast",
        default_index=0,
    )

# Function to handle file uploads
def upload_file(data_type, upload_func, file_type=None):
    file = st.file_uploader(f"Upload a {data_type} file", type=file_type, key=data_type)
    if file:
        data = upload_func(file)
        if data is not None:
            dataset_name = f"STAGE-1 : DATA UPLOAD {data_type} Dataset"
            st.session_state.availableDatasets[dataset_name] = data
            st.success(f"{data_type} file uploaded successfully!")
            st.dataframe(data.head())
        
# Data Upload Section
if selected == "Data Upload":
    col1, col2 = st.columns([1, 2])

    with col1:
        readCsv = st.checkbox("Read Data From CSV")
        readExcel = st.checkbox("Read Data From Excel")
        readJson = st.checkbox("Read Data From JSON")
        readHTML = st.checkbox("Read Data From HTML")

    with col2:
        if readCsv:
            upload_file("CSV", dataextractor.readCsv, ["csv"])
        if readExcel:
            upload_file("Excel", dataextractor.readExcel, ["xls", "xlsx"])
        if readJson:
            st.session_state.readJson = dataextractor.readJson()
            if st.session_state.readJson is not None:
                st.session_state.availableDatasets["JSON Dataset"] = st.session_state.readJson
        if readHTML:
            st.session_state.readHTML = dataextractor.readHTML()
            if st.session_state.readHTML is not None:
                st.session_state.availableDatasets["HTML Dataset"] = st.session_state.readHTML

# Data Cleaning Section
elif selected == "Data Cleaning":
    if st.session_state.availableDatasets:
        # Dataset selection for cleaning
        selected_dataset_name = st.selectbox(
            "Select a dataset to clean",
            list(st.session_state.availableDatasets.keys())
        )

    if selected_dataset_name:
        # Load the selected dataset from session state
        st.session_state.selected_dataset = st.session_state.availableDatasets[selected_dataset_name]
        dataset = st.session_state.selected_dataset
        st.markdown("### Selected Dataset")
        st.dataframe(dataset)
        # Display missing value charts
        st.divider()
        st.markdown("### Missing Value Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        mso.matrix(st.session_state.selected_dataset, ax=ax)
        st.pyplot(fig)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        mso.heatmap(st.session_state.selected_dataset, ax=ax1)
        st.pyplot(fig1)

        # Layout for cleaning options
        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            # Headings for cleaning options
            st.markdown("<p style='color:blue;'>Handle Missing Values</p>", unsafe_allow_html=True)
            apply_backward_fill = st.checkbox("Backward Fill")
            apply_forward_fill = st.checkbox("Forward Fill")
            drop_nan_rows = st.checkbox("Drop Rows with NaN")
            fill_with_value = st.checkbox("Fill Missing Values with a Particular Value")
            interpolate_values = st.checkbox("Interpolate Missing Values")
            st.divider()

            st.markdown("<p style='color:blue;'>Univariate Missing Data Imputers</p>", unsafe_allow_html=True)
            mean_median_imputer = st.checkbox("Mean/Median Imputer")
            end_tail_imputer = st.checkbox("End Tail Imputer")
            random_sample_imputer = st.checkbox("Random Sample Imputer")
            add_missing_indicator = st.checkbox("Add Missing Indicator")
            st.divider()

            # Headings for Outlier Treatment
            st.markdown("<p style='color:blue;'>Outlier Treatment</p>", unsafe_allow_html=True)
            visualize_outliers=st.checkbox("Visualize outliers")
            apply_outlier_treatment = st.checkbox("Apply Outlier Treatment")
                
            st.divider()

        with col2:
            pm = PandasMethods(st.session_state.selected_dataset)
            uim = UnivariateImputers(st.session_state.selected_dataset)
            ot = OutliersTreatment(st.session_state.selected_dataset)  # Initialize OutliersTreatment

            # Apply Backward Fill
            if apply_backward_fill:
                result = pm.backward_fill()
                if st.checkbox("Proceed with Backward Fill", key="confirm_bfill"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: DATA CLEANING OUTPUT - BFILL"] = result
                        st.success("Backward Fill applied successfully!")

            # Apply Forward Fill
            if apply_forward_fill:
                result = pm.forward_fill()
                if st.checkbox("Proceed with Forward Fill", key="confirm_ffill"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: DATA CLEANING OUTPUT - FFILL"] = result
                        st.success("Forward Fill applied successfully!")

            # Drop Rows with NaN
            if drop_nan_rows:
                result = pm.drop_na()
                if st.checkbox("Proceed with Drop NaN Rows", key="confirm_dropna"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: DATA CLEANING OUTPUT - DROP_NAN"] = result
                        st.success("Rows with NaN dropped successfully!")

            # Fill Missing Values
            if fill_with_value:
                result = pm.fill_na()
                if st.checkbox("Proceed with Fill Missing Values", key="confirm_fillna"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: DATA CLEANING OUTPUT - FILL_VALUES"] = result
                        st.success("Missing values filled successfully!")

            # Interpolate Missing Values
            if interpolate_values:
                result = pm.interpolate_missing_values()
                if st.checkbox("Proceed with Interpolation", key="confirm_interpolate"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: DATA CLEANING OUTPUT - INTERPOLATE"] = result
                        st.success("Missing values interpolated successfully!")

            # Apply Mean/Median Imputer
            if mean_median_imputer:
                result = uim.MeanMedianImputer()
                if st.checkbox("Proceed with Mean/Median Imputation", key="confirm_meanmedian"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: UNIVARIATE IMPUTATION - MEAN/MEDIAN"] = result
                        st.success("Mean/Median imputation applied successfully!")

            # Apply End Tail Imputer
            if end_tail_imputer:
                result = uim.EndTailImputer()
                if st.checkbox("Proceed with End Tail Imputation", key="confirm_endtail"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: UNIVARIATE IMPUTATION - END TAIL"] = result
                        st.success("End Tail imputation applied successfully!")

            # Apply Random Sample Imputer
            if random_sample_imputer:
                result = uim.RandomSampleImputer()
                if st.checkbox("Proceed with Random Sample Imputation", key="confirm_randomsample"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: UNIVARIATE IMPUTATION - RANDOM SAMPLE"] = result
                        st.success("Random Sample imputation applied successfully!")

                # Add Missing Indicator
            if add_missing_indicator:
                result = uim.AddMissingIndicator()
                if st.checkbox("Proceed with Adding Missing Indicator", key="confirm_missingindicator"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: UNIVARIATE IMPUTATION - MISSING INDICATOR"] = result
                        st.success("Missing indicator added successfully!")

            # Apply Outlier Treatment
            if apply_outlier_treatment:
                result = ot.apply_outlier_treatment()
                if st.checkbox("Proceed with Outlier Treatment", key="confirm_outlier_treatment"):
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2: OUTLIER TREATMENT"] = result
                        st.success("Outlier treatment applied successfully!")
            if visualize_outliers:
                ot.visualize_outliers()
    else:
        st.warning("No datasets available. Please upload data first.")
elif selected == "Identify & Select the Most Important Features":
    if st.session_state.availableDatasets:
        # Dataset selection for cleaning
        selected_dataset_name = st.selectbox(
            "Select a dataset to clean",
            list(st.session_state.availableDatasets.keys())
        )

    if selected_dataset_name:
        # Load the selected dataset from session state
        st.session_state.selected_dataset = st.session_state.availableDatasets[selected_dataset_name]
        dataset = st.session_state.selected_dataset
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.info("CORRELATION ANALYSIS")
            pearson = st.checkbox("Pearson correlation")
            spearman = st.checkbox("Spearman rank correlation")
            kendall = st.checkbox("Kendall tau correlation")
            point = st.checkbox("Point biserial corelation")
            cramers = st.checkbox("Cramers V correlation")
            st.info("Varience Threshold Method")
            variance = st.checkbox("Varience Threshold")
            st.info("Univariate Selections")
            generic_univariate_select=st.checkbox("Generic Univariate Select")
            select_fdr=st.checkbox("Select FDR")
            select_fpr=st.checkbox("Select FPR")
            select_k_best=st.checkbox("Select K Best")
            select_fwe=st.checkbox("Select FWE")
            select_percentile=st.checkbox("Select Percentile")
            st.info("Create Your Final DataFrame")
            drop_duplicate=st.checkbox("Drop Duplicate Features")
            drop_features=st.checkbox("Drop Features")
            drop_constant=st.checkbox("Drop Constant & Quai Constant Features")
            drop_correlated=st.checkbox("Drop Correlated Featiures")
            drop_high_psi=st.checkbox("Drop HIGH PSI Features")
            select_by_information_value=st.checkbox("Selet Features By Information Values")
            select_by_shuffling=st.checkbox("Select Features By Shuffling")
            select_by_target_mean=st.checkbox("Select features By Target Mean Performance")
            select_by_single_feature = st.checkbox("Select By Single Feature Performance")
            recursive_feature_elimination=st.checkbox("Recursive Feature Elimination")
            recursive_feature_addition=st.checkbox("Recursive Feature Addition")
            select_by_mrmr=st.checkbox("select_by_mrmr")
            
        with col2:
            fe=FeatureSelection(st.session_state.selected_dataset)
            stats=StatisticalFunctions(st.session_state.selected_dataset)
            features=FinalDataSet(st.session_state.selected_dataset)
            if pearson:
                fe.pearson()
            if spearman:
                fe.spearman()
            if kendall:
                fe.kendall()
            if point:
                fe.point()
            if cramers:
                fe.cramers()
            if variance:
                fe.variance_threshold()
            if generic_univariate_select:
                obtainedValue=stats.generic_univariate_select()
            if select_fdr:
                obtainedValue=stats.select_fdr()
            if select_fpr:
                obtainedValue=stats.select_fpr()
            if select_fwe:
                obtainedValue=stats.select_fwe()
            if select_k_best:
                obtainedValue=stats.select_k_best()
            if select_percentile:
                obtainedValue=stats.select_percentile()
            if drop_features:
                obtainedData=features.drop_features()
                st.session_state.availableDatasets["drop_features"]=obtainedData
            if drop_duplicate:
                obtainedData=features.drop_duplicate_features()
                st.session_state.availableDatasets["drop_duplicate_features"]=obtainedData
            if drop_constant:
                obtainedData=features.drop_constant_features()
                st.session_state.availableDatasets["drop_constant_features"]=obtainedData
            if drop_correlated:
                obtainedData=features.drop_correlated_features()
                st.session_state.availableDatasets["drop_correlated_features"]=obtainedData
            if drop_high_psi:
                obtainedData=features.drop_features()
                st.session_state.availableDatasets["drop_correlated_features"]=obtainedData
            if select_by_information_value:
                obtainedData=features.select_by_information_value()
                st.session_state.availableDatasets["select_by_information_value"]=obtainedData
            if select_by_shuffling:
                obtainedData=features.select_by_shuffling()
                st.session_state.availableDatasets["select_by_shuffling"]=obtainedData
            if select_by_target_mean:
                obtainedData=features.drop_features()
                st.session_state.availableDatasets[""]=obtainedData
            if select_by_single_feature:
                obtainedData=features.select_by_single_feature_performance()
                st.session_state.availableDatasets["select_by_single_feature_performance"]=obtainedData
            if select_by_mrmr:
                obtainedData=features.select_by_mrmr()
                st.session_state.availableDatasets["select_by_mrmr"]=obtainedData
            if recursive_feature_elimination:
                obtainedData=features.recursive_feature_elimination()
                st.session_state.availableDatasets["recursive_feature_elimination"]=obtainedData
            if recursive_feature_addition:
                obtainedData=features.recursive_feature_addition()
                st.session_state.availableDatasets["recursive_feature_addition"]=obtainedData
 
                
    else:
        st.warning("Please give the data first")
elif selected=="Encode Categorical Data":
    if st.session_state.availableDatasets:
        # Dataset selection for cleaning
        selected_dataset_name = st.selectbox(
            "Select a dataset to clean",
            list(st.session_state.availableDatasets.keys())
        )

    if selected_dataset_name:
        # Load the selected dataset from session state
        st.session_state.selected_dataset = st.session_state.availableDatasets[selected_dataset_name]
        dataset = st.session_state.selected_dataset
        st.markdown("### Selected Dataset")
        st.dataframe(dataset)
        # Display missing value charts
        st.divider()
        st.markdown("### Missing Value Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        mso.matrix(st.session_state.selected_dataset, ax=ax)
        st.pyplot(fig)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        mso.heatmap(st.session_state.selected_dataset, ax=ax1)
        st.pyplot(fig1)
        # layout for encoders
        col1,col2=st.columns([1,2])
        encoders=Encoders(st.session_state.selected_dataset)
        encoder_methods = {
            "TargetEncoder": encoders.apply_target_encoder,
            "WOEEncoder": encoders.apply_woe_encoder,
            "SummaryEncoder": encoders.apply_summary_encoder,
            "SumEncoder": encoders.apply_sum_encoder,
            "RankhotEncoder": encoders.apply_rankhot_encoder,
            "QuantileEncoder": encoders.apply_quantile_encoder,
            "PolynomialEncoder": encoders.apply_polynomial_encoder,
            "OrdinalEncoder": encoders.apply_ordinal_encoder,
            "OneHotEncoder": encoders.apply_one_hot_encoder,
            "LeaveOneOutEncoder": encoders.apply_leave_one_out_encoder,
            "MEstimateEncoder": encoders.apply_m_estimate_encoder,
            "JamesSteinEncoder": encoders.apply_james_stein_encoder,
            "HelmertEncoder": encoders.apply_helmert_encoder,
            "HashingEncoder": encoders.apply_hashing_encoder,
            "GrayEncoder": encoders.apply_gray_encoder,
            "GeneralizedLinearMixedModel": encoders.generalized_linear_mixed_model,
            "CountEncoder": encoders.apply_count_encoder,
            "CatboostEncoder": encoders.apply_catboost_encoder,
            "BinaryEncoder": encoders.apply_binary_encoder,
            "BasenEncoder": encoders.apply_basen_encoder
        }
        with col1:
            st.subheader("Encoder Techniques")
            for i in encoder_methods.keys():
                if st.checkbox(i):
                    with col2:
                        output=encoder_methods[i]()
                        st.session_state.availableDatasets[i]=output
elif selected=="Normalize or Scale the Features":
    if st.session_state.availableDatasets:
        # Dataset selection for cleaning
        selected_dataset_name = st.selectbox(
            "Select a dataset to clean",
            list(st.session_state.availableDatasets.keys())
        )

    if selected_dataset_name:
        # Load the selected dataset from session state
        st.session_state.selected_dataset = st.session_state.availableDatasets[selected_dataset_name]
        dataset = st.session_state.selected_dataset
        st.markdown("### Selected Dataset")
        st.dataframe(dataset)
        # Display missing value charts
        st.divider()
        st.markdown("### Missing Value Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        mso.matrix(st.session_state.selected_dataset, ax=ax)
        st.pyplot(fig)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        mso.heatmap(st.session_state.selected_dataset, ax=ax1)
        st.pyplot(fig1)
        methods=PreprocessingMethods(st.session_state.selected_dataset)
        discretizers = Descritizers(st.session_state.selected_dataset)
        

        # Create a dictionary mapping technique names to methods
        col1,col2=st.columns([1,2])                
        with col1:
            discretizer_methods = {
                'Equal Width Discretiser': discretizers.equal_width_discretiser,
                'Equal Frequency Discretiser': discretizers.equal_frequency_discretiser,
                'Decision Tree Discretiser': discretizers.decision_tree_discretiser,
                'Geometric Width Discretiser': discretizers.geometric_width_discretiser
            }
            normalizers = {
                "normalizer": methods.normalizer,
                "binarizer": methods.binarizer,
                "label_binarizer": methods.label_binarizer,
                "multi_label_binarizer": methods.multi_label_binarizer,
            
            
            }

            st.subheader("Descritizers")
            st.divider()
            for i in discretizer_methods.keys():
                if st.checkbox(i):
                    with col2:
                        dataframe=discretizer_methods[i]()
                        st.session_state.availableDatasets[i]=dataframe
            st.subheader("Normalizers")
            st.divider()
            for i in normalizers.keys():
                if st.checkbox(i):
                    with col2:
                        dataframe=normalizers[i]()
                        st.session_state.availableDatasets[i]=dataframe
                
elif selected=="Regression":
    if st.session_state.availableDatasets:
        # Dataset selection for cleaning
        selected_dataset_name = st.selectbox(
            "Select a dataset to clean",
            list(st.session_state.availableDatasets.keys())
        )

    if selected_dataset_name:
        # Load the selected dataset from session state
        st.session_state.selected_dataset = st.session_state.availableDatasets[selected_dataset_name]
        dataset = st.session_state.selected_dataset
        st.markdown("### Selected Dataset")
        st.dataframe(dataset)
        # Display missing value charts
        st.divider()
        st.markdown("### Missing Value Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        mso.matrix(st.session_state.selected_dataset, ax=ax)
        st.pyplot(fig)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        mso.heatmap(st.session_state.selected_dataset, ax=ax1)
        st.pyplot(fig1)
        regression=Regression(st.session_state.selected_dataset)
        regression.display()
        
        
            
                        
        
