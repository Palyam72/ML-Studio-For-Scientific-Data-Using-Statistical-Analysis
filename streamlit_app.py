import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from DATAREADERS import DataExtractor
import missingno as mso
import matplotlib.pyplot as plt
from DATACLEANERS import PandasMethods, UnivariateImputers, OutliersTreatment  # Import OutliersTreatment

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
            "Data Upload", "Data Cleaning", "Identify the Most Important Features",
            "Create New Features", "Encode Categorical Data",
            "Normalize or Scale the Features", "Exploratory Data Analysis",
            "Splitting the Data", "Model Selection and Training",
            "Model Evaluation", "Model Download"
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

            # Display the selected dataset
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
