import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from DATAREADERS import DataExtractor
import missingno as mso
import matplotlib.pyplot as plt
from DATACLEANERS import PandasMethods

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

            # Create PandasMethods object for the selected dataset
            pm = PandasMethods(dataset)

            # Display missing matrix chart
            st.divider()
            st.markdown("### Missing Value Matrix")
            fig, ax = plt.subplots(figsize=(10, 5))
            mso.matrix(dataset, ax=ax)
            st.pyplot(fig)
            st.markdown("### Missing Value HeatMap")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            mso.heatmap(dataset, ax=ax1)
            st.pyplot(fig1)

            st.divider()

            # Layout for cleaning options
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("<p style='color:blue;'>Handle Missing Values </p>",unsafe_allow_html=True)
                apply_backward_fill = st.checkbox("Apply Backward Fill")
                apply_forward_fill = st.checkbox("Apply Forward Fill")
                drop_nan_rows = st.checkbox("Drop Rows with NaN")
                fill_with_value = st.checkbox("Fill Missing Values with a Particular Value")
                interpolate_values = st.checkbox("Interpolate Missing Values")
                st.divider()
                st.markdown("<p style='color:blue;'>Handle Outliers </p>",unsafe_allow_html=True)
                outlier_visualize = st.checkbox("Visualize outliers")
                remove_outliers = st.checkbox("Remove Outliers")
                st.divider()
                st.markdown("<p style='color:blue;'>Remove Duplicates </p>",unsafe_allow_html=True)
                remove_duplicates = st.checkbox("Remove Duplicate Rows")
                st.divider()
                st.markdown("<p style='color:blue;'>Convert dtypes + drop misssing values + standadize column names </p>",unsafe_allow_html=True)
                convert_dtypes = st.checkbox("Convert dtypes + drop misssing values + standadize colmn names")

            with col2:
                # Apply Backward Fill
                if apply_backward_fill:
                    result = pm.backward_fill()
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2 : DATA CLEANING OUTPUT - BFILL"] = result
                        st.success("Backward Fill applied successfully!")

                # Apply Forward Fill
                if apply_forward_fill:
                    result = pm.forward_fill()
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2 : DATA CLEANING OUTPUT - FFILL"] = result
                        st.success("Backward Fill applied successfully!")

                # Drop NaN Rows
                if drop_nan_rows:
                    st.subheader("Drop Rows with NaN")
                    result = pm.drop_na()
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2 : DATA CLEANING OUTPUT - DROP_NAN"] = result
                        st.success("Rows with NaN dropped successfully!")

                # Fill Missing Values
                if fill_with_value:
                    result = pm.fill_na()
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2 : DATA CLEANING OUTPUT - FILL_VALUES"] = result
                        st.success("Missing values filled successfully!")

                # Interpolate Missing Values
                if interpolate_values:
                    st.subheader("Interpolate Missing Values")
                    method = st.selectbox("Select Interpolation Method", ['linear', 'polynomial', 'cubic', 'nearest'])
                    result = pm.interpolate_missing_values(method=method, inplace=False)
                    if result is not None:
                        st.dataframe(result)
                        st.session_state.availableDatasets["STAGE 2 : DATA CLEANING OUTPUT - INTERPOLATE"] = result
                        st.success("Missing values interpolated successfully!")

    else:
        st.warning("No datasets available. Please upload data first.")
