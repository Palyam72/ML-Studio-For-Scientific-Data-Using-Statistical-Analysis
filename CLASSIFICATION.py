import streamlit as st

class Classification:
    def __init__(self, dataset):
        self.dataset = dataset

    def display(self):
        # Create three tabs
        tab1, tab2, tab3 = st.tabs(["Perform Operations", "View Operations", "Delete Operations"])

        # Content for Perform Operations tab
        with tab1:
            # Create two columns with a ratio of 1:2
            col1, col2 = st.columns([1, 2])

            # Add a radio button in the first column
            with col1:
                operation = st.radio("Select Operation", ["Train Test Split", "Classifiers"])

            # You can add more content to col2 as needed
            with col2:
                if operation == "Train Test Split":
                    st.write("You selected Train Test Split")
                    # Add more content related to Train Test Split
                elif operation == "Classifiers":
                    st.write("You selected Classifiers")
                    # Add more content related to Classifiers

        # Content for View Operations tab
        with tab2:
            st.write("View Operations content goes here")

        # Content for Delete Operations tab
        with tab3:
            st.write("Delete Operations content goes here")
