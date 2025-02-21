import streamlit as st
import pandas as pd

class Classification:
    def __init__(self, dataset):
        self.dataset = dataset

    def display(self):
        # Create three tabs
        tab1, tab2, tab3 = st.tabs(["Perform Operations", "View Operations", "Delete Operations"])

        # Content for Perform Operations tab
        with tab1:
            # Create two columns with a ratio of 1:2
            col1, col2 = st.columns([1, 2],border=True)
            operation = col1.radio("Select Operation", ["Train Test Split", "Classifiers"])
            if operation == "Train Test Split":
                col2.write("You selected Train Test Split")
            elif operation == "Classifiers":
                col2.write("You selected Classifiers")

        # Content for View Operations tab
        with tab2:
            st.subheader("View All Of Your Operations Result Here:",divider='blue')
            st.dataframe(self.dataset)

        # Content for Delete Operations tab
        with tab3:
            st.write("Delete Operations content goes here")
