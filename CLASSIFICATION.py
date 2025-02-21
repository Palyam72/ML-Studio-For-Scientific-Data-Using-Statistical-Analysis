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
            col1,col2=st.columns([1,2],border=True)
            col1.subheader("select The View Mode",divider='blue')
            options=col1.radio("Options",["View Data Frame","View Missing Information"])
            if options=="View Data Frame":
                col2.dataframe(self.dataset)
            if option=="View Missing Information":
                pass

        # Content for Delete Operations tab
        with tab3:
            st.write("Delete Operations content goes here")
