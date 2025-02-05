import pandas as pd
import chardet
import requests
import streamlit as st

class DataExtractor:
    def __init__(self):
        self.dataset = None

    def detect_encoding(self, file):
        """Detect encoding of a file using chardet."""
        raw_data = file.read(1024)
        file.seek(0)  # Reset the cursor to the beginning
        result = chardet.detect(raw_data)
        encoding = result.get('encoding', 'utf-8')
        return encoding

    def readCsv(self, file):
        """Read a CSV file with the detected encoding."""
        encoding = self.detect_encoding(file)
        try:
            df = pd.read_csv(file, encoding=encoding)
            self.dataset = df
            st.success("CSV file successfully read!")
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None

    def readExcel(self, file):
        """Read an Excel file."""
        try:
            df = pd.read_excel(file)
            self.dataset = df
            st.success("Excel file successfully read!")
            return df
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return None

    def readJson(self):
        """Read JSON data from a file or a URL."""
        col1, col2 = st.columns([1, 2])
        df = None
        with col1:
            option = st.radio("Choose an option to extract JSON data:", ["From File", "From URL"])
        with col2:
            if option == "From File":
                file = st.file_uploader("Upload a JSON file:", type=["json"])
                if file is not None:
                    try:
                        df = pd.read_json(file)
                        self.dataset = df
                        st.success("JSON file successfully read!")
                        st.dataframe(df.head())
                    except Exception as e:
                        st.error(f"Error reading JSON file: {e}")
            elif option == "From URL":
                url = st.text_input("Enter the URL to extract JSON data:")
                if url:
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        json_data = response.json()
                        df = pd.json_normalize(json_data)
                        self.dataset = df
                        st.success("JSON data successfully read from URL!")
                        st.dataframe(df.head())
                    except Exception as e:
                        st.error(f"Error reading JSON from URL: {e}")
        return df

    def readHTML(self):
        """Read HTML file either from URL or uploaded .txt file."""
        col1, col2 = st.columns([1, 2])
        df = None
        with col1:
            option = st.radio("Choose an option:", ["Extract DataFrame from URL", "Extract DataFrame from .txt"])
        with col2:
            if option == "Extract DataFrame from URL":
                url = st.text_input("Enter the URL to extract HTML data:")
                if url:
                    try:
                        dfs = pd.read_html(url)
                        st.success(f"Found {len(dfs)} tables in the HTML!")
                        for i, table in enumerate(dfs):
                            st.write(f"Table {i + 1}:")
                            st.dataframe(table.head())
                        choice = st.number_input("Enter the table number to use:", min_value=1, max_value=len(dfs))
                        df = dfs[int(choice) - 1]
                        self.dataset = df
                    except Exception as e:
                        st.error(f"Error reading HTML from URL: {e}")
            elif option == "Extract DataFrame from .txt":
                file = st.file_uploader("Upload a .txt file containing HTML data:", type=["txt"])
                if file is not None:
                    try:
                        html_content = file.read().decode("utf-8")
                        dfs = pd.read_html(html_content)
                        st.success(f"Found {len(dfs)} tables in the HTML!")
                        for i, table in enumerate(dfs):
                            st.write(f"Table {i + 1}:")
                            st.dataframe(table.head())
                        choice = st.number_input("Enter the table number to use:", min_value=1, max_value=len(dfs))
                        df = dfs[int(choice) - 1]
                        self.dataset = df
                    except Exception as e:
                        st.error(f"Error reading HTML from .txt file: {e}")
        return df
    def makeData(self):
        pass
