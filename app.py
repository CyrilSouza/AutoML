import streamlit as st
from streamlit import file_util
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models, pull as classification_pull, save_model as classification_save_model, load_model as classification_load_model
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models, pull as regression_pull, save_model as regression_save_model, load_model as regression_load_model
import base64

def check_dataset_type(df, target_column):
    # Check data type of target variable
    target_dtype = df[target_column].dtype
    
    if pd.api.types.is_numeric_dtype(target_dtype):
        # If target variable is numeric, it's likely a regression problem
        return "Regression"
    else:
        # If target variable is not numeric, count unique values to determine classification problem
        num_unique_values = df[target_column].nunique()
        if num_unique_values > 2:
            return "Classification - Multi-Class"
        else:
            return "Classification - Binary"

# Initialize the df variable in the session state
if 'df' not in st.session_state:
    st.session_state['df'] = None

def main():
    with st.sidebar:
        st.title("AutoML")
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download Pickle"])
        st.info("This project application helps you build and explore your data.")
    
    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file:
            # Update the df variable in the session state with the uploaded dataset
            st.session_state['df'] = pd.read_csv(file, index_col=None)
            st.session_state['df'].to_csv('uploaded_dataset.csv', index=None)
            st.dataframe(st.session_state['df'])
        elif st.session_state['df'] is not None:
            # Display the dataset if it has already been uploaded
            st.dataframe(st.session_state['df'])
        else:
            # Add a button to clear the current dataset
            if st.button('Clear Dataset'):
                st.session_state['df'] = None

        if st.session_state['df'] is not None:
            target_column = st.selectbox("Select the target column:", options=st.session_state['df'].columns)
            if st.button("Check Dataset Type"):
                result = check_dataset_type(st.session_state['df'], target_column)
                st.write(f"**Result:** {result}")
        else:
            st.warning("Please upload a dataset first.")

    if choice == "Profiling":
        if st.session_state['df'] is not None:
            st.title("Exploratory Data Analysis")
            profile_df = generate_profile_report(st.session_state['df'])
            st_profile_report(profile_df)

            # Add button to download profiling report as PDF
            if st.button('Download Profile Report as PDF'):
                download_profile_report_as_html(profile_df, 'profile_report.html')

        else:
            st.warning("Please upload a dataset first.")

    if choice == "Modelling":
        if st.session_state['df'] is not None:
            target_column = st.selectbox("Select the target column:", options=st.session_state['df'].columns)

            model_type = st.radio("Select Model Type:", ["Regression", "Classification"])
            if model_type == "Regression":
                setup_function = regression_setup
                compare_models_function = regression_compare_models
                pull_function = regression_pull
                save_model_function = regression_save_model
                load_model_function = regression_load_model
            else:
                setup_function = classification_setup
                compare_models_function = classification_compare_models
                pull_function = classification_pull
                save_model_function = classification_save_model
                load_model_function = classification_load_model

            if st.button('Run Modelling'):
                setup_df, compare_df = generate_modeling_results(st.session_state['df'], target_column, setup_function, compare_models_function, pull_function, save_model_function)
                st.dataframe(setup_df)
                st.dataframe(compare_df)
        else:
            st.warning("Please upload a dataset first.")

    if choice == "Download Pickle":
        if st.button('Download Model'):
            st.markdown(get_binary_file_downloader_html('best_model.pkl', 'Model Download'), unsafe_allow_html=True)


@st.cache_data
def generate_profile_report(df):
    return df.profile_report(config_file="")

def download_profile_report_as_html(profile_report, file_name):
    # Save the report to HTML
    html_data = profile_report.to_html()

    # Send the HTML file for download
    html_bytes = html_data.encode()
    st.markdown(get_binary_file_downloader_html_profile(html_bytes, file_name, 'Download Profile Report as HTML'), unsafe_allow_html=True)


@st.cache_data
def generate_modeling_results(df, target_column, _setup_function, _compare_models_function, _pull_function, _save_model_function):
    _setup_function(df, target=target_column, verbose=False)
    setup_df = _pull_function()
    best_model = _compare_models_function()
    compare_df = _pull_function()
    _save_model_function(best_model, 'best_model')  # Save the model to a file
    return setup_df, compare_df

def get_binary_file_downloader_html_profile(bin_data, file_name, button_label='Download'):
    bin_str = base64.b64encode(bin_data).decode()
    href = f'<a href="data:text/html;base64,{bin_str}" download="{file_name}">{button_label}</a>'
    return href

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    return href

if __name__ == "__main__":
    main()
