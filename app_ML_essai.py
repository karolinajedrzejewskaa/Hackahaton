import streamlit as st
import pandas as pd
import pickle
import numpy as np

# File paths
original_data_path = '/Users/lua/wild/Hackathon/jobs_in_data.csv'
cleaned_data_path = '/Users/lua/wild/Hackathon/tableau_nettoye.csv'
model_path = '/Users/lua/wild/Hackathon/model.pkl'
encoders_path = '/Users/lua/wild/Hackathon/encoders.pkl'

# Load data
df = pd.read_csv(cleaned_data_path)

# Load your pre-trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load your pre-trained encoders
with open(encoders_path, 'rb') as file:
    encoders = pickle.load(file)

# Function to encode input data
def encode_input_data(input_data, encoders):
    input_data['work_setting'] = encoders['work_setting'].transform(input_data[['work_setting']])
    input_data['experience_level'] = encoders['experience_level'].transform(input_data[['experience_level']])
    input_data['work_year'] = encoders['work_year'].transform(input_data[['work_year']])
    input_data['company_size'] = encoders['company_size'].transform(input_data[['company_size']])
    input_data['job_title_encoded'] = encoders['job_title'].transform(input_data[['job_title']])
    input_data.drop('job_title', axis=1, inplace=True)
    return input_data

# Initialize session state
if 'view' not in st.session_state:
    st.session_state.view = 'form'

# Streamlit app

st.sidebar.header('Menu')

# Navigation buttons
if st.sidebar.button('Dashboard'):
    st.session_state.view = 'dashboard'

job_titles = df['job_title'].unique()

# Form for input details in an expandable section
with st.sidebar.expander('Prediction App'):
    with st.form(key='salary_form'):
        job_title = st.selectbox('Job Title', job_titles)
        experience_level = st.selectbox('Experience level', ['Entry-level', 'Mid-level', 'Senior', 'Executive'])
        work_setting = st.selectbox('Work setting', ['Remote', 'Hybrid', 'In-person'])
        company_size = st.selectbox('Company size', ['L', 'M', 'S'])
        work_year = st.selectbox('Year', [2023, 2022, 2021, 2020])
        
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.session_state.view = 'prediction'
            st.session_state.job_title = job_title
            st.session_state.experience_level = experience_level
            st.session_state.work_setting = work_setting
            st.session_state.company_size = company_size
            st.session_state.work_year = work_year

# Prediction view
if st.session_state.view == 'prediction':
    st.title('Salary Prediction App')
    st.write('## Prediction Results')
    st.write('You selected:')
    st.write(f'Job Title: {st.session_state.job_title}')
    st.write(f'Experience level: {st.session_state.experience_level}')
    st.write(f'Work setting: {st.session_state.work_setting}')
    st.write(f'Company size: {st.session_state.company_size}')
    st.write(f'Year: {st.session_state.work_year}')

    # Prepare the input data for the model
    input_data = pd.DataFrame({
        'work_setting': [st.session_state.work_setting],
        'experience_level': [st.session_state.experience_level],
        'work_year': [st.session_state.work_year],
        'company_size': [st.session_state.company_size],
        'job_title': [st.session_state.job_title]
    })

    # Encode the input data
    input_data_encoded = encode_input_data(input_data, encoders)

    # Make the prediction
    prediction = model.predict(input_data_encoded)
    predicted_salary = np.expm1(prediction[0])  # Assuming the target variable was log-transformed
    st.write(f'Predicted Salary: ${predicted_salary:,.2f}')
    
    # Filter data based on user input
    filtered_data = df[(df['job_title'] == st.session_state.job_title) & 
                      (df['experience_level'] == st.session_state.experience_level) & 
                      (df['work_setting'] == st.session_state.work_setting) & 
                      (df['company_size'] == st.session_state.company_size) & 
                      (df['work_year'] == st.session_state.work_year)]
    
    if not filtered_data.empty:
        st.write('Filtered Data:')
        st.dataframe(filtered_data)
    else:
        st.write('No data available for the selected criteria.')

# Dashboard view
if st.session_state.view == 'dashboard':
    st.write('## Dashboard')
    
    # Example visualizations
    st.write('### Salary Distribution')
    st.bar_chart(df['salary_in_usd'])
    
    st.write('### Average Salary by Job Title')
    avg_salary_by_job = df.groupby('job_title')['salary_in_usd'].mean().reset_index()
    st.bar_chart(avg_salary_by_job.set_index('job_title'))

# Display the sample data as a table
st.write('Sample Data for Visualization:')
st.dataframe(df)
