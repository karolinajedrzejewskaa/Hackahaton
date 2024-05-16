import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px 
import requests
from io import BytesIO

# File paths
original_data_path = 'https://raw.githubusercontent.com/LuaGeo/hackathon/main/jobs_in_data.csv'
cleaned_data_path = 'https://raw.githubusercontent.com/LuaGeo/hackathon/main/tableau_nettoye.csv'
model_path = 'https://raw.githubusercontent.com/LuaGeo/hackathon/main/model.pkl'
encoders_path = 'https://raw.githubusercontent.com/LuaGeo/hackathon/main/encoders.pkl'

# Load data
df = pd.read_csv(cleaned_data_path)
original_df = pd.read_csv(original_data_path)


# Function to load pickled files from GitHub
def load_pickle(url):
    response = requests.get(url)
    return pickle.load(BytesIO(response.content))

# Load your pre-trained model
model = load_pickle(model_path)

# Load your pre-trained encoders
encoders = load_pickle(encoders_path)

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
    st.session_state.view = 'dashboard'

# Sidebar Navigation
with st.sidebar:
    st.header('Menu')
    if st.button('Dashboard'):
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
    st.write(f'## Predicted Salary: ${predicted_salary:,.2f}')
    
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

# Dashboard Page
if st.session_state.view == 'dashboard':
    st.title('Dashboard')

    # Number of job positions by year
    work_year_counts = original_df['work_year'].value_counts().sort_index()
    fig_year = px.line(
        x=work_year_counts.index,
        y=work_year_counts.values,
        markers=True,
        labels={'x': 'Year', 'y': 'Number of job positions'},
        title='Number of job positions by year',
        template='plotly_dark'
    )
    fig_year.update_traces(line=dict(color='#c63256'))
    fig_year.update_layout(template='plotly_dark')
    st.plotly_chart(fig_year, use_container_width=True)

    # Total salaries by top 10 job titles
    df_job_title_USD = original_df.groupby('job_title')['salary_in_usd'].sum().sort_values(ascending=False).reset_index()
    top_10_job_titles = df_job_title_USD.head(10)
    fig_top_salaries = px.bar(
        top_10_job_titles,
        x='salary_in_usd',
        y='job_title',
        orientation='h',
        labels={'salary_in_usd': 'Salary in USD', 'job_title': 'Job Title'},
        title='Total of salaries by top 10 job titles',
        template='plotly_dark',
        text='salary_in_usd',
        color='job_title'
    )
    fig_top_salaries.update_layout(showlegend=False)
    st.plotly_chart(fig_top_salaries, use_container_width=True)

    # Average salary per job category and experience level
    avg_salary = original_df.groupby(['job_category', 'experience_level'])['salary_in_usd'].mean().reset_index()
    fig_avg_salary = px.bar(
        avg_salary,
        x='job_category',
        y='salary_in_usd',
        color='experience_level',
        barmode='group',
        labels={'salary_in_usd': 'Average Salary in USD', 'job_category': 'Job Category', 'experience_level': 'Experience Level'},
        title='Average salary per job category and experience level',
        template='plotly_dark',
        text='salary_in_usd'
    )
    st.plotly_chart(fig_avg_salary, use_container_width=True)

    # Salary distribution per job category
    fig_job_category_salary = px.box(
        original_df,
        x='job_category',
        y='salary_in_usd',
        labels={'salary_in_usd': 'Salary in USD', 'job_category': 'Job Category'},
        title='Salary distribution per job category',
        template='plotly_dark'
    )
    st.plotly_chart(fig_job_category_salary, use_container_width=True)

    # Average Salary in Data per country
    avg_salary_per_country = original_df.groupby('company_location')['salary_in_usd'].mean().reset_index().round(2)
    avg_salary_per_country = avg_salary_per_country.sort_values(by='salary_in_usd', ascending=False)
    fig_avg_salary_country = px.bar(
        avg_salary_per_country,
        x='company_location',
        y='salary_in_usd',
        labels={'salary_in_usd': 'Average Salary in USD', 'company_location': 'Country'},
        title='Average Salary in Data per country',
        template='plotly_dark',
        text='salary_in_usd'
    )
    fig_avg_salary_country.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_avg_salary_country, use_container_width=True)

    # Number of positions by top 10 Job titles
    df_job_title_count = original_df['job_title'].value_counts().sort_values(ascending=False).reset_index()
    df_job_title_count.columns = ['job_title', 'count']
    top_10_job_titles_count = df_job_title_count.head(10)
    fig_top_job_titles_count = px.bar(
        top_10_job_titles_count,
        x='count',
        y='job_title',
        orientation='h',
        labels={'count': 'Number of job positions', 'job_title': 'Job Title'},
        title='Number of positions by top 10 Job titles',
        template='plotly_dark',
        color='job_title',
        text='count'
    )
    fig_top_job_titles_count.update_layout(showlegend=False)
    st.plotly_chart(fig_top_job_titles_count, use_container_width=True)

    # Number of job positions per country
    country_counts = original_df['company_location'].value_counts().reset_index()
    country_counts.columns = ['company_location', 'count']
    filtered_country_counts = country_counts[country_counts['count'] >= 10]
    fig_country_counts = px.bar(
        filtered_country_counts,
        x='company_location',
        y='count',
        labels={'count': 'Number of Positions', 'company_location': 'Country'},
        title='Number of job positions per country',
        template='plotly_dark',
        text='count'
    )
    st.plotly_chart(fig_country_counts, use_container_width=True)