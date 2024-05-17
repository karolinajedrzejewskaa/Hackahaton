import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import requests
from io import BytesIO

# Set the page configuration
st.set_page_config(page_title="Dashboard and Prediction App", layout="wide")

# File paths
original_data_path = 'https://raw.githubusercontent.com/LuaGeo/hackathon/main/jobs_in_data.csv'
cleaned_data_path = 'https://raw.githubusercontent.com/LuaGeo/hackathon/main/tableau_nettoye.csv'
model_path = 'https://raw.githubusercontent.com/LuaGeo/hackathon/main/model.pkl'
encoders_path = 'https://raw.githubusercontent.com/LuaGeo/hackathon/main/encoders.pkl'
css_path = 'https://raw.githubusercontent.com/LuaGeo/hackathon/main/styles.css'


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
    st.session_state.view = 'home'

# Load CSS content from GitHub
def load_css(path):
    response = requests.get(path)
    return response.text

# Apply CSS
css = load_css(css_path)
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)




# Navigation buttons in the sidebar
if st.sidebar.button('Home'):
    st.session_state.view = 'home'

if st.sidebar.button('Dashboard'):
    st.session_state.view = 'dashboard'

if st.sidebar.button('Conclusion'):
    st.session_state.view = 'conclusion'

if st.sidebar.button('Contact'):
    st.session_state.view = 'contact'


job_titles = df['job_title'].unique()

# Form for input details in an expandable section
with st.sidebar.expander('Prediction App'):
    with st.form(key='salary_form'):
        job_title = st.selectbox('Job Title', job_titles)
        experience_level = st.selectbox('Experience level', ['Entry-level', 'Mid-level', 'Senior', 'Executive'])
        work_setting = st.selectbox('Work setting', ['Remote', 'Hybrid', 'In-person'])
        company_size = st.selectbox('Company size', ['L', 'M', 'S'])
        work_year = st.selectbox('Year', [2023, 2022])
        
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
    st.markdown("""
    <style>
    .prediction-container {
        background-color: #f8f9fa;
        height: 1px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .prediction-title {
        font-size: 36px;
        font-weight: bold;
        color: #343a40;
        
    }
    .prediction-subtitle {
        font-size: 24px;
        font-weight: bold;
        color: #495057;
        
    }
    .prediction-details {
        font-size: 18px;
        color: #6c757d;
        
    }
    .predicted-salary {
        font-size: 28px;
        font-weight: bold;
        color: #28a745;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown('<div class="prediction-title">Salary Prediction App</div>', unsafe_allow_html=True)
    st.markdown('<div class="prediction-subtitle">Prediction Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="prediction-details">You selected:</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-details"><strong>Job Title:</strong> {st.session_state.job_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-details"><strong>Experience level:</strong> {st.session_state.experience_level}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-details"><strong>Work setting:</strong> {st.session_state.work_setting}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="prediction-details"><strong>Company size:</strong> {st.session_state.company_size}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-details"><strong>Year:</strong> {st.session_state.work_year}</div>', unsafe_allow_html=True)

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
    st.markdown(f'<div class="predicted-salary">Predicted Salary: ${predicted_salary:,.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    
    # Filter data based on user input
    filtered_data = df[(df['job_title'] == st.session_state.job_title) & 
                    (df['experience_level'] == st.session_state.experience_level) & 
                    (df['work_setting'] == st.session_state.work_setting) & 
                    (df['company_size'] == st.session_state.company_size) & 
                    (df['work_year'] == st.session_state.work_year)]

# Home view
if st.session_state.view == 'home':
    st.markdown(
    """
    <div style="display: flex; justify-content: center; ">
        <img style= "width: 200px;" src="https://raw.githubusercontent.com/LuaGeo/hackathon/main/logo_with_transparent_background%20(1).png" alt="logo">
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown(
    """
    <div style="display: flex; justify-content: center; margin-bottom: 30px">
        <h1 style="color: #1AE3FC">DATA CAREER CONSULTING</h1>
    </div>
    """,
    unsafe_allow_html=True
)

    st.write("❝ Data Career Consulting is a platform providing insights into data job salaries. \
            With our interactive dashboard, clients can explore salary data by country, \
            job title, and experience level, empowering them to make informed career decisions. ❞")


# Dashboard Page
if st.session_state.view == 'dashboard':
    st.title('Dashboard')

    #----------------------------------------------------------------

    # Calculate the number of unique job titles, unique job categories, and total salaries for 2023
    unique_job_titles_count = original_df['job_title'].nunique()
    unique_job_categories_count = original_df['job_category'].nunique()
    total_salaries_2023 = original_df[original_df['work_year'] == 2023]['salary_in_usd'].sum()

    # Create the indicators
    fig1 = go.Figure(go.Indicator(
        mode="number",
        value=unique_job_titles_count,
        title="Job Titles",
        number={'font': {'color': 'lightblue'}}
    ))

    fig2 = go.Figure(go.Indicator(
        mode="number",
        value=unique_job_categories_count,
        title="Job Categories",
        number={'font': {'color': 'lightblue'}}
    ))

    fig3 = go.Figure(go.Indicator(
        mode="number",
        value=total_salaries_2023,
        title="Total Salaries in 2023",
        number={'prefix': "$", 'font': {'color': 'lightblue'}}
    ))

    fig1.update_layout(width=400, height=300)
    fig2.update_layout(width=400, height=300)
    fig3.update_layout(width=400, height=300)

    # Display the indicators in a single row
    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    col3.plotly_chart(fig3, use_container_width=True)
    

    #----------------------------------------------------------------

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
    fig_year.update_xaxes(tickvals=[2020, 2021, 2022, 2023])
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
        text = top_10_job_titles['salary_in_usd'] / 1000,
        color='job_title'
    )
    fig_top_salaries.update_layout(showlegend=False)
    fig_top_salaries.update_traces(texttemplate='$%{text:.0f}K', textposition='inside')
    st.plotly_chart(fig_top_salaries, use_container_width=True)

    # Average salary per job category and experience level 
    # & Salary distribution per job category

    avg_salary = original_df.groupby(['job_category', 'experience_level'])['salary_in_usd'].mean().sort_values(ascending=False).round().reset_index()

    # Create bar plot
    bar_fig = px.bar(
        avg_salary,
        x='job_category',
        y='salary_in_usd',
        color='experience_level',
        barmode='group',
        labels={'salary_in_usd': 'Average Salary in USD', 'job_category': 'Job Category', 'experience_level': 'Experience Level'},
        title='Average Salary per Job Category and Experience Level',
        template='plotly_dark',
        text=avg_salary['salary_in_usd'] / 1000
    )
    bar_fig.update_layout(
        width=1200,
        height=800,
        yaxis_title='Average Salary in USD',
        xaxis_title='Job Category'
    )
    bar_fig.update_yaxes(tickprefix='$', ticksuffix='K', tickformat=',.0f')
    bar_fig.update_traces(texttemplate='$%{text:.0f}K', textposition='inside')

    # Create box plot
    box_fig = px.box(
        original_df,
        x='job_category',
        y='salary_in_usd',
        labels={'salary_in_usd': 'Salary in USD', 'job_category': 'Job Category'},
        title='Salary distribution per job category',
        template='plotly_dark'
    )
    box_fig.update_layout(
        width=1000, 
        height=600 
    )

    # Combine plots into subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average Salary per Job Category and Experience Level", "Salary distribution per job category"))

    for trace in bar_fig['data']:
        fig.add_trace(trace, row=1, col=1)

    for trace in box_fig['data']:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(
        height=800,
        width=1700,
        template='plotly_dark',
        showlegend=False
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

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
        text=avg_salary_per_country['salary_in_usd'] / 1000
    )
    fig_avg_salary_country.update_layout(xaxis_tickangle=-45)
    fig_avg_salary_country.update_traces(texttemplate='$%{text:.0f}K', textposition='inside')
    st.plotly_chart(fig_avg_salary_country, use_container_width=True)

    # Number of positions by top 10 Job titles
    # & Number of job positions per country
    df_job_title_count = original_df['job_title'].value_counts().sort_values(ascending=False).reset_index()
    df_job_title_count.columns = ['job_title', 'count']
    top_10_job_titles_count = df_job_title_count.head(10)

    # Create the first bar plot
    bar_fig1 = px.bar(
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
    bar_fig1.update_layout(
        width=1200,
        height=600,
        xaxis_title='Number of Positions',
        yaxis_title='Job Title',
        showlegend=False
    )

    # Calculate the number of job positions per country
    country_counts = original_df['company_location'].value_counts().reset_index()
    country_counts.columns = ['company_location', 'count']
    filtered_country_counts = country_counts[country_counts['count'] >= 10]

    # Create the second bar plot
    bar_fig2 = px.bar(
        filtered_country_counts,
        x='company_location',
        y='count',
        labels={'count': 'Number of Positions', 'company_location': 'Country'},
        title='Number of job positions per country',
        template='plotly_dark',
        text='count'
    )
    bar_fig2.update_layout(
        width=1000,
        height=600
    )

    # Combine plots into subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Number of positions by top 10 Job titles", "Number of job positions per country"))

    for trace in bar_fig1['data']:
        fig.add_trace(trace, row=1, col=1)

    for trace in bar_fig2['data']:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(
        height=800,
        width=1700,
        template='plotly_dark',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


# Conclusion view
if st.session_state.view == 'conclusion':
    st.title('Conclusions')
    conclusion_text = """
    - Hausse du nombre de postes entre 2022 et 2023
    - Le jeu de données renseigne en majorité sur les métiers aux États-Unis
    - Les autres régions du monde sont sous-représentées
    - Ecart aléatoire entre les salaires aux Etats-Unis pour le même métier et même niveau d'expérience
    - Catégorie Data Analysis dans le top 5 des budgets salaires alloués
    - L'écart reflète-t-il la disparité des salaires aux Etats-Unis ou est expliqué par l'insuffisance de données?
    
    ### Barrières de l'analyse:
    - Prédiction de salaire possible uniquement pour les métiers aux Etats-Unis
    - Données insuffisantes pour avoir un score du modèle satisfaisant
    
    ### Informations utiles pour améliorer le score:
    - Localisation des entreprises : Etat, ville
    - Secteur d'activité de l'entreprise: tech, industries, ...
    - Niveau d'expérience
    """

    st.markdown(conclusion_text)
    
# Contact view
if st.session_state.view == 'contact':
    st.write('## Contact Us')
    st.write('Please contact Karolina, Lala, Luana, and Patrick for any questions or feedback. 📧')