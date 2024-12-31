import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# Charger l'objet de normalisation
with open('MinMaxScaler.pkl', 'rb') as f:
    minmax_scaler = pickle.load(f)

# Charger les coefficients de pondération des variables par thème
with open('Coefficients.pkl', 'rb') as f:
    coefficients = pickle.load(f)

# Charger le modèle KMeans déployé
with open('KMeans_Model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Ajuster à la largeur de l'écran
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Afficher l'entête
with st.container():
    col = st.columns(spec=1, border=True)
    with col[0]:
        st.markdown("""
            <div style="text-align: center;">
                <img src="ehtp_logo.png"/>
                <br>
                <h2>MSDE6 : Machine Learning Project</h2>
                <h3>Countries Segmentation Application</h3>
                <br>
            </div>
        """, unsafe_allow_html=True)
        

# Séparation entre les blocs
st.write('---')

# Charger la liste de tous les pays du monde dans un ComboBox
countries = pd.read_csv(r'Countries_List.csv', header=None)
with st.container():
    col1, col2, col3 = st.columns(spec=3, gap='medium')
    with col2:
        country = st.selectbox('**Select a Country**', countries[0].values.tolist())

# Séparation entre les blocs
st.write('---')

# Saisir les valeurs des caractéristiques (inputs)
with st.container():
    col1, col2, col3 = st.columns(spec=3, gap='medium', border=True)
    with col1:
        #st.subheader('Geographic & Environmental Indicators')
        st.markdown("""
                        <h4>Geographic & Environmental Indicators</h4>
                    """, unsafe_allow_html=True)
        land_area = st.number_input('**Land area (km2)**', min_value=0.0)
        forested_area_rate = st.number_input('**Forested area rate (%)**', min_value=0.0, max_value=100.0)
        agricultural_land_rate = st.number_input('**Agricultural land rate (%)**', min_value=0.0, max_value=100.0)
        co2_emissions = st.number_input('**Co2-Emissions (ton)**', min_value=0.0)
    with col2:
        #st.subheader('Demographic Indicators')
        st.markdown("""
                        <h4>Demographic Indicators</h4>
                    """, unsafe_allow_html=True)
        population = st.number_input('**Population**', min_value=0.0)
        density = st.number_input('**Density (population/km2)**', min_value=0.0)
        fertility_avg = st.number_input('**Fertility average**', min_value=0.0)
    with col3:
        #st.subheader('Economic Indicators')
        st.markdown("""
                        <h4>Economic Indicators</h4>
                    """, unsafe_allow_html=True)
        gdp = st.number_input('**Gross Domestic Product ($)**', min_value=0.0)
        minimum_wage = st.number_input('**Minimum wage ($)**', min_value=0.0)
        cpi = st.number_input('**Consumer Price Index**', min_value=0.0)
        total_tax_rate = st.number_input('**Total tax rate (%)**', min_value=0.0, max_value=100.0)
        gasoline_price = st.number_input('**Gasoline Price ($)**', min_value=0.0)

with st.container():
    col1, col2, col3 = st.columns(spec=3, gap='medium', border=True)
    with col1:
        #st.subheader('Social Indicators')
        st.markdown("""
                        <h4>Social Indicators</h4>
                    """, unsafe_allow_html=True)
        primary_schooling_rate = st.number_input('**Primary schooling rate (%)**', min_value=0.0, max_value=100.0)
        tertiary_schooling_rate = st.number_input('**Tertiary schooling rate (%)**', min_value=0.0, max_value=100.0)
        labor_force_rate = st.number_input('**Labor force rate (%)**', min_value=0.0, max_value=100.0)
    with col3:
        #st.subheader('Healthcare Indicators')
        st.markdown("""
                        <h4>Healthcare Indicators</h4>
                    """, unsafe_allow_html=True)
        life_expectancy_avg = st.number_input('**Life expectancy**', min_value=0.0)
        physicians_rate = st.number_input('**Physicians rate (per 1000 people)**', min_value=0.0)
        pocket_health_expenditure_rate = st.number_input('**Out of Pocket Health expenditure rate (%)**', min_value=0.0, max_value=100.0)
    with col2:
        st.markdown("""
                        <style>
                            .stButton>button {
                                background-color: #808080;  /* Couleur de fond du bouton */
                                color: white;               /* Couleur du texte */
                                border: none;               /* Pas de bordure */
                                padding: 15px 32px;         /* Espacement interne */
                                text-align: center;         /* Centrer le texte */
                                font-size: 16px;            /* Taille de la police */
                                border-radius: 8px;         /* Bord arrondi */
                                cursor: pointer;           /* Curseur de souris */
                            }

                            .stButton>button:hover {
                                background-color: #C0C0C0;  /* Couleur de fond lors du survol */
                            }
                        </style>
                    """, unsafe_allow_html=True)
        predict_button = st.columns(3)[1].button('**Predict**')
        if predict_button:
            # Créer un dictionnaire de données
            inputs_dict = { 'Land area' : land_area,
                            'Forested area rate' : forested_area_rate,
                            'Agricultural land rate' : agricultural_land_rate,
                            'Co2-Emissions' : co2_emissions,
                            'Population' : population,
                            'Density' : density,
                            'Fertility avg' : fertility_avg,
                            'Primary schooling rate' : primary_schooling_rate,
                            'Tertiary schooling rate' : tertiary_schooling_rate,
                            'Labor force rate' : labor_force_rate,
                            'GDP' : gdp,
                            'Minimum wage' : minimum_wage,
                            'CPI' : cpi,
                            'Total tax rate' : total_tax_rate,
                            'Gasoline Price' : gasoline_price,
                            'Life expectancy avg' : life_expectancy_avg,
                            'Physicians rate' : physicians_rate,
                            'Pocket Health expenditure rate' : pocket_health_expenditure_rate
                           }

            # Créer un dataframe de données
            df_inputs = pd.DataFrame(inputs_dict, index=[0])

            # Normaliser les variables d'entrée en distinguant entre les pourcentages et les autres
            col_per = ['Forested area rate', 'Agricultural land rate', 'Labor force rate', 'Tertiary schooling rate', 'Total tax rate', 'Pocket Health expenditure rate']
            df_inputs[col_per] = df_inputs[col_per].apply(lambda x : x/100)

            col_minmax = ['Land area', 'Co2-Emissions', 'Population', 'Density', 'Fertility avg', 'GDP', 'Minimum wage', 'CPI', 'Gasoline Price', 
                          'Life expectancy avg', 'Physicians rate', 'Primary schooling rate']
            df_inputs[col_minmax] = pd.DataFrame(minmax_scaler.transform(df_inputs[col_minmax]), columns = col_minmax)

            # Réduire les variables en 5 dimensions (indicateurs)
            df_inputs['Geographic & Environmental Indicator'] = df_inputs['Land area'] * coefficients['Geographic & Environmental']['Land area']\
                                                              + df_inputs['Forested area rate'] * coefficients['Geographic & Environmental']['Forested area rate']\
                                                              + df_inputs['Agricultural land rate'] * coefficients['Geographic & Environmental']['Agricultural land rate']

            df_inputs['Demographic Indicator'] = df_inputs['Population'] * coefficients['Demographic']['Population']\
                                               + df_inputs['Fertility avg'] * coefficients['Demographic']['Fertility avg']

            df_inputs['Social Indicator'] = df_inputs['Primary schooling rate'] * coefficients['Social']['Primary schooling rate']\
                                          + df_inputs['Tertiary schooling rate'] * coefficients['Social']['Tertiary schooling rate']\
                                          + df_inputs['Labor force rate'] * coefficients['Social']['Labor force rate']

            df_inputs['Total tax rate'] = 1 - df_inputs['Total tax rate']
            df_inputs['Gasoline Price'] = 1 - df_inputs['Gasoline Price']
            df_inputs['Economic Indicator'] = df_inputs['GDP'] * coefficients['Economic']['GDP']\
                                            + df_inputs['Minimum wage'] * coefficients['Economic']['Minimum wage']\
                                            + df_inputs['Total tax rate'] * coefficients['Economic']['Total tax rate']\
                                            + df_inputs['Gasoline Price'] * coefficients['Economic']['Gasoline Price']

            df_inputs['Pocket Health expenditure rate'] = 1 - df_inputs['Pocket Health expenditure rate']
            df_inputs['Healthcare Indicator'] = df_inputs['Life expectancy avg'] * coefficients['Healthcare']['Life expectancy avg']\
                                              + df_inputs['Physicians rate'] * coefficients['Healthcare']['Physicians rate']\
                                              + df_inputs['Pocket Health expenditure rate'] * coefficients['Healthcare']['Pocket Health expenditure rate']                          

            # Préparer les entrées réduites
            df_inputs_reduced = df_inputs[['Geographic & Environmental Indicator', 
                                           'Demographic Indicator', 
                                           'Social Indicator', 
                                           'Economic Indicator', 
                                           'Healthcare Indicator'
                                          ]]

            # Prédiction avec le modèle KMeans
            prediction = kmeans_model.predict(df_inputs_reduced)
            
            # Afficher la prédiction
            if prediction[0] == 0:
                cluster_dict = {'Cluster 0' : 'Countries with high population growth, with challenges noted in the social, economic and health areas, despite relatively good geographical resources.'}
            elif prediction[0] == 1:
                cluster_dict = {'Cluster 1' : 'Countries with low population growth, highly developed social and health factors, and moderate economic development.'}
            elif prediction[0] == 2:
                cluster_dict = {'Cluster 2' : 'Countries relatively developed in economic and health areas, but with moderate challenges in social areas, and with moderate population growth.'}
            
            result = pd.DataFrame(cluster_dict, index=[0])
            st.markdown(result.style.hide(axis="index").to_html(), unsafe_allow_html=True)
            
        else:
            st.markdown("""
                            <p style="text-align: center;">Click the button to predict the country cluster!</p>
                        """, unsafe_allow_html=True)

