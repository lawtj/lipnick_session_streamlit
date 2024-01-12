import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from redcap import Project
import io
import random

def arms(spo2,sao2):
    return np.sqrt(np.mean((spo2-sao2)**2))

st.set_page_config(layout="wide")

if 'clinimark' not in st.session_state:
    with st.spinner('Loading data...'):
        api_url = 'https://redcap.ucsf.edu/api/'
        api_k = st.secrets['api_k']
        proj = Project(api_url, api_k)
        f = io.BytesIO(proj.export_file(record='3', field='file')[0])
        clinimark = pd.read_excel(f)
        clinimark['finga_bias'] = clinimark['SpO2fnga'] - clinimark['SaO2a']
        clinimark['fingb_bias'] = clinimark['SpO2fngc'] - clinimark['SaO2a']
        clinimark['finga_bias'] = np.abs(clinimark['finga_bias'])
        clinimark['fingb_bias'] = np.abs(clinimark['fingb_bias'])
        st.session_state['clinimark'] = clinimark
else:
    st.caption('Using cached data')
    clinimark = st.session_state['clinimark']

if st.button('Pick random session'):
    random_session = random.choice(clinimark['sida'].unique())
    fig = px.scatter(clinimark[clinimark['sida'] == random_session], x=clinimark[clinimark['sida'] == random_session].index,y=['SaO2a','SpO2fnga', 'finga_bias'], template='plotly_white').update_traces(marker=dict(size=12,opacity=0.8, 
                    line=dict(width=2,color='DarkSlateGrey'))).update_layout(title='SaO2a vs SpO2fnga vs finga_bias for session {}'.format(random_session), yaxis_title='Oxygen Saturation (%)')
    st.plotly_chart(fig, use_container_width=True)
    arms_value = arms(clinimark[clinimark['sida'] == random_session]['SpO2fnga'], clinimark[clinimark['sida'] == random_session]['SaO2a'])
    st.write('Arms: {:.2f}'.format(round(arms_value, 2)))
    st.write('Number of samples: ', len(clinimark[clinimark['sida'] == random_session]['SpO2fnga']))
    st.write('Maximum and minimum SaO2a: {:.2f} and {:.2f}'.format(max(clinimark[clinimark['sida'] == random_session]['SaO2a']), min(clinimark[clinimark['sida'] == random_session]['SaO2a'])))
