import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from redcap import Project
import io
import plotly.graph_objects as go
from session_functions import create_scatter
import openox as ox
import os

st.set_page_config(layout="wide", )

def get_labview_samples():
    api_url = 'https://redcap.ucsf.edu/api/'
    try:
        api_k = st.secrets['api_k']
    except:
        api_k = os.environ['REDCAP_FILE_RESPOSITORY']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='8', field='file')[0])
    labview_samples = pd.read_csv(f)
    return labview_samples

labview_samples = get_labview_samples()

with st.sidebar:
    st.write('## Session Selector')
    st.markdown('### Filters')
    st.write('Show session where maximum bias is >= :')
    
    ## selectbox for session
    max_bias = st.number_input('Show biases greater than', 0, 20, 10, 1)
    sessionlist = labview_samples[labview_samples['bias'] >= max_bias]['session'].unique().tolist()
    st.write('Number of sessions: ', len(sessionlist))
    sessionlist.reverse()
    selected_session = st.selectbox('Select a session', sessionlist)
    
    frame = labview_samples[labview_samples['session'] == selected_session]
    
    criteria_check_tuple, criteria_check_df = ox.session_criteria_check(frame)
    
st.markdown('## Session ' + str(selected_session))
st.write(frame)

plotcolumns = ['so2', 'Nellcor/SpO2','bias']
from session_functions import colormap

# create column 'col_so2_symbol' which is colormap['column'][1] if so2_stable is True, else 'cross'
frame['so2_symbol'] = frame['so2_stable'].apply(lambda x: colormap['so2'][1] if x else 'cross')
frame['Nellcor/SpO2_symbol'] = frame['Nellcor_stable'].apply(lambda x: colormap['Nellcor/SpO2'][1] if x else 'cross')
frame['bias_symbol'] = colormap['bias'][1]


fig = go.Figure()
for column in plotcolumns:
    fig.add_trace(go.Scatter(
        x=frame['sample'], y=frame[column],
        mode='markers',
        name=column,
        marker=dict(
            symbol= frame[column + '_symbol'],
            color= colormap[column][0],
            size=12,
            opacity=0.8,
            line=dict(width=1, color='DarkSlateGrey')
        )
    ))

st.plotly_chart(fig, use_container_width=True)

st.markdown('#### Session Criteria Check')

one, two = st.columns(2)
with one:
    st.dataframe(criteria_check_df.style.map(lambda x: 'background-color: yellow' if x<1 else '', subset=['#so2 in 97-100', '#so2 in 67-73'])
                                        .map(lambda x: 'background-color: yellow' if x<6 else '', subset=['#so2 in 70-80', '#so2 in 80-90', '#so2 in 90-100']))

with two:
    if criteria_check_tuple[0]:
        st.success('at least one so2 data point in 97-100 range')
    else:
        st.error('no so2 data points in 97-100 range')
    if criteria_check_tuple[1]:
        st.success('at least one so2 data point in 67-73 range')
    else:
        st.error('no so2 data points in 67-73 range')
    if criteria_check_tuple[2]:
        st.success('at least 6 so2 data points in each 3 plateaus')
    else:
        st.error('fewer than 6 so2 data points in each 3 plateaus')