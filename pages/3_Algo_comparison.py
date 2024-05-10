import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from redcap import Project
import io
import plotly.graph_objects as go
from session_functions import create_scatter
import openox as ox

st.set_page_config(layout="wide", )

def get_labview_samples():
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = st.secrets['api_k']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='9', field='file')[0])
    labview_samples = pd.read_csv(f)
    return labview_samples

labview_samples = get_labview_samples()

with st.sidebar:
    st.write('## Session Selector')
    st.markdown('### Filters')
    st.write('Show only those sessions containing at least one sample greater than:')

    # filters 
    sample_stability_threshold = st.number_input('Sample Stability Threshold', 0.0, 20.0, 1.5, 0.25)
    ref_stability_threshold = st.number_input('Reference Stability Threshold', 0.0, 20.0, 1.5, 0.25)
    
    ## selectbox for session
    sessionlist = labview_samples['session'].unique().tolist()
    sessionlist.reverse()
    selected_session = st.selectbox('Select a session', sessionlist)

    frame = labview_samples[labview_samples['session'] == selected_session]
    frame, dict = ox.sample_stability_multi(frame, so2_col='so2', nellcor_col='Nellcor/SpO2', timestamp_col='Timestamp', output_newcol_name='algo', so2_bound=sample_stability_threshold, ref_bound=ref_stability_threshold)


st.markdown('## Session ' + str(selected_session))

st.plotly_chart(create_scatter(frame, plotcolumns=['so2', 'Nellcor/SpO2'], stylecolumns=[ 'so2_stable','Nellcor_stable']), use_container_width=True)
st.dataframe(frame.set_index('sample').drop(['Unnamed: 0', 'Timestamp','sample_diff_prev','sample_diff_next'], axis=1).style.format(precision=2).map(lambda x: 'background-color: yellow' if 'reject' in str(x)  else ''))
