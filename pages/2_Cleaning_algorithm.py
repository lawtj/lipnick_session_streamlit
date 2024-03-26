import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from redcap import Project
import io
import plotly.graph_objects as go


st.set_page_config(layout="wide")

if 'labview_samples' not in st.session_state:
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = st.secrets['api_k']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='8', field='file')[0])
    labview_samples = pd.read_csv(f)
    st.session_state['labview_samples'] = labview_samples
else:
    # st.caption('Using cached data')
    labview_samples = st.session_state['labview_samples']

from session_functions import threesamples, recalculate_so2_range, sample_stability, apply_sample_stability, assign_marker_color, assign_marker_style
from session_functions import colormap, create_scatter, arms


with st.sidebar:
    st.write('## Session Selector')
    st.markdown('### Filters')
    st.write('Show only those sessions containing at least one sample greater than:')
    sample_stability_threshold = st.number_input('Sample Stability Threshold', 0.0, 20.0, 1.5, 0.1)
    for i, j in zip(['so2','Masimo 97/SpO2','Nellcor/SpO2'], ['so2','masimo','nellcor']):
        labview_samples = apply_sample_stability(labview_samples, i, j, sample_stability_threshold)
    
    selected_session = st.selectbox('Select a session', labview_samples['session'].unique().tolist())

    frame = labview_samples[labview_samples['session'] == selected_session]

st.markdown('## Session ' + str(selected_session))
st.plotly_chart(create_scatter(frame))
labview_samples_filter = labview_samples[['session','sample', 'so2', 'so2_previous','so2_next','so2_keep', 'Masimo 97/SpO2', 'masimo_previous','masimo_next','masimo_keep','Nellcor/SpO2', 'nellcor_previous','nellcor_next','nellcor_keep']].style.format(precision=2).map(lambda x: 'background-color: yellow' if 'reject' in str(x) else '')
st.write(labview_samples_filter)



