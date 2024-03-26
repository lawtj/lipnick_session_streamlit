import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from redcap import Project
import io
import plotly.graph_objects as go


st.set_page_config(layout="wide")

def get_labview_samples():
    api_url = 'https://redcap.ucsf.edu/api/'
    api_k = st.secrets['api_k']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='9', field='file')[0])
    labview_samples = pd.read_csv(f)
    return labview_samples

labview_samples = get_labview_samples()

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
st.markdown('''
*Legend*:
            
            - Circle: kept by both
            - Hourglass: kept by manual cleaning but rejected by algorithm
            - Diamond: kept by algorithm but rejected by manual cleaning
            - Cross: rejected by both

            ''')
st.plotly_chart(create_scatter(frame))
labview_samples_filter = labview_samples[labview_samples['session']==selected_session][['session','sample', 'so2', 'so2_previous','so2_next','so2_keep','so2_compare', 'manual_clean_so2','Masimo 97/SpO2', 'masimo_previous','masimo_next','masimo_keep','Nellcor/SpO2', 'nellcor_previous','nellcor_next','nellcor_keep']].style.format(precision=2).map(lambda x: 'background-color: yellow' if 'reject' in str(x) else '')
st.write(labview_samples_filter)

so2mask = labview_samples['so2_keep'] != labview_samples['manual_clean_so2']
masimomask = labview_samples['masimo_keep'] != labview_samples['manual_clean_masimo']
manualmask = labview_samples['manual_clean_so2'] == 'reject'
algomask = labview_samples['so2_keep'] == 'reject'
manualrjecets = set(labview_samples[manualmask].index)
algorejects = set(labview_samples[algomask].index)

st.write('numnber of samples rejected by manual cleaning: ', labview_samples[manualmask].shape[0])
st.write('number of samples rejected by algorithm: ', labview_samples[algomask].shape[0],'\n')
st.write('number of samples rejected by both:', len(manualrjecets.intersection(algorejects)))
st.write('number of samples rejected by manual but not algorithm:', len(manualrjecets.difference(algorejects)))
st.write('number of samples rejected by algorithm but not manual:', len(algorejects.difference(manualrjecets)),'\n')

st.write(labview_samples['so2_keep'].value_counts(normalize=True),'\n')
st.write(labview_samples['manual_clean_so2'].value_counts(normalize=True),'\n')