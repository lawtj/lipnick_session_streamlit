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

from exclude_unclean import drop_dict
def label_manual_samples(labview_samples, drop_dict):
    # label samples to be dropped
    for session, samples in drop_dict.items():
        labview_samples.loc[(labview_samples['session'] == session), 'manual_so2'] = 'keep'
        labview_samples.loc[(labview_samples['session'] == session) & (labview_samples['sample'].isin(samples)), 'manual_so2'] = 'reject'
    # now compare the manual_so2 column with the algo column
    labview_samples['manual_algo_compare'] = None
    # start with the samples that were rejected by the manual_so2 column but kept by the algo column
    labview_samples.loc[(labview_samples['manual_so2'] == 'reject') & (labview_samples['so2_stable'] == True),'manual_algo_compare'] = 'manual reject'
    # now the samples that were kept by the manual_so2 column but rejected by the algo column
    labview_samples.loc[(labview_samples['manual_so2'] == 'keep') & (labview_samples['so2_stable'] == False),'manual_algo_compare'] = 'manual keep'
    # label keep both
    labview_samples.loc[(labview_samples['manual_so2'] == 'keep') & (labview_samples['so2_stable'] == True),'manual_algo_compare'] = 'keep (both)'
    # label reject both
    labview_samples.loc[(labview_samples['manual_so2'] == 'reject') & (labview_samples['so2_stable'] == False),'manual_algo_compare'] = 'reject (both)'

label_manual_samples(labview_samples, drop_dict)


with st.sidebar:
    st.write('## Session Selector')
    st.markdown('### Filters')
    
    limit_to_manual_sessions = st.checkbox('Limit to sessions undergoing manual review')

    ## selectbox for session
    max_bias = st.number_input('Show session where maximum bias is >= :', 0, 20, 10, 1)
    sessionlist = labview_samples[(abs(labview_samples['bias']) >= max_bias) & (labview_samples['so2_stable']==True)]['session'].unique().tolist()
    if limit_to_manual_sessions:
        set1 = set(sessionlist)
        set2 = set(labview_samples[labview_samples['manual_so2'] == 'reject']['session'].unique().tolist())
        sessionlist = list(set1.intersection(set2))
        sessionlist.sort()
    st.write('Number of sessions: ', len(sessionlist))
    sessionlist.reverse()
    selected_session = st.selectbox('Select a session', sessionlist)
    
    frame = labview_samples[labview_samples['session'] == selected_session]
    frame = frame.drop(columns=['session'])
    if frame['Nellcor PM1000N-1/SpO2'].sum() >0:
        frame = frame.drop(columns=['Masimo 97/SpO2','Nellcor/SpO2', 'Nellcor/SpO2_diff_prev','Nellcor/SpO2_diff_next', 'Masimo 97/SpO2_diff_prev', 'Masimo 97/SpO2_diff_next'])
        frame = frame.rename(columns={'Nellcor PM1000N-1/SpO2':'Nellcor/SpO2',
                                      'Nellcor PM1000N-1/SpO2_diff_prev':'Nellcor/SpO2_diff_prev',
                                      'Nellcor PM1000N-1/SpO2_diff_next':'Nellcor/SpO2_diff_next',
                                      'Rad97-60/SpO2':"Masimo 97/SpO2",
                                        'Rad97-60/SpO2_diff_prev':'Masimo 97/SpO2_diff_prev',
                                        'Rad97-60/SpO2_diff_next':'Masimo 97/SpO2_diff_next'
                                      })
    else:
        frame = frame.drop(columns=['Masimo HB/SpO2','Masimo HB/SpO2_diff_prev','Masimo HB/SpO2_diff_next', 'Nellcor PM1000N-1/SpO2', 'Nellcor PM1000N-1/SpO2_diff_prev', 'Nellcor PM1000N-1/SpO2_diff_next', 'Rad97-60/SpO2', 'Rad97-60/SpO2_diff_prev', 'Rad97-60/SpO2_diff_next'])
    criteria_check_tuple, criteria_check_df = ox.session_criteria_check(frame)

    #calculate stats on sessions that were manually reviewed
    manual_stats_df = labview_samples[labview_samples['manual_so2'].notnull()]
    st.write(manual_stats_df['manual_algo_compare'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

    
st.markdown('## Session ' + str(selected_session))
st.write(frame.set_index('sample').drop(columns=['sample_diff_prev', 'sample_diff_next', ]))

plotcolumns = ['so2', 'Nellcor/SpO2','bias']
from session_functions import colormap

# create column 'col_so2_symbol' which is colormap['column'][1] if so2_stable is True, else 'cross'
frame['so2_symbol'] = frame['so2_stable'].apply(lambda x: colormap['so2'][1] if x else 'cross')
frame['Nellcor/SpO2_symbol'] = frame['Nellcor_stable'].apply(lambda x: colormap['Nellcor/SpO2'][1] if x else 'cross')
# frame['bias_symbol'] = colormap['bias'][1]
# bias should be circle unless bias is greater than max_bias then it should be other
frame['bias_symbol'] = np.where(abs(frame['bias']) > max_bias, 'hexagram', 'circle')

# ACTUALLY so2_line is always DarkSlateGrey
frame['so2_line'] = 'DarkSlateGrey'

# same for Nellcor/SpO2
frame['Nellcor/SpO2_line'] = np.where((frame['Nellcor_stable'] == True) & (abs(frame['bias']) > max_bias), 'red', 'DarkSlateGrey')
# bias line should be blue if bias is greater than max_bias
frame['bias_line'] = np.where(abs(frame['bias']) > max_bias, 'blue', 'DarkSlateGrey')


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
            line=dict(width=1.5, color=frame[column + '_line'])
        ),
    ))
# add text labeling points for manual_algo_compare
for index, row in frame.iterrows():
    if row['manual_algo_compare'] == 'manual reject':
        fig.add_annotation(x=row['sample'], y=row['so2']+10, text='manual reject', showarrow=True)
    if row['manual_algo_compare'] == 'manual keep':
        fig.add_annotation(x=row['sample'], y=row['so2'], text='manual keep', showarrow=True)

st.write('''
         * Crosses indicate that the data point was rejected by the algorithm (either so2 or Nellcor). 
         * Nellcor: Red outlines indicate Nellcor values where the bias is > threshold, but were not cleaned out.
        * Bias: Blue outlines indicate bias values > threshold.
         ''')
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