import streamlit as st
import pandas as pd
from redcap import Project
import io
import plotly.graph_objects as go
from session_functions import colormap
import os
import numpy as np

st.set_page_config(layout="wide", )

def get_labview_samples():
    api_url = 'https://redcap.ucsf.edu/api/'
    try:
        api_k = st.secrets['api_k']
    except:
        api_k = os.environ['REDCAP_FILE_RESPOSITORY']
    proj = Project(api_url, api_k)
    f = io.BytesIO(proj.export_file(record='11', field='file')[0])
    labview_samples = pd.read_csv(f)
    return labview_samples

cleaned_merged = get_labview_samples()
# keep to dorsal only for ITA
cleaned_merged = cleaned_merged[cleaned_merged['group'] == 'Dorsal (B)']

def clean_outlier_spo2(df, delta_threshold):
    """
    Checks the stability of spo2 samples in the DataFrame, rejecting unstable readings.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing spo2 readings.
        delta_threshold (int): Threshold for deciding if a sample is stable.
    
    Returns:
        pandas.DataFrame: The original dataFrame with 3 additional columns indicating sample stability: 'spo2_delta_previous', 'spo2_delta_next', 'spo2_stable'.
    """
    df = df.sort_values('sample').reset_index(drop=True)
    
    df_no_duplicates = df.drop_duplicates(subset=['session', 'sample'], keep='first').reset_index(drop=True)
    
    # calculate previous and next deltas
    df_no_duplicates['spo2_delta_previous'] = df_no_duplicates['saturation'].diff().abs()
    df_no_duplicates['spo2_delta_next'] = df_no_duplicates['saturation'].diff(-1).abs()
    
    # set 0s for the first and last samples
    df_no_duplicates['spo2_delta_previous'].iloc[0] = 0
    df_no_duplicates['spo2_delta_next'].iloc[-1] = 0
    
    # df_no_duplicates['spo2_stable'] = (df_no_duplicates[['spo2_delta_previous', 'spo2_delta_next']].min(axis=1) < delta_threshold)
    df_no_duplicates['spo2_stable'] = np.where(
        df_no_duplicates.index == 0, 
        df_no_duplicates['spo2_delta_next'] < delta_threshold,
        np.where(
            df_no_duplicates.index == len(df_no_duplicates) - 1,
            df_no_duplicates['spo2_delta_previous'] < delta_threshold,
            (df_no_duplicates[['spo2_delta_previous', 'spo2_delta_next']].min(axis=1) < delta_threshold)
        )
    )
    
    df = pd.merge(df, df_no_duplicates[['session', 'sample', 'spo2_delta_previous', 'spo2_delta_next', 'spo2_stable']], on=['session', 'sample'], how='left')

    return df

with st.sidebar:
    st.write('## Session Selector')
    st.markdown('### Filters')
    st.write('Show session where SpO2 difference between adjacent samples is > :')
    
    max_bias = st.number_input('Show adjacent samples SpO2 difference greater than', 0, 20, 10, 1)
    
    ### Apply the function
    cleaned_merged = pd.concat([clean_outlier_spo2(group, max_bias) for _, group in cleaned_merged.groupby(['session', 'device'])], ignore_index=True)
    
    sessionlist = cleaned_merged[~cleaned_merged['spo2_stable']]['session'].unique().tolist()
    devicelist = cleaned_merged[~cleaned_merged['spo2_stable']]['device'].unique().tolist()
    st.write(f'Found {len(devicelist)} devices in {len(sessionlist)} sessions with unstable SpO2 readings')
    selected_session = st.selectbox('Select a session', sessionlist)
    selected_device = st.selectbox('Select a device', cleaned_merged[(cleaned_merged['session'] == selected_session) & (~cleaned_merged['spo2_stable'])]['device'].unique().tolist())
    
    frame = cleaned_merged[(cleaned_merged['session'] == selected_session) & (cleaned_merged['device'] == selected_device)][['sample', 'so2', 'saturation', 'spo2_delta_previous', 'spo2_delta_next', 'spo2_stable']]
    frame = frame.rename(columns={'so2': 'sao2', 'saturation': 'spo2'})
    
st.markdown('## Session ' + str(selected_session) + ', Device ' + str(selected_device))
st.write(frame)

plotcolumns = ['sao2', 'spo2']

fig = go.Figure()
for column in plotcolumns:
    fig.add_trace(go.Scatter(
        x=frame['sample'], y=frame[column],
        mode='markers',
        name=column,
        marker=dict(
            symbol= colormap[column][1],
            color= colormap[column][0],
            size=12,
            opacity=0.8,
            line=dict(width=1, color='DarkSlateGrey')
        )
    ))

st.plotly_chart(fig, use_container_width=True)


    
    


