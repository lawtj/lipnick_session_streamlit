import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from redcap import Project
import io

st.set_page_config(layout="wide")


api_url = 'https://redcap.ucsf.edu/api/'
api_k = st.secrets['api_k']
proj = Project(api_url, api_k)
sessionlist = [194.0,
 197.0,
 198.0,
 195.0,
 196.0,
 201.0,
 200.0,
 202.0,
 204.0,
 203.0,
 205.0,
 199.0,
 172.0,
 173.0,
 175.0,
 174.0,
 176.0,
 177.0,
 178.0,
 179.0,
 180.0,
 181.0,
 51.0,
 52.0,
 53.0,
 54.0,
 56.0,
 58.0,
 59.0,
 60.0,
 61.0,
 62.0,
 63.0,
 64.0,
 65.0,
 66.0,
 68.0,
 69.0,
 70.0,
 71.0,
 76.0,
 77.0,
 331.0,
 78.0,
 75.0,
 82.0,
 85.0,
 131.0,
 132.0,
 138.0,
 83.0,
 133.0,
 134.0,
 84.0,
 135.0,
 136.0,
 137.0,
 139.0,
 86.0,
 87.0,
 88.0,
 89.0,
 90.0,
 91.0,
 92.0,
 94.0,
 95.0,
 96.0,
 97.0,
 99.0,
 100.0,
 101.0,
 104.0,
 105.0,
 106.0,
 108.0,
 109.0,
 110.0,
 111.0,
 182.0,
 79.0,
 80.0,
 81.0,
 147.0,
 1.0,
 2.0,
 3.0,
 4.0,
 5.0,
 6.0,
 7.0,
 8.0,
 9.0,
 10.0,
 11.0,
 12.0,
 13.0,
 14.0,
 15.0,
 16.0,
 17.0,
 18.0,
 19.0,
 20.0,
 21.0,
 22.0,
 23.0,
 24.0,
 25.0,
 27.0,
 28.0,
 29.0,
 30.0,
 31.0,
 32.0,
 33.0,
 34.0,
 35.0,
 36.0,
 37.0,
 38.0,
 39.0,
 40.0,
 41.0,
 42.0,
 43.0,
 44.0,
 45.0,
 46.0,
 47.0,
 48.0,
 49.0,
 50.0,
 160.0,
 161.0,
 162.0,
 167.0,
 168.0,
 169.0,
 170.0,
 171.0,
 26.0,
 207.0,
 206.0,
 1100.0,
 295.0,
 112.0,
 297.0,
 317.0,
 302.0,
 301.0,
 233.0,
 55.0,
 67.0,
 93.0,
 102.0,
 98.0,
 103.0,
 148.0,
 117.0,
 114.0,
 318.0,
 336.0,
 313.0,
 337.0,
 314.0,
 74.0,
 339.0,
 288.0,
 289.0,
 290.0,
 296.0,
 300.0,
 303.0,
 304.0,
 309.0,
 311.0,
 312.0,
 338.0,
 340.0,
 107.0,
 235.0,
 341.0,
 342.0,
 343.0,
 345.0,
 346.0,
 347.0,
 348.0,
 349.0,
 350.0,
 351.0,
 320.0,
 321.0,
 322.0,
 323.0,
 324.0,
 335.0,
 352.0,
 353.0,
 354.0,
 355.0,
 356.0,
 357.0,
 333.0,
 325.0,
 326.0,
 327.0,
 328.0,
 329.0,
 330.0,
 332.0,
 334.0,
 113.0,
 115.0,
 116.0,
 118.0,
 119.0,
 120.0,
 358.0,
 359.0,
 361.0,
 362.0,
 360.0,
 363.0,
 365.0,
 368.0,
 364.0,
 366.0,
 367.0,
 369.0,
 370.0,
 371.0,
 372.0,
 373.0,
 374.0,
 375.0,
 376.0,
 377.0,
 378.0,
 379.0,
 381.0,
 382.0,
 380.0,
 383.0,
 384.0,
 385.0,
 386.0,
 387.0,
 388.0,
 389.0,
 390.0,
 392.0,
 391.0,
 393.0,
 394.0,
 395.0,
 396.0,
 397.0,
 398.0,
 399.0,
 400.0,
 401.0,
 402.0,
 403.0,
 404.0,
 408.0,
 409.0,
 410.0,
 411.0,
 418.0,
 419.0,
 420.0,
 421.0,
 422.0,
 423.0,
 429.0,
 427.0,
 426.0,
 425.0,
 424.0,
 413.0,
 414.0,
 416.0,
 431.0,
 432.0,
 433.0,
 434.0,
 435.0,
 436.0,
 437.0,
 439.0,
 440.0,
 444.0,
 443.0,
 442.0,
 441.0]
if 'labview_session_abg' not in st.session_state:
    f = io.BytesIO(proj.export_file(record='2', field='file')[0])
    labview_session_abg = pd.read_parquet(f)
    labview_session_abg = labview_session_abg[labview_session_abg['session'].isin(sessionlist)]
    labview_session_abg['masimo_abs_bias'] = np.abs(labview_session_abg['masimo_bias'])
    labview_session_abg['nellcor_abs_bias'] = np.abs(labview_session_abg['nellcor_bias'])
else:
    labview_session_abg = st.session_state['labview_session_abg']

with st.sidebar:
    st.write('## Session Selector')
    st.markdown('Click [here](https://docs.google.com/spreadsheets/d/1fFLITWYsDQYx_rDuYvF1bGgqf-PqbKUVcvQ4I0OTXe0/edit?usp=sharing) to enter comments on the Google Sheet')
    st.markdown('### Filters')
    st.write('Show only those sessions containing at least one sample greater than:')
    so2_range_thresh = st.slider('SO2 Range Threshold', 0, 20, 2)
    masimo_bias_thresh = st.slider('Masimo Bias Threshold', 0, 20, 6)
    nellcor_bias_thresh = st.slider('Nellcor Bias Threshold', 0, 20, 0)
    sessions = labview_session_abg.query('masimo_abs_bias > @masimo_bias_thresh and nellcor_abs_bias > @nellcor_bias_thresh')['session'].sort_values().unique().tolist()
    st.divider()
    st.write('Total number of sessions: ', len(labview_session_abg['session'].unique()))
    st.write(len(sessions), 'sessions match these criteria')
    session = st.selectbox('Select a session', sessions)
    
frame = labview_session_abg[labview_session_abg['session']==session][['session', 'Sample', 'masimo_bias', 'so2_range', 'so2', 'Masimo 97/SpO2', 'nellcor_bias','Nellcor/SpO2']]
colormap = {'Masimo 97/SpO2':'IndianRed',
            'masimo_bias':'IndianRed',
            'Nellcor/SpO2':'palegreen',
            'nellcor_bias':'palegreen',
            'so2':'powderblue',
            'so2_range':'powderblue'}

fig = (px.scatter(
    frame, x='Sample', y=['masimo_bias', 'so2_range', 'so2', 'Masimo 97/SpO2', 'nellcor_bias','Nellcor/SpO2'], color_discrete_map=colormap,)
    .update_traces(marker=dict(size=12,opacity=0.8, 
                    line=dict(width=2,color='DarkSlateGrey')))
)


st.markdown('## Session ' + str(session))
one, two = st.columns(2)
with one:
    st.plotly_chart(fig)
with two:
    st.dataframe(frame, hide_index=True)
