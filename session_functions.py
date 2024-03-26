import plotly.graph_objects as go
import numpy as np

def arms(spo2,sao2):
    return np.sqrt(np.mean((spo2-sao2)**2))


colormap = {'Masimo 97/SpO2':'IndianRed',
            'masimo_bias':'IndianRed',
            'Nellcor/SpO2':'palegreen',
            'nellcor_bias':'palegreen',
            'so2':'powderblue',
            'so2_range':'powderblue',
            'so2_compare':'powderblue',}

def threesamples(abg):
    ## goal: if the CRC was doing 3 runs for the blood sample, throw out the so2 value that is off
    # group by encountercol and sample and count the number of so2 values
    # if there are only two so2 values, then the so2 range is the difference between the two so2 values
    # else if there are more than two samples, calculate the so2 range for the two samples that has so2 value less than 0.5 apart.
    # else if there are no samples that are less than 0.5 apart, then the so2 range is the difference between the highest and lowest so2 values - werid if has
    exclude_list = []

    for group, sample in abg.groupby(['session','sample']):
        #group is a tuple of (current session, current sample), and sample is the dataframe filtered 
            ## --------------- calculate the difference between the median and the so2 value here
            median_so2 = sample['so2'].median()
            sample['diff'] = sample['so2'] - median_so2
            ## --------------- if any difference greater than 1.5, we need to skip this sample - might be a trailing 0s problem or something else weird
            # if len(sample[sample['diff'].abs() >= 1.5]) > 0:
            #     # print(group[0])
            #     abg.loc[(abg['session'] == group[0]) & (abg['sample'] == group[1]), 'so2_range'] = sample['so2'].max() - sample['so2'].min()
            #     continue
    
            # now we've skipped those samples that have so2 values that are more than 1.5 apart
            if len(sample) == 2:
                # if the two so2 values are more than 1 apart, then we need to exclude this sample
                if sample['so2'].max() - sample['so2'].min() > 1:
                    exclude_list.append([group[0], group[1], sample['so2'].max(), sample['so2'].min(), sample['so2'].max() - sample['so2'].min()])
                abg.loc[(abg['session'] == group[0]) & (abg['sample'] == group[1]), 'so2_range'] = sample['so2'].max() - sample['so2'].min()

            elif len(sample) > 2:
                try:
                    # if none of the samples are less than 0.5 apart, then we need to exclude this sample
                    if len(sample[sample['diff'].abs() <= 0.5]) < 2:
                        abg.drop(abg[(abg['session'] == group[0]) & (abg['sample'] == group[1])].index, inplace=True)
                    else:
                        # if we do have two so2 values that are no more than 0.5 apart, drop the row where the difference is greater than 0.5
                        abg.drop(sample[sample['diff'].abs() > 0.5].index, inplace=True)
                        # now calculate the difference between the highest and lowest so2 values
                        abg.loc[(abg['session'] == group[0]) & (abg['sample'] == group[1]), 'so2_range'] = sample['so2'].max() - sample['so2'].min()
                except Exception as e:
                    print('Exception!: ', e)

    # ###---------------------------------------------------------------------------------------------------------------------------------------------------------###
    # drop rows of where session and sample is in the exclude list
    for row in exclude_list:
        abg.drop(abg[(abg['session'] == row[0]) & (abg['sample'] == row[1])].index, inplace=True)

def recalculate_so2_range(df, encountercol):
    if 'so2_range' in df.columns:
        df.drop('so2_range', axis=1, inplace=True)
    ###---------------------------------------------------------------------------------------------------------------------------------------------------------###
    for encounter in df['session'].unique():
        df_encounter = df[df['session'] == encounter]
        for sample in df_encounter['sample'].unique():
            df_encounter_sample = df_encounter[df_encounter['sample'] == sample].copy()
            
            ## --------------- calculate the difference between the median and the so2 value here
            median_so2 = df_encounter_sample['so2'].median()
            df_encounter_sample['diff'] = df_encounter_sample['so2'] - median_so2
            ## --------------- if any difference greater than 1.5, we need to skip this sample - might be a trailing 0s problem or something else weird
            ## we comment out this part of the code now since we start reviewing session in the QA script 
            # if len(df_encounter_sample[df_encounter_sample['diff'].abs() >= 1.5]) > 0:
            #     df.loc[(df['session'] == encounter) & (df['sample'] == sample), 'so2_range'] = df_encounter_sample['so2'].max() - df_encounter_sample['so2'].min()
            #     continue
            
            # now we skip those samples that have so2 values that are more than 1.5 apart
            if len(df_encounter_sample) == 2:
                df.loc[(df['session'] == encounter) & (df['sample'] == sample), 'so2_range'] = df_encounter_sample['so2'].max() - df_encounter_sample['so2'].min()
            elif len(df_encounter_sample) > 2:
                # if we do have two so2 values that are no more than 0.5 apart, drop the row where the difference is greater than 0.5
                if len(df_encounter_sample[df_encounter_sample['diff'].abs() <= 0.5]) > 0:
                    df.drop(df_encounter_sample[df_encounter_sample['diff'].abs() > 0.5].index, inplace=True)
                    # now calculate the difference between the highest and lowest so2 values
                    df.loc[(df['session'] == encounter) & (df['sample'] == sample), 'so2_range'] = df_encounter_sample['so2'].max() - df_encounter_sample['so2'].min()
                else:
                    df.loc[(df['session'] == encounter) & (df['sample'] == sample), 'so2_range'] = df_encounter_sample['so2'].max() - df_encounter_sample['so2'].min()
    ###---------------------------------------------------------------------------------------------------------------------------------------------------------###
    
    so2_range = df.groupby([encountercol,'sample'])['so2'].agg(so2_range=lambda x: x.max() - x.min()).reset_index()
    # # df = df.merge(so2_range, on=[encountercol,'sample'])
    # so2_count = len(df[abs(df['so2_range']) > so2thresh])
    # fig = px.scatter(df, x='sample', y='so2_range', color='sample', template='plotly_white', hover_data=['so2','so2_range','sample','session']) 
    # fig.update_layout(title='So2 Range by Sample: Number of so2 data points that has range > {} = {}'.format(so2thresh, so2_count))
    # return fig
    return df

# create a function that checks the value of so2 in each row and compares it to the so2 value of the previous row
def sample_stability(df, value_to_check, newcol_name, bound):
    sample_stability_previous = []
    sample_stability_next = []
    value_keep = []
    # iterate through the so2 values and compare the current so2 value to the previous so2 value and the next so2 value
    for i in range(1, len(df)): # check each sample against the previous sample
        sample_stability_previous.append(df[value_to_check].iloc[i] - df[value_to_check].iloc[i-1])
        if i != len(df)-1: # if we are not at the last sample, also check the next sample
            sample_stability_next.append(df[value_to_check].iloc[i] - df[value_to_check].iloc[i+1])        
    # the first sample has no previous sample, so we insert a 0 at the beginning of the list
    sample_stability_previous.insert(0,0)
    sample_stability_next.insert(0,0)

    # the last sample has no next sample, so we insert a 0 at the end of the list
    sample_stability_next.append(0)

    for i in range(len(sample_stability_previous)):
        if abs(sample_stability_previous[i]) > bound: 
            if abs(sample_stability_next[i]) > bound:
                value_keep.append('reject')
            else:
                value_keep.append('keep')
        else:
            value_keep.append('keep')

    df[newcol_name + '_previous'] = sample_stability_previous
    df[newcol_name + '_next'] = sample_stability_next
    df[newcol_name+ '_keep'] = value_keep

    return df

def apply_sample_stability(df, value_to_check, newcol_name, bound):
    df = sample_stability(df, value_to_check, newcol_name, bound)
    return df

# Define a function to assign marker style based on the values of name_keep
def assign_marker_style(name_keep):
    if name_keep == 'keep':
        return 'circle'
    elif name_keep == 'manual rejection':
        return 'bowtie'
    elif name_keep == 'algorithm rejection':
        return 'hourglass'
    elif name_keep == 'reject':
        return 'cross'

def assign_marker_color(name_keep, column):
    if name_keep == 'keep':
        return colormap[column]
    else:
        return 'purple'  # You can define additional conditions and colors as needed

def create_scatter(frame):
    fig = go.Figure()

    # Adding traces with customized marker style based on name_keep
    # for column, keepcolumn in zip(['Masimo 97/SpO2', 'Nellcor/SpO2', 'so2'], ['masimo_keep', 'nellcor_keep', 'so2_compare']):
    for column, keepcolumn in zip(['so2'], ['so2_compare']):
        fig.add_trace(go.Scatter(
            x=frame['sample'], y=frame[column],
            mode='markers',
            name=column,
            marker=dict(
                symbol=[assign_marker_style(style) for style in frame[keepcolumn]],
                color=[assign_marker_color(style, column) for style in frame[keepcolumn]],
                # color=frame[column],  # Marker color based on y value
                # colorscale='Viridis',  # You can specify a color scale if needed
                size=12,
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=column
        ))

    # Update layout and show the plot
    fig.update_layout(
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def so2_compare(row):
    if (row['manual_clean_so2'] == 'reject') & (row['so2_keep'] == 'keep'):
        return 'manual rejection'
    elif (row['manual_clean_so2'] == 'keep') & (row['so2_keep'] == 'reject'):
        return 'algorithm rejection'
    elif (row['manual_clean_so2'] == 'reject') & (row['so2_keep'] == 'reject'):
        return 'reject'
    else:
        return 'keep'