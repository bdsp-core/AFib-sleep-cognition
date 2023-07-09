import datetime
from collections import defaultdict
from itertools import groupby
import re
import xmltodict
import numpy as np
import pandas as pd
import mne


def delete_start_end(signals, sleep_stages, start_time, Fs, stage, pad_sec=0):
    if np.isnan(stage):
        isgood = ~np.isnan(sleep_stages)
    else:
        isgood = sleep_stages!=stage
    pad = int(round(pad_sec*Fs))
    
    start = 0
    for k,l in groupby(isgood):
        ll = len(list(l))
        if not k:
            start = ll
        break
    start = max(0,start-pad)

    end = 0
    for k,l in groupby(isgood[::-1]):
        ll = len(list(l))
        if not k:
            end = ll
        break
    end = len(sleep_stages)-end
    end = min(signals.shape[1],end+pad)
    return signals[:,start:end], sleep_stages[start:end], start_time+datetime.timedelta(seconds=start/Fs)


def load_mros_sof_data(signal_path, annot_path):
    edf = mne.io.read_raw_edf(signal_path, stim_channel=None, preload=False, verbose=False)
    Fs = edf.info['sfreq']

    ch_names = ['C3', 'C4', 'A2', 'A1', 'SaO2']
    signals = edf.get_data(picks=ch_names)
    signals = np.array([
            (signals[0]-signals[2])*1e6,
            (signals[1]-signals[3])*1e6,
            signals[4]])
    ch_names = ['C3-A2', 'C4-A1', 'SaO2']

    stage_dict = {'0':5, '1':3, '2':2, '3':1, '4':1, '5':4}
    with open(annot_path, encoding='utf-8') as f:
        info_dict = xmltodict.parse(f.read())
    sleep_stages_node = info_dict['CMPStudyConfig']['SleepStages']['SleepStage']
    epoch_time = float(info_dict['CMPStudyConfig']['EpochLength'])
    epoch_size = int(round(Fs*epoch_time))
    sleep_stages = np.zeros(signals.shape[1])+np.nan
    for si, s in enumerate(sleep_stages_node):
        start = si*epoch_size
        end = min(start+epoch_size, signals.shape[1])
        sleep_stages[start:end] = stage_dict.get(s, np.nan)
    if np.all(np.isnan(sleep_stages)):
        raise ValueError('No sleep stages.')

    df_event = pd.DataFrame(info_dict['CMPStudyConfig']['ScoredEvents']['ScoredEvent'])
    df_event.loc[df_event.Name=='Hypopnea','Name'] = 'Hypopnea (AirFlowReduction30-50%)'
    df_event.loc[df_event.Name=='Unsure','Name'] = 'Hypopnea (AirFlowReduction>50%)'
    df_event['Start'] = df_event.Start.astype(float)
    df_event['Duration'] = df_event.Duration.astype(float)
    df_event = df_event.sort_values('Start', ignore_index=True, ascending=True)
    df_event = df_event.rename(columns={'Input':'Channel'})
    #TODO ch_mapping = {ch1:ch2 for ch1,ch2 in zip(ch_names_data, ch_names)}
    #TODO df_event['Channel'] = df_event.Channel.map(ch_mapping)

    spo2_id = ch_names.index('SaO2')
    for i in range(len(df_event)):
        if df_event.Name.iloc[i]=='SpO2 artifact' or df_event.Name.iloc[i]=='SaO2 artifact':
            start_idx = int(round(df_event.Start.iloc[i]*Fs))
            end_idx = int(round((df_event.Start.iloc[i]+df_event.Duration.iloc[i])*Fs))
            signals[spo2_id, start_idx:end_idx] = np.nan

    params = {'Fs':Fs, 'start_time':edf.info['meas_date'].replace(tzinfo=None),
            'ch_names':ch_names}#, 'ch_names_type':ch_names_type}
    return signals, sleep_stages, df_event, params

