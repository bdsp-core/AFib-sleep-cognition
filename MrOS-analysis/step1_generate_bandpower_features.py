"""
spindle/SO features from project AD_PD_prediction_from_sleep (YASA)
BAI from project AD_PD_prediction_from_sleep
band power features are computed here, using pre-computed spectrogram saved in mat files during BAI computation above
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import linregress
from tqdm import tqdm


def main():
    df = pd.read_csv(r'D:\projects\AD_PD_prediction_from_sleep\mastersheet_MrOS_SOF.csv')
    df = df[df.Dataset=='MrOS'].reset_index(drop=True)
    channel = 'C' # assumes only one channel
    stages = ['W', 'N1', 'N2', 'N3', 'R']
    stages_num = [5,3,2,1,4]
    bands = ['delta', 'theta', 'alpha']
    band_ranges = [[1,4], [4,8], [8,12]]
    total_freq_range = [0.3,35]

    mat_folder = r'D:\projects\AD_PD_prediction_from_sleep\step5_extract_brain_age\features_MrOS_SOF'

    for si in tqdm(range(len(df))):
        sid = df.SID.iloc[si]
        mat_path = os.path.join(mat_folder, f'features_{sid}.mat')
        if not os.path.exists(mat_path):
            continue

        mat = sio.loadmat(mat_path)#, variable_names=['EEG_specs','EEG_frequency','sleep_stages'])
        combined_EEG_channels = mat['combined_EEG_channels'].tolist()
        combined_EEG_channels_ids = mat['combined_EEG_channels_ids'].tolist()
        ch_idx = combined_EEG_channels_ids[combined_EEG_channels.index(channel)]
        specs = mat['EEG_specs']
        freq = mat['EEG_frequency'].flatten()
        sleep_stages = mat['sleep_stages'].flatten()
        epoch_start_idx = mat['epoch_start_idx'].flatten()
        Fs = mat['Fs'].item()
        epoch_start_hour = epoch_start_idx/Fs/3600
        dfreq = freq[1]-freq[0]
        specs = specs[...,(freq>=total_freq_range[0])&(freq<=total_freq_range[1])]

        for s, sn in zip(stages, stages_num):
            stage_ids = sleep_stages==sn
            if stage_ids.sum()<=2:
                continue
            for b, br in zip(bands, band_ranges):
                freq_ids = (freq>=br[0])&(freq<br[1])
                spec_ = specs[stage_ids][:,ch_idx]
                bp = spec_[...,freq_ids].sum(axis=-1)*dfreq
                bp_db = 10*np.log10(bp)
                tp = spec_.sum(axis=-1)*dfreq
                df.loc[si, f'bp_{b}_abs_mean_{s}_{channel}'] = bp_db.mean()
                df.loc[si, f'bp_{b}_rel_mean_{s}_{channel}'] = (bp/tp).mean()
                if s=='N3' and b=='delta':  # delta band power slope during N3
                    t = epoch_start_hour[stage_ids]
                    bp_db = bp_db.mean(axis=1)
                    df.loc[si, f'bp_{b}_abs_slope_{s}_{channel}'] = linregress(t,bp_db).slope

    df = df.drop(columns=['Site', 'Dataset', 'Fs', 'StartTime', 'Channel', 'EDFPath', 'AnnotPath'])
    print(df)
    import pdb;pdb.set_trace()
    df.to_csv('bandpower_features2.csv', index=False)
                

if __name__=='__main__':
    main()
