import datetime
from itertools import product
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne
from sklearn.impute import KNNImputer
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 12.6})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_style('ticks')
from myfunctions_psg import *
#from pattern_detection import *


def main(display_type):
    df = pd.read_excel('../dataset.xlsx')
    As = [0,1]

    """
    df_res = pd.read_csv('significant_features.csv')
    feature_names = df_res.Name.values
    ref_vals = {
        0: df_res['value(-)'].values,
        1: df_res['value(+)'].values,
        }
    weights = np.array([1,1,10,1,5,10,1,1,1,5,5,10,1,1,5,1])
    important_feature_ids = np.where(weights>1)[0]
    assert len(feature_names)==len(weights)

    K = 5
    example_ids = {}
    for a in [0,1]:
        ids = np.where(df['A_AF_ECG']==a)[0]
        X = df.loc[ids, feature_names].values
        Xmean = np.nanmean(X, axis=0)
        Xstd = np.nanstd(X, axis=0)
        X = (X-Xmean)/Xstd
        X = KNNImputer(n_neighbors=10).fit_transform(X)
        ref = (ref_vals[a]-Xmean)/Xstd
        dists = np.sqrt(np.sum((X - ref)**2*weights, axis=1))
        example_ids[a] = ids[np.argsort(dists)][:K]

    id0s = []; id1s = []
    age0s = []; age1s = []
    age_diff = []
    consistent_sign_ratio = []
    consistent_sign_ratio2 = []
    for id0, id1 in product(example_ids[0], example_ids[1]):
        id0s.append(id0)
        id1s.append(id1)
        age0s.append(df.iloc[id0].L_VSAGE1)
        age1s.append(df.iloc[id1].L_VSAGE1)
        age_diff.append(np.abs(df.iloc[id0].L_VSAGE1-df.iloc[id1].L_VSAGE1))
        consistent_sign = []
        for j, fn in enumerate(feature_names):
            s1 = df_res.effect[df_res.Name==fn].iloc[0]
            s2 = df.loc[id1,fn]-df.loc[id0,fn]
            consistent_sign.append(np.sign(s1)==np.sign(s2))
        consistent_sign_ratio.append(np.array(consistent_sign).mean())
        consistent_sign_ratio2.append(np.array(consistent_sign)[important_feature_ids].mean())
    df_choose_example = pd.DataFrame(data={
        'ID0':id0s, 'ID1':id1s,
        'Age0':age0s, 'Age1':age1s,
        'AgeAbsDiff':age_diff,
        'ConsistentSignRatio':consistent_sign_ratio,
        'ImportantConsistentSignRatio':consistent_sign_ratio2,
        })
    df_choose_example = df_choose_example.sort_values(
        ['ImportantConsistentSignRatio', 'AgeAbsDiff', 'ConsistentSignRatio'],
        ascending=[False, True, False], ignore_index=True)
    print(df_choose_example)
    df_choose_example.to_excel(f'df_choose_example_top{K}.xlsx',index=False)
    """
    example_ids = {0:2095, 1:855}
    print({x:df.ID.iloc[y] for x,y in example_ids.items()})
    #rclone copy box:"PSG MrOS SOF/EDF/sd/sd8164.edf" D:\projects\AFib-sleep-cognition\MrOS-analysis\figures\ -P
    #rclone copy box:"PSG MrOS SOF/EDF/sd/sd8164.edf.XML" D:\projects\AFib-sleep-cognition\MrOS-analysis\figures\ -P
    base_folder = r'D:\projects\AFib-sleep-cognition\MrOS-analysis\figures'
    paths = {
        a:[os.path.join(base_folder, f'{df.ID.iloc[example_ids[a]]}.edf'),
            os.path.join(base_folder, f'{df.ID.iloc[example_ids[a]]}.edf.XML')] for a in As}
    print(paths)

    notch_freq = 60  # [Hz]
    epoch_time = 30  # [s]
    Fs = 512

    data_path = '../big_files/example_data.pickle'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as ff:
            res = pickle.load(ff)
        start_times = res['start_times']
        sleep_stages = res['sleep_stages']
        specs_db = res['specs_db']
        freqs = res['freqs']
        eegs = res['eegs']
        apnea_events = res['apnea_events']
        spo2 = res['spo2']
        spindle_detections = res['spindle_detections']
    else:
        start_times = {}
        sleep_stages = {}
        eegs = {}
        specs_db = {}
        freqs = {}
        apnea_events = {}
        spo2 = {}
        spindle_detections = {}
        for a in As:
            signals, sleep_stages_, df_event, params = load_mros_sof_data(paths[a][0], paths[a][1])
            Fs = params['Fs']
            start_time = params['start_time']
            ch_names = params['ch_names']
            signals, sleep_stages_, start_time = delete_start_end(signals, sleep_stages_, start_time, Fs, 5, pad_sec=180)
            eeg = signals[:2]
            spo2_ = signals[2]
            spo2_[spo2_>100] = 100
            q1, q3 = np.nanpercentile(spo2_, (25,75))
            spo2_[spo2_<q1-(q3-q1)*4] = np.nan

            if notch_freq<Fs/2:
                eeg = mne.filter.notch_filter(eeg, Fs, notch_freq, verbose=False)
            eeg = mne.filter.filter_data(eeg, Fs, 0.3, 35, verbose=False)
            epoch_size = int(round(Fs*epoch_time))
            epoch_ids = np.arange(0, eeg.shape[1]-epoch_size+1, epoch_size)
            epochs = np.array([eeg[:,x:x+epoch_size] for x in epoch_ids])
            BW = 10*2/epoch_time
            spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.5, fmax=20, bandwidth=BW, normalization='full', verbose=False)

            sp_freq = get_spindle_peak_freq(spec, freq, sleep_stages_[epoch_ids], freq_range=[11,15])
            if np.all(np.isnan(sp_freq)):
                sp_freq[:] = 13.5
            else:
                sp_freq[np.isnan(sp_freq)] = np.nanmedian(sp_freq)
            sp_res = my_spindle_detect(
                eeg, sleep_stages_, Fs, ch_names[:2],
                include=[2,1],
                freq_sp=[[x-1,x+1] for x in sp_freq],
                thresh={'corr':0.73, 'rel_pow':0.07, 'rms':1},
                verbose=False)#, return_filtered_signal=True)

            start_times[a] = start_time
            sleep_stages[a] = sleep_stages_[epoch_ids]
            specs_db[a] = 10*np.log10(spec)
            freqs[a] = freq
            eegs[a] = eeg
            apnea_events[a] = df_event[df_event.Name.str.contains('pnea', case=False)].reset_index(drop=True)
            spo2[a] = spo2_
            spindle_detections[a] = sp_res['detection']
        with open(data_path, 'wb') as ff:
            pickle.dump({
                'start_times':start_times,
                'sleep_stages':sleep_stages,
                'specs_db':specs_db,
                'freqs':freqs,
                'eegs':eegs,
                'apnea_events':apnea_events,
                'spo2':spo2,
                'spindle_detections':spindle_detections,
                }, ff)
        

    figsize = (12,7)
    save_name = 'example_signals'
    colors = {0:'b', 1:'r'}
    labels = {0:'AFib-', 1:'AFib+'}

    plt.close()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2,2,width_ratios=[5,3])
    gs_psg = {0:gs[0,0].subgridspec(3,1,height_ratios=[5,10,5],hspace=0.01),
              1:gs[1,0].subgridspec(3,1,height_ratios=[5,10,5],hspace=0.01)}
    gs_spec = gs[0,1].subgridspec(2,3)
    gs_sp   = gs[1,1].subgridspec(2,1,hspace=0,wspace=0)

    ## plot PSG

    for a in As:
        tt = np.arange(len(sleep_stages[a]))*epoch_time/3600
        xticks = np.arange(0, np.floor(tt.max())+1) 
        xticklabels = []
        for j, x in enumerate(xticks):
            dt = start_times[a]+datetime.timedelta(hours=x)
            xx = datetime.datetime.strftime(dt, '%H:%M')#\n%m/%d/%Y')
            xticklabels.append(xx)
        age = df.L_VSAGE1.iloc[example_ids[a]]
        BMI = df.L_HWBMI.iloc[example_ids[a]]
        AHI = df.L_AHI3.iloc[example_ids[a]]
        
        # plot hypnogram
        ax = fig.add_subplot(gs_psg[a][0,0]); ax0 = ax
        ax.step(tt, sleep_stages[a], color='k', where='post')
        ss_r = np.array(sleep_stages[a]); ss_r[ss_r!=4] = np.nan
        ax.step(tt, ss_r, color='r', where='post', lw=2)
        ax.text(-0.06, 1, chr(ord('a')+a), ha='right', va='top', transform=ax.transAxes, fontweight='bold', fontsize=14)
        ax.yaxis.grid(True)
        ax.set_ylim([0.7,5.2])
        ax.set_yticks([1,2,3,4,5])
        ax.set_yticklabels(['N3', 'N2', 'N1', 'R', 'W'])
        ax.set_xlim([tt.min(), tt.max()])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        sns.despine()
        plt.setp(ax.get_xlabel(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        
        # plot spectrogram
        ax = fig.add_subplot(gs_psg[a][1,0], sharex=ax0)
        specs_db_ch = specs_db[a].mean(axis=1)
        ax.imshow(
                specs_db_ch.T, aspect='auto', origin='lower', cmap='turbo',
                vmin=-5, vmax=25,
                extent=(tt.min(), tt.max(), freqs[a].min(), freqs[a].max()))
        txt = ax.text(0.01, 0.95, f'{labels[a]}: Age={age}y, Sex=Male, BMI={BMI:.1f}kg/m2, AHI={AHI:.1f}/hour', ha='left', va='top', transform=ax.transAxes)
        txt.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax.set_ylabel('freq (Hz)')
        ax.set_yticks([1,5,10,13])
        plt.setp(ax.get_xlabel(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)

        # plot apnea and spo2
        ax = fig.add_subplot(gs_psg[a][2,0], sharex=ax0)
        for ii in range(len(apnea_events[a])):
            loc = (apnea_events[a].Start.iloc[ii]+apnea_events[a].Duration.iloc[ii]/2)/3600
            ax.plot([loc,loc], [100,103], lw=0.8, color='r')
        spo2_tt = np.arange(len(spo2[a]))/Fs/3600
        ax.plot(spo2_tt[::2], spo2[a][::2], c='k', lw=0.5)
        ax.set_ylim(85,103)
        ax.set_yticks([85,90,95,100])
        ax.yaxis.grid(True)
        ax.set_ylabel('SpO2')
        sns.despine()

    # stage-wise spectrum

    for si, (stagenum,stage) in enumerate(zip([3,2,1,4,5], ['N1','N2','N3','R','W'])):
        gs_spec_ = gs_spec[si//3,si%3].subgridspec(1,1,hspace=0)
        if si==0:
            ax = fig.add_subplot(gs_spec_[0,0])
            ax0 = ax
        else:
            ax = fig.add_subplot(gs_spec_[0,0], sharex=ax0, sharey=ax0)
        #ax2 = fig.add_subplot(gs_spec_[1,0], sharex=ax)
        for a in As:
            spec_abs = specs_db[a][sleep_stages[a]==stagenum].mean(axis=(0,1))
            aa = np.power(10, specs_db[a]/10)
            aa = aa/aa.sum(axis=2, keepdims=True)
            ax.plot(freqs[a], spec_abs, c=colors[a], lw=1.5, alpha=0.75)
            #spec_rel = aa[sleep_stages[a]==stagenum].mean(axis=(0,1))
            #spec_rel = np.log(spec_rel)
            #ax2.plot(freqs[a], spec_rel, c=colors[a], lw=1.5, alpha=0.75, ls='--')
        ax.text(0.1,0.98,stage,ha='left',va='top',transform=ax.transAxes)
        if si==0:
            ax.text(-0.08, 1, 'c', ha='right', va='top', transform=ax.transAxes, fontweight='bold', fontsize=14)
        ax.set_xlim(freqs[0].min(), freqs[0].max())
        #ax.set_xscale('log')
        ax.set_xticks([1,5,10,13,20])
        #plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlim(0,20)
        ax.xaxis.grid(True)
        #ax2.set_xticks([1,5,10,13,20])
        #ax2.xaxis.grid(True)
        #if si%3==0:
        #    #ax.set_ylabel('PSD (dB)')
        #    pass
        #else:
        plt.setp(ax.get_ylabel(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        #plt.setp(ax2.get_ylabel(), visible=False)
        #plt.setp(ax2.get_yticklabels(), visible=False)
        if si//3==1:
            #ax.set_xlabel('freq (Hz)')
            ax.set_xticks([1,5,10,13,20])
        else:
            plt.setp(ax.get_xlabel(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            #plt.setp(ax2.get_xlabel(), visible=False)
            #plt.setp(ax2.get_xticklabels(), visible=False)
        sns.despine()
    ax = fig.add_subplot(gs_spec[1,2])
    for a in As:
        ax.plot([2,3], [3,4], c=colors[a], lw=1.5, label=f'{labels[a]} abs')
    #for a in As:
    #    ax.plot([2,3], [3,4], c=colors[a], lw=1.5, label=f'{labels[a]} rel', ls='--')
    ax.legend(frameon=False, loc='center')
    ax.set_xlim(0,1)
    ax.axis('off')

    # spindles
    #np.random.seed(2023)
    K = 9
    for a in As:
        amp = spindle_detections[a].Amplitude.mean()
        ids = np.argsort(np.abs(spindle_detections[a].Amplitude.values-amp))[:K]
        gs_sp_ = gs_sp[a].subgridspec(3,3,hspace=0,wspace=0)
        for j in range(K):
            if j==0:
                ax = fig.add_subplot(gs_sp_[0,0])
                ax0 = ax
            else:
                ax = fig.add_subplot(gs_sp_[j//3,j%3], sharex=ax0, sharey=ax0)
            if j==K-1 and a==1:
                # ruler
                ax.plot([-0.5-0.1,0.5-0.1],[-25,-25],c='k')
                ax.plot([0.5-0.1,0.5-0.1],[-25,25],c='k')
                ax.text(0-0.1,-25-5,'1s',ha='center',va='top')
                ax.text(0.5-0.1+0.03,0,r'50$\mu$V',ha='left',va='center')
            else:
                start = spindle_detections[a].iloc[ids[j]].Start-0.5
                end = spindle_detections[a].iloc[ids[j]].End+0.5
                chid = spindle_detections[a].iloc[ids[j]].IdxChannel
                start_idx = int(round(Fs*start))
                end_idx = int(round(Fs*end))
                duration_idx = end_idx-start_idx
                sp_wave = eegs[a][chid, start_idx:end_idx]
                sp_tt = np.linspace(-duration_idx/Fs/2,(duration_idx+1)/Fs/2, duration_idx)
                ax.plot(sp_tt, sp_wave, c=colors[a], lw=0.5)
            if j==0 and a==0:
                ax.text(-0.08, 1, 'd', ha='right', va='top', transform=ax.transAxes, fontweight='bold', fontsize=14)
            ax.set_xlim(-1.25,1.25)
            ax.set_ylim(-50,50)
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07)
    if display_type=='pdf':
        plt.savefig(save_name+'.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig(save_name+'.png', bbox_inches='tight', pad_inches=0.05)
    elif display_type=='svg':
        plt.savefig(save_name+'.svg', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()


if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        elif 'svg' in sys.argv[1].lower():
            display_type = 'svg'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf/svg'%__file__)
    main(display_type)
