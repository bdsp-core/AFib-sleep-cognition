import re
import os
import numpy as np
import pandas as pd


def main():
    mros_folder = r'R:\CDAC Sleep\HaoqiSun\dataset_MrOSFEB23'

    df_vs1_psg = pd.read_sas(os.path.join(mros_folder, 'POSFEB23/POSFEB23.SAS7BDAT'))
    df_vs1_psg['ID'] = df_vs1_psg.ID.str.decode("utf-8")
    df_vs1_psg = df_vs1_psg[
        #(df_vs1_psg.POAPHYUN==0)&  # SCORING APNEA/HYPOPNEA UNRELIABLE
        #(df_vs1_psg.POARUNR==0)&   # SCORING AROUSALS UNRELIABLE
        (df_vs1_psg.PODLBEG==0)&   # DATA LOST AT BEGINNING OF STUDY
        (df_vs1_psg.PODLBW==0)&    # DATA LOST - RECORDING ENDED BEFORE WAKE
        (df_vs1_psg.PODLEND==0)&   # DATA LOST AT END OF STUDY
        #(df_vs1_psg.POPRREM==0)&   # SCORING REM/NREM UNRELIABLE
        (df_vs1_psg.POPRSLP==0)&   # SCORING WAKE/SLEEP UNRELIABLE
        (df_vs1_psg.POPRSTAG==0)   # NO STAGING AVAILABLE
        ].reset_index(drop=True)
    cols = ['POSTDYDT', 'POAHI3', 'PODSSLP3', 'POPCSA90', 'POAVGPLM', 'POAI_ALL', 'POSLPRDP', 'POWASO', 'POTMST1P', 'POTMST2P', 'POTMS34P', 'POTMREMP', 'POTMST1', 'POTMST2', 'POTMST34', 'POTMREM']
    cols2 = ['date_PSG', 'L_AHI3', 'L_ODI3', 'L_hypoxemia_perc', 'M_PLMI', 'M_AI', 'M_macro_TST', 'M_macro_WASO', 'M_macro_N1Perc', 'M_macro_N2Perc', 'M_macro_N3Perc', 'M_macro_REMPerc', 'M_macro_N1Time', 'M_macro_N2Time', 'M_macro_N3Time', 'M_macro_REMTime']
    df_vs1_psg = df_vs1_psg.rename(columns={x:x2 for x,x2 in zip(cols,cols2)})
    df_vs1_psg['M_macro_TST'] /= 60

    df_vs1_sleep = pd.read_csv(r'D:\projects\AD_PD_prediction_from_sleep\dataset_MrOS.csv')
    df_vs1_sleep = df_vs1_sleep.rename(columns={'id':'ID'})
    df_vs1_sleep2 = pd.read_csv('bandpower_features.csv')  # use ones computed here
    df_vs1_sleep2 = df_vs1_sleep2.rename(columns={'SID':'ID'})
    df_vs1_sleep2['ID'] = df_vs1_sleep2.ID.str.upper()
    df_vs1_sleep = df_vs1_sleep.drop(columns=[x for x in df_vs1_sleep.columns if 'bp_' in x]).merge(df_vs1_sleep2, on='ID', how='left', validate='1:1')
    feat_cols = [x for x in df_vs1_sleep.columns if re.match(r'^(?:bp_|sp_|sw_|EMG_|REM_|HB_|BAI|ArtifactRatio|NumMissingStage)', x, re.I)]#,'log' CPC is not accurate in AFib
    feat_cols += ['macro_SL', 'macro_RL', 'macro_SFI']+[x for x in df_vs1_sleep.columns if 'macro' in x and 'to' in x]  # use other macro from dataset precomputed
    #feat_cols.remove('log_CPC_l2h_XueSong')
    #feat_cols.remove('log_CPC_vl2lh_XueSong')
    df_vs1_sleep = df_vs1_sleep.rename(columns={x:'M_'+x for x in feat_cols})

    """ outcomes
    'vstbs', 'v4tbs', 'v2tbs', 'v3tbs',
    'vstms', 'v3tms', 'v2tms', 'v4tms',
    'v3dement', 'v3dementt', 'v4dement', 'v4dementt', 'dement3', 'dement4', 'inc_dementia', 'prevalent_dementia',
    'v1alzh', 'v2alzh', 'v3alzh', 'v4alzh', 'vsalzh',
    'tmms2sc', 'tmms3sc', 'tmms4sc',
    'v1park', 'vspark', 'vs2park', 'v2park', 'v3park', 'v4park',
    'efstatus', 'dadead', 'fucytime', 'efvsstat', 'efv2stat', 'efv3stat', 'efvs2sta', 'efv4stat', 'fuv2yt', 'fuvsyt', 'fuv3yt', 'fuv4yt',
    """
    #'actnap2p', 'aclsepmp', 'acsminmp', 'acslatmp', 'acwasomp', 'acsefnmp', 'acnap5mp', 'acnap10mp',
    #'slnap', 'slnaphr', 'slnaphwk',
    #?? 'pqpslmed', 'slsa',

    df = df_vs1_sleep.merge(df_vs1_psg, on='ID', how='inner', validate='1:1')

    df_vs1_ecg = pd.read_sas(os.path.join(mros_folder, 'AYSAUG07/AYSAUG07.SAS7BDAT'))
    df_vs1_ecg['ID'] = df_vs1_ecg.ID.str.decode("utf-8")
    df_vs1_ecg = df_vs1_ecg[
        (df_vs1_ecg.AYMISS==0)  # REASONS FOR CANNOT EVALUATE VALUES
        ].reset_index(drop=True)
    df = df.merge(df_vs1_ecg[[
            'ID', 'AYAFF',
        ]].rename(columns={
            'AYAFF':'A_AF_ECG',
        }), on='ID', how='inner', validate='1:1')

    df_med = pd.read_sas(os.path.join(mros_folder, 'MSAUG16/MSAUG16.SAS7BDAT'))
    df_med['ID'] = df_med.ID.str.decode("utf-8")
    df = df.merge(df_med[[
            'ID', 'M1ADEPR', 'M1BENZO',
        ]].rename(columns={
            'M1ADEPR':'L_AntiDep',
            'M1BENZO':'L_Benzo'
        }), on='ID', how='inner', validate='1:1')

    df_vs1 = pd.read_sas(os.path.join(mros_folder, 'VSFEB23/VSFEB23.SAS7BDAT'))
    df_vs1['ID'] = df_vs1.ID.str.decode("utf-8")
    cov_cols = [
    'VSAGE1',
    'MHMI', 'MHMIT', 'MHSTRK', 'MHSTRKT', 'MHBP', 'MHBPT', 'MHDIAB', 'MHDIABT', 'MHCHF', 'MHCHFT',
    'EPEPWORT', 'QLFXST51',
    'HWBMI', 'PASCORE', 'DPGDS15', 'TURSMOKE',
    'VSCPAP',]
    # colinear/duplicated columns
    #'TUSMKNOW', 'DPGDSYN', 'EPEDS', 'VSSLPMED'
    df = df.merge(df_vs1[
        ['ID', 'VSDATE']+cov_cols+[
            #'BPBPSYSM', 'BPBPDIAM',
            'MHAFIB', 'TMMSCORE', 'TBSECON', 'DVTMSCOR'
        ]].rename(columns={
            'VSDATE':'date_vs1',
            #'BPBPDIAM':'L_BPDIA', 'BPBPSYSM':'L_BPSYS',
            'MHAFIB':'A_AF_history',
            'TMMSCORE':'Y_Cog3MS_vs1', 'TBSECON':'Y_CogTMB_time_vs1', 'DVTMSCOR':'Y_CogDVT_time_vs1'
        }|{x:'L_'+x for x in cov_cols}), on='ID', how='inner', validate='1:1')
    #df['L_BP'] = df.L_BPSYS/3 + df.L_BPDIA/3*2
    df['L_MHMI'] = ((df['L_MHMI'].fillna(0).values+df['L_MHMIT'].fillna(0).values)>0).astype(int); df = df.drop(columns=['L_MHMIT'])
    df['L_MHSTRK'] = ((df['L_MHSTRK'].fillna(0).values+df['L_MHSTRKT'].fillna(0).values)>0).astype(int); df = df.drop(columns=['L_MHSTRKT'])
    df['L_MHBP'] = ((df['L_MHBP'].fillna(0).values+df['L_MHBPT'].fillna(0).values)>0).astype(int); df = df.drop(columns=['L_MHBPT'])
    df['L_MHDIAB'] = ((df['L_MHDIAB'].fillna(0).values+df['L_MHDIABT'].fillna(0).values)>0).astype(int); df = df.drop(columns=['L_MHDIABT'])
    df['L_MHCHF'] = ((df['L_MHCHF'].fillna(0).values+df['L_MHCHFT'].fillna(0).values)>0).astype(int); df = df.drop(columns=['L_MHCHFT'])

    df_v1 = pd.read_sas(os.path.join(mros_folder, 'V1FEB23/V1FEB23.SAS7BDAT'))
    df_v1['ID'] = df_v1.ID.str.decode("utf-8")
    cov_cols = ['GIEDUC', 'GIERACE']
    df = df.merge(df_v1[['ID', 'EFDATE']+cov_cols].rename(
        columns={'EFDATE':'date_enroll'}|{x:'L_'+x for x in cov_cols}
        ), on='ID', how='inner', validate='1:1')

    df_v2 = pd.read_sas(os.path.join(mros_folder, 'V2FEB23/V2FEB23.SAS7BDAT'))
    df_v2['ID'] = df_v2.ID.str.decode("utf-8")
    df = df.merge(df_v2[[
            'ID', 'V2DATE', 'TMMSCORE', 'TBSECON',
        ]].rename(columns={
            'V2DATE':'date_v2',
            'TMMSCORE':'Y_Cog3MS_v2', 'TBSECON':'Y_CogTMB_time_v2',
        }), on='ID', how='left', validate='1:1')

    df_v3 = pd.read_sas(os.path.join(mros_folder, 'V3FEB23/V3FEB23.SAS7BDAT'))
    df_v3['ID'] = df_v3.ID.str.decode("utf-8")
    df = df.merge(df_v3[[
            'ID', 'V3DATE', 'TMMSCORE', 'TBSECON',
        ]].rename(columns={
            'V3DATE':'date_v3',
            'TMMSCORE':'Y_Cog3MS_v3', 'TBSECON':'Y_CogTMB_time_v3',
        }), on='ID', how='left', validate='1:1')

    df_vs2 = pd.read_sas(os.path.join(mros_folder, 'VS2FEB23/VS2FEB23.SAS7BDAT'))
    df_vs2['ID'] = df_vs2.ID.str.decode("utf-8")
    df = df.merge(df_vs2[[
            'ID', 'VS2DATE', 'TMMSCORE', 'TBSECON', 'DVTMSCOR'
        ]].rename(columns={
            'VS2DATE':'date_vs2',
            'TMMSCORE':'Y_Cog3MS_vs2', 'TBSECON':'Y_CogTMB_time_vs2', 'DVTMSCOR':'Y_CogDVT_time_vs2'
        }), on='ID', how='left', validate='1:1')

    df_v4 = pd.read_sas(os.path.join(mros_folder, 'V4FEB23/V4FEB23.SAS7BDAT'))
    df_v4['ID'] = df_v4.ID.str.decode("utf-8")
    df = df.merge(df_v4[[
            'ID', 'V4DATE', 'TMMSCORE', 'TBSECON',
        ]].rename(columns={
            'V4DATE':'date_v4',
            'TMMSCORE':'Y_Cog3MS_v4', 'TBSECON':'Y_CogTMB_time_v4',
        }), on='ID', how='left', validate='1:1')

    df = df.rename(columns={x:x.replace('N2+N3','N2N3') for x in df.columns})  # R cannot have + in column names
    # remove lots of NA columns
    df = df.drop(columns=[
        'M_sp_sw_coupl_phase_mean_C',
        'M_sp_sw_coupl_phase_std_C',
        'M_sp_sw_coupl_pac_mi_C',
        'M_sw_slope_neg_rem_change_rel_N2N3_C',
        'M_sw_slope_pos_rem_change_rel_N2N3_C',
        ])
    # dummy race
    df['L_RACE_AFR'] = (df.L_GIERACE==2).astype(int)
    df['L_RACE_ASI'] = (df.L_GIERACE==3).astype(int)
    df['L_RACE_HIS'] = (df.L_GIERACE==4).astype(int)
    df['L_RACE_OTH'] = (df.L_GIERACE==5).astype(int)
    df = df.drop(columns=['L_GIERACE'])

    date_cols = ['date_enroll', 'date_vs1', 'date_PSG', 'date_v2', 'date_v3', 'date_vs2', 'date_v4']
    A_cols = [x for x in df.columns if x.startswith('A_')]
    L_cols = [x for x in df.columns if x.startswith('L_')]
    Y_cols = [x for x in df.columns if x.startswith('Y_')]
    M_cols = [x for x in df.columns if x.startswith('M_')]
    cols = ['ID']+date_cols+A_cols+L_cols+Y_cols+M_cols
    df = df[cols]

    df = df.rename(columns={'M_ArtifactRatio':'EEGArtifactRatio', 'M_NumMissingStage':'NumMissingStage'})
    print(df)
    df.to_excel('dataset.xlsx', index=False)


if __name__=='__main__':
    main()

