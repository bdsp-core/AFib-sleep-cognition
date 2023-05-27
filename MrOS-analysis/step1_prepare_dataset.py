import os
import numpy as np
import pandas as pd


mros_folder = r'D:\projects\MrOSFEB23_dataset'

df_vs1_sleep = pd.read_sas(os.path.join(mros_folder, 'POSFEB23/POSFEB23.SAS7BDAT'))
df_vs1_sleep = df_vs1_sleep[
    (df_vs1_sleep.POAPHYUN==0)&  # SCORING APNEA/HYPOPNEA UNRELIABLE
    #(df_vs1_sleep.POARUNR==0)&   # SCORING AROUSALS UNRELIABLE
    (df_vs1_sleep.PODLBEG==0)&   # DATA LOST AT BEGINNING OF STUDY
    (df_vs1_sleep.PODLBW==0)&    # DATA LOST - RECORDING ENDED BEFORE WAKE
    (df_vs1_sleep.PODLEND==0)&   # DATA LOST AT END OF STUDY
    #(df_vs1_sleep.POPRREM==0)&   # SCORING REM/NREM UNRELIABLE
    (df_vs1_sleep.POPRSLP==0)&   # SCORING WAKE/SLEEP UNRELIABLE
    (df_vs1_sleep.POPRSTAG==0)   # NO STAGING AVAILABLE
    ].reset_index(drop=True)
df_vs1_sleep['ID'] = df_vs1_sleep.ID.str.decode("utf-8")

df_vs1_eeg = pd.read_csv(r'D:\projects\AD_PD_prediction_from_sleep\features.csv')
df_vs1_eeg['SID'] = df_vs1_eeg.SID.str.upper()
df_vs1_eeg = df_vs1_eeg[df_vs1_eeg.Dataset=='MrOS'].reset_index(drop=True).drop(columns='Dataset')
df_vs1_eeg = df_vs1_eeg.rename(columns={'SID':'ID'})
df_vs1_eeg = df_vs1_eeg.rename(columns={x:'M_'+x for x in df_vs1_eeg.columns if x!='ID'})
df = df_vs1_eeg.merge(df_vs1_sleep[[
        'ID', 'POSTDYDT', 'POAHI3', 'POAHI4', 'POAVGPLM'
    ]].rename(columns={
        'POSTDYDT':'date_PSG',
        'POAHI3':'L_AHI3',
        'POAHI4':'L_AHI4',
        'POAVGPLM':'L_PLMI'
    }), on='ID', how='inner', validate='1:1')

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

df_vs1 = pd.read_sas(os.path.join(mros_folder, 'VSFEB23/VSFEB23.SAS7BDAT'))
df_vs1['ID'] = df_vs1.ID.str.decode("utf-8")
df = df.merge(df_vs1[[
        'ID', 'VSDATE', 'VSAGE1', 'EPEPWORT', 'PASCORE',
        'BPBPSYSM', 'BPBPDIAM',
        'MHAFIB', 'TMMSCORE', 'TBSECON', 'DVTMSCOR'
    ]].rename(columns={
        'VSDATE':'date_vs1',
        'VSAGE1':'L_Age', 'EPEPWORT':'L_ESS', 'PASCORE':'L_PASE',
        'BPBPDIAM':'L_BPDIA', 'BPBPSYSM':'L_BPSYS',
        'MHAFIB':'A_AF_history',
        'TMMSCORE':'Y_Cog3MS_vs1', 'TBSECON':'Y_CogTMB_time_vs1', 'DVTMSCOR':'Y_CogDVT_time_vs1'
    }), on='ID', how='inner', validate='1:1')
df['L_BP'] = df.L_BPSYS/3 + df.L_BPDIA/3*2

df_v1 = pd.read_sas(os.path.join(mros_folder, 'V1FEB23/V1FEB23.SAS7BDAT'))
df_v1['ID'] = df_v1.ID.str.decode("utf-8")
df_v1 = df_v1[df_v1.GIERACE==1].reset_index(drop=True)  # take white only... since others are sparse
df_v1['L_QoL_SF12'] = (df_v1.QLPCS12.values+df_v1.QLMCS12.values)/2
df = df.merge(df_v1[[
        'ID', 'EFDATE', 'GIEDUC', 'HWBMI','L_QoL_SF12',
    ]].rename(columns={
        'EFDATE':'date_enroll',
        'GIEDUC':'L_educ', 'HWBMI':'L_BMI',
    }), on='ID', how='inner', validate='1:1')

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

date_cols = [x for x in df.columns if x.startswith('date_')]
A_cols = [x for x in df.columns if x.startswith('A_')]
L_cols = [x for x in df.columns if x.startswith('L_')]
Y_cols = [x for x in df.columns if x.startswith('Y_')]
M_cols = [x for x in df.columns if x.startswith('M_')]
cols = ['ID']+date_cols+A_cols+L_cols+Y_cols+M_cols
df = df[cols]
df = df.rename(columns={x:x.replace('N2+N3','N2N3') for x in df.columns})  # R cannot have + in column names
print(df)
df.to_excel('dataset.xlsx', index=False)

