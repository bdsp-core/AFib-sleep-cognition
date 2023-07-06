import os
import numpy as np
import pandas as pd


def main():
    mros_folder = r'D:\MrOSFEB23_dataset'

    df_vs1_psg_status = pd.read_sas(os.path.join(mros_folder, 'POSFEB23/POSFEB23.SAS7BDAT'))
    df_vs1_psg_status = df_vs1_psg_status[
        #(df_vs1_psg_status.POAPHYUN==0)&  # SCORING APNEA/HYPOPNEA UNRELIABLE
        #(df_vs1_psg_status.POARUNR==0)&   # SCORING AROUSALS UNRELIABLE
        (df_vs1_psg_status.PODLBEG==0)&   # DATA LOST AT BEGINNING OF STUDY
        (df_vs1_psg_status.PODLBW==0)&    # DATA LOST - RECORDING ENDED BEFORE WAKE
        (df_vs1_psg_status.PODLEND==0)&   # DATA LOST AT END OF STUDY
        #(df_vs1_psg_status.POPRREM==0)&   # SCORING REM/NREM UNRELIABLE
        (df_vs1_psg_status.POPRSLP==0)&   # SCORING WAKE/SLEEP UNRELIABLE
        (df_vs1_psg_status.POPRSTAG==0)   # NO STAGING AVAILABLE
        ].reset_index(drop=True)
    df_vs1_psg_status['ID'] = df_vs1_psg_status.ID.str.decode("utf-8")

    df_vs1_sleep = pd.read_csv(r'D:\projects\AD_PD_prediction_from_sleep\dataset_MrOS.csv')
    feat_cols = [x for x in df_vs1_sleep if x.split('_')[0] in ['macro','bp','sp','sw','EMG','REM','HB']]#,'log' CPC is not accurate in AFib
    feat_cols.extend([
    #'powaso', 'potmst1p', 'potmst2p', 'potms34p', 'potmremp', 'poslpeff', 'posllatp',
    #'poavgplm', 'poavplma',
    #'poordi3', 'poordi4', 'pocai4p',
    'popcsa90', 'poai_all', ])
    #feat_cols.remove('log_CPC_l2h_XueSong')
    #feat_cols.remove('log_CPC_vl2lh_XueSong')

    cov_cols = ['vsage1',# 'v2age1', 'v3age1', 'v4age1',
    'gieduc', 'girace',
    'mhparkt', 'mhmi', 'mhstrk', 'mhbp', 'mhdiab', 'tusmknow', 'tusmkcgn', 'dpgdsyn',
    'epepwort', 'epeds', 'qlfxst51',
    'hwbmi', 'pascore', 'dpgds15', 'tursmoke',
    'vsbenzo', 'vsslpmed', 'vscpap',]

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

    df_vs1_sleep = df_vs1_sleep.rename(columns={'id':'ID'})
    df_vs1_sleep = df_vs1_sleep.rename(columns={x:'M_'+x for x in feat_cols})
    df_vs1_sleep = df_vs1_sleep.rename(columns={x:'L_'+x for x in cov_cols})
    df = df_vs1_sleep.merge(df_vs1_psg_status[[
            'ID', 'POSTDYDT', 'POAHI3', 'POAHI4', 'POAVGPLM'
        ]].rename(columns={
            'POSTDYDT':'date_PSG',
            'POAHI3':'M_AHI3',
            'POAHI4':'M_AHI4',
            'POAVGPLM':'M_PLMI'
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
            'ID', 'VSDATE',# 'VSAGE1', 'EPEPWORT', 'PASCORE',
            #'BPBPSYSM', 'BPBPDIAM',
            'MHAFIB', 'TMMSCORE', 'TBSECON', 'DVTMSCOR'
        ]].rename(columns={
            'VSDATE':'date_vs1',
            #'VSAGE1':'L_Age', 'EPEPWORT':'L_ESS', 'PASCORE':'L_PASE',
            #'BPBPDIAM':'L_BPDIA', 'BPBPSYSM':'L_BPSYS',
            'MHAFIB':'A_AF_history',
            'TMMSCORE':'Y_Cog3MS_vs1', 'TBSECON':'Y_CogTMB_time_vs1', 'DVTMSCOR':'Y_CogDVT_time_vs1'
        }), on='ID', how='inner', validate='1:1')
    #df['L_BP'] = df.L_BPSYS/3 + df.L_BPDIA/3*2

    """
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
    """

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
    # remove lots of NA columns
    df = df.drop(columns=[
        'M_sp_sw_coupl_phase_mean_C',
        'M_sp_sw_coupl_phase_std_C',
        'M_sp_sw_coupl_pac_mi_C',
        'M_sw_slope_neg_rem_change_rel_N2N3_C',
        'M_sw_slope_pos_rem_change_rel_N2N3_C',
        'L_tusmkcgn', 'L_mhparkt'])
    # remove colinear/duplicated columns
    df = df.drop(columns=[
        'tusmknow', 'dpgdsyn', 'epeds', 'vsslpmed'
        ])
    print(df)
    df.to_excel('dataset.xlsx', index=False)


if __name__=='__main__':
    main()

