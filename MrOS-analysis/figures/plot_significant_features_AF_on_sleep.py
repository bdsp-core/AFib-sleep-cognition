import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_style('ticks')


def main(display_type):
    #suffix = '_macrostructure'
    #suffix = '_age_interaction'
    suffix = ''

    if suffix=='_macrostructure':
        df = pd.read_excel(f'../AF_as_exposure.xlsx')
    else:
        df = pd.read_excel(f'../AF_as_exposure{suffix}.xlsx')
    if suffix == '_age_interaction':
        df2 = pd.read_excel('../AF_as_exposure.xlsx')
        df = pd.concat([df, df2[df2.method=='adj_bart']], ignore_index=True, axis=0)
    if suffix=='_macrostructure':
        names = ['M_AI', 'M_macro_N1Perc', 'M_macro_N1Time', 'M_macro_N1toN1', 'M_macro_N1toN2', 'M_macro_N1toN3', 'M_macro_N1toR', 'M_macro_N1toW', 'M_macro_N2Perc', 'M_macro_N2Time', 'M_macro_N2toN1', 'M_macro_N2toN2', 'M_macro_N2toN3', 'M_macro_N2toR', 'M_macro_N2toW', 'M_macro_N3Perc', 'M_macro_N3Time', 'M_macro_N3toN1', 'M_macro_N3toN2', 'M_macro_N3toN3', 'M_macro_N3toR', 'M_macro_N3toW', 'M_macro_REMPerc', 'M_macro_REMTime', 'M_macro_RL', 'M_macro_RtoN1', 'M_macro_RtoN2', 'M_macro_RtoN3', 'M_macro_RtoR', 'M_macro_RtoW', 'M_macro_SFI', 'M_macro_SL', 'M_macro_TST', 'M_macro_WASO', 'M_macro_WtoN1', 'M_macro_WtoN2', 'M_macro_WtoN3', 'M_macro_WtoR', 'M_macro_WtoW']
    else:
        names = ['M_BAI', 'M_bp_alpha_abs_mean_N1_C', 'M_bp_alpha_abs_mean_N2_C', 'M_bp_alpha_abs_mean_N3_C', 'M_bp_alpha_abs_mean_R_C', 'M_bp_alpha_abs_mean_W_C', 'M_bp_delta_abs_mean_N1_C', 'M_bp_delta_abs_mean_N2_C', 'M_bp_delta_abs_mean_N3_C', 'M_bp_delta_abs_mean_R_C', 'M_bp_delta_abs_mean_W_C', 'M_bp_delta_abs_slope_N3_C', 'M_bp_theta_abs_mean_N1_C', 'M_bp_theta_abs_mean_N2_C', 'M_bp_theta_abs_mean_N3_C', 'M_bp_theta_abs_mean_R_C', 'M_bp_theta_abs_mean_W_C', 'M_sp_amp_N2N3_C', 'M_sp_dens_N2N3_C', 'M_sp_dur_N2N3_C', 'M_sp_dur_total_N2N3_C', 'M_sp_freq_N2N3_C', 'M_sp_sw_coupl_perc_C', 'M_sp_sym_N2N3_C', 'M_sw_amp_N2N3_C', 'M_sw_amp_neg_N2N3_C', 'M_sw_amp_pos_N2N3_C', 'M_sw_dur_perc_N2N3_C', 'M_sw_dur_total_N2N3_C', 'M_sw_freq_N2N3_C', 'M_sw_slope_neg_N2N3_C', 'M_sw_slope_pos_N2N3_C']
    df = df[np.in1d(df.Name, names)].reset_index(drop=True)
    #sig_col = 'sig_bonf'
    #df[sig_col] = df.pval<0.05/len(names)
    sig_col = 'sig_fdr_bh'
    methods = df.method.unique()
    for m in methods:
        ids = df.method==m
        df.loc[ids, sig_col] = multipletests(df[ids].pval.values, method='fdr_bh')[0].astype(bool)

    patterns = df.Name.unique()
    df_res = []
    for i, p in enumerate(patterns):
        if df[sig_col][df.Name==p].sum()>=2 and df.pval[(df.Name==p)&(df.method=='dr')].iloc[0]<0.05:
            df_res.append(df[(df.Name==p)&(df.method=='dr')])
    df_res = pd.concat(df_res, axis=0).sort_values('pval', ignore_index=True)
    df_res.to_csv(f'significant_features{suffix}.csv', index=False)

    for i in range(len(df_res)):
        name = df_res.Name.iloc[i]
        if 'M_macro' in name and 'to' in name:
            df_res.loc[i,'effect'] *= 100
            df_res.loc[i,'lb'] *= 100
            df_res.loc[i,'ub'] *= 100
    types = ['M_']#[r'^M_(?:AI|AHI|bp|sp)', 'M_HB', 'M_macro']
    df_res_types = [df_res[df_res.Name.str.contains(x)].sort_values('effect', ascending=False).reset_index(drop=True) for x in types]
    assert sum([len(x) for x in df_res_types])==len(df_res)
    print(df_res)

    name_mapping = {
    'M_AHI3':'AHI (/h)',
    'M_AI':'Arousal Index (/h)',

    'M_bp_delta_abs_slope_N3_C':r'$\delta$ band power slope overnight(dB/h), N3',
    'M_bp_delta_abs_mean_N1_C':r'$\delta$ band power (dB), N1',
    'M_bp_delta_abs_mean_R_C':r'$\delta$ band power (dB), R',
    'M_bp_alpha_abs_mean_W_C':r'$\alpha$ band power (dB), W',
    'M_bp_alpha_abs_mean_R_C':r'$\alpha$ band power (dB), R',
    'M_bp_alpha_abs_mean_N1_C':r'$\alpha$ band power (dB), N1',
    'M_bp_alpha_abs_mean_N2_C':r'$\alpha$ band power (dB), N2',
    'M_bp_alpha_abs_mean_N3_C':r'$\alpha$ band power (dB), N3',
    'M_bp_theta_abs_mean_N1_C':r'$\theta$ band power (dB), N1',
    'M_bp_theta_abs_mean_N2_C':r'$\theta$ band power (dB), N2',
    'M_bp_theta_abs_mean_N3_C':r'$\theta$ band power (dB), N3',
    'M_bp_theta_abs_mean_R_C':r'$\theta$ band power (dB), R',
    'M_bp_beta_abs_mean_N2_C':r'$\beta$ band power (dB), N2',
    'M_bp_beta_abs_mean_N3_C':r'$\beta$ band power (dB), N3',

    'M_sp_amp_N2N3_C':r'spindle amp ($\mu$V), N2 or N3',

    'M_HB_apnea':'hypoxic burden (%min/h)',

    'M_macro_N1toW':'N1 to W transition prob (%)',
    'M_macro_N1toN1':'N1 to N1 transition prob (%)',
    'M_macro_SFI':'sleep fragmentation index (/h)',
    }

    figsize = (8,max(2,len(df_res)*0.5))
    save_name = f'AF_on_sleep{suffix}'

    plt.close()
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(types),1,height_ratios=[len(x) for x in df_res_types])

    for ii, df_ in enumerate(df_res_types):
        ax = fig.add_subplot(gs[ii,0])
        ys = []
        for i in range(len(df_)):
            y = i+1#len(df_)-i
            ax.plot([df_.lb.iloc[i], df_.ub.iloc[i]], [y,y], c='k', lw=1)
            ys.append(y)
            ax.scatter([df_.effect.iloc[i]], [y], s=70, c='k', marker='s', alpha=0.3)
            ax.scatter([df_.effect.iloc[i]], [y], s=8, c='k')
        ax.axvline(0, color='k', ls='--')
        ax.set_yticks(ys)
        ax.set_ylim(min(ys)-0.5, max(ys)+0.5)
        ax.set_yticklabels([name_mapping[x] for x in df_.Name], rotation=0)
        if ii==len(df_res_types)-1:
            ax.set_xlabel('AFib+ vs. AFib- (reference)')
        sns.despine()

    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.1, wspace=0.24)
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
