import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_style('ticks')


def main(display_type):
    suffix = '_holdAHI2'
    df = pd.read_excel(f'../AF_as_exposure{suffix}.xlsx')

    patterns = df.Name.unique()
    df_res = []
    sig_col = 'sig_fdr_bh'
    for i, p in enumerate(patterns):
        if p=='M_HB_desat':
            continue
        if 'slope' in p:
            continue
        if df[sig_col][df.Name==p].sum()>=2:
            df_res.append(df[(df.Name==p)&(df.method=='dr')])
    df_res = pd.concat(df_res, axis=0).sort_values('pval', ignore_index=True)
    print(df_res)
    df_res.to_csv(f'significant_features{suffix}.csv', index=False)

    for i in range(len(df_res)):
        name = df_res.Name.iloc[i]
        if 'M_bp' in name or ('M_macro' in name and 'to' in name):
            df_res.loc[i,'effect'] *= 100
            df_res.loc[i,'lb'] *= 100
            df_res.loc[i,'ub'] *= 100
    types = [r'^M_(?:AI|AHI|bp|sp)', 'M_HB', 'M_macro']
    df_res_types = [df_res[df_res.Name.str.contains(x)].sort_values('effect', ascending=False).reset_index(drop=True) for x in types]
    assert sum([len(x) for x in df_res_types])==len(df_res)

    name_mapping = {
    'M_AHI3':'AHI (/h)',
    'M_AI':'Arousal Index (/h)',

    'M_bp_delta_rel_mean_N1_C':r'avg rel $\delta$ power(%), N1',
    'M_bp_delta_rel_mean_R_C':r'avg rel $\delta$ power(%), R',
    #'M_bp_delta_rel_slope_N2_C':r'slope of rel $\delta$ power (%/h) overnight, N2',
    'M_bp_alpha_rel_mean_W_C':r'avg rel $\alpha$ power(%), W',
    'M_bp_alpha_rel_mean_R_C':r'avg rel $\alpha$ power(%), R',
    'M_bp_alpha_rel_mean_N1_C':r'avg rel $\alpha$ power(%), N1',
    'M_bp_alpha_rel_mean_N2_C':r'avg rel $\alpha$ power(%), N2',
    'M_bp_alpha_rel_mean_N3_C':r'avg rel $\alpha$ power(%), N3',
    'M_bp_theta_rel_mean_N2_C':r'avg rel $\theta$ power(%), N2',
    'M_bp_theta_rel_mean_N3_C':r'avg rel $\theta$ power(%), N3',
    'M_bp_theta_rel_mean_R_C':r'avg rel $\theta$ power(%), R',
    'M_bp_beta_rel_mean_N2_C':r'avg rel $\beta$ power(%), N2',
    'M_bp_beta_rel_mean_N3_C':r'avg rel $\beta$ power(%), N3',

    'M_sp_amp_N2N3_C':r'spindle amp ($\mu$V), N2 or N3',

    'M_HB_apnea':'hypoxic burden (%min/h)',

    'M_macro_N1toW':'N1 to W transition prob (%)',
    'M_macro_N1toN1':'N1 to N1 transition prob (%)',
    'M_macro_SFI':'sleep fragmentation index (/h)',
    }

    figsize = (8.8,5.2)
    save_name = f'AF_on_sleep{suffix}'

    plt.close()
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(types),1,height_ratios=[len(x) for x in df_res_types])

    for ii, df_ in enumerate(df_res_types):
        print(df_)
        ax = fig.add_subplot(gs[ii,0])
        ys = []
        for i in range(len(df_)):
            y = len(df_)-i
            ax.plot([df_.lb.iloc[i], df_.ub.iloc[i]], [y,y], c='k', lw=1)
            ys.append(y)
            ax.scatter([df_.effect.iloc[i]], [y], s=70, c='k', marker='s', alpha=0.3)
            ax.scatter([df_.effect.iloc[i]], [y], s=8, c='k')
        ax.axvline(0, color='k', ls='--')
        ax.set_yticks(ys)
        ax.set_ylim(min(ys)-0.5, max(ys)+0.5)
        ax.set_yticklabels([name_mapping[x] for x in df_.Name], rotation=0)
        if ii==len(df_res_types)-1:
            ax.set_xlabel('effect of having AFib')
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
