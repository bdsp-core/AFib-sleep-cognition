import re
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
sns.set_style('ticks')


def main(display_type):
    df = pd.read_excel('../sleep_on_cog.xlsx')
    method = 'adj_bart'
    df = df[df.method==method].reset_index(drop=True)

    df = df[df.pval<0.01/len(df)].reset_index(drop=True)

    figure_folder = f'sleep_on_cog_{method}'
    os.makedirs(figure_folder, exist_ok=True)

    for i in tqdm(range(len(df))):
        Aname = df.Aname.iloc[i]
        Yname = df.Yname.iloc[i]
        As = df[[x for x in df.columns if re.match(r'A\d+',x)]].iloc[i].values
        Ys = df[[x for x in df.columns if re.match(r'Y\d+',x)]].iloc[i].values
        Ys_lb = df[[x for x in df.columns if re.match(r'Y_lb\d+',x)]].iloc[i].values
        Ys_ub = df[[x for x in df.columns if re.match(r'Y_ub\d+',x)]].iloc[i].values

        figsize = (8,6)
        save_name = os.path.join(figure_folder, f'rank{i+1}_{Aname}_on_{Yname}_N{df.N.iloc[i]}')

        plt.close()
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(111)
        ax.fill_between(As, Ys_lb, Ys_ub, color='k', alpha=0.2)
        ax.plot(As, Ys, lw=2, c='k')
        ax.set_xlabel(Aname)
        ax.set_ylabel(Yname)
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
