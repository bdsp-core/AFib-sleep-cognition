"""
https://gwas.mrcieu.ac.uk/
UKB
"""

import os
import numpy as np
import pandas as pd

folder = r'../big_files'
key_cols = ['ID','REF','ALT']

df_apnea_ = pd.read_csv(os.path.join(folder, 'ukb-b-16781.vcf.gz'), comment='#', sep='\t', header=None)
df_apnea = df_apnea_[9].str.split(':',expand=True)
df_apnea.columns = df_apnea_[8].iloc[0].split(':')
df_apnea['ES'] = df_apnea.ES.astype(float)
df_apnea['SE'] = df_apnea.SE.astype(float)
df_apnea['LP'] = df_apnea.LP.astype(float)
df_apnea['AF'] = df_apnea.AF.astype(float)
df_apnea['REF'] = df_apnea_[3].values
df_apnea['ALT'] = df_apnea_[4].values
df_apnea = df_apnea.rename(columns={x:x+'_apnea' for x in df_apnea.columns if x not in key_cols})

df_af_ = pd.read_csv(os.path.join(folder, 'ukb-b-964.vcf.gz'), comment='#', sep='\t', header=None)
df_af = df_af_[9].str.split(':',expand=True)
df_af.columns = df_af_[8].iloc[0].split(':')
df_af['ES'] = df_af.ES.astype(float)
df_af['SE'] = df_af.SE.astype(float)
df_af['LP'] = df_af.LP.astype(float)
df_af['AF'] = df_af.AF.astype(float)
df_af['REF'] = df_af_[3].values
df_af['ALT'] = df_af_[4].values
df_af = df_af.rename(columns={x:x+'_af' for x in df_af.columns if x not in key_cols})

df = df_apnea.merge(df_af, on=key_cols, how='inner', validate='1:1')
df2 = df[(df.LP_apnea>3)&(df.LP_af>3)].reset_index(drop=True)
print(df2)
"""
       ES_apnea     SE_apnea LP_apnea  AF_apnea         ID REF ALT        ES_af        SE_af    LP_af     AF_af
0   0.000577126  0.000170444  3.14874   0.26589  rs1778211   A   G  0.000914174  0.000265281  3.24413   0.26589
1  -0.000544822  0.000149323  3.58503  0.443631  rs4788693   T   C  0.000935437  0.000232417  4.24413  0.443631
2  -0.000549397  0.000149367  3.63827  0.443379  rs4788694   C   G  0.000949117  0.000232485  4.34679  0.443379
"""

or_apnea = np.exp(df2.ES_apnea)
print(or_apnea)
"""
0    1.000577
1    0.999455
2    0.999451
Name: ES_apnea, dtype: float64
"""

or_af = np.exp(df2.ES_af)
print(or_af)
"""
0    1.000915
1    1.000936
2    1.000950
Name: ES_af, dtype: float64
"""

bl_prev_apnea = 0.8
rr_apnea = or_apnea/(1-bl_prev_apnea+bl_prev_apnea*or_apnea)
print(rr_apnea)
"""
0    1.000115
1    0.999891
2    0.999890
Name: ES_apnea, dtype: float64
"""

bl_prev_af = 119/2471
rr_af = or_af/(1-bl_prev_af+bl_prev_af*or_af)
print(rr_af)
"""
0    1.000871
1    1.000891
2    1.000904
Name: ES_af, dtype: float64
"""