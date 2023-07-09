"""
https://ngdc.cncb.ac.cn/twas/
"""
import os
import numpy as np
import pandas as pd

folder = r'.'

df_ins_ = pd.read_excel(os.path.join(folder, 'TWASAtlas-TraitAssociationTable-Insomnia.xlsx'))
df_ins_['Effect Size'] = df_ins_['Effect Size'].astype(str).str.replace("^'-$",'nan').str.replace("'",'').astype(float)
df_ins_['P-value'] = df_ins_['P-value'].astype(str).str.replace("^'-$",'nan').str.replace("'",'').astype(float)

df_af_ = pd.read_excel(os.path.join(folder, 'TWASAtlas-TraitAssociationTable-AFib.xlsx'))
df_af_['Effect Size'] = df_af_['Effect Size'].astype(str).str.replace("^'-$",'nan').str.replace("'",'').astype(float)
df_af_['P-value'] = df_af_['P-value'].astype(str).str.replace("^'-$",'nan').str.replace("'",'').astype(float)

print(set(df_af_['Ensembl ID'])&set(df_ins_['Ensembl ID']))
# ENSG00000138175.8

print(df_af_[df_af_['Ensembl ID']=='ENSG00000138175.8'])
"""
Reported Trait     Atrial Fibrillation
Mapped Trait       Atrial Fibrillation
Tissues                 Left Ventricle
Gene Symbol                       ARL3
Ensembl ID           ENSG00000138175.8
Gene Type               protein-coding
Method/Software               MetaXcan
Effect Size                  -3.30E-01
P-value                       5.28e-15
Publication                TWASP000005
"""

print(df_ins_[df_ins_['Ensembl ID']=='ENSG00000138175.8'])
"""
Reported Trait                                              Insomnia
Mapped Trait                                    Insomnia Measurement
Tissues                                             Brain Cerebellum
Gene Symbol                                                     ARL3
Ensembl ID                                         ENSG00000138175.8
Gene Type                                             protein-coding
Method/Software    Probabilistic Transcriptome-Wide Association(P...
Effect Size
P-value                                                     0.000064
Publication                                              TWASP000141
Name: 176, dtype: object
"""

or_af= np.exp(-3.30E-01)
print(or_af)  # 0.7189237334319262

bl_prev_af = 119/2471
rr_af = or_af/(1-bl_prev_af+bl_prev_af*or_af)
print(rr_af)  # 0.7287888014267364
print(1/rr_af)  # 1.3721396350250148