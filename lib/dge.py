#!/usr/bin/env python3


import os.path as osp
import json

import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from typing import Union,Tuple,List
import PIL

import utils as ut

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats as st


def dge_analysis(counts : np.ndarray,
                 full_design : np.ndarray,
                 ):

    def glm(x,d):
        return sm.GLM(x,
                      d,
                      family=sm.families.NegativeBinomial(),
                      missing = 'drop',
                      )


    nS,nG = counts.shape
    res = np.ones((nG,3)) * np.nan

    for gene in range(nG):

        if counts.values[:,gene].sum() < 2:
            continue

        try:
            model_full = glm(counts.values[:,gene],
                            full_design)
            res_full = model_full.fit()

            ll_full = res_full.llf

            model_red = glm(counts.values[:,gene],
                            full_design[:,0:-1])

            res_red = model_red.fit()

            ll_red = res_red.llf


            dd = -2*(ll_red - ll_full)
            pval = st.chi2.sf(dd,df = 1)

            lfc = np.log2(np.exp(res_full.params[-1]))

            res[gene,1] = pval
            res[gene,0] = lfc
        except:
            pass

    is_not_na = np.isnan(res[:,1]) == False
    mht = multipletests(res[is_not_na,1],method = 'fdr_bh', alpha = 0.05)
    res[is_not_na,2] = mht[1]
    res = pd.DataFrame(res,
                       index = counts.columns,
                       columns = ['l2fc','pval','adj_pval']
                       )

    return res















