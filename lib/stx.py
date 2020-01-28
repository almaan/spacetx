#!/usr/bin/env python3


import os.path as osp
import json

import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from typing import Union,Tuple
import PIL

import utils as ut

class STdata:
    def __init__(self,
                 cnt : Union[str,pd.DataFrame],
                 img : Union[str,PIL.Image.Image] = None,
                 mta : Union[str,pd.DataFrame] = None,
                 scf : Union[str,dict] = None,
                 delim : str = None,
                 img_type : str = 'hires',
                 )->None:

        self.delim = ('\t' if delim is \
                      None else delim)

        self.img_type = img_type

        self.cnt = (self._read_data(cnt) if \
                    isinstance(cnt,str) else cnt)

        self.img = (self._read_image(img) if \
                    isinstance(img,str) else img)

        self.mta = (self._read_data(mta) if \
                    isinstance(mta,str) else mta)

        self.sfdict = (self._read_json(scf) if \
                    isinstance(scf,str) else scf)

        self._update()

    def _read_data(self,
                   pth : str,
                   )-> Union[pd.DataFrame,None]:

        if pth is None:
            return None
        else:
            try:
                return pd.read_csv(pth,
                                   sep = self.delim,
                                   header = 0,
                                   index_col  = 0)
            except:
                print("[WARNING] : Unsupported data")
                return None

    def _read_image(self,
                    pth : str,
                    ) -> Union[PIL.Image.Image,None]:

        if pth is None:
            return None
        else:
            try:
                return PIL.Image.open(pth).convert('RGB')
            except:
                print("[WARNING] : Unsupported image")
                return None

    def _read_json(self,
                   pth : str,
                   ) -> Union[dict,None]:

        if pth is None:
            return None
        else:
            try:
                with open(pth,'r') as fopen:
                    scf = json.load(fopen)
                return scf
            except:
                print("[WARNING] : Unsupported json file ")
                return None

    def _match_data(self,
                    )->None:
        if self.mta is not None:
            if isinstance(self.mta,pd.DataFrame):
                inter = self.cnt.index.intersection(self.mta.index)
                self.cnt = self.cnt.loc[inter,:]
                self.mta = self.mta.loc[inter,:]
            else:
                print('[WARNING] : Unsupported meta data')

    def _update(self,)->None:
        self._match_data()
        self.S = self.cnt.shape[0]
        self.G = self.cnt.shape[1]

        self.foi = np.zeros((self.S,4))
        self.foi[:,2:4] = 1.0 
        self.title = ''

        self.genes = self.cnt.columns.values
        self._set_crd()
        self._set_radius()
        self._set_scale_factor()


    def _set_radius(self,
                    )->None :

        if self.sfdict is not None:
            try:
                diameter_key = 'spot_diameter_fullres'
                diameter = float(self.sfdict[diameter_key])
                self.r = diameter 
            except:
                self.r = None
                print("[WARNING] : Unsupported json file")
        else:
            self.r = None

    def _set_scale_factor(self,)->None:
        if self.sfdict is not None:
            try:
                scale_factor_key = '_'.join(['tissue',
                                            self.img_type,
                                            'scalef'])
                self.sf = float(self.sfdict[scale_factor_key])
            except:
                print("[WARNING] : Unsupported json file")
        else:
            self.sf = 1.0

    def _set_crd(self,
                )->np.ndarray:

        self.crd = np.array([x.split('x') for \
                            x in self.cnt.index])

        self.crd = self.crd.astype(float)
    
    def __getitem__(self,
                    item)->Union[pd.DataFrame,Tuple[pd.DataFrame,pd.DataFrame]]:

        if item >= self.S or item < 0:
            print("[WARNING] : Item out of reach")
            return None
        
        if self.mta is not None:
            return (self.cnt[item,:],self.mta[item,:])
        else:
            return self.cnt[item,:]

    def plot(self,
             fig : plt.Figure = None,
             ax : plt.Axes = None,
             marker_size : float = None,
             overlay : bool = True,
             figsize : Tuple[float,float] = (20,20),
             cmap = plt.cm.Blues,
             alpha : float = 1,
             )-> Tuple[plt.Figure,plt.Axes]:


        if ax is None:
            fig,ax = plt.subplots(1,
                                  1,
                                  figsize = figsize)

        if overlay:
            if self.img is not None:
                ax.imshow(self.img)

        if marker_size is None:
            if self.r is not None:
                marker_size = (self.r * self.sf / fig.dpi * 72)
            else:
                marker_size = 10

        

        if len(self.foi.shape) > 1:
            cmap = None
            alpha = None
        else:
            alpha = np.clip(alpha,a_min = 0, a_max = 1)

        ax.scatter(self.crd[:,1] * self.sf,
                   self.crd[:,0] * self.sf,
                   c = self.foi,
                   cmap = cmap,
                   s = marker_size,
                   alpha = alpha,
                   )

        ax.set_title(self.title)
        ax.set_aspect('equal')

        ax = ut.clean_spines(ax)
        ax = ut.clean_ticks(ax)

        if fig is not None:
            return (fig,ax)
        else:
            return ax

    def set_foi(self,
                feature : Union[str,np.ndarray],
                )->None:

        if isinstance(feature,str):
            if feature in self.genes:
                self.foi = np.zeros((self.S,4))
                self.foi[:,2] = 1
                self.foi[:,3] = self.cnt[feature].values
                mx = self.foi[:,3].max()
                if mx > 0:
                    self.foi[:,3] /= mx
                self.title = feature
            elif feature in self.mta.columns:
                self.foi = self.mta[feature].values.flatten()
            else:
                print("[ERROR] : {} is not a valid Feature of Interest".format(feature))
        elif isinstance(feature,np.ndarray):
            self.foi = feature
            self.title = ''
        else:
            print("[ERROR] : Provided feature not supported")

    def filter_genes(self,
                     pattern : str,
                     )-> None:

        keep = [not bool(re.match(pattern.upper(),x.upper())) for x in self.genes]

        self.cnt = self.cnt.iloc[:,keep]
        self._update()



def metric_plt(func):
    def wrapper(cnt : pd.DataFrame,
                 nbins = 100,
                 ax : plt.Axes = None,
                 facecolor : str = 'gray',
                 )-> Tuple[Union[plt.Figure,None],plt.Axes]:

        if ax is None:
            fig,ax = plt.subplots(1,1)
        else:
            fig = None

        ax,vals = func(cnt,ax)
        nbins = np.min((nbins,int(vals.shape[0] / 2)))

        ax.hist(vals,
                facecolor = 'gray',
                edgecolor = 'black',
                bins = nbins)

        ax.axvline(x = vals.mean(),
                   linewidth = 2,
                   linestyle = 'dashed',
                   color = 'red')

        for pos in ['right','top']:
            ax.spines[pos].set_visible(False)

        return fig,ax
    return wrapper

@metric_plt
def obs_per_gene(cnt,ax):
    vals = (cnt.values).sum(axis = 0)
    vals = np.log1p(vals)

    ax.set_title("Observations per gene")
    ax.set_xlabel("log(1 + genes)")
    ax.set_ylabel("Genes")

    return ax,vals


@metric_plt
def unique_genes_per_spot(cnt : pd.DataFrame,
                          ax : plt.Axes = None,
                          )-> Tuple[plt.Axes,np.ndarray]:

    vals = (cnt.values > 0 ).sum(axis = 1)

    ax.set_title("Unique genes per spots")
    ax.set_xlabel("Unique Genes")
    ax.set_ylabel("Spots")

    return (ax,vals)

@metric_plt
def obs_per_spot(cnt : pd.DataFrame,
                 ax : plt.Axes = None,
                 )-> Tuple[plt.Axes,np.ndarray]:

    vals = cnt.values.sum(axis = 1)

    ax.set_title("Observations per spots")
    ax.set_xlabel("Observations")
    ax.set_ylabel("Spots")

    return (ax, vals)


def topNgenes(cnt : pd.DataFrame,
              N : int = 10,
              )->pd.DataFrame:

    df = pd.DataFrame(np.zeros((N,3)))
    df.columns = ['Rank','SumTotal','Mean']
    sms  = cnt.values.sum(axis=0)
    ordr = np.argsort(sms)[::-1][0:N]
    df.index = cnt.columns.values[ordr]
    df['Rank'] = np.arange(1,N+1)
    df['Mean'] = cnt.values.mean(axis = 0)[ordr]
    df['SumTotal'] = sms[ordr]

    return df

def normalize_cnt(cnt : pd.DataFrame):

    vals = 2.0 * np.sqrt(cnt.values + 3.0 / 8.0)
    rsms = vals.sum(axis = 1).reshape(-1,1)
    vals = np.divide(vals,rsms,where = rsms > 0)

    return vals
