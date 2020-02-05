#!/usr/bin/env python3

import sys
import os.path as osp

if "lib" not in sys.path:
    sys.path.append("lib")

import numpy as np
import pandas as pd

import json

import PIL.Image as Image 

import matplotlib.pyplot as plt
from typing import Union, Tuple

import stx
import utils as ut

import argparse as arp

def hex2rgb(hexa):
    h = hexa.replace("#","")
    return np.array(tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))


def _single_viz(ax : plt.Axes,
                img : np.ndarray,
                crd : np.ndarray,
                vals : np.ndarray,
                color : Union[np.ndarray,list] = [0,0,0],
                title : str = None,
                plt_dict : dict = None,
                ) -> plt.Axes:

    n_spots = crd.shape[0]

    if plt_dict is None:
        plt_dict =  dict(s = 5,
                         edgecolor = "none",
                         )
    else:
        if 's' not in plt_dict.keys:
            plt_dict.update({"s":5})


    rgba = np.zeros((crd.shape[0],4))
    rgba[:,0:3] = color
    rgba /= 255
    rgba[:,-1] = vals.flatten()

    ec = np.zeros(rgba.shape)
    ec[:,-1] = rgba[:,-1]


    ax.imshow(img,plt.cm.gray,alpha = 0.5)
    ax.scatter(crd[:,1],
               crd[:,0],
               c = rgba,
               **plt_dict
               )

    ax.set_title(title)

    return ax

def _matchref(cluster_name : str,
              ref : pd.DataFrame,
              # old_col : str = "class_label",
              # new_col : str = "class_label",
              old_col : str = "cluster_id",
              new_col : str = "cluster_label",
              col_col : str = "cluster_color",
              )-> Tuple[str,str]:

    pos = np.where(ref[old_col].values.astype(str) == str(cluster_name))[0]
    new_name = ref[new_col].values[pos][0]
    out_color = ref[col_col].values[pos][0]
    out_color = hex2rgb(out_color)

    return (new_name,out_color)

def montage(crd : np.ndarray,
            img : np.ndarray,
            props : pd.DataFrame,
            ref : pd.DataFrame,
            n_cols = 10,
            normalize : bool = True,
            old_col : str = "cluster_id",
            new_col : str = "cluster_label",
            col_col : str = "cluster_color",
            )-> Tuple[plt.Figure,plt.Axes]:

    n_types = props.shape[1]
    n_rows = int(np.ceil(n_types / n_cols))

    typenames = props.columns.values
    figsize = (4 * n_cols, 4 * n_rows)
    fig,axs = plt.subplots(n_rows, n_cols,figsize = figsize)
    axs = axs.flatten()

    for tp in range(n_types):
        title,color  = _matchref(typenames[tp],ref,old_col,new_col,col_col)
        print("Rendering type {} | No. {}/{} ".format(title,tp +1,n_types))
        vals = props.values[:,tp].flatten()

        if normalize:
            vals /= vals.max()

        axs[tp] = _single_viz(axs[tp],
                             img,
                             crd,
                             vals,
                             color,
                             title,
                             )
    for ax in axs:
        ax = ut.clean_spines(ax)
        ax = ut.clean_ticks(ax)

    return (fig,axs)



def main(img_pth : str,
         prop_pth : str,
         ref_pth : str,
         jsn_pth : str,
         old_col : str = "cluster_id",
         new_col : str = "cluster_label",
         col_col : str = "cluster_color",
         )-> None:

    img = np.asarray(Image.open(img_pth).convert("L"))
    prop = pd.read_csv(prop_pth, sep = '\t',index_col = 0,header = 0)
    crd = np.array( [ x.split('x') for x in  prop.index.values ]).astype(float)

    ref = pd.read_csv(ref_pth, header = 0,sep = ',')
    print(ref.head())

    with open(jsn_pth,'r') as fopen:
        scf = json.load(fopen)

    crd *= float(scf['tissue_hires_scalef'])

    fig,ax = montage(crd = crd,
                     img = img,
                     props = prop,
                     ref = ref,
                     old_col = old_col,
                     new_col = new_col,
                     col_col = col_col)

    return fig,ax


def cli():

    prs = arp.ArgumentParser()

    prs.add_argument("-p",
                     "--proportions",
                     required = True,
                     help = '',
                     )

    prs.add_argument("-i",
                     "--image",
                     required = True,
                     help = '',
                     )

    prs.add_argument("-j",
                     "--scale_factors",
                     required = True,
                     help = '',
                     )

    prs.add_argument("-r",
                     "--reference",
                     required = True,
                     help = '',
                     )

    prs.add_argument("-t",
                     "--tag",
                     required = False,
                     default = None,
                     help = '',
                     )

    prs.add_argument("-od",
                     "--out_dir",
                     required = False,
                     default = "/tmp",
                     help = '',
                     )

    prs.add_argument("-oc",
                     "--old_col",
                     default = "cluster_id",
                     help = '',
                     )

    prs.add_argument("-nc",
                     "--new_col",
                     default = None,
                     help = '',
                     )


    prs.add_argument("-cc",
                     "--col_col",
                     default = "cluster_color",
                     help = '',
                     )

    args = prs.parse_args()

    if args.new_col is None:
        args.new_col = args.old_col

    fig,ax = main(args.image,
                  args.proportions,
                  args.reference,
                  args.scale_factors,
                  args.old_col,
                  args.new_col,
                  args.col_col,
                  )

    bname = osp.basename(args.proportions).replace('tsv','png')

    if args.tag is not None:
        bname = args.tag + '-' + bname 

    fig.savefig(osp.join(args.out_dir,bname))

if __name__ == '__main__':
    cli()
