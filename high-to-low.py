#!/usr/bin/env python3

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

import argparse as arp

from enum import Enum
import sys

class Visium(Enum):
    d = 55.0
    r = d / 2
    c2c = 100.0


def generate_hexagonal(spot_crd : np.ndarray,
                       )-> np.ndarray:

    mx = np.max(spot_crd,axis = 0)
    mn = np.min(spot_crd,axis = 0)

    dx = Visium.d.value
    dy = np.sqrt(dx**2 + (dx/2)**2)

    xs = np.arange(mn[0],mx[0],dx)
    ys = np.arange(mn[1],mx[1],dy)

    x,y = np.meshgrid(xs,ys)
    x[::2] += dx/2

    grid_crd = np.hstack((x.reshape(-1,1),
                        y.reshape(-1,1)))

    return grid_crd




def assign_to_spot(data : pd.DataFrame,
                   spot_crd : np.ndarray,
                   grid_crd : np.ndarray,
                   )->ad.AnnData:


    n_spots = grid_crd.shape[0]
    uni_genes = np.unique(data["gene"].values)
    vis_data = np.zeros((n_spots,
                         uni_genes.shape[0]))
    vis_data = pd.DataFrame(vis_data,
                            columns = pd.Index(uni_genes))

    for s in range(data.shape[0]):

        dist = np.linalg.norm(grid_crd - spot_crd[s,:],
                              axis = 1)

        if dist.min() < Visium.r.value:
            pos = np.argmin(dist)
            vis_data.loc[pos,data.loc[s,"gene"]] +=1
        else:
            pos = -1

    sms = vis_data.values.sum(axis = 1)
    vis_data = vis_data.iloc[sms > 0,:]
    n_grid_crd = grid_crd[sms > 0,:]

    spot_names = [str(x) + "x" + str(y) for\
                  x,y in zip(n_grid_crd[:,0],
                             n_grid_crd[:,1])]

    spot_names = pd.Index(spot_names)
    vis_data.index = spot_names

    obs = pd.DataFrame(n_grid_crd,
                       index = spot_names,
                       columns = ["x","y"])

    var = pd.DataFrame(vis_data.columns.values,
                       index = vis_data.columns,
                       columns = ["gene"])

    adata = ad.AnnData(vis_data,
                       obs = obs,
                       var = var,
                       )

    return adata




pth ="/tmp/ISS_3_spot_table.csv"

data = pd.read_csv(pth,
                   sep = ",",
                   header = 0,
                   index_col = 0)



def main(pth : str,
        opth : str,
        data_type : str,
        )->None:


    if data_type == "iss":
        data = pd.read_csv(pth,
                        sep = ",",
                        header = 0,
                        index_col = 0)


        data.columns = ["gene","x","y"]

        spot_crd = data[["x","y"]].values 

    else:
        print("[ERROR] : Sorry data not supported")
        sys.exit(-1)


    hex_grid = generate_hexagonal(spot_crd)

    pseudo_visium = assign_to_spot(data,
                                   spot_crd,
                                   hex_grid,
                                   )

    pseudo_visium.write_h5ad(opth)



if __name__ == "__main__":

    prs = arp.ArgumentParser()
    aa = prs.add_argument

    aa("-i",
       "--input_file",
       required = True,
       help = "input file",
       )

    aa("-d",
       "--data_type",
       required = True,
       choices = ["iss"],
       help = "specify data type",
       )

    aa("-o",
       "--out_name",
       required = True,
       help = "output full path",
       )

    args = prs.parse_args()


    main(args.input_file,
         args.out_name,
         args.data_type,
         )

