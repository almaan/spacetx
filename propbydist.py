#!/usr/bin/env python3


import numpy as np
import pandas as pd
import anndata as ad

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from loess.loess_1d import loess_1d

import os.path as osp
import sys
import argparse as arp

from typing import List,Tuple,Dict,Union

import matplotlib.pyplot as plt

def eprint(s : str,
          )-> None:

    print("[ERROR] : {}".format(s))

def iprint(s : str,
          )-> None:

    print("[INFO] : {}".format(s))


def col_major_ordr(n_col : int,
              n_row : int,
              )-> np.ndarray:

    o = list()
    n = n_col*n_row
    for i in range(n_col):
        for j in range(n_row):
            x = i + j*n_col
            o.append(x)

    return np.array( o )

def add_edge_iss(data : ad.AnnData,
                 N : int = 6,
                 eps : float = 0.1,
                 )->None:

    eprint("Assay not implemented yet. Exiting.")
    sys.exit(-1)



def add_edge_visium(data : ad.AnnData,
                    N : int = 6,
                    eps : float = 0.1,
                    )->None:

    if "edge" not in data.obs.keys():

        crd = data.obs[["x","y"]].values
        dmat = cdist(crd,crd)
        dmat[dmat == 0] = np.inf
        sf = np.min(dmat)
        crd = crd / sf
        del dmat

        kd = KDTree(crd)
        dist,nbrs = kd.query(crd,
                            k = N+1,
                            p = 2,
                            distance_upper_bound = np.sqrt(2) + eps)
        is_edge = np.zeros(crd.shape[0])

        for k,d in enumerate( dist ):
            if np.sum(d != np.inf) < N+1:
                is_edge[k] = 1

        data.obs["edge"] = pd.DataFrame(is_edge.astype(int),
                                        index = data.obs.index,
                                        columns = ["edge"],
                                        )

def add_distance_to_edge(data : ad.AnnData)->None:
    if "edge_distance" not in data.obs.keys():
        if "edge" in data.obs.keys():
            crd = data.obs[["x","y"]].values
            dmat = cdist(crd[data.obs["edge"]==0],
                        crd[data.obs["edge"]==1])
            mind = np.min(dmat,axis=1)
            dist = np.zeros(crd.shape[0])
            dist[data.obs["edge"]==0] = mind

            data.obs["edge_distance"] = dist / dist.max()




def smoothing(dist : np.ndarray,
              prop : np.ndarray,
              frac : float = 0.24,
              )-> Tuple[np.ndarray,np.ndarray]:

    xout, yout,_ = loess_1d(dist,
                            prop,
                            frac = frac,
                            )

    ordr = np.argsort(xout)

    return (xout[ordr],yout[ordr])





def smooth_layers(data : ad.AnnData,
                  layers : List[str],
                  frac : float = 0.24,
                  dist_attr : Tuple[str,str] = ("obs","edge_distance"),
                  prop_attr : Tuple[str,str] = ("obsm","proportions_class"),
                  )-> Dict[str,Union[List[np.ndarray],np.ndarray]]:

    xs = list()
    ys = list()
    yr = list()

    dist = eval("data."+dist_attr[0] +\
                "['{}']".format(dist_attr[1]))
    dist = dist.values.reshape(-1,)
    prop = eval("data."+prop_attr[0] +\
                "['{}']".format(prop_attr[1]))

    for layer in layers:
        _yr = prop[layer].values.flatten()
        _x,_y = smoothing(dist,
                          _yr,
                          )

        xs.append(_x)
        ys.append(_y)
        yr.append(_yr)

    res = dict(x_smooth = xs,
               y_smooth = ys,
               x_raw = dist,
               y_raw = yr,
               )
    return res

def plot_smooth(ax : plt.Axes,
              xx : np.ndarray,
              yy : np.ndarray,
              color : np.ndarray = np.zeros(4),
              title : str = "",
              )->None:

    ax.fill_between(xx,
                       0,
                       yy,
                       color = color,
                       alpha = 0.4)
    ax.plot(xx,
            yy,
            "-o",
            c = color,
            markersize = 1,
            )

    ax.set_title(title)
    ax.set_xlabel("Relative distance to edge")
    ax.set_ylabel("Proportion Value")

    for side in ["top","right"]:
        ax.spines[side].set_visible(False)


def plot_raw(ax : plt.Axes,
             dist : np.ndarray,
             prop : np.ndarray,
             color : np.ndarray = np.zeros(4),
             )->None:

    ax.scatter(dist,
               prop,
               alpha = 0.1,
               s = 2,
               c = color.reshape(1,-1),
               )

def visualize_distribution(x_smooth : List[np.ndarray],
                           y_smooth : List[np.ndarray],
                           x_raw : np.ndarray,
                           y_raw : List[np.ndarray],
                           layers : List[str],
                           n_cols : int = 2,
                           width : float = 4,
                           height : float = 1.5,
                           )-> Tuple[plt.Figure,plt.Axes]:

    colormap = plt.cm.rainbow
    colors = [colormap(l/len(layers)) for\
              l in range(len(layers))]


    n_rows = int( np.ceil(len(layers)/n_cols) )
    fig,ax = plt.subplots(nrows = n_rows,
                          ncols = n_cols,
                          figsize = (width * n_cols,
                                     height * n_rows),
                          )

    ax = ax.flatten()[col_major_ordr(n_cols,n_rows)]

    for k,layer in enumerate(layers):
        plot_raw(ax[k],
                 x_raw,
                 y_raw[k],
                 color = np.array(colors[k]),
                 )


        plot_smooth(ax[k],
                    x_smooth[k],
                    y_smooth[k],
                    color = colors[k],
                    )

        ax[k].set_title("Layer {}".format(layer))

    fig.tight_layout()

    return (fig,ax)



def main():

    prs = arp.ArgumentParser()
    aa = prs.add_argument

    aa("-d",
       "--data",
       required = True,
       help = "path to .h5ad file holding"\
       " the data",
      )

    aa("-ss",
       "--subset",
       default = None,
       nargs = 2,
       help = "substting of spatial locations"\
       " first argument : key to 'obs' object"\
       " holding the labels to subset w.r.t."\
       " second argumnet : labels to include",
      )

    aa("-l",
       "--layers",
       nargs = '+',
       required = True,
       help = "(1) Space separated list"\
       " of layer cell type names."\
       " (2) File with layer cell types"\
       " listed (one per row)",
      )

    aa("-a",
       "--assay",
       default = "visium",
       choices = ["visium",
                  "iss"],
       help = "assay from which data"\
       " is collected",
      )

    aa("-o",
       "--out_dir",
       default = None,
       help = "output directory",
      )

    aa("-t",
       "--tag",
       default = None,
       help = "tag to prepend result image"\
       " with. Default is none.",
      )


    args = prs.parse_args()

    iprint("Processing file {}".format(args.data))
    prp = ad.read_h5ad(args.data)

    try:
        add_edge = eval("add_edge_" + args.assay)
    except:
        eprint("Assay not implemented yet. Exiting")
        sys.exit(-1)

    add_edge(prp)
    add_distance_to_edge(prp)

    if args.subset is not None:
        prp = prp[prp.obs[args.subset[0]]==int(args.subset[1]),:]

    if osp.isfile(args.layers[0]):
        with open(args.layers[0],"r+") as f:
            args.layers = f.readlines()

        iprint("Only analyzing layers:\n{}".\
               format(''.join(args.layers)))

        args.layers = [x.rstrip("\n") for x in args.layers]

    smooth_res = smooth_layers(prp,layers = args.layers)

    fig,ax = visualize_distribution(**smooth_res,
                                    layers = args.layers,
                                    )

    bname = "type-by-dist.png"
    if args.tag is not None:
        bname = args.tag + "-" + bname

    fig.savefig(osp.join(args.out_dir,
                         bname))



if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append("-h")
    main()
