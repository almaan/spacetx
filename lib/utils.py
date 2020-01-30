#!/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mlp

from typing import Tuple

from IPython.display import display, HTML

def clean_spines(ax : plt.Axes,
               ) -> plt.Axes:

    for sp in ax.spines.values():
        sp.set_visible(False)

    return ax


def clean_ticks(ax : plt.Axes,
               ) -> plt.Axes:

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return ax


def fancy_print(df : pd.DataFrame)-> None:
    display(HTML(df.to_html()))


def plot_member_distribution(cidx : np.ndarray,
                            cmap : mlp.colors.ListedColormap,
                            style_dict : dict = {},
                            )-> Tuple[plt.Figure,plt.Axes]:

    cluster_id,n_members = np.unique(cidx,
                                    return_counts=True)

    n_clusters = cluster_id.shape[0]

    cluster_color = lambda x : cmap(x / cluster_id.max())

    fig, ax = plt.subplots(1,1,figsize= (20,8))
    bars = ax.bar(cluster_id,
                n_members,
                edgecolor = 'black'
                )
    for k in range(n_clusters):
        bars[k].set_color(cluster_color(cluster_id[k]))
        ax.text(cluster_id[k],
                n_members[k] + 10 ,
                str(n_members[k]),
                horizontalalignment = 'center',
                **style_dict,
                )


    ax.set_yticks([])
    ax.set_yticklabels([])
    
    ax.set_xticks(np.arange(n_clusters))
    ax.set_xticklabels(['Cluster ' + str(x) for \
                        x in range(n_clusters)],
                    rotation = 90,**style_dict)


    clean_spines(ax)

    return fig, ax


def cmap_legend(n_members,
               cmap,
               ) -> Tuple[plt.Figure,plt.Axes]:

    fig, ax = plt.subplots(1,1)

    ax.scatter(np.arange(n_members),
            np.zeros(n_members),
            s = 150,
            cmap = cmap,
            c = np.arange(n_members))

    ax.set_ylim([-1,1])

    for x in range(n_members):
        ax.text(x = x,
                y = 0.2,
                s = str(x),
                horizontalalignment='center',
                fontsize = 24)


    clean_spines(ax)
    clean_ticks(ax)

    return (fig,ax)


