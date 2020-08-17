
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os.path as osp
import os


from typing import Dict,Union,List,Optional,Tuple,Callable



def load_results(dirname : str,
                 filters : Optional[List[Callable]] = None,
                )->List[Union[pd.DataFrame,pd.Index]]:

    files = os.listdir(dirname)

    if filters is not None:
        if callable(filters):
            filters = [filters]

        for fun in filters:
            files = list(filter(fun,files))

    results = dict()

    columns = pd.Index([])
    indices = pd.Index([])

    for f in files:
        sample = osp.basename(f).split(".")[0]
        _res = pd.read_csv(osp.join(dirname,f),
                           sep = '\t',
                           header = 0,
                           index_col = 0)

        columns = columns.union(_res.columns)
        indices = indices.union(_res.index)

        results.update({sample:_res})

    n_row = len(indices)
    n_col = len(columns)

    for k,v in results.items():
        _tmp = pd.DataFrame(np.zeros((n_row,n_col)),
                            index = indices,
                            columns = columns,
                           )
        inter_r = indices.intersection(v.index)
        inter_c = columns.intersection(v.columns)

        _tmp.loc[inter_r,inter_c] = v.loc[inter_r,
                                          inter_c].values
        results[k] = _tmp

    return [results,
            columns,
            indices,
            ]

def merge_results(results : Dict[str,pd.DataFrame],
                  columns : pd.Index,
                  indices : pd.Index,
                 )->np.ndarray:

    n_row = len(indices)
    n_col = len(columns)
    n_samples = len(results)

    results_tensor = np.zeros((n_samples,
                               n_row,n_col))

    for k,v in enumerate(results.values()):
        results_tensor[k,:,:] = v.values

    return results_tensor


def plot_results(ax : plt.Axes,
                 x_vals : np.ndarray,
                 obs : np.ndarray,
                 std : Optional[np.ndarray] = None,
                 theo : Optional[np.ndarray] = None,
                 vertical_reference : Optional[float] = None,
                 use_log_scale : bool = True,
                 use_legend : bool = True,
                 neg_to_nan : bool = True,
                 **kwargs,
                 )->None:

    if theo is not None:
        if neg_to_nan:
            _theo = theo.copy()
            _theo[_theo < 0] = np.nan

        ax.plot(x_vals,
                _theo,
                "-",
                label = kwargs.get("theo_label",
                                   "null"),
                alpha = 1.0,
                color = "red",
                )
    if neg_to_nan:
        _obs = obs.copy()
        _obs[_obs < 0] = np.nan

    ax.plot(x_vals,
            _obs,
            "--",
            label = kwargs.get("obs_label",
                               "data"),
            alpha = 1.0,
            color = "black",
            )

    if std is not None:
        for m in range(2):
            y_vals = obs + ((-1)**m)*std
            y_vals[y_vals < 0] = np.nan

            label = (r"$\pm$1" + "sd" if\
                     m == 0 else None)

            ax.plot(x_vals,
                    y_vals,
                    "--",
                    color = "blue",
                    alpha = 0.4,
                    label = label,
                    )

    if use_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if vertical_reference is not None:
        ax.axvline(x = vertical_reference,
                linewidth = 1,
                color = "green",
                linestyle = "dashed")

    if use_legend:
        ax.legend()


def visualize_types(ax : plt.Axes,
                    crd : np.ndarray,
                    labels : np.ndarray,
                    selected : Union[str,int,List[Union[str,int]]],
                    cmap : Optional[str] = None,
                    reference_distance : Optional[float] = None,
                    use_background : bool = True,
                    use_legend : bool = True,
                    )->None:


    if isinstance(selected,str) or \
       isinstance(selected,int):
        selected = [selected]


    to_array = lambda x : np.array(x).reshape(1,-1)
    while cmap is None or isinstance(cmap,str):
        if cmap is None:
            _cm = lambda x: plt.cm.jet( x /\
                                         len(selected) *\
                                         plt.cm.jet.N)

            cmap = lambda x : to_array(_cm(x))


        elif cmap is "one_two":
            if len(selected) <= 2:
                cmap = lambda x : ["red","blue"][x]
            else:
                cmap = None
        else:
            try: 
                cm_name = eval("plt.cm."+ cmap)
                _cm = lambda x: cm_name(x /\
                                        len(selected)*\
                                        cm_name.N)

                cmap = lambda x : to_array(_cm(x))

            except:
                cmap = None


    if use_background:

        ax.scatter(crd[:,0],
                   crd[:,1],
                   c = "black",
                   alpha = 0.2,
                   s = 40,
                   )

    for k,sel in enumerate(selected):
        pos = labels == sel

        ax.scatter(crd[pos,0],
                   crd[pos,1],
                   c = cmap(k),
                   s = 80,
                   alpha = 0.8,
                   label = sel,
                   )


    if reference_distance is not None:
        mx = crd.max(axis = 0)
        mn = crd.min(axis = 0)


        ax.plot((mx[0],mx[0]-reference_distance),
                (mn[1],mn[1]),
                c = "green",
                linewidth  = 3, 
            )

        ax.text(mx[0]-reference_distance/2,max(20,mn[1]*2),
                "{} LU".format(reference_distance),
                horizontalalignment = "center")

    if use_legend:
        ax.legend()

