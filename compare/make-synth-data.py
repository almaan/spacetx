#!/usr/bin/env python3


import pandas as pd
import numpy as np
import anndata as ad
import os.path as osp


def synth_prop_1(n_obs : int,
                 n_vars : int,
                 alpha : float = 0.5,
                 ) -> np.ndarray:

    props = np.random.dirichlet( np.ones(n_vars) * alpha,
                                 n_obs)
    return props

def synth_prop_2(base : np.ndarray,
                 std : float = None,
                )->np.ndarray:


    n_obs,n_vars = base.shape
    if std is None:
        scale = np.array([0.01,
                          0.05,
                          0.1,
                          0.5,
                          1.0,
                          2.0,
                          ])

        _std = np.std(base,axis = 0) * np.random.choice(scale,
                                                        size = n_vars,
                                                        replace = True,
                                                        )
    else:
        _std = std

    noise = np.random.normal(loc = np.zeros(n_vars), scale = _std, size = base.shape)
    _X = np.abs(base + noise)
    _X = _X / _X.sum(axis = 1,keepdims = True)
    return _X


def main():
    ori_pth = "/home/alma/w-projects/stx/data/example/allen-2.h5ad"
    ori_data = ad.read_h5ad(ori_pth)

    out_dir = "/home/alma/w-projects/stx/scripts/compare/synth-data"

    n_fake = 2
    fake_name = "method"
    base_key = "proportions_class"

    n_spots,n_types = ori_data.obsm[base_key].shape
    base_index = ori_data.obsm[base_key].index
    base_cols = ori_data.obsm[base_key].columns

    data = ori_data.copy()

    for k in ori_data.obsm.keys():
        del data.obsm[k]

    for ii in range(n_fake):
        # _tmp = synth_prop(n_spots,n_types)
        _tmp = synth_prop_2(ori_data.obsm[base_key].values)
        _tmp = pd.DataFrame(_tmp,
                            columns = base_cols,
                            index = base_index,
                            )
        data.obsm[fake_name + "_{}".format(ii + 1)] = _tmp

    data.write_h5ad(osp.join(out_dir,"example-data-1.h5ad"))

if __name__ == "__main__":
    main()
