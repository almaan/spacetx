#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json

import argparse as arp

from PIL import Image as im

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import os.path as osp


# class taken from : 
class SelectFromCollection(object):


    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    # cnt_pth = "~/w-projects/stx/data/200123-allen/curated_data/Allen-2/Allen-2-count-matrix.tsv.gz"
    # img_pth = "/home/alma/w-projects/stx/data/200123-allen/raw_data/Allen-2/spatial/tissue_hires_image.png"
    # jsn_pth = "/home/alma/w-projects/stx/data/200123-allen/raw_data/Allen-2/spatial/scalefactors_json.json"

    prs = arp.ArgumentParser()

    prs.add_argument("-c","--count_data",required=True)
    # prs.add_argument("-s","--scale_factors",default = None)
    # prs.add_argument("-i","--image",default = None)
    prs.add_argument("-he","--he_overlay",nargs = 2, default = None)
    prs.add_argument("-o","--out_dir",default = None)

    args = prs.parse_args()


    if args.he_overlay is not None:
        use_img =True
    else:
        use_img = False

    if args.out_dir is None:
        args.out_dir = osp.dirname(args.count_data)


    cnt = pd.read_csv(args.count_data,
                      sep = '\t',
                      engine = 'c',
                      header = 0,
                      index_col = 0)

    data = np.array([x.split('x') for x in cnt.index]).astype(float)

    if use_img:
        with open(args.he_overlay[1],"r+") as fopen:
            scf = json.load(fopen)

        img = np.asarray(im.open(args.he_overlay[0]))
        data *= float(scf['tissue_hires_scalef'])


    meta = pd.DataFrame(np.zeros((cnt.shape[0],1)),
                        index = cnt.index,
                        columns = ['region'],
                        )
    num = 0
    interact = {'v': False,'a': True } 
    while interact['a']:
        num += 1
        interact['v'] = False
        while not interact['v']:
            sidx = {'selected_idx':None,
                    'selected_crd':None}

            fig, ax = plt.subplots()

            ax.set_aspect("equal")

            if use_img: ax.imshow(img)

            pts = ax.scatter(data[:, 1],
                            data[:, 0],
                            s=10,
                             c = meta.values.flatten(),
                             # c = 'k',
                             )

            selector = SelectFromCollection(ax, pts)


            def accept(event):
                if event.key == "enter":

                    sidx['selected_idx'] = selector.ind
                    sidx['selected_crd'] = selector.xys[selector.ind]

                    selector.disconnect()
                    ax.set_title("")
                    plt.close("all")

            fig.canvas.mpl_connect("key_press_event", accept)
            ax.set_title("Press enter to accept selected points.")
            ax.set_aspect("equal")
            plt.show()

            fig,ax = plt.subplots()

            if use_img : ax.imshow(img)

            ax.scatter(sidx['selected_crd'][:,0],
                    sidx['selected_crd'][:,1],
                    c = 'k',
                    s = 10)

            def yesno(event):
                doAsk = True
                while doAsk: 
                    if event.key == 'y':
                        interact['v'] = True
                        doAsk = False
                    elif event.key == 'n':
                        interact['v'] = False
                        doAsk = False

                plt.close("all")


            ax.set_title("Use selection (y/n)")
            fig.canvas.mpl_connect("key_press_event",yesno)
            plt.show()

        meta.iloc[sidx['selected_idx'],0] = num
        qa = input("Add more regions (y/n) >> ")
        if qa == 'y':
            interact['a'] = True
        else:
            interact['a'] = False

    new_name = input("Name of new count file >> ")
    if new_name.split('.')[-1] != 'tsv':
        new_name += '.tsv'

    meta.to_csv(osp.join(args.out_dir,new_name),
               sep = '\t',
               header = True,
               index = True,
               )
    
