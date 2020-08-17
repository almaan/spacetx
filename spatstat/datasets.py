import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from abc import ABC, abstractmethod,abstractproperty
from typing import Dict,Union,List,Optional,Tuple

from sklearn.preprocessing import LabelEncoder as LE
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

import anndata as ad

import os.path as osp


class DataSet(ABC):
    """Class to hold expression data"""
    def __init__(self,
                 pth : Optional[str] = None,
                 data : Optional[Union[pd.DataFrame,ad.AnnData]] = None,
                )->None:


        # get data
        self.pth = pth
        if pth is not None:
            self._data : pd.DataFrame = self._from_pth(pth)
        elif data is not None:
            self._data : pd.DataFrame= self._from_data(data)

        #TODO: Implement else catch


        # number of data points
        self.N = self._data.shape[0]
        # width and height of region of study
        # when applicable
        self.width : Optional[float] = None
        self.height : Optional[float] = None
        self.minside : Optional[float] = None

        # generate distance matrix
        # self.D : np.ndarray = self._get_weights()
        # self.max_dist = self.D[self.D > 0].max()
        # self.min_dist = self.D[self.D > 0].min()

        # get ranges of data region
        self.ranges : Dict[str,np.ndarray] = self._get_ranges()
        # compute are for data region
        self.area : float = self._get_area()

        # numeric versions of labels
        self.label_to_num : Optional[LE] = None

        # unique cell types present in data
        self.uni_types : np.ndarray = np.unique(self.types)

    @abstractmethod
    def _from_pth(self,
                  pth : str,
                  )->pd.DataFrame:
        """read data from path"""
        pass

    def _from_data(self,
                  pth : str,
                  )->pd.DataFrame:
        """read data from path"""
        pass


    @abstractproperty
    def crd(self,)->np.ndarray:
        """get coordinates"""
        pass

    @abstractmethod
    def _get_area(self,
                  *args,
                  **kwargs,
                 )->float:
        """compute area of data region"""
        pass

    @abstractmethod
    def _get_ranges(self,
                  )->Dict[str,np.ndarray]:

        """compute ranges of data region"""

        pass

    @abstractproperty
    def types(self,)->np.ndarray:
        """vector with cell type of each data point"""
        pass

    @property
    def numeric_types(self,
                     )->np.ndarray:
        """numeric version of cell type labels"""
        if self.label_to_num is None:
            self.label_to_num = LE().fit(self.types)
        return self.label_to_num.transform(self.types)

    @abstractmethod
    def sample_crd(self,
                   npts : int,
                  )->np.ndarray:
        """sample coordinates from data region"""
        pass

    @abstractmethod
    def _get_edgedistances(self,
                           )->None:
        pass


    def _get_weights(self,
                    )->np.ndarray:
        """compute distance weights"""

        mat =  csr_matrix(cdist(self.crd,
                                self.crd,
                                metric = "euclidean"),
                          )
        return mat


    def simulate(self,
                 npts : int,
                )->np.ndarray:
        """simulate CSR in the data region"""

        npts = np.random.poisson(npts)
        crd = sample_crd(npts)

    def __getitem__(self,x)->pd.DataFrame:
        return self._data.iloc[x,:]

    def __len__(self,)->int:
        return self.N

    def get_ext(self,
                )->None:

        ext = osp.basename(self.pth)\
                 .split(".")[-1]

        return ext


# Define child of DataSet class to hold MERFISH data

class MerfishData(DataSet):
    def __init__(self,
                 pth : Optional[str] = None,
                 data : Optional[Union[pd.DataFrame,ad.AnnData]] = None,
                 type_colname : str = "types",
                )->None:

        self.type_colname = type_colname

        super().__init__(pth = pth, data = data)

        self.reposition()
        self.sides,self.corners = self._get_sides()


    def _from_pth(self,
                  pth : str,
                  )->pd.DataFrame:


        ext =  self.get_ext()
        if ext == "csv":
            data = pd.read_csv(pth,
                            sep = ',',
                            header = 0,
                            index_col = 0,
                            )

        elif ext == "h5ad":
            data = ad.read_h5ad(pth).obs

        return data

    def _from_data(self,
                   data : Optional[Union[pd.DataFrame,ad.AnnData]] = None,
                   )->pd.DataFrame:

        if isinstance(data,ad.AnnData):
            return pd.DataFrame(data.obs)
            
        elif isinstance(data,pd.DataFrame):
            return data
        else:
            raise NotImplementedError


    @property
    def crd(self,)->np.ndarray:
        return self._data[["x","y"]].values


    @property
    def types(self,
              )->np.ndarray:
        return self._data[self.type_colname].values


    def reposition(self,)->None:

        x_crds = self.crd[:,0] - self.ranges["x"][0]
        y_crds = self.crd[:,1] - self.ranges["y"][0]

        self._data["x"] = x_crds
        self._data["y"] = y_crds

        self.ranges = self._get_ranges()
        self.sids = self._get_sides()

    def _get_sides(self,)->np.ndarray:

        # clockwise order
        # start in bottom
        # left corner

        x = self.ranges["x"]
        y = self.ranges["y"]

        # sides = np.zeros((4,2))
        corners = np.zeros((4,2))


        corners[0,:] = (x[0],y[0])
        corners[1,:] = (x[0],y[1])
        corners[2,:] = (x[1],y[1])
        corners[3,:] = (x[1],y[0])


        sides = corners[[1,2,3,0],:] - corners[[0,1,2,3],:]

        return (sides,corners)

    def _get_ranges(self,
                   )->Dict[str,np.ndarray]:


        mxs = self.crd.max(axis =0)
        mns = self.crd.min(axis =0)

        _ranges = dict(x = np.array((mns[0],mxs[0])),
                       y = np.array((mns[1],mxs[1])),
                      )

        self.width = mxs[0] - mns[0]
        self.height = mxs[1] - mns[0]
        self.minside = np.min((self.width,self.height))

        return _ranges

    def _get_area(self,
                 )->float:

        dx = self.ranges["x"][1]-self.ranges["x"][0]
        dy = self.ranges["y"][1]-self.ranges["y"][0]

        return dx*dy

    def proj(self,
             x : np.ndarray,
             v : np.ndarray)->np.ndarray:

        vn = v / np.linalg.norm(v,axis=1,keepdims=True)
        xv = np.einsum("ij,kj->i",x,vn).reshape(-1,1)

        return xv * vn

    def perp(self,
             x : np.ndarray,
             v : np.ndarray)->np.ndarray:

        return x - self.proj(x,v)


    def _get_edgedistances(self,
                           )->None:

        ed = np.zeros((self.N,4))

        for s in range(4):
            
            ncrd = self.crd - self.corners[s,:]
            perp_x_s = self.perp(ncrd,
                                 self.sides[s,:].reshape(1,-1))

            ed[:,s] = np.linalg.norm(perp_x_s,axis = 1)

        return ed

    def weights(self,
                t : Optional[float] = None,
                )->np.ndarray:

        w = np.ones(self.N)
        if t is None:
            print("hej")
            return w
        else:

            assert t <= self.minside / 2,\
                "distance can max be half"\
                "of the shortest side"

            ed = self._get_edgedistances()
            ed = np.sort(ed,axis = 1)
            nc = np.sum(ed - t < 0,axis = 1)
            for k,n in enumerate(nc):
                d1 = ed[k,0]
                d2 = ed[k,1]
                if n == 1:
                    e = np.sqrt(t**2-d1**2)
                    a = np.arccos(d1/t)
                    w[k] = np.pi*t**2 / (e*d1 + (np.pi -a)*t**2)

                elif n == 2:
                    e1 = np.sqrt(t**2-d1**2)
                    e2 = np.sqrt(t**2-d2**2)
                    a1 = np.arccos(d1/t)
                    a2 = np.arccos(d2/t)

                    if t**2 > d1**2 + d2**2:
                        w[k] = np.pi*t**2
                        w[k] /= (d1*d2+0.5*(e1*d1+e2*d2) +\
                                 (0.75*np.pi - 0.5*a1-0.5*a2)*t**2)

                    elif t**2 < d1**2 + d2**2:
                        w[k] = np.pi*t**2
                        w[k] /= (e1*d1 + e2*d2 + (np.pi - a1 - a2)*t**2)
            return w


    def sample_crd(self,
                  npts: int,
                  )->np.ndarray:

        xs = np.random.uniform(low = self.ranges["x"][0],
                               high = self.ranges["x"][1],
                               size = npts,
                              ).reshape(-1,1)

        ys = np.random.uniform(low = self.ranges["x"][0],
                               high = self.ranges["x"][1],
                               size = npts,
                              ).reshape(-1,1)

        return np.hstack((xs,ys))
