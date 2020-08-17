#cython: language_level=3

from numpy cimport ndarray as ar
from numpy cimport abs as abs
cimport cython




@cython.boundscheck(False)
@cython.wraparound(False)
def PCF_r(double r,
          ar[double,ndim =2] crd1,
          ar[double,ndim =2] crd2,
          double A,
          double h,
          )->float:
    
    """Pair Correlation Function
    
    Parameters:
    ----------
    
    r : double
        distance to compute metric for
    crd1 : ar[double,ndim=2]
        coordinates for each observation
        format [n_obsx2]. Source.
    crd2 : ar[double,ndim=2]
        coordinates for each observation
        format [n_obsx2]. Target.
    A : double
        area of region
    h : double
        bandwidth

    """

    def eh(double t)->float:
        cdef double kh
        kh =  3.0 / 4.0 / h * (1.0-(t/h)**2)
        return kh

    cdef double pi = 3.141592653589793

    cdef int n1 = crd1.shape[0]
    cdef int n2 = crd2.shape[0]
    cdef double s = 0.0
    cdef double dr
    cdef int i,j

    for i in range(n1):
        for j in range(n2):
            d = (crd1[i,0]-crd2[j,0])**2 + (crd1[i,1]-crd2[j,1])**2
            d = d**0.5
            dr = abs(d - r)
            s += (eh(dr) if dr <= h else 0)

    s *= (A / (float(n1) * float(n2) * 2 * r * pi))

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
def K_uni(double t,
          ar[double,ndim =2] D,
          double A,
          ):

    """Ripley's K Univariate

    Parameters:
    -----------

    t : double
        distance to compute metric for
    D : ndarray[double,ndim=2]
        distance matrix
    A : double
        area of region

    Returns:
    --------
    Ripley's K at distance t

    """

    cdef double s  = 0.0
    cdef int i,j
    cdef int n_pts = D.shape[0]

    for i in range(n_pts-1):
        for j in range(i+1,n_pts):
            s += int(D[i,j] < t)

    s = s * A
    s = s / float(n_pts)**2

    return s*2.0


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def K_mul(double t,
#           ar[double,ndim =2] D,
#           double A,
#           ):
    
#     """Ripley's K Multivariate
    
#     Parameters:
#     -----------
    
#     t : double
#         distance to compute metric for
#     D : ndarray[double,ndim=2]
#         distance matrix
#     A : double
#         area of region
    
#     Returns:
#     --------
#     Ripley's K at distance t
    
#     """
    
#     cdef double s  = 0.0
#     cdef int i,j
#     cdef int n_1 = D.shape[0]
#     cdef int n_2 = D.shape[1]
    
#     for i in range(n_1):
#         for j in range(n_2):
#             s += int(D[i,j] < t)
            
#     s = s * A
#     s = s / float(n_1*n_1)
    
#     return s

@cython.boundscheck(False)
@cython.wraparound(False)
def K_uni(double t,
          ar[double,ndim =2] crd,
          double A,
          ar[double,ndim=1] ws,
          ):

    """Ripley's K Univariate
    with control for edge effects

    Parameters:
    -----------

    t : double
        distance to compute metric for
    crd : ar[double,ndim=2]
        coordinates for each observation
        format [n_obsx2]
    A : double
        area of region
    ws : ar[double,ndim=1]
        weights to use in
        edge effect correction

    Returns:
    --------
    Ripley's K at distance t

    """

    cdef double s  = 0.0
    cdef int i,j
    cdef ar[long,ndim=1] ce
    cdef int n_pts = crd.shape[0]

    for i in range(n_pts-1):
        for j in range(i+1,n_pts):
            d = (crd[i,0]-crd[j,0])**2 +\
                (crd[i,1]-crd[j,1])**2
            d = d**0.5
            s += int(d < t)*ws[i]


    s = s * A
    s = s / float(n_pts)**2

    return s*2.0


@cython.boundscheck(False)
@cython.wraparound(False)
def K_mul(double t,
          ar[double,ndim =2] crd1,
          ar[double,ndim =2] crd2,
          double A,
          ar[double,ndim =1] ws,
          ):

    """Ripley's K Multivariate
    with control 
    for edge effects

    
    Parameters:
    -----------
    
    t : double
        distance to compute metric for
    crd1 : ar[double,ndim=2]
        coordinates for each observation
        format [n_obsx2]. Source.
    crd2 : ar[double,ndim=2]
        coordinates for each observation
        format [n_obsx2]. Target.
    A : double
        area of region
    ws : ar[double,ndim=1]
        weights to use in
        edge effect correction
    
    Returns:
    --------
    Ripley's K at distance t
    
    """
   
    cdef double s  = 0.0
    cdef int i,j
    cdef int n_1 = crd1.shape[0]
    cdef int n_2 = crd2.shape[1]
    
    for i in range(n_1):
        for j in range(n_2):
            d = (crd1[i,0]-crd2[j,0])**2 + (crd1[i,1]-crd2[j,1])**2
            d = d**0.5
            s += int(d < t)*ws[i]
           
    s = s * A
    s = s / float(n_1*n_2)
    
    return s

