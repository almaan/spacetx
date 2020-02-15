# Visium Data, Results and Examples related to SpaceJam2

## Data

In total 4 samples (also referred to as sections) were produced for the
SpaceJam2 event. These are all from mouse brain, including - but not only
covering- the visual isocortex. The samples are given the identifiers Allen-x
with x being one of the numbers 1,2,3 or 4. The two samples Allen-1 and Allen-2
originate from the same mouse, whilst Allen-3 and Allen-4 are generated from
another individual.

All data is found within the ```data``` directory (the directory ```_data```
contains files used in the jupyter notebooks, do **not** use these). We've
adopted the ```.h5ad``` format ([read
more](https://anndata.readthedocs.io/en/latest)), when representing both our
data and format. There are 4 gziped ```.h5ad``` files - one for each sample -
in the ```data``` directory. Short descriptions of the content of these files are given below :

* X : The raw count matrix [n\_spots x n\_genes].
* obs 
  * barcodes : 10x barcodes identifiers
  * under_tissue : Binary indiator if spot is under tissue or not (1 = is under,
    0 = is outside)
  * _x : array x-coordinates
  * _y : array y-coordinates
  * x : pixel x-coordinates (use these for visualization)
  * y : pixel y-coordinates (use these for visualization)
  * n_counts : Total number of observed UMI's (transcripts) in a given spot
  * sample : string indicating which sample a spot belongs to
* var
  * n_counts : total number of observed UMI's (transcripts) of a given gene
* uns
  * spot\_diameter\_fullres : diameter of spots for the full resolution image
  * tissue\_hires\_scalef : scaling factor, transforms full resolution pixel
    coordinates (obs.x and obs.y) to coordinates compatible with the hires image.
  * fiducial\_diameter\_fullres : diameter of fiducials for the full resolution image
  * image\_hires : HE-image 
* obsm
  * proportions_cluster : proportions for every cluster within every spot
  * proportions_class : proportions for every class within every spot
 
## Results

The provided single cell data has been mapped to all of the 4 Visium samples
using [Stereoscope](https://github.com/almaan/stereoscope), resulting in
proportion estimates for each type (cluster or class) within every spatial
capture location (spot). 

As mentioned above these results are found in the ```.h5ad``` files attached to the
attribute ```obsm``` with the interpretation:

* __proportions_cluster__ : proportions for every cluster within every spot
* __proportions_class__ : proportions for every class within every spot


# Examples of ST/Visium data analysis

Very basic examples of ST/Visium data analysis, given as jupyter notebooks. The
two images below show an excerpt from the analysis (unsupervised clustering of
the isocortex, found in example 2)

<div align="center">

![sample-1](https://github.com/almaan/spacetx/blob/master/_data/img/sample-1-cluster.png?raw=true)
![sample-2](https://github.com/almaan/spacetx/blob/master/_data/img/sample-2-cluster.png?raw=true)

</div>


1. Standard explorative analysis of one section
2. Specific analysis of Isocortex, clustering and DGE
3. Single Cell data integration using Stereoscope

The notebooks can be found in the directory "Examples"
