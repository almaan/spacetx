load_results = True
prep_results = True
make_tensor = True

use_edgecorrection = False

files = os.listdir(OUT_DIR)
if use_edgecorrection:
    files = list(filter(lambda x: "-EC" in x,files))
else:
    files = list(filter(lambda x: "-OR" in x,files))

results = dict()

cell_types = pd.Index([])
indices = pd.Index([])

if load_results:
    for f in files:
        sample = osp.basename(f).split(".")[0]
        _res = pd.read_csv(osp.join(OUT_DIR,f),
                           sep = '\t',
                           header = 0,
                           index_col = 0)
        
        cell_types = cell_types.union(_res.columns)
        indices = indices.union(_res.index)
        
        results.update({sample:_res})
    

    
if prep_results:
    n_row = len(indices)
    n_col = len(cell_types)
    for k,v in results.items():
        _tmp = pd.DataFrame(np.zeros((n_row,n_col)),
                            index = indices,
                            columns = cell_types,
                           )
        inter_r = indices.intersection(v.index)
        inter_c = cell_types.intersection(v.columns)
        
        _tmp.loc[inter_r,inter_c] = v.loc[inter_r,inter_c].values
        results[k] = _tmp
        
if make_tensor:
    n_row = len(indices)
    n_col = len(cell_types)
    n_samples = len(files)
    
    results_tensor = np.zeros((n_samples,n_row,n_col))
    for k,v in enumerate(results.values()):
        results_tensor[k,:,:] = v.values
