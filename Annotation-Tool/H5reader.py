import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

def read_H5(fn, dataset="mask_data"):
    with h5py.File(fn, 'r') as fh: 
        fh = fh['hypercube']  
        if dataset not in fh.keys():  return None
        data = np.array(  fh[dataset][:]  )     
    
    if data.dtype==np.uint8:         pass
    else:
        if data.max()<=1:  data = (data*255).astype(np.uint8)

    return data

path = r"C:\Users\gille\Downloads\hypercubes_rhys/hypercube_cut"
alle_files = glob.glob(path + "/*")
i = 0
jep = input()

while jep != "stop":
    x = read_H5(alle_files[i], dataset = "data")
    plt.imshow(x[:,:,9])
    plt.show(block = True)
    i+=1