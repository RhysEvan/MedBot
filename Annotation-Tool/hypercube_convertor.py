import os
import h5py
import matplotlib.pyplot as plt
import easygui as eg
import numpy as np
import glob

def labelled_hypercube_to_image_and_mask(h5, *bands):
    path, fn = os.path.split(h5)
    name,ext = fn.split('.')
    image_path = path + r'/image/'
    mask_path = path + r'/mask/'
    if os.path.exists(image_path) == False:
        os.mkdir(image_path)
    if os.path.exists(mask_path) == False:
        os.mkdir(mask_path)

    f = h5py.File(h5, "r")
    hypercube = f['hypercube']['data']
    new_image = np.array([])
    for i in bands:
        new_band = hypercube[:, :, i][..., np.newaxis].copy()
        if new_image.size == 0:
            new_image = new_band.copy()
        else:
            new_image = np.concatenate((new_image, new_band), axis=2)
    plt.imsave(image_path +name+'.jpg',new_image.squeeze())
    mask = f['hypercube']['labels'][:]
    mask[mask==1] = 255
    plt.imsave(mask_path + name + '_mask.gif', mask, cmap='gray')

def import_h5(loc):
    if os.path.exists(loc):
        filenames = glob.glob(directory + "/*.h5")
        filenames = [fn for fn in filenames if fn.find("_labeled_hypercube") != -1]
        print(filenames)
        for i in range(len(filenames)):
            labelled_hypercube_to_image_and_mask(filenames[i], [5,6,7])

directory = eg.diropenbox("Choose map with the labeled hypercubes")
import_h5(directory)
