import os
import glob
place = r"C:\Users\gille\OneDrive\Documenten\gilles mapppeke\Machine Learning\data\hypercube_cut"
filenames = glob.glob(place + r'/*h5')

to_displace = "cut_labeled"
to_displace_with = "labeled"

for i, fn in enumerate(filenames):
    if fn.find(to_displace)!=-1:
        fn = fn.replace("_"+ to_displace + "_hypercube", "_" + to_displace_with + "_hypercube")
        print(f"oud: {filenames[i]}")
        print(f"nieuw: {fn}")
        os.rename(filenames[i], fn)
