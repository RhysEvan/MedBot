import numpy as np
import open3d as o3d
import pptk
import matplotlib.pyplot as plt
x = np.arange(np.pi/2,8,0.1)   # start,stop,step
y = np.sin(x)
plt.plot(x,y)
plt.show()