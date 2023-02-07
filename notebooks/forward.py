
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

#############################################        
######################################

def update_position(mot_pos, new_active, active):

    mot_pos = set_active(mot_pos, new_active, active)
    Positions = forward(*mot_pos)  

    return Positions

def generate_maps(alpha, theta, radius, dists, active, limits):
    
    inputs, target = [],[]
    for x in range(500):
        mot_pos = random_positions( alpha, theta, radius, dists, active, limits)
        act_pos = get_active(mot_pos, active, limits)

        Positions = forward(*mot_pos)[:,:3]
        EndPos = Positions[-1]

        inputs.append(EndPos)
        target.append(act_pos)

    inputs = np.array(inputs)
    target = np.array(target)    
    
    return inputs, target

def random_positions( alpha, theta, radius, dists, active, limits):
    
    mot_pos = np.array([alpha,theta,radius,dists])
    
    for n,limit in enumerate(limits):
        if active[n]=="": continue
        if len(limit)==0: continue

        v = np.random.uniform(limit[0],limit[1])

        if active[n]=="t":            mot_pos[1,n] = v
        if active[n]=="r":            mot_pos[2,n] = v
    
    return mot_pos
             
def get_active(mot_pos, active):
    
    out_pos = []
    for n,a in enumerate(active):
        if a=="": continue
        if a=="t":          v = mot_pos[1,n]    
        if a=="r":          v = mot_pos[2,n]
        #else: continue
            
        out_pos.append(v)
    
    return np.array(out_pos)

def set_active(mot_pos, new_active, active):
    
    i = 0
    for n in range(mot_pos.shape[1]):
        if active[n]=="": continue
        v = new_active[i]

        if active[n]=="r":            mot_pos[2,n] = v
        elif active[n]=="t":          mot_pos[1,n] = v
        else: continue
        i += 1
            
    return mot_pos

#######################
#######################

def plot_joints(positions, fig=None):
    
    if fig is None: fig = plt.figure()
        
    x,y,z = positions
    n = range(len(x))
    ax = fig.axes
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    ax.plot3D(x, y, z, 'gray', linewidth=5)
    ax.scatter3D(x, y, z, c=n, s= 100, cmap='hsv')
    plot_plane(ax)
    plot_axes(ax)
    
def plot_plane(ax):
    
    x = np.linspace(-100, 100, 10)
    y = np.linspace(-100, 100, 30)
    X, Y = np.meshgrid(x, y)
    Z = X*Y*0
    ax.plot_surface(X, Y, Z, alpha=0.2)
    #ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
def plot_axes(ax):
    zero = [0,0]
    x = [0,100]
    y = [0,100]
    z = [0,100]
    ax.plot3D(x, zero, zero, 'red')
    ax.plot3D(zero, y, zero, 'green')
    ax.plot3D(zero, zero, z, 'blue')


def make_path():
    PositionList = []
    for n,limit in enumerate(limits):
        if active[n]=="": continue
        if len(limit)==0: continue

        values = np.linspace(limit[0],limit[1],10)
        for v in values:
            if active[n]=="r":            radius[n] = v
            if active[n]=="t":            theta[n] = v
            Positions = forward_all(alpha, theta, radius, dists)
            PositionList.append(Positions)


def get_DH_params(model):
    
    alpha = model["alpha"]
    theta =  model["theta"]
    radius = model["radius"]
    dists = model["dists"]
    active = model["active"]
    limits = model["limits"]
    
    return alpha, theta, radius, dists, active, limits
