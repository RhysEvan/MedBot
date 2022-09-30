import numpy as np
import matplotlib.pyplot as plt

#from forward import *

from matplotlib.widgets import Slider, Button
from functools import partial

class interface():
    
    def __init__(self,a,t,r,d, active, limits):
        self.alpha = a
        self.theta = t
        self.radius = r 
        self.dists = d
        self.active = active
        self.limits = limits 
        
        self.update_position()
             
        x,y,z = self.positions
        
        n = range(len(x))
        fig, ax = plt.subplots()
        ax = fig.axes
        ax = plt.axes(projection='3d')
        self.plot_plane(ax)
        self.plot_axes(ax)
        # Data for a three-dimensional line
        line = ax.plot3D(x, y, z, 'gray', linewidth=5)[0]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylabel('z')

        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(left=0.05, right=0.5,
                            bottom=0.1, top=0.99
                           )

        self.init_sliders(active,limits)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = plt.axes([.6, 0.025, 0.3, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')
        
        
        button.on_clicked(self.reset)
        self.line = line
        self.fig = fig
        self.ax = ax
        self.button = button
        plt.show()
        
    def init_sliders(self,active,limits):
        
        sliders = []
        i = 0
        for n,limit in enumerate(limits):
            if len(limit)==0: continue
            if active[n]=="r":            idx = 2
            elif active[n]=="t":          idx = 1
            else: continue
                            
            # Make a horizontal slider to control the frequency.
            ypos = 0.9-i*0.1
            ax = plt.axes([0.6, ypos, 0.35, 0.05])
            slider = Slider( ax=ax, label="Joint" + str(i), 
                          valmin=limit[0], valmax=limit[1],
                          valinit=limit[0])
            i += 1 
            func = partial(self.set_motor,n)
            slider.on_changed(func)
            sliders.append(slider)
            
        self.sliders = sliders        
        
    def set_motor(self,idx,value):
        t = self.theta
        r = self.radius
        active = self.active

        if active[idx]=="r":            r[idx] = value
        elif active[idx]=="t":          t[idx] = value

        self.theta = t
        self.radius = r
        self.update()
        
    def update(self):
        self.update_position()
        x,y,z = self.positions
        self.line.set_data_3d(x,y,z)
        #self.line.set_zdata(z)
        self.fig.canvas.draw_idle()
        
    def reset(self,x):
        return

        
    def plot_plane(self,ax):
    
        x = np.linspace(-100, 100, 10)
        y = np.linspace(-100, 100, 30)
        X, Y = np.meshgrid(x, y)
        Z = X*Y*0
        ax.plot_surface(X, Y, Z, alpha=0.2)
        #ax.contour3D(X, Y, Z, 50, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    def plot_axes(self,ax):
        zero = [0,0]
        x = [0,100]
        y = [0,100]
        z = [0,100]
        ax.plot3D(x, zero, zero, 'red')
        ax.plot3D(zero, y, zero, 'green')
        ax.plot3D(zero, zero, z, 'blue')
        
    def update_position(self):
        a = self.alpha
        t = self.theta
        r = self.radius
        d = self.dists
        position = forward_all(a,t,r,d)
        self.positions = position

def forward_all(alpha, theta, radius, dists):
    origin = np.array((0,0,0,1))
    positions = [origin]
    A = np.array(origin)[None,:]
    
    DHts = make_transforms(alpha, theta, radius, dists)

    for DH in DHts:
        B = forward_transform(origin, DH)
        positions.append(B)
        A = B.copy()
            
    return np.array(positions)[:,:3].T

def make_transforms(alpha, theta, radius, dists):
    
    DHts = []
    T = np.eye(4)
    for i in range(len(alpha)):
        a = np.radians(alpha[i])
        t = np.radians(theta[i])
        r = radius[i]
        d = dists[i]
        DHt = DH_transform(a,t,r,d) 
        DHt = np.matmul(T,DHt)
        T = DHt.copy()
        DHts.append(DHt)
    return np.array(DHts)

def forward_transform(A, DHt):
    B = np.matmul(DHt,A.T).T   
    return B

    
def DH_transform(a,t,r,z):
    Zt = Z_transform(a,t,r,z)
    Xt = X_transform(a,t,r,z)
    DHt = np.matmul(Zt,Xt)
    return DHt


def X_transform(a,t,r,d):
    
    Xt = [[1,0,0,r],
          [0,np.cos(a), -1*np.sin(a),0],
          [0,np.sin(a),  np.cos(a),0],
          [0,0,0,1]]
    Xt = np.array(Xt)

    return Xt
    
def Z_transform(a,t,r,d):
    
    Zt = [[np.cos(t), -1*np.sin(t),0,0],
          [np.sin(t), np.cos(t), 0,0],
          [0,0,1,d],
          [0,0,0,1]]
    Zt = np.array(Zt)
    return Zt


## Hanging arm
alpha =  [0,0,180, 90,90, 90]
theta =  [0,0,90, 45,40,0] 
radius = [0,0,100,10,75,50]
dists =  [0,200,0,0,0,0]
active = ["r","","r","t","t","t"]
limits = [[0,200],[],[20,200],[0,270],[0,180],[0,350]]

x = interface(alpha, theta, radius, dists, active, limits)