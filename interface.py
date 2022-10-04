#libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from functools import partial
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

#files
from forward import *

################################################
class interface(QWidget):
    
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.alpha, self.theta, self.radius, self.dists, self.active, self.limits = get_DH_params(preset_models["HangingArm"])
        
        self.update_position()
        self.setup_fig()
       

    def setup_fig(self):
        x,y,z = self.positions
        n = range(len(x))
        self.fig = Figure()
        self.widget = FigureCanvasQTAgg(self.fig)
        self.layoutvertical = QVBoxLayout(self)
        self.layoutvertical.addWidget(self.widget)
        self.axes = self.widget.figure.add_subplot(111, projection='3d')
        ax = self.axes
        self.plot_plane(ax)
        self.plot_axes(ax)
        # Data for a three-dimensional line
        line = ax.plot3D(x, y, z, 'gray', linewidth=5)[0]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylabel('z')
        # adjust the main plot to make room for the sliders
        self.line = line
        self.ax = ax
        self.widget.show()

    def set_motor(self,idx,value):
        t = self.theta
        r = self.radius
        active = self.active

        if active[idx]=="r":            r[idx] = int(value)
        elif active[idx]=="t":          t[idx] = int(value)
        print(r)
        print(t)
        self.theta = t
        self.radius = r
        self.update()
        
    def update(self):
        self.update_position()
        x,y,z = self.positions
        self.line.set_data_3d(x,y,z)
        #self.line.set_zdata(z)
        self.fig.canvas.draw_idle()
        
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