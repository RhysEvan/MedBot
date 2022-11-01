#libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from functools import partial
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import animation

#files
from kinematics import forward_all, get_DH_params
import presets

################################################
class interface(QWidget):
    
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        atrdal = get_DH_params(presets.preset_models["HangingArm"])

        self.alpha, self.theta, self.radius, self.dists, self.active, self.limits = atrdal

        self.update_position()
        self.setup_fig()
       
    def setup_fig(self):
        x,y,z = self.positions
        n = range(len(x))
        self.fig = Figure()
        self.widget = FigureCanvasQTAgg(self.fig)
        self.layoutvertical = QVBoxLayout(self)
        self.layoutvertical.addWidget(self.widget)
        self.axes = self.widget.figure.add_subplot(111, projection='3d', position=(0,0,1,1))
        #plt.tight_layout()

        ax = self.axes
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylabel('z')

        self.plot_plane(ax)
        self.plot_axes(ax)

        # Data for a three-dimensional line
        line = ax.plot3D(x, y, z, 'gray', linewidth=5)[0]
        # adjust the main plot to make room for the sliders
        self.line = line
        self.ax = ax
        self.widget.show()

    ######################################
    #  will remove references to kinematics 
    def set_active(self, new_active):
        
        mot_pos = self.get_motor_positions()
        active = self.active

        i = 0
        for n in range(mot_pos.shape[1]):
            if active[n]=="": continue
            v = new_active[i]

            if active[n]=="r":            mot_pos[2,n] = v
            elif active[n]=="t":          mot_pos[1,n] = v
            else: continue
            i += 1
            
        self.set_motor_positions(mot_pos) 

        return mot_pos

    def set_motor_positions(self, mot_pos):
        
        a,t,r,d = mot_pos   
        self.alpha = a
        self.theta = t
        self.radius = r 
        self.dists = d  

    def set_active_motor(self,idx,value):
        
        t,r = self.theta, self.radius

        if self.active[idx]=="r":            r[idx] = int(value)
        elif self.active[idx]=="t":          t[idx] = int(value)
        
        self.theta,self.radius  = t,r

        self.update()
        
    def update_position(self):
        a, t = self.alpha, self.theta
        r, d = self.radius, self.dists
        self.positions = forward_all(a,t,r,d)
    #################################
    
    def update(self):
        self.update_position()
        x,y,z = self.positions
        self.line.set_data_3d(x,y,z)
        self.fig.canvas.draw_idle()

    def rerender(self):

        ax = self.axes
        self.plot_plane(ax)
        self.plot_axes(ax)
        self.update()

    def plot_plane(self,ax):
    
        x = np.linspace(-100, 250, 10)
        y = np.linspace(-10, 250, 30)
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
        
    def draw_path(self, position_list):

        x,y,z = np.array(position_list).T[:3]
        path = self.ax.plot3D(x,y,z, 'bo-', linewidth=1)[0]
        self.path = path

    def animate(self, position_list):
        frms = len(position_list)
        self.posp = position_list
        self.animation = animation.FuncAnimation(self.widget.figure, self.update_animation, frames = frms, interval = 1,repeat = False)
    
    def update_animation(self,i):
        x,y,z = self.posp[i]
        self.line.set_data_3d(x,y,z)