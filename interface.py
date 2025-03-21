#libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from functools import partial
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import animation
import copy
#files
from Inverse_kinematics_ML.kinematics import Kinematics,forward_all, get_DH_params

################################################
class interface(QWidget):
    
    def __init__(self, parent, robot="Prismatic3"):
        super(QWidget, self).__init__(parent)
        self.kin = Kinematics(robot, json_path="./Inverse_kinematics_ML/presets.json")
        self.setup_fig()

    def setup_fig(self):
        x,y,z = self.kin.position
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

    def set_active_motor(self,idx,value):
        self.kin.set_active_motor(idx, value)
        self.update()
   
    def update(self):
        self.kin.update_position()
        x,y,z = self.kin.position
        self.line.set_data_3d(x,y,z)
        self.fig.canvas.draw_idle()

    def rerender(self):
        ax = self.axes
        ## TODO Clear all object in the axis to rerender?
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
        
    def draw_path(self, position_list, i):
        x,y,z = np.array(position_list).T[:3]
        if i == 0:
            self.path = self.ax.plot3D(x,y,z, 'bo-', linewidth=1)[0]
        else :
            self.path.set_data_3d(x,y,z)
        self.update()
        self.fig.canvas.draw()
    
    def hide_path(self):
        self.path.set_data_3d([],[],[])
        self.update()
        self.fig.canvas.draw()

    def animate(self, position_list):
        frms = len(position_list)
        self.posp = position_list
        self.update()
        self.animation = animation.FuncAnimation(self.widget.figure, self.update_animation, frames = frms, interval = 50,repeat = False)
    
    def update_animation(self,i):
        x,y,z = self.posp[i]
        self.line.set_data_3d(x,y,z)