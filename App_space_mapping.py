import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib import colors

class visual_pointcloud(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.fig = Figure()
        self.widget = FigureCanvasQTAgg(self.fig)
        self.layoutvertical = QVBoxLayout(self)
        self.layoutvertical.addWidget(self.widget)
        self.axes = self.widget.figure.add_subplot(111)
        self.widget.show()