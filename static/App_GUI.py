# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'static/App_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1836, 927)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setContentsMargins(-1, 11, -1, -1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 5, 7, 1, 1)
        self.alfacoord = QtWidgets.QLineEdit(self.centralwidget)
        self.alfacoord.setObjectName("alfacoord")
        self.gridLayout_2.addWidget(self.alfacoord, 6, 6, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 5, 9, 1, 1)
        self.zcoord = QtWidgets.QLineEdit(self.centralwidget)
        self.zcoord.setObjectName("zcoord")
        self.gridLayout_2.addWidget(self.zcoord, 5, 10, 1, 1)
        self.ycoord = QtWidgets.QLineEdit(self.centralwidget)
        self.ycoord.setObjectName("ycoord")
        self.gridLayout_2.addWidget(self.ycoord, 5, 8, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setObjectName("label_19")
        self.gridLayout_2.addWidget(self.label_19, 6, 9, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 0, 7, 1, 2)
        self.text_eabs = QtWidgets.QLabel(self.centralwidget)
        self.text_eabs.setObjectName("text_eabs")
        self.gridLayout_2.addWidget(self.text_eabs, 7, 4, 1, 1)
        self.xcoord = QtWidgets.QLineEdit(self.centralwidget)
        self.xcoord.setObjectName("xcoord")
        self.gridLayout_2.addWidget(self.xcoord, 5, 6, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setObjectName("label_20")
        self.gridLayout_2.addWidget(self.label_20, 6, 11, 1, 1)
        self.aabs = QtWidgets.QSlider(self.centralwidget)
        self.aabs.setOrientation(QtCore.Qt.Horizontal)
        self.aabs.setObjectName("aabs")
        self.gridLayout_2.addWidget(self.aabs, 3, 1, 1, 3)
        self.eabs = QtWidgets.QSlider(self.centralwidget)
        self.eabs.setOrientation(QtCore.Qt.Horizontal)
        self.eabs.setObjectName("eabs")
        self.gridLayout_2.addWidget(self.eabs, 7, 1, 1, 3)
        self.cabs = QtWidgets.QSlider(self.centralwidget)
        self.cabs.setOrientation(QtCore.Qt.Horizontal)
        self.cabs.setObjectName("cabs")
        self.gridLayout_2.addWidget(self.cabs, 5, 1, 1, 3)
        self.dabs = QtWidgets.QSlider(self.centralwidget)
        self.dabs.setOrientation(QtCore.Qt.Horizontal)
        self.dabs.setObjectName("dabs")
        self.gridLayout_2.addWidget(self.dabs, 6, 1, 1, 3)
        self.homing = QtWidgets.QPushButton(self.centralwidget)
        self.homing.setObjectName("homing")
        self.gridLayout_2.addWidget(self.homing, 9, 0, 1, 1)
        self.fabs = QtWidgets.QSlider(self.centralwidget)
        self.fabs.setOrientation(QtCore.Qt.Horizontal)
        self.fabs.setObjectName("fabs")
        self.gridLayout_2.addWidget(self.fabs, 8, 1, 1, 3)
        self.babs = QtWidgets.QSlider(self.centralwidget)
        self.babs.setOrientation(QtCore.Qt.Horizontal)
        self.babs.setObjectName("babs")
        self.gridLayout_2.addWidget(self.babs, 4, 1, 1, 3)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 8, 0, 1, 1, QtCore.Qt.AlignRight)
        self.compiling = QtWidgets.QPushButton(self.centralwidget)
        self.compiling.setObjectName("compiling")
        self.gridLayout_2.addWidget(self.compiling, 9, 3, 1, 1)
        self.betacoord = QtWidgets.QLineEdit(self.centralwidget)
        self.betacoord.setObjectName("betacoord")
        self.gridLayout_2.addWidget(self.betacoord, 6, 8, 1, 1)
        self.text_aabs = QtWidgets.QLabel(self.centralwidget)
        self.text_aabs.setObjectName("text_aabs")
        self.gridLayout_2.addWidget(self.text_aabs, 3, 4, 1, 1)
        self.motorlist = QtWidgets.QListWidget(self.centralwidget)
        self.motorlist.setMinimumSize(QtCore.QSize(300, 0))
        self.motorlist.setObjectName("motorlist")
        self.gridLayout_2.addWidget(self.motorlist, 1, 7, 3, 5)
        self.remove_motor = QtWidgets.QPushButton(self.centralwidget)
        self.remove_motor.setObjectName("remove_motor")
        self.gridLayout_2.addWidget(self.remove_motor, 9, 5, 1, 1)
        self.text_cabs = QtWidgets.QLabel(self.centralwidget)
        self.text_cabs.setObjectName("text_cabs")
        self.gridLayout_2.addWidget(self.text_cabs, 5, 4, 1, 1)
        self.executing = QtWidgets.QPushButton(self.centralwidget)
        self.executing.setObjectName("executing")
        self.gridLayout_2.addWidget(self.executing, 9, 4, 1, 1)
        self.recording = QtWidgets.QPushButton(self.centralwidget)
        self.recording.setObjectName("recording")
        self.gridLayout_2.addWidget(self.recording, 9, 2, 1, 1)
        self.animating = QtWidgets.QPushButton(self.centralwidget)
        self.animating.setObjectName("animating")
        self.gridLayout_2.addWidget(self.animating, 9, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 3, 0, 1, 1, QtCore.Qt.AlignRight)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 2, 0, 1, 1)
        self.information_window = QtWidgets.QTextBrowser(self.centralwidget)
        self.information_window.setObjectName("information_window")
        self.gridLayout_2.addWidget(self.information_window, 10, 0, 1, 12)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 7, 0, 1, 1, QtCore.Qt.AlignRight)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 6, 0, 1, 1, QtCore.Qt.AlignRight)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 5, 0, 1, 1, QtCore.Qt.AlignRight)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 4, 0, 1, 1, QtCore.Qt.AlignRight)
        self.coordlist = QtWidgets.QListWidget(self.centralwidget)
        self.coordlist.setMinimumSize(QtCore.QSize(300, 0))
        self.coordlist.setObjectName("coordlist")
        self.gridLayout_2.addWidget(self.coordlist, 1, 12, 10, 1)
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setObjectName("label_21")
        self.gridLayout_2.addWidget(self.label_21, 0, 12, 1, 1)
        self.gammacoord = QtWidgets.QLineEdit(self.centralwidget)
        self.gammacoord.setObjectName("gammacoord")
        self.gridLayout_2.addWidget(self.gammacoord, 6, 10, 1, 1)
        self.text_dabs = QtWidgets.QLabel(self.centralwidget)
        self.text_dabs.setObjectName("text_dabs")
        self.gridLayout_2.addWidget(self.text_dabs, 6, 4, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setObjectName("label_18")
        self.gridLayout_2.addWidget(self.label_18, 6, 7, 1, 1)
        self.text_babs = QtWidgets.QLabel(self.centralwidget)
        self.text_babs.setObjectName("text_babs")
        self.gridLayout_2.addWidget(self.text_babs, 4, 4, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 5, 11, 1, 1)
        self.visual = interface(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.visual.sizePolicy().hasHeightForWidth())
        self.visual.setSizePolicy(sizePolicy)
        self.visual.setMinimumSize(QtCore.QSize(450, 450))
        self.visual.setAutoFillBackground(False)
        self.visual.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.visual.setObjectName("visual")
        self.gridLayout_2.addWidget(self.visual, 1, 0, 1, 7)
        self.text_fabs = QtWidgets.QLabel(self.centralwidget)
        self.text_fabs.setObjectName("text_fabs")
        self.gridLayout_2.addWidget(self.text_fabs, 8, 4, 1, 1)
        self.camera_left = QtWidgets.QLabel(self.centralwidget)
        self.camera_left.setMinimumSize(QtCore.QSize(500, 473))
        self.camera_left.setStyleSheet("background-color: rgb(0,0,0);")
        self.camera_left.setObjectName("camera_left")
        self.gridLayout_2.addWidget(self.camera_left, 1, 13, 2, 1)
        self.camera_right = QtWidgets.QLabel(self.centralwidget)
        self.camera_right.setMinimumSize(QtCore.QSize(500, 373))
        self.camera_right.setStyleSheet("background-color:rgb(0,0,0);")
        self.camera_right.setObjectName("camera_right")
        self.gridLayout_2.addWidget(self.camera_right, 4, 13, 7, 1)
        self.coord_query = QtWidgets.QPushButton(self.centralwidget)
        self.coord_query.setObjectName("coord_query")
        self.gridLayout_2.addWidget(self.coord_query, 7, 6, 1, 1)
        self.submit = QtWidgets.QPushButton(self.centralwidget)
        self.submit.setObjectName("submit")
        self.gridLayout_2.addWidget(self.submit, 7, 8, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_15.setText(_translate("MainWindow", "X"))
        self.label_16.setText(_translate("MainWindow", "Y"))
        self.label_19.setText(_translate("MainWindow", "β"))
        self.label_14.setText(_translate("MainWindow", "List of Motor Positions"))
        self.text_eabs.setText(_translate("MainWindow", "TextLabel"))
        self.label_20.setText(_translate("MainWindow", "γ"))
        self.homing.setText(_translate("MainWindow", "Home"))
        self.label_6.setText(_translate("MainWindow", "F"))
        self.compiling.setText(_translate("MainWindow", "Save"))
        self.text_aabs.setText(_translate("MainWindow", "TextLabel"))
        self.remove_motor.setText(_translate("MainWindow", "Remove"))
        self.text_cabs.setText(_translate("MainWindow", "TextLabel"))
        self.executing.setText(_translate("MainWindow", "Run"))
        self.recording.setText(_translate("MainWindow", "Record"))
        self.animating.setText(_translate("MainWindow", "Animate"))
        self.label.setText(_translate("MainWindow", "A"))
        self.label_13.setText(_translate("MainWindow", "Motor Positions"))
        self.information_window.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">By pressing ENTER the current slider positions will be sent to the motors.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Whenever the &quot;Record&quot; button has been pressed the current slider positions will be saved.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The &quot;Save&quot; button will make a json file with all the &quot;Recorded&quot; motor positions.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The &quot;Run&quot; button will initiate the recorded list file that you create when pressing &quot;Save&quot;.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The &quot;Remove last&quot; button removes the last motor positon input. </p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The recording options beneath are in xyz recorded forms. Submit adds it to the list and Query initiates the list. By Pressing ESC you remove the last value from the list.</p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "E"))
        self.label_4.setText(_translate("MainWindow", "D"))
        self.label_3.setText(_translate("MainWindow", "C"))
        self.label_2.setText(_translate("MainWindow", "B"))
        self.label_21.setText(_translate("MainWindow", "List of coordinates"))
        self.text_dabs.setText(_translate("MainWindow", "TextLabel"))
        self.label_18.setText(_translate("MainWindow", "α"))
        self.text_babs.setText(_translate("MainWindow", "TextLabel"))
        self.label_17.setText(_translate("MainWindow", "Z"))
        self.text_fabs.setText(_translate("MainWindow", "TextLabel"))
        self.camera_left.setText(_translate("MainWindow", "TextLabel"))
        self.camera_right.setText(_translate("MainWindow", "TextLabel"))
        self.coord_query.setText(_translate("MainWindow", "Query"))
        self.submit.setText(_translate("MainWindow", "Submit"))
from interface import interface
