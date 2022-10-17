# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'App_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(3823, 1729)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.visual = interface(self.centralwidget)
        self.visual.setGeometry(QtCore.QRect(9, 10, 1591, 1091))
        self.visual.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.visual.setObjectName("visual")
        self.Homing = QtWidgets.QPushButton(self.centralwidget)
        self.Homing.setGeometry(QtCore.QRect(20, 1490, 200, 50))
        self.Homing.setObjectName("Homing")
        self.coordlist = QtWidgets.QListWidget(self.centralwidget)
        self.coordlist.setGeometry(QtCore.QRect(2120, 70, 501, 1651))
        self.coordlist.setObjectName("coordlist")
        self.xcoord = QtWidgets.QLineEdit(self.centralwidget)
        self.xcoord.setGeometry(QtCore.QRect(1010, 1160, 81, 20))
        self.xcoord.setText("")
        self.xcoord.setObjectName("xcoord")
        self.Submit = QtWidgets.QPushButton(self.centralwidget)
        self.Submit.setGeometry(QtCore.QRect(1130, 1240, 91, 30))
        self.Submit.setObjectName("Submit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1103, 1160, 16, 23))
        self.label.setObjectName("label")
        self.ycoord = QtWidgets.QLineEdit(self.centralwidget)
        self.ycoord.setGeometry(QtCore.QRect(1130, 1160, 81, 20))
        self.ycoord.setText("")
        self.ycoord.setObjectName("ycoord")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1220, 1160, 16, 23))
        self.label_2.setObjectName("label_2")
        self.zcoord = QtWidgets.QLineEdit(self.centralwidget)
        self.zcoord.setGeometry(QtCore.QRect(1240, 1160, 81, 20))
        self.zcoord.setText("")
        self.zcoord.setObjectName("zcoord")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1330, 1160, 16, 23))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(2260, 20, 221, 23))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 1200, 16, 31))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(30, 1250, 16, 31))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(30, 1310, 16, 31))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(30, 1360, 21, 31))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(30, 1410, 16, 31))
        self.label_9.setObjectName("label_9")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(210, 1150, 251, 31))
        self.label_11.setObjectName("label_11")
        self.camera_left = QtWidgets.QLabel(self.centralwidget)
        self.camera_left.setGeometry(QtCore.QRect(2630, 0, 1181, 921))
        self.camera_left.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.camera_left.setText("")
        self.camera_left.setObjectName("camera_left")
        self.camera_right = QtWidgets.QLabel(self.centralwidget)
        self.camera_right.setGeometry(QtCore.QRect(2630, 930, 1181, 791))
        self.camera_right.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.camera_right.setText("")
        self.camera_right.setObjectName("camera_right")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(1330, 1200, 16, 23))
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(1220, 1200, 16, 23))
        self.label_12.setObjectName("label_12")
        self.gammacoord = QtWidgets.QLineEdit(self.centralwidget)
        self.gammacoord.setGeometry(QtCore.QRect(1240, 1200, 81, 20))
        self.gammacoord.setText("")
        self.gammacoord.setObjectName("gammacoord")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(1100, 1200, 16, 23))
        self.label_13.setObjectName("label_13")
        self.betacoord = QtWidgets.QLineEdit(self.centralwidget)
        self.betacoord.setGeometry(QtCore.QRect(1130, 1200, 81, 20))
        self.betacoord.setText("")
        self.betacoord.setObjectName("betacoord")
        self.alfacoord = QtWidgets.QLineEdit(self.centralwidget)
        self.alfacoord.setGeometry(QtCore.QRect(1010, 1200, 81, 20))
        self.alfacoord.setText("")
        self.alfacoord.setObjectName("alfacoord")
        self.aabs = QtWidgets.QSlider(self.centralwidget)
        self.aabs.setGeometry(QtCore.QRect(70, 1190, 500, 50))
        self.aabs.setMinimum(-100)
        self.aabs.setMaximum(200)
        self.aabs.setOrientation(QtCore.Qt.Horizontal)
        self.aabs.setObjectName("aabs")
        self.babs = QtWidgets.QSlider(self.centralwidget)
        self.babs.setGeometry(QtCore.QRect(60, 1250, 500, 50))
        self.babs.setMinimum(20)
        self.babs.setMaximum(200)
        self.babs.setProperty("value", 100)
        self.babs.setOrientation(QtCore.Qt.Horizontal)
        self.babs.setObjectName("babs")
        self.cabs = QtWidgets.QSlider(self.centralwidget)
        self.cabs.setGeometry(QtCore.QRect(60, 1410, 500, 50))
        self.cabs.setMaximum(270)
        self.cabs.setProperty("value", 270)
        self.cabs.setOrientation(QtCore.Qt.Horizontal)
        self.cabs.setObjectName("cabs")
        self.dabs = QtWidgets.QSlider(self.centralwidget)
        self.dabs.setGeometry(QtCore.QRect(70, 1350, 500, 50))
        self.dabs.setMaximum(180)
        self.dabs.setProperty("value", 80)
        self.dabs.setOrientation(QtCore.Qt.Horizontal)
        self.dabs.setObjectName("dabs")
        self.eabs = QtWidgets.QSlider(self.centralwidget)
        self.eabs.setGeometry(QtCore.QRect(60, 1300, 500, 50))
        self.eabs.setMaximum(180)
        self.eabs.setProperty("value", 40)
        self.eabs.setOrientation(QtCore.Qt.Horizontal)
        self.eabs.setObjectName("eabs")
        self.coord_query = QtWidgets.QPushButton(self.centralwidget)
        self.coord_query.setGeometry(QtCore.QRect(1010, 1240, 93, 30))
        self.coord_query.setObjectName("coord_query")
        self.text_aabs = QtWidgets.QLabel(self.centralwidget)
        self.text_aabs.setGeometry(QtCore.QRect(590, 1200, 100, 30))
        self.text_aabs.setObjectName("text_aabs")
        self.text_babs = QtWidgets.QLabel(self.centralwidget)
        self.text_babs.setGeometry(QtCore.QRect(590, 1250, 100, 30))
        self.text_babs.setObjectName("text_babs")
        self.text_dabs = QtWidgets.QLabel(self.centralwidget)
        self.text_dabs.setGeometry(QtCore.QRect(590, 1360, 100, 30))
        self.text_dabs.setObjectName("text_dabs")
        self.text_cabs = QtWidgets.QLabel(self.centralwidget)
        self.text_cabs.setGeometry(QtCore.QRect(590, 1320, 100, 30))
        self.text_cabs.setObjectName("text_cabs")
        self.text_eabs = QtWidgets.QLabel(self.centralwidget)
        self.text_eabs.setGeometry(QtCore.QRect(590, 1410, 100, 30))
        self.text_eabs.setObjectName("text_eabs")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(1720, 20, 341, 23))
        self.label_14.setObjectName("label_14")
        self.motorlist = QtWidgets.QListWidget(self.centralwidget)
        self.motorlist.setGeometry(QtCore.QRect(1620, 70, 491, 1651))
        self.motorlist.setObjectName("motorlist")
        self.recording = QtWidgets.QPushButton(self.centralwidget)
        self.recording.setGeometry(QtCore.QRect(220, 1490, 200, 50))
        self.recording.setStyleSheet("")
        self.recording.setObjectName("recording")
        self.compiling = QtWidgets.QPushButton(self.centralwidget)
        self.compiling.setGeometry(QtCore.QRect(430, 1490, 200, 50))
        self.compiling.setObjectName("compiling")
        self.executing = QtWidgets.QPushButton(self.centralwidget)
        self.executing.setGeometry(QtCore.QRect(640, 1490, 200, 50))
        self.executing.setObjectName("executing")
        self.information_window = QtWidgets.QTextBrowser(self.centralwidget)
        self.information_window.setGeometry(QtCore.QRect(20, 1550, 1581, 171))
        self.information_window.setObjectName("information_window")
        self.remove_motor = QtWidgets.QPushButton(self.centralwidget)
        self.remove_motor.setGeometry(QtCore.QRect(850, 1490, 271, 50))
        self.remove_motor.setObjectName("remove_motor")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Homing.setText(_translate("MainWindow", "Home"))
        self.Submit.setText(_translate("MainWindow", "Submit"))
        self.label.setText(_translate("MainWindow", "X"))
        self.label_2.setText(_translate("MainWindow", "Y"))
        self.label_3.setText(_translate("MainWindow", "Z"))
        self.label_4.setText(_translate("MainWindow", "List of coordinates"))
        self.label_5.setText(_translate("MainWindow", "A"))
        self.label_6.setText(_translate("MainWindow", "B"))
        self.label_7.setText(_translate("MainWindow", "C"))
        self.label_8.setText(_translate("MainWindow", "D"))
        self.label_9.setText(_translate("MainWindow", "E"))
        self.label_11.setText(_translate("MainWindow", "Motor Positions"))
        self.label_10.setText(_translate("MainWindow", "γ"))
        self.label_12.setText(_translate("MainWindow", "β"))
        self.label_13.setText(_translate("MainWindow", "α"))
        self.coord_query.setText(_translate("MainWindow", "Query"))
        self.text_aabs.setText(_translate("MainWindow", "TextLabel"))
        self.text_babs.setText(_translate("MainWindow", "TextLabel"))
        self.text_dabs.setText(_translate("MainWindow", "TextLabel"))
        self.text_cabs.setText(_translate("MainWindow", "TextLabel"))
        self.text_eabs.setText(_translate("MainWindow", "TextLabel"))
        self.label_14.setText(_translate("MainWindow", "List of Motor Positions"))
        self.recording.setText(_translate("MainWindow", "Record"))
        self.compiling.setText(_translate("MainWindow", "Save"))
        self.executing.setText(_translate("MainWindow", "Run"))
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
        self.remove_motor.setText(_translate("MainWindow", "Remove Last"))
from interface import interface
