# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'a:\Computer vision\HW1.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1102, 823)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 251, 371))
        self.groupBox.setObjectName("groupBox")
        self.Loadfolder = QtWidgets.QPushButton(self.groupBox)
        self.Loadfolder.setGeometry(QtCore.QRect(10, 40, 231, 81))
        self.Loadfolder.setObjectName("Loadfolder")
        self.Loadimage_L = QtWidgets.QPushButton(self.groupBox)
        self.Loadimage_L.setGeometry(QtCore.QRect(10, 150, 231, 81))
        self.Loadimage_L.setObjectName("Loadimage_L")
        self.Loadimage_R = QtWidgets.QPushButton(self.groupBox)
        self.Loadimage_R.setGeometry(QtCore.QRect(10, 260, 231, 81))
        self.Loadimage_R.setObjectName("Loadimage_R")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(290, 20, 251, 371))
        self.groupBox_2.setObjectName("groupBox_2")
        self.Findcorners = QtWidgets.QPushButton(self.groupBox_2)
        self.Findcorners.setGeometry(QtCore.QRect(10, 40, 231, 31))
        self.Findcorners.setObjectName("Findcorners")
        self.Findintrinsic = QtWidgets.QPushButton(self.groupBox_2)
        self.Findintrinsic.setGeometry(QtCore.QRect(10, 80, 231, 31))
        self.Findintrinsic.setObjectName("Findintrinsic")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 140, 231, 101))
        self.groupBox_3.setObjectName("groupBox_3")
        self.Findextrinsic = QtWidgets.QPushButton(self.groupBox_3)
        self.Findextrinsic.setGeometry(QtCore.QRect(10, 60, 211, 31))
        self.Findextrinsic.setObjectName("Findextrinsic")
        self.spinBox = QtWidgets.QSpinBox(self.groupBox_3)
        self.spinBox.setGeometry(QtCore.QRect(70, 30, 101, 22))
        self.spinBox.setObjectName("spinBox")
        self.Finddistortion = QtWidgets.QPushButton(self.groupBox_2)
        self.Finddistortion.setGeometry(QtCore.QRect(10, 260, 231, 31))
        self.Finddistortion.setObjectName("Finddistortion")
        self.Showresult = QtWidgets.QPushButton(self.groupBox_2)
        self.Showresult.setGeometry(QtCore.QRect(10, 300, 231, 31))
        self.Showresult.setObjectName("Showresult")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(560, 20, 251, 371))
        self.groupBox_4.setObjectName("groupBox_4")
        self.Showwordonbroad = QtWidgets.QPushButton(self.groupBox_4)
        self.Showwordonbroad.setGeometry(QtCore.QRect(10, 260, 231, 31))
        self.Showwordonbroad.setObjectName("Showwordonbroad")
        self.Showwordvertical = QtWidgets.QPushButton(self.groupBox_4)
        self.Showwordvertical.setGeometry(QtCore.QRect(10, 300, 231, 31))
        self.Showwordvertical.setObjectName("Showwordvertical")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_4)
        self.textEdit.setGeometry(QtCore.QRect(10, 30, 231, 221))
        self.textEdit.setObjectName("textEdit")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(830, 20, 251, 371))
        self.groupBox_5.setObjectName("groupBox_5")
        self.stereodisparity = QtWidgets.QPushButton(self.groupBox_5)
        self.stereodisparity.setGeometry(QtCore.QRect(10, 180, 231, 31))
        self.stereodisparity.setObjectName("stereodisparity")
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(290, 410, 251, 371))
        self.groupBox_6.setObjectName("groupBox_6")
        self.Keypoints = QtWidgets.QPushButton(self.groupBox_6)
        self.Keypoints.setGeometry(QtCore.QRect(10, 200, 231, 31))
        self.Keypoints.setObjectName("Keypoints")
        self.LoadImage2 = QtWidgets.QPushButton(self.groupBox_6)
        self.LoadImage2.setGeometry(QtCore.QRect(10, 130, 231, 31))
        self.LoadImage2.setObjectName("LoadImage2")
        self.LoadImage1 = QtWidgets.QPushButton(self.groupBox_6)
        self.LoadImage1.setGeometry(QtCore.QRect(10, 60, 231, 31))
        self.LoadImage1.setObjectName("LoadImage1")
        self.MatchedKeypoints = QtWidgets.QPushButton(self.groupBox_6)
        self.MatchedKeypoints.setGeometry(QtCore.QRect(10, 270, 231, 31))
        self.MatchedKeypoints.setObjectName("MatchedKeypoints")
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(570, 410, 251, 371))
        self.groupBox_7.setObjectName("groupBox_7")
        self.ModelStructure = QtWidgets.QPushButton(self.groupBox_7)
        self.ModelStructure.setGeometry(QtCore.QRect(10, 100, 231, 31))
        self.ModelStructure.setObjectName("ModelStructure")
        self.AugmentedImages = QtWidgets.QPushButton(self.groupBox_7)
        self.AugmentedImages.setGeometry(QtCore.QRect(10, 60, 231, 31))
        self.AugmentedImages.setObjectName("AugmentedImages")
        self.LoadImage = QtWidgets.QPushButton(self.groupBox_7)
        self.LoadImage.setGeometry(QtCore.QRect(10, 20, 231, 31))
        self.LoadImage.setObjectName("LoadImage")
        self.ShowAcc = QtWidgets.QPushButton(self.groupBox_7)
        self.ShowAcc.setGeometry(QtCore.QRect(10, 140, 231, 31))
        self.ShowAcc.setObjectName("ShowAcc")
        self.Inference = QtWidgets.QPushButton(self.groupBox_7)
        self.Inference.setGeometry(QtCore.QRect(10, 180, 231, 31))
        self.Inference.setObjectName("Inference")
        self.label = QtWidgets.QLabel(self.groupBox_7)
        self.label.setGeometry(QtCore.QRect(10, 220, 221, 16))
        self.label.setObjectName("label")
        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox_7)
        self.graphicsView.setGeometry(QtCore.QRect(10, 240, 221, 121))
        self.graphicsView.setObjectName("graphicsView")
        self.label_2 = QtWidgets.QLabel(self.groupBox_7)
        self.label_2.setGeometry(QtCore.QRect(80, 290, 81, 20))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1102, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Load Image"))
        self.Loadfolder.setText(_translate("MainWindow", "Load folder"))
        self.Loadimage_L.setText(_translate("MainWindow", "Load Image_L"))
        self.Loadimage_R.setText(_translate("MainWindow", "Load Image_R"))
        self.groupBox_2.setTitle(_translate("MainWindow", "1. Calibration"))
        self.Findcorners.setText(_translate("MainWindow", "1.1 Find corners"))
        self.Findintrinsic.setText(_translate("MainWindow", "1.2 Find intrinsic"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Calibration"))
        self.Findextrinsic.setText(_translate("MainWindow", "1.3 Find extrinsic"))
        self.Finddistortion.setText(_translate("MainWindow", "1.4 Find distortion"))
        self.Showresult.setText(_translate("MainWindow", "1.5 Show result"))
        self.groupBox_4.setTitle(_translate("MainWindow", "2. Augmented Reality"))
        self.Showwordonbroad.setText(_translate("MainWindow", "2.1 show word on broad"))
        self.Showwordvertical.setText(_translate("MainWindow", "2.1 show word vertical"))
        self.groupBox_5.setTitle(_translate("MainWindow", "3. Stereo disparity map"))
        self.stereodisparity.setText(_translate("MainWindow", "3.1 stereo disparity map"))
        self.groupBox_6.setTitle(_translate("MainWindow", "4. SIFT"))
        self.Keypoints.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.LoadImage2.setText(_translate("MainWindow", "Load Image 2"))
        self.LoadImage1.setText(_translate("MainWindow", "Load Image 1"))
        self.MatchedKeypoints.setText(_translate("MainWindow", "4.2 Matched Keypoints"))
        self.groupBox_7.setTitle(_translate("MainWindow", "5. VGG19"))
        self.ModelStructure.setText(_translate("MainWindow", "5.2 Show Model Structure"))
        self.AugmentedImages.setText(_translate("MainWindow", "5.1 Show Augmented Images"))
        self.LoadImage.setText(_translate("MainWindow", "Load Image"))
        self.ShowAcc.setText(_translate("MainWindow", "5.3 Show Acc and Loss"))
        self.Inference.setText(_translate("MainWindow", "5.4 Inference"))
        self.label.setText(_translate("MainWindow", "Predict = "))
        self.label_2.setText(_translate("MainWindow", "Inference Image"))