import os
import sys
import cv2
import torch
from VGG19 import VGG
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
import torchvision.transforms as trans
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import Ui_HW1


class MyWindow (QMainWindow) :
    def __init__(self):
        self.dir = ""
        self.Filelist = []
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        super(MyWindow, self).__init__()
        self.ui = Ui_HW1.Ui_MainWindow()  # 創建UI對象
        self.ui.setupUi(self)
## ====================== Load ====================== ##
        self.ui.Loadfolder.clicked.connect(self.Loadfolder)
        self.ui.Loadimage_L.clicked.connect(self.Loadimage_L)
        self.ui.Loadimage_R.clicked.connect(self.Loadimage_R)
## ======================  Q1  ====================== ##
        self.ui.Findcorners.clicked.connect(self.Findcorners)
        self.ui.Findintrinsic.clicked.connect(self.Findintrinsic)
        self.ui.Findextrinsic.clicked.connect(self.Findextrinsic)
        self.ui.Finddistortion.clicked.connect(self.Finddistortion)
        self.ui.Showresult.clicked.connect(self.Showresult)
## ======================  Q2  ====================== ##
        self.ui.Showwordonbroad.clicked.connect(self.Showwordonbroad)
        self.ui.Showwordvertical.clicked.connect(self.Showwordvertical)
## ======================  Q3  ====================== ##
        self.ui.stereodisparity.clicked.connect(self.stereodisparity)
## ======================  Q4  ====================== ##
        self.ui.LoadImage1.clicked.connect(self.LoadImage1)
        self.ui.LoadImage2.clicked.connect(self.LoadImage2)
        self.ui.Keypoints.clicked.connect(self.Keypoints)
        self.ui.MatchedKeypoints.clicked.connect(self.MatchedKeypoints)
## ======================  Q5  ====================== ##
        self.ui.AugmentedImages.clicked.connect(self.AugmentedImages)
        self.ui.ModelStructure.clicked.connect(self.ModelStructure)
        self.ui.LoadImage.clicked.connect(self.LoadImage)
        self.ui.ShowAcc.clicked.connect(self.ShowAcc)
        self.ui.Inference.clicked.connect(self.Inference)
## ====================== Load ====================== ##
    def Loadfolder(self): 
        self.dir = ""
        self.Filelist = []

        dir = QFileDialog.getExistingDirectory(None, 'Choose Folder', os.getcwd())
        if dir :
            print(dir)
            self.dir = dir
            print("-----------------------------")
            for files in os.listdir(dir):
                if os.path.isfile(os.path.join(dir, files)) :
                    file_path = os.path.join(dir,files)
                    self.Filelist.append(file_path)
                    print(file_path)
    def Loadimage_L(self):
        imageL = QFileDialog.getOpenFileName(None,'Load ImageL',os.getcwd())
        if imageL : 
            print(imageL[0])
            image = cv2.imread(imageL[0])
            image = cv2.resize(image, (704,429))
            self.imageL_DIR = imageL[0]
            self.imageL_RGB = image
            self.imageL_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("ImageL", self.imageL_RGB)
            # cv2.setMouseCallback("ImageL", self.mouse)
            cv2.setMouseCallBack("ImageL", self.mouse)
            cv2.waitKey(0)
    def Loadimage_R(self):
        imageR = QFileDialog.getOpenFileName(None,'Load ImageR',os.getcwd())
        if imageR : 
            print(imageR[0])
            image = cv2.imread(imageR[0])
            image = cv2.resize(image, (704,429))
            self.imageR_DIR = imageR[0]
            self.imageR_RGB = image
            self.imageR_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("ImageR", self.imageR_RGB)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
## ======================  Q1  ====================== ##
    def Findcorners(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for i in self.Filelist :
            image = cv2.imread(i)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray_image, (11,8), None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(image, (11,8), corners2, ret)
                image = cv2.resize(image, (720,720))
                cv2.imshow('12*9 ChessBoard', image)
                cv2.waitKey(1000)
            cv2.destroyAllWindows()        
    def Findintrinsic(self):
        oplist = []
        iplist = []
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        imageshape = ()
        for i in self.Filelist :
            image = cv2.imread(i)
            imageshape = image.shape[::-2]
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image, (11,8), None)
            if ret :
                oplist.append(objp)
                iplist.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(oplist, iplist, imageshape, None, None)
        print("Instrinsic:")
        print(mtx)
    def Findextrinsic(self):
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        index = self.ui.spinBox.value()
        print(self.Filelist[int(index)])
        image = cv2.imread(self.Filelist[int(index)])
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_image, (11,8), None)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray_image.shape[::-1], None, None)
        rvecs = cv2.Rodrigues(rvecs[0])
        print(rvecs[0])
    def Finddistortion(self):
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        oplist = []
        iplist = []
        imageshape = ()
        for i in self.Filelist:
            image = cv2.imread(i)
            imageshape = image.shape[::-2]
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image, (11,8), None)
            if ret :
                oplist.append(objp)
                iplist.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(oplist, iplist, imageshape, None, None)
        print("Distrotion:")
        print(dist)
    def Showresult(self):
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        imageshape = ()
        objpoints = []
        imagepoints = []
        for i in self.Filelist:
            image = cv2.imread(i)
            imageshape = image.shape[::-2]
            ret, corners = cv2.findChessboardCorners(image, (11,8))
            if ret:
                objpoints.append(objp)
                imagepoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imagepoints, imageshape, None, None)

        for i in self.Filelist:
            image = cv2.imread(i)
            h, w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
            dst = np.append(dst, image, axis = 1)
            dst = cv2.resize(dst, (1024, 512))
            cv2.imshow('Result',dst)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()
## ======================  Q2  ====================== ##
    def ShiftLine(self, line, shift):
        for i in range(3):
            line[0][i] += shift[i]
            line[1][i] += shift[i]
        return line
    def Showwordonbroad(self):
        dir = (self.dir) + "/Q2_lib/alphabet_lib_onboard.txt"
        textfile = cv2.FileStorage(dir, cv2.FileStorage_READ)
        text = (self.ui.textEdit.toPlainText()).upper()
        print(text)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        imageshape = ()
        objlist = []
        imagelist = []
        for i in self.Filelist:
            image = cv2.imread(i)
            imageshape = image.shape[::-2]
            ret, corners = cv2.findChessboardCorners(image, (11,8), None)
            if ret:
                objlist.append(objp)
                imagelist.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objlist, imagelist, imageshape, None, None)
        for i ,file in enumerate(self.Filelist, start=1):
            image = cv2.imread(file)
            for j in range(len(text)):
                temp = textfile.getNode(text[j]).mat()
                for lines in temp:
                    lines = self.ShiftLine(lines, [7-j%3*3, 5-int(j/3)*3, 0])
                    lines = np.float32(lines).reshape(-1,3)
                    image_lines, jac = cv2.projectPoints(lines, rvecs[i-1], tvecs[i-1], mtx, dist)
                    pt1 = tuple(map(int,image_lines[0].ravel()))
                    pt2 = tuple(map(int,image_lines[1].ravel()))
                    image = cv2.line(image, pt1, pt2, (0,0,255), 5)
            image = cv2.resize(image, (1024, 1024))
            cv2.imshow('Result',image)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()
    def Showwordvertical(self):
        dir = (self.dir) + "/Q2_lib/alphabet_lib_vertical.txt"
        textfile = cv2.FileStorage(dir, cv2.FileStorage_READ)
        text = (self.ui.textEdit.toPlainText()).upper()
        print(text)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        imageshape = ()
        objlist = []
        imagelist = []
        for i in self.Filelist:
            image = cv2.imread(i)
            imageshape = image.shape[::-2]
            ret, corners = cv2.findChessboardCorners(image, (11,8), None)
            if ret:
                objlist.append(objp)
                imagelist.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objlist, imagelist, imageshape, None, None)
        for i ,file in enumerate(self.Filelist, start=1):
            image = cv2.imread(file)
            for j in range(len(text)):
                temp = textfile.getNode(text[j]).mat()
                for lines in temp:
                    lines = self.ShiftLine(lines, [7-j%3*3, 5-int(j/3)*3, 0])
                    lines = np.float32(lines).reshape(-1,3)
                    image_lines, jac = cv2.projectPoints(lines, rvecs[i-1], tvecs[i-1], mtx, dist)
                    pt1 = tuple(map(int,image_lines[0].ravel()))
                    pt2 = tuple(map(int,image_lines[1].ravel()))
                    image = cv2.line(image, pt1, pt2, (0,0,255), 5)
            image = cv2.resize(image, (1024, 1024))
            cv2.imshow('Result',image)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()
## ======================  Q3  ====================== ##
    def mouse(self, click, x, y, flags, data):
        if click == cv2.EVENT_LBUTTONDOWN:
            if self.imageR_DIR:
                image = cv2.imread(self.imageR_DIR)
                image = cv2.resize(image, (704,429))
                stereo = cv2.StereoBM_create(256, 15)
                data = stereo.compute(self.imageL_GRAY, self.imageR_GRAY)
                print(str(x) + "----" + str(y))
                if data[y][x] > 0:
                    cv2.circle(image, (x-int(data[y][x]/16),y), 5, (0,0,255), -1)
                    cv2.imshow("ImageR", image)
                else :
                    print("Fail")
    def stereodisparity(self):
        if not (self.imageL_DIR and self.imageR_DIR):
            print("Load image first")
            return
        imageL = cv2.imread(self.imageL_DIR)
        imageR = cv2.imread(self.imageR_DIR)
        imLG = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
        imRG = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(256, 15)
        data = stereo.compute(imLG, imRG)
        data = cv2.resize(data,(704, 429))
        cv2.imshow("Stereo",data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
## ======================  Q4  ====================== ##
    def LoadImage1(self):
        dir = QFileDialog.getOpenFileName(None, 'Load Image 1',os.getcwd())
        print (dir[0])
        image = cv2.imread(dir[0])
        self.image1_DIR = dir
        self.image1_RGB = image
        self.image1_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    def LoadImage2(self):
        dir = QFileDialog.getOpenFileName(None, 'Load Image 2',os.getcwd())
        print (dir[0])
        image = cv2.imread(dir[0])
        self.image2_DIR = dir
        self.image2_RGB = image
        self.image2_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    def Keypoints(self):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(self.image1_RGB, None)
        
        image = cv2.drawKeypoints(self.image1_GRAY, keypoints, None, color=(0,255,0))
        image = cv2.resize(image,(1024,1024))
        cv2.imshow("image1", cv2.resize((self.image1_RGB), (1024,1024)))
        cv2.imshow("result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def MatchedKeypoints(self):
        sift = cv2.SIFT_create()
        kp1, dp1 = sift.detectAndCompute(self.image1_RGB, None)
        kp2, dp2 = sift.detectAndCompute(self.image2_RGB, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(dp1, dp2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance :
                good.append(m)
        image = cv2.drawMatchesKnn(self.image1_GRAY, kp1, self.image2_GRAY, kp2, [good],None,flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        image = cv2.resize(image,(1024,512))
        cv2.imshow("result",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
## ======================  Q5  ====================== ##
    def LoadImage(self):
        dir = QFileDialog.getOpenFileName(None, 'Load Image',os.getcwd())
        image = cv2.imread(dir[0])
        cv2.resize(image, (128,128))
        self.image5_DIR = dir[0]
        self.image5_RGB = image
        print(self.image5_DIR)
        self.ui.label_2.setText("")
        pixmap = QPixmap(self.image5_DIR)
        graphicsView = self.ui.graphicsView
        pixmap = pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        graphicsView.setScene(scene)
    def AugmentedImages(self):
        image_folder = "Q5_image/Q5_1/"
        image_path = []
        image_name = []
        for filename in os.listdir(image_folder):
            print(filename.replace(".png",""))
            image_name.append(filename.replace(".png",""))
            image_path.append(os.path.join(image_folder,filename))

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(3):
            for j in range(3):
                image = Image.open(image_path[i*3+j])
                image = transforms.RandomHorizontalFlip()(image)
                image = transforms.RandomVerticalFlip()(image)
                image = transforms.RandomRotation(30)(image)
                axes[i,j].imshow(image)
                axes[i,j].axis('off')
                axes[i,j].set_title(image_name[i*3+j])
        plt.show()
    def ModelStructure(self):
        model = models.vgg19_bn(num_classes=10)
        summary(model, (3, 32, 32))
    def ShowAcc(self):
        if os.path.exists(".\Q5_VGG19\output.jpg"):
            image = cv2.imread(".\Q5_VGG19\output.jpg")
            fig, ax = plt.subplots(figsize=(17, 10))
            ax.imshow(image, aspect="auto")
            ax.axis('off')  
            plt.tight_layout()
            plt.show()
    def Inference(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load('.\Q5_VGG19\model.pth',map_location='cpu')
        
        model.eval()
        image_path = self.image5_DIR
        mean = [x/255 for x in [125.3, 23.0, 113.9]] 
        std = [x/255 for x in [63.0, 62.1, 66.7]]
        data_transforms_test = trans.Compose([
                                 trans.ToTensor(),
                                 trans.Normalize(mean, std)
                             ])                            
        image = Image.open(image_path)
        image_tensor = data_transforms_test(image).float()
        image_tensor = image_tensor.unsqueeze_(0).to(device)
        probabilities = F.softmax(model(image_tensor)).data.cpu().numpy()
        output = probabilities[0]
        predicted_class = output.argmax().item()
        prediction = self.classes[np.argmax(output)]
        score = max(output)
        print(score, prediction)

        predicted_class_name = self.classes[predicted_class]
        # 绘制概率分布的直方图
        plt.figure(figsize=(6, 4))
        plt.bar(range(10), probabilities[0])
        plt.xticks(range(10), [self.classes[i] for i in range(10)]) 
        plt.xlabel('Class Label')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.title(f'Predicted Class: {predicted_class_name}')
        self.ui.label.setText("Predict = " + predicted_class_name)
        plt.show()
        
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    MainWindow = MyWindow()
    MainWindow.show()
    sys.exit(app.exec_())
