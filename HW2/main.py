import os
import sys
import cv2
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
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
import Ui_HW2
from Ui_HW2 import Painter
import random
import matplotlib.image
import torch.nn as nn
import torchsummary

class MyWindow (QMainWindow) :
    def __init__(self):
        self.dir = ""
        self.Filelist = []

        super(MyWindow, self).__init__()
        self.ui = Ui_HW2.Ui_MainWindow()  # 創建UI對象
        self.ui.setupUi(self)
# ====================== Load ====================== ##
        self.ui.LoadImage.clicked.connect(self.LoadImage)
        self.ui.LoadVideo.clicked.connect(self.LoadVideo)
# ======================  Q1  ====================== ##
        self.ui.BackgroundSubstraction.clicked.connect(self.BackgroundSubstraction)
# ======================  Q2  ====================== ##
        self.ui.Preprocessing.clicked.connect(self.Preprocessing)
        self.ui.Vediotracking.clicked.connect(self.Vediotracking)
# ======================  Q3  ====================== ##
        self.ui.DimensionReduction.clicked.connect(self.DimensionReduction)
# ======================  Q4  ====================== ##
        self.ui.ShowModelStructure.clicked.connect(self.ShowModelStructure)
        self.ui.ShowAccuracyandLoss.clicked.connect(self.ShowAccuracyandLoss)
        self.ui.Predict.clicked.connect(self.Predict)
        self.ui.Reset.clicked.connect(self.Reset)
# ## ======================  Q5  ====================== ##
        self.ui.ResnetLoadImage.clicked.connect(self.ResnetLoadImage)
        self.ui.ShowImages.clicked.connect(self.ShowImages)
        self.ui.ShowComprasion.clicked.connect(self.ShowComprasion)
        self.ui.ShowModelStructureQ5.clicked.connect(self.ShowModelStructureQ5)
        self.ui.Inference.clicked.connect(self.Inference)

## ====================== Load ====================== ##
    def LoadImage(self):
        Path = QFileDialog.getOpenFileName(None,'Load Path',os.getcwd())
        if Path : 
            print(Path[0])
            image = cv2.imread(Path[0])
            self.image_DIR = Path[0]
            self.image_RGB = image
            self.image_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("image", self.image_RGB)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    def LoadVideo(self):
        Video = QFileDialog.getOpenFileName(None,'Load Video',os.getcwd())
        if Video : 
            print(Video[0])
            cap = cv2.VideoCapture(Video[0])
            self.video_DIR = Video[0]
## ====================== Sub ====================== ##
    def BackgroundSubstraction(self):
        history = 500  
        # dist2Threshold = 256  
        detectShadows = True 

        # subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist2Threshold, detectShadows=detectShadows)
        subtractor = cv2.createBackgroundSubtractorKNN(history=history, detectShadows=detectShadows)


        if not self.video_DIR:
            print ("No video")
        else:
            cap = cv2.VideoCapture(self.video_DIR)
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

                # 3.2)
                mask = subtractor.apply(blurred_frame)

                # 3.3)
                result_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)

                # 顯示原始影片和處理後的影片
                combined_frame = cv2.hconcat([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result_frame])

                # 顯示原始影片、高斯模糊後的影片和僅包含移動物體的影片
                cv2.imshow('Combined Frames', combined_frame)

                # 按下 'q' 鍵退出迴圈
                if cv2.waitKey(1) == ord('q'):
                    break
            # 釋放資源
            cap.release()
            cv2.destroyAllWindows()
## ====================== Optical ====================== ##
    def Preprocessing(self):
        max_corners = 1
        quality_level = 0.3
        min_distance = 7
        block_size = 7
        if not self.video_DIR:
            print ("No video")
        else:
            cap = cv2.VideoCapture(self.video_DIR)
            ret, frame = cap.read()
            if not ret:
                exit()
            gray_first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray_first_frame, max_corners, quality_level, min_distance, blockSize=block_size)
            corners = np.int0(corners)
            x, y = corners.ravel()
            marker_size = 20
            cv2.line(frame, (x - marker_size, y), (x + marker_size, y), (0, 0, 255), 2)
            cv2.line(frame, (x, y - marker_size), (x, y + marker_size), (0, 0, 255), 2)

            # 顯示帶有紅色十字標記的影像
            resized_frame = cv2.resize(frame, (1080, 720))
            cv2.imshow('', resized_frame)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cap.release()
    def Vediotracking(self):
        max_corners = 1
        quality_level = 0.3
        min_distance = 7
        block_size = 7
        if not self.video_DIR:
            print ("No video")
        else:
            cap = cv2.VideoCapture(self.video_DIR)
            ret, prev_frame = cap.read()
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_corners = cv2.goodFeaturesToTrack(prev_gray, max_corners, quality_level, min_distance, blockSize=block_size)
            prev_corners = np.float32(prev_corners)
            trajectory_frame = np.zeros_like(prev_frame)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 使用calcOpticalFlowPyrLK追蹤角點
                next_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_corners, None)

                # 選取追蹤成功的角點
                good_new = next_corners[status == 1]
                good_old = prev_corners[status == 1]

                # 繪製軌跡
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    color = (0, 0, 255)  # 使用黃色 (BGR格式)
                    frame = cv2.line(frame, (int(c), int(d)), (int(a), int(b)), color, 2)
                    frame = cv2.line(frame, (int(a) - 20, int(b)), (int(a) + 20, int(b)), color, 2)
                    frame = cv2.line(frame, (int(a), int(b) - 20), (int(a), int(b) + 20), color, 2)

                    trajectory_frame = cv2.line(trajectory_frame, (int(c), int(d)), (int(a), int(b)), color, 2)

                combine_frame =cv2.add(frame,trajectory_frame)
                cv2.imshow('Original Video with Trajectory', combine_frame)

                if cv2.waitKey(1) == ord('q'):
                    break

                # 更新上一幀的角點和灰度影格
                prev_corners = good_new.reshape(-1, 1, 2)
                prev_gray = gray

            # 釋放資源
            cap.release()
            cv2.destroyAllWindows()
## ====================== DimensionReduction ====================== ##
    def DimensionReduction(self):
        if not self.image_DIR:
            print("no image")
        else:
            normalizer = MinMaxScaler()
            normalized_gray = normalizer.fit_transform(self.image_GRAY)
            maxcomponents = min((self.image_GRAY).shape)
            
            for n_components in range(1,maxcomponents+1):

                pca = PCA(n_components = n_components)
                reduced_image = pca.fit_transform(normalized_gray)
                reconstructed_image = normalizer.inverse_transform(pca.inverse_transform(reduced_image))

                
                mse = mean_squared_error(self.image_GRAY, reconstructed_image)
                if mse <= 3.0:
                    break

            print(f"MSE <= 3.0 min:{n_components}")

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(self.image_GRAY, cmap='gray')
            plt.title('Gray image')

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title(f'Reconstructed image (n={n_components},MSE={mse:.4f})')

            plt.show()
## ====================== MNIST ====================== ##
    def ShowModelStructure(self):
        model = models.vgg19_bn(num_classes=10)
        summary(model, (3, 32, 32))
    def ShowAccuracyandLoss(self):
        # if os.path.exists(".vgg19_bn_training_results.png"):
        image = cv2.imread("./vgg19_bn_training_results.png")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(image, aspect="auto")
        ax.axis('off')  
        plt.tight_layout()
        plt.show()
    def Predict(self):
        image = self.ui.painter.get_canvas_image()
        model = models.vgg19_bn(num_classes=10)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        check = torch.load('./Q4_model/my_model_weights_vgg19_Q4.pth', map_location=torch.device('cpu'))
        model.load_state_dict(check)
        
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        image = transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
        predicted_class = torch.argmax(output).item()
        self.ui.label.setText(f"Predicted Class: {predicted_class}")
        probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        print(output)
        self.show_probability_distribution(probabilities)
    def Reset(self):
        self.ui.label.setText(f"Predicted Class:")
        self.ui.painter.reset_drawing()
    def show_probability_distribution(self, probabilities):
        plt.figure(figsize=(12, 8))
        plt.bar(range(10), probabilities)
        plt.xticks(range(10), [str(i) for i in range(10)]) 
        plt.title('Probability Distribution')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.show()
## ====================== MNIST ====================== ##
    def ResnetLoadImage(self):
        dir = QFileDialog.getOpenFileName(None, 'Load Image',os.getcwd())
        image = cv2.imread(dir[0])
        cv2.resize(image, (128,128))
        self.image5_DIR = dir[0]
        self.image5_RGB = image
        print(self.image5_DIR)
        self.ui.label_2.setText("")
        pixmap = QPixmap(self.image5_DIR)
        graphicsView = self.ui.graphicsView_2
        pixmap = pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        graphicsView.setScene(scene)
    def ShowImages(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        cat = "./dataset/inference_dataset/Cat"
        dog = "./dataset/inference_dataset/Dog"
        acat = os.path.join(cat, random.choice(os.listdir(cat)))
        adog = os.path.join(dog, random.choice(os.listdir(dog)))

        plt.figure(figsize=(10,8))

        plt.subplot(1, 2, 1)
        acat = Image.open(acat)
        acat = transform(acat)
        acat = np.transpose(acat.numpy(), (1, 2, 0))
        plt.imshow(acat)
        plt.title('Cat')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        adog = Image.open(adog)
        adog = transform(adog)
        adog = np.transpose(adog.numpy(), (1, 2, 0))
        plt.imshow(adog)
        plt.title('Dog')
        plt.axis('off')
        plt.show()
    def ShowModelStructureQ5(self):
        model = models.resnet50()
        fil = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fil, 1),
            nn.Sigmoid()
        )
        torchsummary.summary(model,(3,224,224))
    def ShowComprasion(self):
        Image = matplotlib.image.imread("compare.png")
        plt.figure(figsize=(10, 8))
        plt.imshow(Image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    def Inference(self):
        model = torch.load('./Q5_model_erase/ResNet50_catdog_ereasing_weights.pth', map_location=torch.device('cpu'))
        model.eval()
        transform_Q5 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(self.image5_DIR)
        image = transform_Q5(image)
        image = image.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image)
            predicted = (outputs >= 0.5).int()  
            print(predicted)
        
        classes = ['Cat', 'Dog']
        predicted_class = classes[predicted.item()]
        self.ui.label_2.setText(f'Predicted Class: {predicted_class}')
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    MainWindow = MyWindow()
    MainWindow.show()
    sys.exit(app.exec_())