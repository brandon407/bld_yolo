import os

import torch

from ultralytics import YOLO
import cv2
import numpy as np

from PySide6.QtGui import QIcon
from PySide6 import QtWidgets, QtCore, QtGui


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_gui()
        self.model = None
        self.timer = QtCore.QTimer()
        self.timer1 = QtCore.QTimer()
        self.cap = None
        self.video = None
        self.timer.timeout.connect(self.camera_show)
        self.timer1.timeout.connect(self.video_show)

    def init_gui(self):
        self.setFixedSize(960, 440)
        self.setWindowTitle('Bilibili：秋芒时不知')
        self.setWindowIcon(QIcon("🅱️"))
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        topLayout = QtWidgets.QHBoxLayout()
        self.oriVideoLabel = QtWidgets.QLabel(self)
        self.detectlabel = QtWidgets.QLabel(self)
        self.oriVideoLabel.setMinimumSize(448, 336)
        self.detectlabel.setMinimumSize(448, 336)
        self.oriVideoLabel.setStyleSheet('border:1px solid #D7E2F9;')
        self.detectlabel.setStyleSheet('border:1px solid #D7E2F9;')
        # 960 540  1920 960

        topLayout.addWidget(self.oriVideoLabel)
        topLayout.addWidget(self.detectlabel)

        mainLayout.addLayout(topLayout)

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)

        bottomLayout = QtWidgets.QVBoxLayout(groupBox)

        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QHBoxLayout()
        self.selectModel = QtWidgets.QPushButton('📂选择模型')
        self.selectModel.setFixedSize(100, 50)
        self.selectModel.clicked.connect(self.load_model)
        self.openPictureBtn = QtWidgets.QPushButton('🧩图片文件')
        self.openPictureBtn.setFixedSize(100, 50)
        self.openPictureBtn.clicked.connect(self.picture_show)
        self.openVideoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.openVideoBtn.setFixedSize(100, 50)
        self.openVideoBtn.clicked.connect(self.start_video)
        self.openVideoBtn.setEnabled(False)
        self.openCamBtn = QtWidgets.QPushButton('📹摄像头')
        self.openCamBtn.setFixedSize(100, 50)
        self.openCamBtn.clicked.connect(self.start_camera)
        self.stopDetectBtn = QtWidgets.QPushButton('🛑停止')
        self.stopDetectBtn.setFixedSize(100, 50)
        self.stopDetectBtn.setEnabled(False)
        self.stopDetectBtn.clicked.connect(self.stop_detect)
        self.exitBtn = QtWidgets.QPushButton('⏹退出')
        self.exitBtn.setFixedSize(100, 50)
        self.exitBtn.clicked.connect(self.close)
        btnLayout.addWidget(self.selectModel)
        btnLayout.addWidget(self.openPictureBtn)
        btnLayout.addWidget(self.openVideoBtn)
        btnLayout.addWidget(self.openCamBtn)
        btnLayout.addWidget(self.stopDetectBtn)
        btnLayout.addWidget(self.exitBtn)
        bottomLayout.addLayout(btnLayout)

    def start_camera(self):
        self.timer1.stop()
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            # exit()
            self.timer.start(50)
            pass
        self.stopDetectBtn.setEnabled(True)

    def picture_show(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取图片文件")
        if os.path.isfile(fileName):
            file_name, file_extension = os.path.splitext(fileName)
            if file_extension in {".jpg", ".png", ".bmp", ".mpo", ".tif", ".webp", ".jpeg", ".pfm", ".dng", ".tiff"}:
                frame = cv2.imread(fileName)
                frame = cv2.resize(frame, (448, 352))
                if self.model is not None:
                    frame1 = self.model(frame, imgsz=[448, 352], device='cuda') if torch.cuda.is_available() \
                        else self.model(frame, imgsz=[448, 352], device='cpu')
                    frame1 = cv2.cvtColor(frame1[0].plot(), cv2.COLOR_RGB2BGR)
                    frame1 = QtGui.QImage(frame1.data, frame1.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                    self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame1))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
                self.oriVideoLabel.setScaledContents(True)
        else:
            print("Reselect video")

    def camera_show(self):
        ret, frame = self.cap.read()
        if ret:
            if self.model is not None:
                frame = cv2.resize(frame, (448, 352))
                frame1 = self.model(frame, imgsz=[448, 352], device='cuda') if torch.cuda.is_available() \
                    else self.model(frame, imgsz=[448, 352], device='cpu')
                frame1 = cv2.cvtColor(frame1[0].plot(), cv2.COLOR_RGB2BGR)
                frame1 = QtGui.QImage(frame1.data, frame1.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame1))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
            self.oriVideoLabel.setScaledContents(True)
        else:
            pass

    def start_video(self):
        if self.timer.isActive():
            self.timer.stop()
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取视频文件", filter='*.mp4')
        if os.path.isfile(fileName):
            file_name, file_extension = os.path.splitext(fileName)
            if file_extension in {".avi", ".mp4", ".mpeg", ".gif", ".mpg", ".mkv", ".ts", ".webm", ".asf", ".wmv"}:
                # capture = cv2.VideoCapture(fileName)
                # frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video = cv2.VideoCapture(fileName)
                fps = self.video.get(cv2.CAP_PROP_FPS)
                self.timer1.start(int(1/fps))
            else:
                print("Reselect video")
        else:
            print("Reselect video")

    def video_show(self):
        ret, frame = self.video.read()
        if ret:
            if self.model is not None:
                frame = cv2.resize(frame, (448, 352))
                frame1 = self.model(frame, imgsz=[448, 352], device='cuda') if torch.cuda.is_available() \
                    else self.model(frame, imgsz=[448, 352], device='cpu')
                frame1 = cv2.cvtColor(frame1[0].plot(), cv2.COLOR_RGB2BGR)
                frame1 = QtGui.QImage(frame1.data, frame1.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame1))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
            self.oriVideoLabel.setScaledContents(True)
        else:
            self.timer1.stop()
            img = cv2.cvtColor(np.zeros((500, 500), np.uint8), cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(img))
            self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(img))
            self.video.release()
            self.video = None

    def load_model(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取模型权重", filter='*.pt')
        if fileName.endswith('.pt'):
            self.model = YOLO(fileName)
        else:
            print("Reselect model")

        self.openVideoBtn.setEnabled(True)
        self.stopDetectBtn.setEnabled(True)

    def stop_detect(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.timer1.isActive():
            self.timer1.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video = None
        img = cv2.cvtColor(np.zeros((500, 500), np.uint8), cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(img))
        self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(img))

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.timer.isActive():
            self.timer.stop()
        exit()


if __name__ == '__main__':
    app = QtWidgets.QApplication()
    window = MyWindow()
    window.show()
    app.exec()

#
#
# import numpy as np
# from cv2 import cv2
#
# cap = cv2.VideoCapture(0)  # 设备号为0
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
# while (True):
#     if cap.isOpened() == False:
#         print('can not open camera')
#         break
#     ret, frame = cap.read()  # 读取图像
#     if ret == False:  # 图像读取失败则直接进入下一次循环
#         continue
#
#         cv2.namedWindow("frame")
#         cv2.imshow('frame', frame)
#
#         mykey = cv2.waitKey(1)
#     # 按q退出循环，0xFF是为了排除一些功能键对q的ASCII码的影响
#     if mykey & 0xFF == ord('q'):
#         break
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()