import sys
import cv2
import re
import os
from os.path import isfile, join
import numpy as np
import xlsxwriter
import matplotlib
matplotlib.use("TkAgg")
from identifikasi import Classification
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QAction, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
class ShowImage(QMainWindow):

    def __init__(self):
        super(ShowImage,self).__init__()
        loadUi('Lalulintashitung.ui',self)
        self.image=None

        self.buttonLoadImage_2.clicked.connect(self.load2Clicked)
        self.savebutton_2.clicked.connect(self.save2Clicked)
        self.buttonLoadImage.clicked.connect(self.loadClicked)
        self.savebutton.clicked.connect(self.saveClicked)
        self.buttonprocessing.clicked.connect(self.processingImageClicked)
        self.buttondetection.clicked.connect(self.detectionClicked)
        self.buttonXlsxwriter.clicked.connect(self.exportClicked)

    @pyqtSlot()
    def exportClicked(self):
        col_frames = os.listdir('frames/')

        col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

        # empty list to store the frames
        col_images = []

        for i in col_frames:
            # read the frames
            img = cv2.imread('frames/' + i)
            # append the frames to the list
            col_images.append(img)
        i = 13

        for frame in [i, i + 1]:
            plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
            plt.title("frame: " + str(frame))
            # plt.show()

        grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(col_images[i + 1], cv2.COLOR_BGR2GRAY)

        # plot the image after frame differencing
        # plt.imshow(cv2.absdiff(grayB, grayA), cmap='gray')
        # plt.show()

        diff_image = cv2.absdiff(grayB, grayA)

        # perform image thresholding
        ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

        # plot image after thresholding
        # plt.imshow(thresh, cmap='gray')
        # plt.show()

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # plot dilated image
        # plt.imshow(dilated, cmap='gray')
        # plt.show()

        # plt.imshow(dilated)
        cv2.line(dilated, (0, 50), (256, 50), (100, 0, 0))
        # plt.show()

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        valid_cntrs = []

        for i, cntr in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cntr)
            if (x <= 200) & (y >= 50) & (cv2.contourArea(cntr) >= 25):
                valid_cntrs.append(cntr)

        # count of discovered contours
        len(valid_cntrs)

        dmy = col_images[13].copy()

        cv2.drawContours(dmy, valid_cntrs, -1, (127, 200, 0), 2)
        cv2.line(dmy, (0, 80), (256, 80), (100, 255, 255))

        # plt.imshow(dmy)
        # plt.show()

        kernel = np.ones((4, 4), np.uint8)

        # font style
        font = cv2.FONT_HERSHEY_SIMPLEX

        # directory to save the ouput frames
        pathIn = "contour_frames_3/"

        for i in range(len(col_images) - 1):

            # frame differencing
            grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(col_images[i + 1], cv2.COLOR_BGR2GRAY)
            diff_image = cv2.absdiff(grayB, grayA)

            # image thresholding
            ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

            # image dilation
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            # find contours
            contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # shortlist contours appearing in the detection zone
            valid_cntrs = []
            for cntr in contours:
                x, y, w, h = cv2.boundingRect(cntr)
                if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):
                    if (y >= 90) & (cv2.contourArea(cntr) < 40):
                        break
                    valid_cntrs.append(cntr)

            # add contours to original frames
            dmy = col_images[i].copy()
            cv2.drawContours(dmy, valid_cntrs, -1, (127, 200, 0), 2)

            cv2.putText(dmy, "vehicles detected: " + str(len(valid_cntrs)), (55, 15), font, 0.6, (0, 180, 0), 2)
            cv2.line(dmy, (0, 80), (256, 80), (100, 255, 255))
            cv2.imwrite(pathIn + str(i) + '.png', dmy)
        # plt.imshow(dmy)
        # plt.show()
        self.image=dmy
        self.displayImage(2)
        pathOut = 'vehicle_detection_v3.mp4'
        fps = 14.0
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        for i in range(len(files)):
            filename = pathIn + files[i]

            # read frames
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)

            # inserting the frames into an image array
            frame_array.append(img)
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])



        out.release()
        # plt.imread(out)
        # plt.imshow(out)
        # plt.show()

    @pyqtSlot()
    def detectionClicked(self):
        folder = 'testsa'
        os.mkdir(folder)
        vidcap = cv2.VideoCapture('vehicle.mp4')

        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
            if hasFrames:
                cv2.imwrite(os.path.join(folder,'image' + str(count) + '.jpg'), image)  # save frame as JPG file
            return hasFrames

        sec = 0
        frameRate = 0.095  # //it will capture image in each 0.5 second
        count = 1
        success = getFrame(sec)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec)
        # folder = 'testsa'
        # os.mkdir(folder)
        # cap = cv2.VideoCapture('vehicle.mp4')
        # i = 0
        # while (cap.isOpened()):
        #     ret, frame = cap.read()
        #     if ret == False:
        #         break
        #     cv2.imwrite(os.path.join(folder,'image' + str(i) + '.jpg'), frame)
        #     i += 1
        #
        # cap.release()
        # cv2.destroyAllWindows()
        # folder = 'test'
        # os.mkdir(folder)
        # vidcap = cv2.VideoCapture(
        #     QFileDialog.getOpenFileName(self, 'Open File', 'D:\\@PCD\\PycharmProjects\\LaluLintas\\citraasli\\',
        #                                 "Video Files(*.mp4)"))
        # count = 0
        # while True:
        #     success, image = vidcap.read()
        #     if not success:
        #         break
        #     cv2.imwrite(os.path.join(folder, "frame{:d}.jpg".format(count)), image)  # save frame as JPEG file
        #     count += 1
        # print("{} images are extacted in {}.".format(count, folder))
    @pyqtSlot()
    def processingImageClicked(self):
        col_frames = os.listdir('frames/')

        col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

        # empty list to store the frames
        col_images = []

        for i in col_frames:
            # read the frames
            img = cv2.imread('frames/' + i)
            # append the frames to the list
            col_images.append(img)
        i = 13

        for frame in [i, i + 1]:
            plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
            plt.title("frame: " + str(frame))
            plt.show()

        grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(col_images[i + 1], cv2.COLOR_BGR2GRAY)

        # plot the image after frame differencing
        plt.imshow(cv2.absdiff(grayB, grayA), cmap='gray')
        plt.show()

        self.image=grayA
        self.displayImage(1)
        diff_image = cv2.absdiff(grayB, grayA)

        # perform image thresholding
        ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

        # plot image after thresholding
        plt.imshow(thresh, cmap='gray')
        plt.show()
        self.image = thresh
        self.displayImage(1)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # plot dilated image
        plt.imshow(dilated, cmap='gray')
        plt.show()
        self.image = dilated
        self.displayImage(1)

        plt.imshow(dilated)
        cv2.line(dilated, (0, 80), (256, 80), (100, 0, 0))
        # plt.show()
        self.image = dilated
        self.displayImage(1)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        valid_cntrs = []

        for i, cntr in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cntr)
            if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):
                valid_cntrs.append(cntr)

        # count of discovered contours
        len(valid_cntrs)

        dmy = col_images[13].copy()

        cv2.drawContours(dmy, valid_cntrs, -1, (127, 200, 0), 2)
        cv2.line(dmy, (0, 80), (256, 80), (100, 255, 255))

        plt.imshow(dmy)
        plt.show()
        self.image = dmy
        self.displayImage(1)
        # img = cv2.resize(self.image, (512, 512))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        # img = cv2.GaussianBlur(img, (3, 3), 0)
        # img = cv2.medianBlur(img, 3)
        #
        # img = cv2.Canny(img, 30, 150, L2gradient=True)
        # kernel = np.ones((3, 3), np.uint8)
        # imgh = cv2.dilate(img, kernel, iterations=3)
        # self.image = imgh
        # self.displayImage(2)
        # pathIn = 'frames/'
        # pathOut = 'vehicle.mp4'
        # fps = 14.0
        # frame_array = []
        # files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        # files.sort(key=lambda f: int(re.sub('\D', '', f)))
        #
        # for i in range(len(files)):
        #     filename = pathIn + files[i]
        #
        #     # read frames
        #     img = cv2.imread(filename)
        #     height, width, layers = img.shape
        #     size = (width, height)
        #
        #     # inserting the frames into an image array
        #     frame_array.append(img)
        # out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        #
        # for i in range(len(frame_array)):
        #     # writing to a image array
        #     out.write(frame_array[i])
        #
        # out.release()




    def saveClicked(self):
        flname, filter = QFileDialog.getSaveFileName(self, 'save file', 'D:\\', "Images Files(*.png)")

        if flname:
            cv2.imwrite(flname, self.image)
        else:
            print('Saved')
    def save2Clicked(self):
        flname1, filter = QFileDialog.getSaveFileName(self, 'save file', 'D:\\', "Images Files(*.png)")

        if flname1:
            cv2.imwrite(flname1, self.image)
        else:
            print('Saved')

    @pyqtSlot()
    def loadClicked(self):

        flname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'D:\\@PCD\\PycharmProjects\\LaluLintas\\citraasli\\', "Image Files(*.png)")
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

    @pyqtSlot()
    def loadImage(self, flname):
        self.image = cv2.imread(flname)
        self.displayImage(1)

    @pyqtSlot()
    def load2Clicked(self):

        flname1, filter = QFileDialog.getOpenFileName(self, 'Open File',
                                                     'D:\\@PCD\\PycharmProjects\\LaluLintas\\citraasli\\',
                                                     "Image Files(*.png)")
        if flname1:
            self.loadImage(flname1)
        else:
            print('Invalid Image')

    @pyqtSlot()
    def load2Image(self, flname1):
        self.image = cv2.imread(flname1)
        self.displayImage(2)

    @pyqtSlot()
    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

        img = img.rgbSwapped()
        if windows == 1:
            self.pertama.setPixmap(QPixmap.fromImage(img))
            self.pertama.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.pertama.setScaledContents(True)
        if windows == 2:
            self.pertama_2.setPixmap(QPixmap.fromImage(img))
            self.pertama_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.pertama_2.setScaledContents(True)






if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    window=ShowImage()
    window.setWindowTitle('LALU LINTAS')
    window.show()
    sys.exit(app.exec_())