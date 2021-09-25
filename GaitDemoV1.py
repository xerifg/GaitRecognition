import sys
from PyQt5 import QtCore, QtWidgets, uic, QtGui
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
import cv2
import os
import pickle
import realsense_depth as rsd
import time
import PoseTrackingModel as ptk

# Loading the UI window
qtCreatorFile = "GaitUI.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


def pickle2(filename, data, compress=False):
    fo = open(filename, "wb")
    pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)  # 序列化对象，并将结果数据流写入到文件对象中
    fo.close()


def unpickle2(filename):
    fo = open(filename, 'rb')
    dict = pickle.load(fo)  # 反序列化对象,将文件中的数据解析为一个Python对象
    fo.close()
    return dict


class GaitDemo(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        '''
        Set initial parameters here.
        Note that the demo window size is 1920*1080, you can edit this via Qtcreator.
        In this demo, we take 20 frames of profiles to generate a GEI. You can edit this number by your self.
        '''
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.showFullScreen()
        self.setupUi(self)
        # get camera images
        self.capture = rsd.DepthCamera()  # through D435 camera
        # self.capture = cv2.VideoCapture(0)  # through computer cam
        self.currentFrame = np.array([])
        self.thresh = np.array([])
        self.register_state = False
        self.recognition_state = False
        self.save_on = False
        self.gei_fix_num = 20  # the number of calculate GEI
        self.cTime = 0
        self.pTime = 0
        self.detector = ptk.PoseDetector()

        # Set two window for raw video and segmentation.
        self.video_lable = QtWidgets.QLabel(self.centralwidget)
        self.seg_label = QtWidgets.QLabel(self.centralwidget)
        self._timer = QtCore.QTimer(self)  # open Qt timer
        self.video_lable.setGeometry(50, 200, 640, 480)
        self.seg_label.setGeometry(800, 200, 640, 480)
        self.load_dataset()
        self._timer.timeout.connect(self.play)  # response function of Qt timer

        # Waiting for you to push the button.
        # The slot functions from Qt
        self.register_2.clicked.connect(self.register_show)
        self.save_gei.clicked.connect(self.save_gei_f)
        self.recognize.clicked.connect(self.recognition_show)
        self._timer.start(27)  # the end time of Qt timer, it means get a frame to synthesis GEI every about 27 ms
        self.update()

    def save_gei_f(self):
        '''
        Waiting the save button.
        '''
        self.save_on = True
        self.state_print.setText('Saving!')

    def register_show(self):
        '''
        To record the GEI into gait database.
        '''
        self.register_state = True
        self.recognition_state = False
        self.state_print.setText('Register!')
        self.gei_current = np.zeros((128, 88), np.single)
        self.numInGEI = 0

    def load_dataset(self):
        '''
        Load gait database if existing.
        '''
        self.data_path = './GaitData'
        if os.path.exists(self.data_path):
            dic = unpickle2(self.data_path)
            self.num = dic['num']
            self.gei = dic['gei']
            self.name = dic['name']
        else:
            self.num = 0  # The total number of GEIs that have recorded
            self.gei = np.zeros([100, 128, 88], np.uint8)  # save all GEIs
            self.name = []
            dic = {'num': self.num, 'gei': self.gei, 'name': self.name}
            pickle2(self.data_path, dic, compress=False)
        self.id_num.setText('%d' % self.num)
        self.state_print.setText('Running!')

    def recognition_show(self):
        '''
        Working now and just recognizing the one in front of this camera.
        '''
        self.recognition_state = True
        self.register_state = False
        self.gei_current = np.zeros((128, 88), np.single)
        self.numInGEI = 0
        self.state_print.setText('Recognition!')


    def play(self):
        '''
        Main program.
        '''
        # ret, frame = self.capture.read()  # Read video from a camera.
        ret, depth_frame, frame = self.capture.get_frame()
        if (ret == True):
            # frame = cv2.resize(frame, (512, 384))
            # Apply background subtraction method.
            """####################  preprocess  ########################"""
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # bgr to gray
            # gray = cv2.GaussianBlur(gray, (3, 3), 0)
            img_pose = self.detector.findPose(frame)  # get pose from frame
            # remove background and create Silhouettes
            annotated_image = np.zeros((480, 640), dtype=np.uint8)
            try:
                mask = self.detector.results.segmentation_mask  # dtype:float32
                condition = mask > 0.5  # element type is bool  threshold:0.5
                annotated_image[:] = 255
                bg_image = np.zeros(mask.shape, dtype=np.uint8)
                self.thresh = np.where(condition, annotated_image, bg_image)
                # cv2.imshow("Silhouettes", annotated_image)
            except TypeError:
                print("Fail to capture human pose")
            # self.thresh = cv2.dilate(self.thresh, None, iterations=2)  # dilate image
            # self.thresh = cv2.erode(self.thresh, None, iterations=2)
            max_rec = 0
            #####################################################################
            # Find the max box.
            sil_y_list, sil_x_list = (self.thresh > 100).nonzero()  # return the position of pix value>100
            x_topleft, y_topleft = sil_x_list.min(), sil_y_list.min()
            x_botright, y_botright = sil_x_list.max(), sil_y_list.max()
            w = x_botright - x_topleft
            h = y_botright - y_topleft
            max_rec = w * h
            # If exist  box.
            if max_rec > 0:
                cv2.rectangle(frame, (x_topleft, y_topleft), (x_botright, y_botright), (0, 255, 0), 2)
                # when click register or recognition every time,the GEI will be calculate again
                if self.register_state or self.recognition_state:
                    # Get coordinate position.
                    ty, tx = sil_y_list, sil_x_list  # return the position of pix value>100
                    sy, ey = ty.min(), ty.max() + 1
                    sx, ex = tx.min(), tx.max() + 1
                    h = ey - sy
                    w = ex - sx
                    if h > w:  # Normal human should be like this, the height shoud be greater than wideth.
                        # Calculate the frame for GEI
                        cx = int(tx.mean())
                        cenX = h / 2
                        start_w = (h - w) / 2
                        if max(cx - sx, ex - cx) < cenX:
                            start_w = cenX - (cx - sx)
                        tim = np.zeros((h, h), np.single)
                        start_w = int(start_w)
                        tim[:, start_w:start_w + w] = self.thresh[sy:ey, sx:ex]
                        rim = Image.fromarray(np.uint8(tim)).resize((88, 128),
                                                                    Image.ANTIALIAS)  # from ndtype to PIL,
                        # then resize the image to (88,128)
                        tim = np.array(rim)[:]
                        if self.numInGEI < self.gei_fix_num:
                            self.gei_current += tim  # Add up until reaching the fix number.
                            print(self.numInGEI)
                        self.numInGEI += 1

                    if self.numInGEI > self.gei_fix_num:
                        if self.save_on:
                            # Save the GEI.
                            self.gei[self.num, :, :] = self.gei_current / self.gei_fix_num
                            Image.fromarray(np.uint8(self.gei_current / self.gei_fix_num)).save(
                                './gei/gei%02d%s.jpg' % (
                                    self.num, self.id_name.toPlainText()))  # save the user GEI to local
                            self.name.append(self.id_name.toPlainText())  # save user name
                            self.num += 1
                            self.id_num.setText('%d' % self.num)  # show total number of users
                            dic = {'num': self.num, 'gei': self.gei,
                                   'name': self.name}  # save user name and GEI with dictionary
                            pickle2(self.data_path, dic, compress=False)
                            self.save_on = False
                            self.state_print.setText('Saved!')
                        elif self.recognition_state:
                            # Recognition.
                            self.gei_query = self.gei_current / self.gei_fix_num  # get current GEI
                            score = np.zeros(self.num)
                            self.gei_to_com = np.zeros([128, 88], np.single)  # the GEI from the database
                            for q in range(self.num):
                                self.gei_to_com = self.gei[q, :, :]
                                score[q] = np.exp(-(((self.gei_query[:] - self.gei_to_com[:]) / (
                                        128 * 88)) ** 2).sum())  # Compare with gait database.
                            q_id = score.argmax()
                            if True:
                                id_rec = '%s' % self.name[q_id]
                                cv2.putText(frame, id_rec, (x_topleft + 20, y_topleft + 20),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2,
                                            color=(0, 0, 255))
            else:
                self.gei_current = np.zeros((128, 88), np.single)
                self.numInGEI = 0

            # Show results.
            self.currentFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.currentSeg = Image.fromarray(self.thresh).convert('RGB')
            self.currentSeg = ImageQt(self.currentSeg)
            height, width = self.currentFrame.shape[:2]
            # show fps
            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            cv2.putText(self.currentFrame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)

            img = QtGui.QImage(self.currentFrame,
                               width,
                               height,
                               QtGui.QImage.Format_RGB888)

            img = QtGui.QPixmap.fromImage(img)

            self.video_lable.setPixmap(img)
            seg = QtGui.QImage(self.currentSeg)
            seg = QtGui.QPixmap(seg)
            self.seg_label.setPixmap(seg)

    def keyPressEvent(self, event):  # 重新实现了keyPressEvent()事件处理器。
        # 按住键盘事件
        # 这个事件是PyQt自带的自动运行的，当我修改后，其内容也会自动调用
        if event.key() == QtCore.Qt.Key_Escape:  # 当我们按住键盘是esc按键时
            self.close()  # 关闭程序


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = GaitDemo()
    window.show()
    sys.exit(app.exec_())
