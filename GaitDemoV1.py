import sys
from PyQt5 import QtCore, QtWidgets, uic, QtGui
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
import cv2
import os
import pickle
import realsense_depth as rsd

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
        self.firstFrame = None
        self.register_state = False
        self.recognition_state = False
        self.save_on = False
        self.gei_fix_num = 20


        # Set two window for raw video and segmentation.
        self.video_lable = QtWidgets.QLabel(self.centralwidget)
        self.seg_label = QtWidgets.QLabel(self.centralwidget)
        self._timer = QtCore.QTimer(self)
        self.video_lable.setGeometry(50, 200, 640, 480)
        self.seg_label.setGeometry(800, 200, 640, 480)
        self.load_dataset()
        self._timer.timeout.connect(self.play)

        # Waiting for you to push the button.
        self.register_2.clicked.connect(self.register_show)
        self.recognize.clicked.connect(self.recognition_show)
        self.updater.clicked.connect(self.update_bk)
        self.save_gei.clicked.connect(self.save_gei_f)
        self._timer.start(27)
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
            self.num = 0
            self.gei = np.zeros([100, 128, 88], np.uint8)
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

    def update_bk(self):
        '''
        If you moved the camera.
        '''
        self.firstFrame = self.FrameForUpdate

    def play(self):
        '''
        Main program.
        '''
        # ret, frame = self.capture.read()  # Read video from a camera.
        ret, depth_frame, frame = self.capture.get_frame()
        if (ret == True):
            # frame = cv2.resize(frame, (512, 384))
            # Apply background subtraction method.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            if self.firstFrame is None:
                self.firstFrame = gray  # Set this frame as the background.
            frameDelta = cv2.absdiff(self.firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
            self.FrameForUpdate = gray
            thresh = cv2.dilate(thresh, None, iterations=2)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            thresh = np.array(thresh)
            max_rec = 0

            # Find the max box.
            for c in cnts:
                if cv2.contourArea(c) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)

                if w > 25 and h > 50:
                    if max_rec < w * h:
                        max_rec = w * h
                        (x_max, y_max, w_max, h_max) = cv2.boundingRect(c)
            # If exist max box.
            if max_rec > 0:
                cv2.rectangle(frame, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 0), 2)
                if x_max > 20:  # To ignore some regions which contain parts of human body.
                    if self.register_state or self.recognition_state:
                        nim = np.zeros([thresh.shape[0] + 10, thresh.shape[1] + 10],
                                      np.single)  # Enlarge the box for better result.
                        nim[y_max + 5:(y_max + h_max + 5), x_max + 5:(x_max + w_max + 5)] = thresh[
                                                                                            y_max:(y_max + h_max),
                                                                                            x_max:(x_max + w_max)]
                        offsetX = 20
                        # Get coordinate position.
                        ty, tx = (nim > 100).nonzero()
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
                            tim[:, start_w:start_w + w] = nim[sy:ey, sx:ex]
                            rim = Image.fromarray(np.uint8(tim)).resize((128, 128), Image.ANTIALIAS)
                            tim = np.array(rim)[:, offsetX:offsetX + 88]
                            if self.numInGEI < self.gei_fix_num:
                                self.gei_current += tim  # Add up until reaching the fix number.
                            self.numInGEI += 1

                        if self.numInGEI > self.gei_fix_num:
                            if self.save_on:
                                # Save the GEI.
                                self.gei[self.num, :, :] = self.gei_current / self.gei_fix_num
                                Image.fromarray(np.uint8(self.gei_current / self.gei_fix_num)).save(
                                    './gei/gei%02d%s.jpg' % (self.num, self.id_name.toPlainText()))
                                self.name.append(self.id_name.toPlainText())
                                self.num += 1
                                self.id_num.setText('%d' % self.num)
                                dic = {'num': self.num, 'gei': self.gei, 'name': self.name}
                                pickle2(self.data_path, dic, compress=False)
                                self.save_on = False
                                self.state_print.setText('Saved!')
                            elif self.recognition_state:
                                # Recognition.
                                self.gei_query = self.gei_current / (self.gei_fix_num)
                                score = np.zeros(self.num)
                                self.gei_to_com = np.zeros([128, 88], np.single)
                                for q in range(self.num):
                                    self.gei_to_com = self.gei[q, :, :]
                                    score[q] = np.exp(-(((self.gei_query[:] - self.gei_to_com[:]) / (
                                                128 * 88)) ** 2).sum())  # Compare with gait database.
                                q_id = score.argmax()
                                if True:
                                    id_rec = '%s' % self.name[q_id]
                                    cv2.putText(frame, id_rec, (x_max + 20, y_max + 20),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2,
                                                color=(0, 0, 255))
            else:
                self.gei_current = np.zeros((128, 88), np.single)
                self.numInGEI = 0

            # Show results.
            self.currentFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.currentSeg = Image.fromarray(thresh).convert('RGB')
            self.currentSeg = ImageQt(self.currentSeg)
            height, width = self.currentFrame.shape[:2]
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
