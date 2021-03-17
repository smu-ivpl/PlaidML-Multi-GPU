from PyQt5 import QtCore, QtGui, QtWidgets, uic
from threading import Thread
import numpy as np
import cv2

from src.util import FrameQue

class Dialog(QtWidgets.QDialog):
    def __init__(self, parent=None, **kwargs):
        QtWidgets.QDialog.__init__(self, parent=parent)

        self.mode = None
        self.video = None

        self.perf = []

        self.args = kwargs

        self.merge_que = FrameQue()
        self.frame_que = FrameQue()

        self.th_capture = None
        self.th_update = None
        self.th_merge = None

        self.ui = uic.loadUi('demo.ui')
        self.ui.btn_single.clicked.connect(self.btn_single_clicked)
        self.ui.btn_multi.clicked.connect(self.btn_multi_clicked)

        self.window_width = self.ui.widget_view.frameSize().width()
        self.window_height = self.ui.widget_view.frameSize().height()

        self.ui.widget_view = OwnImageWidget(self.ui.widget_view)

        self.ui.show()

    def merge(self):
        last = 1
        while not self.args['running'].wait(1e-9):
            if self.mode == 'Single':
                if not self.args['out_que_1'].empty():
                    self.merge_que.put(self.args['out_que_1'].get())
            if self.mode == 'Multi':
                if last % 2 == 0:
                    if not self.args['out_que_2'].empty():
                        self.merge_que.put(self.args['out_que_2'].get())
                        last += 1
                else:
                    if not self.args['out_que_1'].empty():
                        self.merge_que.put(self.args['out_que_1'].get())
                        last += 1
        print('Merge thread finished')

    def capture(self):
        count = 0
        while not self.args['running'].wait(1e-9):
            count += 1

            ret, frame = self.video.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.frame_que.put(frame)

            small = cv2.resize(frame, self.args['shape'][:2])
            small = np.expand_dims(small, axis=0)

            if self.mode == 'Single':
                self.args['in_que_1'].put(small)
            else:
                if count % 2 == 0:
                    self.args['in_que_2'].put(small)
                else:
                    self.args['in_que_1'].put(small)

        self.video.release()
        print('Capture thread finished')

    def start(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self)
        if len(fname[0]) == 0:
            print('File is not selected')
            self.btn_back()
            return

        self.video = cv2.VideoCapture(fname[0])
        if not self.video.isOpened():
            print('Incorrect video source')
            self.btn_back()
            return

        self.th_merge = Thread(target=self.merge)
        self.th_update = Thread(target=self.update)
        self.th_capture = Thread(target=self.capture)

        self.th_merge.start()
        self.th_update.start()
        self.th_capture.start()

    def btn_back(self):
        if self.mode == 'Single':
            self.btn_single_clicked()
        if self.mode == 'Multi':
            self.btn_multi_clicked()

    def clear_buffer(self, que):
        while not que.empty():
            que.get()

    def stop(self):
        self.args['running'].set()
        self.perf.clear()
        self.clear_buffer(self.args['in_que_1'])
        self.clear_buffer(self.args['in_que_2'])
        self.clear_buffer(self.args['out_que_1'])
        self.clear_buffer(self.args['out_que_2'])
        self.clear_buffer(self.merge_que)
        self.clear_buffer(self.frame_que)
        print('All buffer cleared')

    def btn_single_clicked(self):
        if self.ui.btn_single.text() == 'Stop':
            self.stop()

            self.ui.btn_multi.setEnabled(True)
            self.ui.btn_single.setText('Single')
        else:
            self.mode = self.ui.btn_single.text()
            self.ui.btn_multi.setEnabled(False)
            self.ui.btn_single.setText('Stop')

            self.args['running'].clear()
            self.start()

    def btn_multi_clicked(self):
        if self.ui.btn_multi.text() == 'Stop':
            self.stop()

            self.ui.btn_single.setEnabled(True)
            self.ui.btn_multi.setText('Multi')
        else:
            self.mode = self.ui.btn_multi.text()
            self.ui.btn_single.setEnabled(False)
            self.ui.btn_multi.setText('Stop')

            self.args['running'].clear()
            self.start()

    def draw_bbox(self, frame, result):
        info = result[0]
        self.perf.append(result[1])

        det_label = info[0][:, 0]
        det_conf = info[0][:, 1]
        det_xmin = info[0][:, 2]
        det_ymin = info[0][:, 3]
        det_xmax = info[0][:, 4]
        det_ymax = info[0][:, 5]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.8]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * frame.shape[1]))
            ymin = int(round(top_ymin[i] * frame.shape[0]))
            xmax = int(round(top_xmax[i] * frame.shape[1]))
            ymax = int(round(top_ymax[i] * frame.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = self.args['classes'][label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)

            if self.mode == 'Single':
                time = np.mean(self.perf) * 1000
            else:
                time = np.mean(self.perf) / 2 * 1000

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            cv2.putText(frame, display_txt, (xmin, ymin - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Throuput: {:5.2f} ms'.format(time), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def closeEvent(self, event):
        self.args['running'].set()

    def update(self):
        while not self.args['running'].wait(1e-9):
            if not self.frame_que.empty() and not self.merge_que.empty():
                frame = self.draw_bbox(self.frame_que.get(), self.merge_que.get())
                self.display(frame)

        print('Update thread finished')

    def display(self, frame):
        h, w, c = frame.shape

        self.ui.gbox.setMinimumHeight(h)
        self.ui.gbox.setMinimumWidth(w)

        bpl = c * w
        frame = QtGui.QImage(frame.data, w, h, bpl, QtGui.QImage.Format_RGB888)
        self.ui.widget_view.setImage(frame)

class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()