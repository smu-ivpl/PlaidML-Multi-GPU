from multiprocessing import Process
import numpy as np
import time
import os

VOC_CLASS = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
             'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
             'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
             'Sheep', 'Sofa', 'Train', 'Tvmonitor']

COCO_CLASS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag',
              'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
              'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot dog',
              'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
              'toilet', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
              'sink',
              'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
              'toothbrush']

def Detect(gpu_no, **kwargs):
    if gpu_no == 1:
        model = kwargs['model_1']
        in_que = kwargs['in_que_1']
        out_que = kwargs['out_que_1']
    else:
        model = kwargs['model_2']
        in_que = kwargs['in_que_2']
        out_que = kwargs['out_que_2']

    while not kwargs['stop_event'].wait(1e-9):
        if not kwargs['running'].wait(1e-9):
            if not in_que.empty() and in_que.qsize() > 60:
                small = in_que.get()
                result, time = kwargs['detector'].detect(model, small)
                out_que.put((result, time))

class Detector(Process):
    def __init__(self, shape):
        Process.__init__(self)
        self.bbox_util = None
        self.input_shape = shape

    def switch_backend(self, use_gpu):
        if use_gpu:
            os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
        else:
            os.environ['KERAS_BACKEND'] = 'tensorflow'

    def detect(self, ssd_model, frame):
        start_time = time.time()
        preds = ssd_model.predict(frame, batch_size=1)
        end_time = time.time()

        result = self.bbox_util.detection_out(preds)

        return result, (end_time - start_time)

    def build(self, mode='multi'):
        assert mode == 'cpu' or mode == 'single' or mode == 'multi'

        num_classes = len(VOC_CLASS) + 1
        # input_shape = (640, 640, 3)
        weights = 'weights/VGG_VOC0712_SSD_300x300_iter_120000.h5'

        if mode == 'cpu':
            self.switch_backend(False)
            return self.make(self.input_shape, num_classes, weights, mode)

        if mode == 'single' or mode == 'multi':
            self.switch_backend(True)
            return self.make(self.input_shape, num_classes, weights, mode)

    def make(self, input_shape, num_classes, weights, mode):
        from src.model.ssd300VGG16_V2 import SSD
        from src.model.ssd_utils import BBoxUtility

        self.bbox_util = BBoxUtility(num_classes)

        net1 = SSD(input_shape, num_classes)
        net1.load_weights(weights, by_name=True)

        print("Running initial batch with dummy input data (compiling tile program)")
        dummy = np.expand_dims(np.ones(input_shape), axis=0)
        net1.predict(dummy, batch_size=1)

        if mode == 'cpu' or mode == 'single':
            return net1

        import plaidml.keras.backend as K
        K.IVPL_DEVICE_NO = 1
        net2 = SSD(input_shape, num_classes)
        net2.load_weights(weights, by_name=True)

        print("Running initial batch with dummy input data (compiling tile program)")
        net2.predict(dummy, batch_size=1)

        return net1, net2


if __name__ == "__main__":
    pass