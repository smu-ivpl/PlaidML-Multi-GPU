import sys
from PyQt5 import QtWidgets
from threading import Thread, Event
from src.ssd import VOC_CLASS, Detector, Detect
from src.gui import Dialog
from src.util import FrameQue

def main():
    args = {}

    args['classes'] = VOC_CLASS
    args['shape'] = (300, 300, 3)
    args['running'] = Event()
    args['stop_event'] = Event()

    args['in_que_1'] = FrameQue()
    args['in_que_2'] = FrameQue()
    args['out_que_1'] = FrameQue()
    args['out_que_2'] = FrameQue()

    args['detector'] = Detector(args['shape'])
    args['model_1'], args['model_2'] = args['detector'].build()
    t1 = Thread(target=Detect, args=(1,), kwargs=args)
    t2 = Thread(target=Detect, args=(2,), kwargs=args)
    t1.start()
    t2.start()

    try:
        winapp = QtWidgets.QApplication(sys.argv)
        demo = Dialog(None, **args)
        sys.exit(winapp.exec_())

    except Exception as ex:
        print('Error: ', ex)

    finally:
        args['stop_event'].set()

if __name__ == "__main__":
    main()