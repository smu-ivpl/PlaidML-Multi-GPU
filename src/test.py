#!/usr/bin/env python
import numpy as np
import os
import time

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# import keras
import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend as K
import keras.applications as kapp
from keras.datasets import cifar10

(x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
batch_size = 8
x_train = x_train[:batch_size]
# x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)
x_train = np.resize(x_train, (x_train.shape[0], 299, 299, x_train.shape[-1]))

K.IVPL_DEVICE_NO = 0
model_1 = kapp.InceptionResNetV2()
model_1.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

print("Running initial batch (compiling tile program)")
model_1._function_kwargs['dev_no'] = 0
model_1.predict(x=x_train, batch_size=batch_size)

K.IVPL_DEVICE_NO = 1
model_2 = kapp.InceptionResNetV2()
model_2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

print("Running initial batch (compiling tile program)")
model_2._function_kwargs['dev_no'] = 1
model_2.predict(x=x_train, batch_size=batch_size)

def Predict(model, x, size):
    # Now start the clock and run 10 batches
    print("Timing inference...")
    start = time.time()
    for i in range(10000):
        y = model.predict(x=x, batch_size=size)
        print(i)
    print("Ran in {} seconds".format(time.time() - start))

import threading

t1 = threading.Thread(target=Predict, args=(model_1, x_train, batch_size))
t2 = threading.Thread(target=Predict, args=(model_2, x_train, batch_size))

t1.daemon = True
t2.daemon = True

t1.start()
t2.start()

t1.join()
t2.join()

print("END")






####################################

def SetProcess(**kwargs):
    pid = os.getpid()

    if kwargs['use_gpu'] is True:
        print('GPU(PlaidML)-Process ID: %d' % (pid))
        from src.ssd import Detector
        detector = Detector('multi')
    else :
        print('CPU(Tensorflow)-Process ID: %d' % (pid))
        from src.ssd import Detector
        detector = Detector('cpu')

    def Detect(**kwargs):
        import numpy as np
        dummy = np.ones((1, 300, 300, 3))
        while True:
            if kwargs['running']:
                kwargs['detector'].detect(kwargs['model'], dummy)
                # if not kwargs['in_que'].empty():
                    # org, small = kwargs['in_que'].get()
                    # result, time = kwargs['detector'].detect(kwargs['model'], small)
                    # kwargs['out_que'].put((org, result, time))

    threads = []
    args = {}
    args['running'] = kwargs['running']
    args['detector'] = detector
    if kwargs['use_gpu'] is True:
        args['in_que'] = kwargs['in_que_gpu_1']
        args['out_que'] = kwargs['out_que_1']
        args['model'] = detector.ssd_model[0]
        t1 = Thread(target=Detect, kwargs=args, name='Thread-PlaidML-1')
        threads.append(t1)

        args['in_que'] = kwargs['in_que_gpu_2']
        args['out_que'] = kwargs['out_que_2']
        args['model'] = detector.ssd_model[1]
        t2 = Thread(target=Detect, kwargs=args, name='Thread-PlaidML-2')
        threads.append(t2)
    else:
        args['in_que'] = kwargs['in_que_cpu']
        args['out_que'] = kwargs['out_que_1']
        args['model'] = detector.ssd_model
        t = Thread(target=Detect, kwargs=args, name='Thread-Tensorflow')
        threads.append(t)

    for t in threads:
        t.start()

    while True:
        continue