import numpy as np
import h5py
import scipy.io

import keras
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping #ModelCheckpoint
from keras import backend as K
print ('imports done')

config = tf.ConfigProto()
#config.log_device_placement = True #see where all the nodes are being placed
config.allow_soft_placement = True #allows adjustments for variables on GPU
sess = tf.Session(config=config)
#Used for tfdbg
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
K.set_session(sess)

for d in ['/gpu:0', '/gpu:1']:
    with tf.device(d):
            print ('loading data into ' + d)
            trainmat = h5py.File('everything/train.mat')
            validmat = scipy.io.loadmat('everything/valid.mat')
            testmat = scipy.io.loadmat('everything/test.mat')
            print ('done')

            #correcting data format
            X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
            y_train = np.array(trainmat['traindata']).T


            #building model
            model = Sequential()

            model.add(Conv1D(input_dim=4,
                             input_length=1000,
                             nb_filter=320,
                             filter_length=26,
                             border_mode="same",
                             activation="relu",
                             subsample_length=1))

            model.add(MaxPooling1D(pool_length=13, stride=  13)) #wtf why 13

            model.add(Dropout(0.2))
            model.add(Flatten())

            model.add(Dense(input_dim=76*640, output_dim=925))
            model.add(Activation('relu'))

            model.add(Dense(input_dim=925, output_dim=919))
            model.add(Activation('sigmoid'))

            print 'compiling model'
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy']) #class_mode="binary")

            print 'running at most 20 epochs'

            #checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel.hdf5", verbose=1, save_best_only=True)
            earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
            callback = keras.callbacks.TensorBoard(log_dir='./logs',
                                                   histogram_freq=0,
                                                   write_graph=True,
                                                   write_grads=False,
                                                   write_images=True,
                                                   embeddings_freq=0,
                                                   embeddings_layer_names=None,
                                                   embeddings_metadata=None)

            modela = model.fit(X_train[0:10000], y_train[0:10000], 
                      batch_size=100, 
                      epochs=20,
                      shuffle="batch", 
                      validation_data=(np.transpose(validmat['validxdata'], axes=(0,2,1)), validmat['validdata']), 
                      verbose=1,
                      callbacks=[earlystopper, callback])

with tf.device('/gpu:0'):
    tresults = tf.to_float(model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)), testmat['testdata']))
    print sess.run(tresults)

sess.close()
