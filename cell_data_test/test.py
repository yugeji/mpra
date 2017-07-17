import tensorflow as tf
import scipy.io
import h5py
import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping #ModelCheckpoint
from keras import backend as K
print ('imports done')

config = tf.ConfigProto()
#config.log_device_placement = True #see where all the nodes are being placed
config.allow_soft_placement=True #allows adjustments for variables on GPU
sess = tf.Session(config=config)
K.set_session(sess) 

cluster = tf.train.ClusterSpec({"local": ["localhost:5000", "localhost:5001"]})

#

for d in ["/gpu:0", "/gpu:1"]: #["/job:local/task:1", "/job:local/task:0"]:
    with tf.device(d):
        print ("loading training data")
        trainmat = h5py.File('../everything/train.mat')
        X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
        y_train = np.array(trainmat['traindata']).T

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
        print ('loading validation data')
        validmat = scipy.io.loadmat('../everything/valid.mat')
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        #callback = keras.callbacks.TensorBoard(log_dir='./logs',
        #                                       histogram_freq=0,
        #                                       write_graph=True,
        #                                       write_grads=False,
        #                                       write_images=True,
        #                                       embeddings_freq=0,
        #                                       embeddings_layer_names=None,
        #                                       embeddings_metadata=None)
        print ('running at least 20 epochs')
        model.fit(X_train[0:1000], y_train[0:1000],
                  batch_size=100,
                  epochs=20,
                  shuffle="batch",
                  validation_data=(np.transpose(validmat['validxdata'], axes=(0,2,1)), validmat['validdata']),
                  verbose=1,
                  callbacks=[])#earlystopper, callback])
        
        #x = tf.constant(2)
        #y2 = x - 66
        #y1 = x + 300

#with tf.device("/job:local/task:1"):
    

#with tf.device("/job:local/task:0"):
    

#with tf.device("/job:local/task:1"):
   # y = y1 + y2


#with tf.Session("grpc://localhost:5000") as sess:
#    result = sess.run(y)
#    print(result)
