import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

from retreive_data import unpack_gzip

print "unpacking data"
x_data, y_data = unpack_gzip("chr9.txt.gz")
n = len(x_data)
try:
    assert (n == len(y_data))
except AssertionError:
    print str(n) + " and " + str(len(y_data)) + " do not match"

#4/6 as training data, 1/6 as test, 1/6 as validation
x, y = (4, 500)
#y_data = y_data.reshape(len(y_data), 1)
train = (x_data[2*n/6:], y_data[2*n/6:])
#train[1] = train[1].reshape(train[1].shape[0], 1)
train_data = train[0].reshape(train[0].shape[0], x, y, 1)
test = (x_data[:n/6], y_data[:n/6])
test_data = test[0].reshape(test[0].shape[0], x, y, 1)
val = (x_data[n/6:2*n/6], y_data[n/6:2*n/6])
val_data = val[0].reshape(val[0].shape[0], x, y, 1)

print "training shape: " + str(train_data.shape)
print "building model"

model = Sequential()
model.add(Conv2D(filters=320,
                  input_shape=(x, y, 1), #input dim: 4 x 500 (x, y, 1)
                  kernel_size= (4, 40), #26, #(4, 40),
                  padding="same",
                  activation="relu",
#                 subsample_length=1
                  strides=(1, 3)
))
model.add(MaxPooling2D(pool_size= (4, 13), padding="same"))
model.add(Dropout(.5))
model.add(Flatten())
#model.add(Dense(320, activation="relu"))
#model.add(Activation('relu'))
#model.add(Conv2D(filters=32,
#                 kernel_size=(4, 20),
#                 activation='relu',
#                 padding='same'))
#model.add(Dropout(.5))
model.add(Dense(100, activation="softmax"))
model.add(Dense(1, init="uniform", activation="sigmoid")) #what does init=normal do?

print "compiling"

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

print model.summary()

print "running at most 20 epochs"
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model.fit(train_data, train[1],
          batch_size=100,
          epochs=20,
          validation_data=(val_data, val[1]),
          shuffle='batch',
#          show_metrics=True,
          verbose=1,
          callbacks=[earlystopper])

print model.evaluate(test_data, test[1], verbose=1)

