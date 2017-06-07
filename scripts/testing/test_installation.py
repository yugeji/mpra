import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils


def generate_separable_dataset(num=(50, 50)):
    a = np.random.multivariate_normal([.2, .2], [[.05, 0], [0, .05]], num[0])
    b = np.random.multivariate_normal([.8, .8], [[.05, 0], [0, .05]], num[1])
    X = np.vstack((a, b))
    labels = np.hstack((np.zeros(50), np.ones(50)))
    return X, np_utils.to_categorical(labels)

X, y = generate_separable_dataset()
#plt.scatter(X[:,0], X[:,1], c=y[:,0], alpha=0.4)
#plt.show()


model = Sequential()
model.add(Dense(2, input_dim=2, init='uniform'))
model.add(Activation('softmax'))
sgd = SGD(lr=0.2)
model.compile(loss='mean_squared_error', optimizer=sgd)


X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True)
print model.evaluate(X_test, y_test, batch_size=16)
