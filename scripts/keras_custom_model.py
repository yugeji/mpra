from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import activations,initializations, regularizers, constraints
import numpy as np
from theano.tensor.nnet import conv2d
import theano


def scale_motif(motif, weight):
    return motif*weight


class MotifLayer(Layer):
    def __init__(self, motif_tensor, **kwargs):
        _dict = kwargs
        print(_dict)

        self.motif_shape = motif_tensor.shape
        self.M = K.variable(motif_tensor)
        self.num_motifs = self.motif_shape[0]
        #InputSpec should specify dtype = dtype, shape = shape, ndim = ndim
        # self.init = initalizations.get(init)
        # self.activation = activations.get(activation)

        # self.W_regularizer = regularizers.get(W_regularizer)
        # self.b_regularizer = regularizers.get(b_regularizer)
        # self.activity_regularizer = regularizers.get(activity_regularizer)
        # self.input_spec = InputSpec(ndim=3)

        # self.bias = bias
        # self.inital_weights = weights
        # self.input
        self.bias = False
        super(MotifLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_shape = (self.num_filters,)
        self.B_shape = (1,)
        self.W = self.add_weight(self.W_shape,
                                 initalizer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(self.B_shape, initalizer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.inital_weights is not None:
            self.set_weights(self.inital_weights)
            del self.inital_weights

        #super(MotifLayer, self).build()
        super(MotifLayer, self).build()
        self.built = True

    def call(self, X, mask=None):
        self.MW = self.W.dimshuffle(0, 'x', 'x', 'x') * self.M
        output = conv2d(X, self.MW, border_mode='valid', filter_flip=False)
        # output = conv2d(input=x,filters=self.weightedMotifs,
        #                 border_mode = 'valid', fliter_flip=True)
        if self.bias:
            output += K.reshape(self.b, (1, self.num_filters, 1, 1))
        output = self.activation(output)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
