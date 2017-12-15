from keras import backend as K
from keras.engine.topology import Layer

class Norm_layer(Layer):
    def __init__(self,**kwargs):
        super(Norm_layer,self).__init__(**kwargs)

    def build(self,input_shape):
        super(Norm_layer,self).build(input_shape)

    def call(self,x):
        return K.l2_normalize(x,1)

    def compute_output_shape(self,input_shape):
        return input_shape
