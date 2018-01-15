import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, UpSampling1D, MaxPooling2D, Reshape, Lambda, Concatenate, Add, Maximum
from keras.models import Model
from spatial_transformer import *
from keras.models import load_model
import os

class TransformingAutoencoder ():
    def __init__(self):

        inputs = Input(shape=(28, 28, 1))
        trans = Input(shape=(2,))
        outputs = Input(shape=(28, 28, 1))

        x = Conv2D(32, 5, strides=(1, 1), padding='same', activation='relu')(inputs)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, 3, strides=(1, 1), padding='same', activation='relu')(x)

        flatten_input = Flatten()(x)

        capsule_inputs = Dense(32 * 6, activation=None)(flatten_input)
        capsule_inputs = Lambda(lambda x: K.reshape(x, (-1, 32, 6)))(capsule_inputs)

        p = Dense(1, activation='sigmoid')(capsule_inputs)

        xy = Dense(2, activation='relu')(capsule_inputs)
        xy = Reshape((32, 2))(xy)

        trans2 = Reshape((1, 2))(trans)
        xy = Lambda(lambda x: x[0] + x[1])([xy, trans2])

        generation_units = Dense(192, activation='relu')(xy)
        generation_units = Dense(784, activation='relu')(generation_units)
        
        p_generation_units = Lambda(lambda x: x[0] * x[1])([generation_units, p])
        output = Lambda(lambda x: K.sum(x, axis=1))(p_generation_units)
        output = Reshape((28, 28, 1))(output)

        print output.shape
        
        self.model = Model(inputs=[inputs, trans], outputs=output)
        self.model.compile(optimizer='rmsprop',
                    loss='mse',
                    metrics=['accuracy'])
        self.model.summary()

        if os.path.isfile('model.h5'):
            self.model = load_model('model.h5')

transforming_autoencoder = TransformingAutoencoder()