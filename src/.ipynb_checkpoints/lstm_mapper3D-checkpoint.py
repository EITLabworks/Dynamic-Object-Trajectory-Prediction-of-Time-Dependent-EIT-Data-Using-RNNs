import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    TimeDistributed,
    Reshape,
    Activation,
    LSTM,
)
from tensorflow.keras.models import Model


def lstm_mapper3D_model(input_shape=(4, 64,64, 1), output_shape=8):
    
    mapper_inputs = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(2, strides=(2, 2), kernel_size=(3, 3), activation="elu"))(mapper_inputs)
    
    x = TimeDistributed(Conv2D(2, strides=(2, 2), kernel_size=(3, 3), activation="elu"))(x)
   
    x = TimeDistributed(Flatten())(x)

    x = TimeDistributed(Dense(64, activation="elu"))(x)

    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(16, return_sequences=False)(x)

    mapper_output = Dense(output_shape)(x)
    
    model = Model(mapper_inputs, mapper_output, name="lstm_mapper")
    
    return model