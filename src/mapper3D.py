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
    Dropout
)
from tensorflow.keras.models import Model


def no_lstm_mapper3D_model(input_shape=(4, 64, 64, 1), output_shape=8):
    
    mapper_inputs = Input(shape=input_shape)
   
    x = TimeDistributed(Conv2D(2, strides=(2, 2), kernel_size=(3, 3), activation="elu"))(mapper_inputs)
    
   
    x = TimeDistributed(Conv2D(2, strides=(2, 2), kernel_size=(3, 3), activation="elu"))(x)
    
    x = TimeDistributed(Flatten())(x)
    
    x = TimeDistributed(Dense(64, activation="elu"))(x)
    
    x = Flatten()(x)
    
    x = Dense(32, activation="elu")(x)
    x = Dense(16, activation="elu")(x)
    
    mapper_output = Dense(output_shape)(x)
    
    model = Model(mapper_inputs, mapper_output, name="3D_no_lstm_mapper")
    
    return model
