# models.py
# Arnav Ghosh
# 15th Nov. 2018

from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Activation #Dense, MaxPooling2D, Conv2D, Flatten, Lambda, Input
from keras.models import Sequential, Model
from keras.activations import softmax

def create_seg_net(num_classes, init_kernel_size, mid_kernel_size, init_num_filters):
    model = models.Sequential()

    # Encoder
    # module 1
    model.add(Conv1D(init_num_filters, init_kernel_size, activation='relu', name="conv1"))
    model.add(MaxPooling1D(pool_length=2, border_mode='same', name='maxpool1'))

    # module 2
    model.add(Conv1D(init_num_filters * 2, mid_kernel_size, activation='relu', name="conv2"))
    model.add(MaxPooling1D(pool_length=2, border_mode='same', name='maxpool2'))

    # module 3
    model.add(Conv1D(init_num_filters * 4, mid_kernel_size, activation='relu', name="conv3"))
    model.add(MaxPooling1D(pool_length=2, border_mode='same', name='maxpool3'))

    # Decoder (3 modules because pooling size is assumed to be 2)
    # module 1
    model.add(UpSampling1D(size=2, name='upsample1'))
    model.add(Conv1D(init_num_filters * 4, mid_kernel_size, activation='relu', name="post_up1_conv"))

    # module 2
    model.add(UpSampling1D(size=2, name='upsample2'))
    model.add(Conv1D(init_num_filters * 2, mid_kernel_size, activation='relu', name="post_up2_conv"))

    # module 3
    model.add(UpSampling1D(size=2, name='upsample3'))
    model.add(Conv1D(init_num_filters, mid_kernel_size, activation='relu', name="post_up3_conv"))

    # Pixel Classification #TODO check if activation necessary (paper uses it)
    model.add(Conv1D(num_classes, 1, activation='relu', name="class_conv"))

    # TODO Reshape to to get probabilities across channel or apply softmax somehow
    # TODO how to add loss function
    # model.add(Activation(softmax()))

    return model

def create_unet(num_classes, init_kernel_size, mid_kernel_size, init_num_filters):
    #TODO
