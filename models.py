# models.py
# Arnav Ghosh
# 15th Nov. 2018

from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Activation, Reshape, Input
from keras.models import Sequential, Model
from keras.activations import softmax

#TODO change to 2D instead of 1D?
def create_seg_net(num_classes, input_shape, init_kernel_size, mid_kernel_size, init_num_filters):
    model = Sequential()

    # Encoder
    # module 1
    model.add(Conv1D(init_num_filters, init_kernel_size, activation='relu', border_mode='same', name="conv1", input_shape=input_shape))
    model.add(MaxPooling1D(pool_length=2, border_mode='same', name='maxpool1'))

    # module 2
    model.add(Conv1D(init_num_filters * 2, mid_kernel_size, activation='relu', border_mode='same', name="conv2"))
    model.add(MaxPooling1D(pool_length=2, border_mode='same', name='maxpool2'))

    # module 3
    model.add(Conv1D(init_num_filters * 4, mid_kernel_size, activation='relu', border_mode='same', name="conv3"))
    model.add(MaxPooling1D(pool_length=2, border_mode='same', name='maxpool3'))

    # Decoder (3 modules because pooling size is assumed to be 2)
    # module 1
    model.add(UpSampling1D(size=2, name='upsample1'))
    model.add(Conv1D(init_num_filters * 4, mid_kernel_size, activation='relu', border_mode='same', name="post_up1_conv"))

    # module 2
    model.add(UpSampling1D(size=2, name='upsample2'))
    model.add(Conv1D(init_num_filters * 2, mid_kernel_size, activation='relu', border_mode='same', name="post_up2_conv"))

    # module 3
    model.add(UpSampling1D(size=2, name='upsample3'))
    model.add(Conv1D(init_num_filters, mid_kernel_size, activation='relu', border_mode='same', name="post_up3_conv"))

    # Pixel Classification #TODO check if activation necessary (paper uses it)
    model.add(Conv1D(num_classes, 1, activation='relu',  border_mode='same', name="class_conv"))

    # TODO CHECK ALL AFTER THIS
    # Reshape in prep for softmax
    model_width = model.output_shape[-2]
    model.add(Reshape((num_classes, model_width)))

    # Softmax --> for each chennel (which corr. to each input)
    model.add(Activation('softmax'))
    model.add(Reshape((model_width, num_classes)))

    # Note: Choosing not to choose argmin here because doesn't really provide too much ground for training
    #       if we wanted to provide an output with 1 channel, then use sigmoid and compare to binary_entropy
    #       Instead, here the output is two channels for each 'pixel' where each channel represents the probaility
    #       of havin the ith class. Use categorical_crossentropy as the loss function with one hot vectors at each step

    return model

# def create_seg_net(num_classes, input_shape, init_kernel_size, mid_kernel_size, init_num_filters):
#     #TODO
