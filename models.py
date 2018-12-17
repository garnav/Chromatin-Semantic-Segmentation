# models.py
# Arnav Ghosh
# 15th Nov. 2018

from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Activation, Reshape, Input, Permute, BatchNormalization, concatenate
from keras.models import Sequential, Model
from keras.activations import softmax

def create_seg_net(num_classes, input_shape, init_kernel_size, mid_kernel_size, init_num_filters):
    model = Sequential()

    # Encoder
    # module 1
    model.add(Conv1D(init_num_filters, init_kernel_size, border_mode='same', name="conv1", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2, border_mode='same', name='maxpool1'))

    # module 2
    model.add(Conv1D(init_num_filters * 2, mid_kernel_size, border_mode='same', name="conv2"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2, border_mode='same', name='maxpool2'))

    # module 3
    model.add(Conv1D(init_num_filters * 4, mid_kernel_size, border_mode='same', name="conv3"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2, border_mode='same', name='maxpool3'))

    model.add(Conv1D(init_num_filters * 8, mid_kernel_size, border_mode='same', name="conv4"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Decoder (3 modules because pooling size is assumed to be 2)
    # module 1
    model.add(UpSampling1D(size=2, name='upsample1'))
    model.add(Conv1D(init_num_filters * 4, mid_kernel_size, border_mode='same', name="post_up1_conv"))
    model.add(BatchNormalization())

    # module 2
    model.add(UpSampling1D(size=2, name='upsample2'))
    model.add(Conv1D(init_num_filters * 2, mid_kernel_size, border_mode='same', name="post_up2_conv"))
    model.add(BatchNormalization())

    # module 3
    model.add(UpSampling1D(size=2, name='upsample3'))
    model.add(Conv1D(init_num_filters, mid_kernel_size, border_mode='same', name="post_up3_conv"))
    model.add(BatchNormalization())

    # Pixel Classification #TODO check if activation necessary (paper uses it)
    model.add(Conv1D(num_classes, 1, border_mode='same', name="class_conv"))

    # TODO CHECK ALL AFTER THIS
    # Reshape in prep for softmax
    model_width = model.output_shape[-2]
    model.add(Reshape((num_classes, model_width)))
    model.add(Permute((2, 1)))
    # Softmax --> for each chennel (which corr. to each input)
    model.add(Activation('softmax'))
    model.add(Reshape((model_width, num_classes)))

    # Note: Choosing not to choose argmin here because doesn't really provide too much ground for training
    #       if we wanted to provide an output with 1 channel, then use sigmoid and compare to binary_entropy
    #       Instead, here the output is two channels for each 'pixel' where each channel represents the probaility
    #       of havin the ith class. Use categorical_crossentropy as the loss function with one hot vectors at each step

    return model

def create_unet(num_classes, input_shape, init_kernel_size, mid_kernel_size, init_num_filters):
    # Use functional API because of difficulty in using Sequential() for this

    # Encoder as with Segnet
    # module 1
    input = Input(shape=input_shape)
    conv1 = Conv1D(init_num_filters, init_kernel_size, border_mode='same', name="conv1")(input)
    bn1 = BatchNormalization()(conv1)
    ac1 = Activation('relu')(bn1)
    mp1 = MaxPooling1D(pool_length=2, border_mode='same', name='maxpool1')(ac1)

    # module 2
    conv2 = Conv1D(init_num_filters * 2, mid_kernel_size, border_mode='same', name="conv2")(mp1)
    bn2 = BatchNormalization()(conv2)
    ac2 = Activation('relu')(bn2)
    mp2 = MaxPooling1D(pool_length=2, border_mode='same', name='maxpool2')(ac2)

    # module 3
    conv3 = Conv1D(init_num_filters * 4, mid_kernel_size, border_mode='same', name="conv3")(mp2)
    bn3 = BatchNormalization()(conv3)
    ac3 = Activation('relu')(bn3)
    mp3 = MaxPooling1D(pool_length=2, border_mode='same', name='maxpool3')(ac3)

    conv4 = Conv1D(init_num_filters * 8, mid_kernel_size, border_mode='same', name="conv4")(mp3)
    bn4 = BatchNormalization()(conv4)
    ac4 = Activation('relu')(bn4)

    # Decoder (as with segnet) with merged layers
    # module 1
    merge1 = concatenate([UpSampling1D(size=2, name='upsample1')(ac4), conv3])
    post_up1_conv = Conv1D(init_num_filters * 4, mid_kernel_size, border_mode='same', name="post_up1_conv")(merge1)
    bn5 = BatchNormalization()(post_up1_conv)

    # module 2
    merge2 = concatenate([UpSampling1D(size=2, name='upsample2')(bn5), conv2])
    post_up2_conv = Conv1D(init_num_filters * 2, mid_kernel_size, border_mode='same', name="post_up2_conv")(merge2)
    bn6 = BatchNormalization()(post_up2_conv)

    # module 3
    merge3 = concatenate([UpSampling1D(size=2, name='upsample3')(bn6), conv1])
    post_up3_conv = Conv1D(init_num_filters, mid_kernel_size, border_mode='same', name="post_up3_conv")(merge3)
    bn7 = BatchNormalization()(post_up3_conv)

    # Pixel Classification #TODO check if activation necessary (paper uses it)
    class_conv = Conv1D(num_classes, 1, border_mode='same', name="class_conv")(bn7)

    # TODO CHECK ALL AFTER THIS
    # Reshape in prep for softmax
    model_width = input_shape[0] #should be the same as width
    r1 = Reshape((num_classes, model_width))(class_conv)
    p1 = Permute((2, 1))(r1)
    # Softmax --> for each chennel (which corr. to each input)
    final_ac = Activation('softmax')(p1)
    result = Reshape((model_width, num_classes))(final_ac)

    return Model(inputs=input, outputs=result)
