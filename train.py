# train.py
# Arnav Ghosh
# 15th Nov. 2018

import models
import data_manipulation

from keras.callbacks import ModelCheckpoint

# ------- DIRECTORIES ------- #
CHKPT_DIR = "checkpoints"

# ------- CONSTANTS ------- #
NUM_EPOCHS = 30
NUM_BATCH_SIZE = 32

I_KERNEL_SIZES = [3, 5] #PARAM
M_KERNEL_SIZE = 3 #PARAM
INPUT_SHAPE = (256, 1) # length, channels PARAM

def train(chk_path, model_name, model, train_x, train_y, val_x, val_y):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']))

    checkpoint = ModelCheckpoint(chk_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(train_x, train_y, batch_size=NUM_BATCH_SIZE, epochs=NUM_EPOCHS,
              validation_data=(val_x, val_y), shuffle=True, callbacks=[checkpoint])

    # save final model
    model.save_weights("{0}_final.h5".format(model_name))

def train_segnet(train_x, train_y, val_x, val_y):
    for kernel_size in I_KERNEL_SIZES:
        seg_chkpth_filepath = os.path.join(CHKPT_DIR,
                                           "segnet-weights-"+str(kernel_size)+"-improvement-{epoch:02d}-{val_loss:.2f}.hdf5")
        segnet = models.create_seg_net(num_classes=2, input_shape=INPUT_SHAPE,
                                       init_kernel_size=kernel_size,
                                       mid_kernel_size=M_KERNEL_SIZE,
                                       init_num_filters=32)
        train(seg_chkpth_filepath, "segnet-"+str(kernel_size), segnet, train_x, train_y, val_x, val_y)

def train_unet(train_x, train_y, val_x, val_y):
    for kernel_size in I_KERNEL_SIZES:
        unet_chkpth_filepath =os.path.join(CHKPT_DIR,
                                           "unet-weights-"+str(kernel_size)+"-improvement-{epoch:02d}-{val_loss:.2f}.hdf5")
        unet = models.create_unet(num_classes=2,
                                  init_kernel_size=kernel_size,
                                  mid_kernel_size=M_KERNEL_SIZE,
                                  init_num_filters=32) #TODO
        train(unet_chkpth_filepath, "unet-"+str(kernel_size), unet, train_x, train_y, val_x, val_y)

def main():
    
    train_segnet(train_x, train_y, val_x, val_y)
