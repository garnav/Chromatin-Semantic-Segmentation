# train.py
# Arnav Ghosh
# 15th Nov. 2018

import models
import data_manipulation

from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import numpy as np
import os
import tensorflow as tf

# ------- DIRECTORIES ------- #
CHKPT_DIR = "checkpoints"
DATA_DIR = "data"
SEGNET_JSON = "segnet-17.json"

# ------- CONSTANTS ------- #
NUM_EPOCHS = 30
NUM_BATCH_SIZE = 32

I_KERNEL_SIZES = [3, 5] #PARAM
M_KERNEL_SIZE = 3 #PARAM
INPUT_SHAPE = (64, 1) # length, channels PARAM

def train(chk_path, model_name, model, train_x, train_y, val_x, val_y):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

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
        write_model(segnet, SEGNET_JSON)
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

def write_model(model, name):
    model_json = model.to_json()
    with open(name, "w") as json_file:
        json_file.write(model_json)

#### DEBUGGING ####
def test_model():
    with open(SEGNET_JSON, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("segnet-5_final.h5")

    return loaded_model

def load_datasets():
    val_x = np.load(os.path.join(DATA_DIR, "val_x.npy"))
    val_y = np.load(os.path.join(DATA_DIR, "val_y.npy"))

    return val_x, val_y

def check_gradients(loaded_model):
    val_x, val_y = load_datasets()
    #loaded_model = test_model()

    loaded_model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
    out = loaded_model.output
    lstVar = loaded_model.trainable_weights
    gradients = K.gradients(out, lstVar)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    evaluate_gradients = sess.run(gradients, feed_dict={model.input:val_x[0:1, :, :]})

    return evaluate_gradients

def main():
    train_x = np.load(os.path.join(DATA_DIR, "train_x.npy"))
    train_y = np.load(os.path.join(DATA_DIR, "train_y.npy"))
    val_x = np.load(os.path.join(DATA_DIR, "val_x.npy"))
    val_y = np.load(os.path.join(DATA_DIR, "val_y.npy"))
    train_segnet(train_x, train_y, val_x, val_y)

if __name__ == '__main__':
    main()
