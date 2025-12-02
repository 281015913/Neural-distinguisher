import speck as sp
import tensorflow as tf
import numpy as np
import random
import os
import csv
import keras
from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation,concatenate
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LSTM
from tensorflow.keras.layers import LSTM, Dropout
#1111231为
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 使用第二块GPU（从0开始）

physical_gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)]
)
logical_gpus = tf.config.list_logical_devices("GPU")
"""

def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first
    # 设置全局随机数种子
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    # 设置GPU计算顺序和确定性
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 设置操作级别的随机数种子
    rng = tf.random.Generator.from_seed(SEED)

    # 设置keras初始化器、正则化器、约束器等的种子
    initializer = keras.initializers.GlorotUniform(seed=SEED)
    regularizer = keras.regularizers.l2(0.01)
    constraint = keras.constraints.MaxNorm(max_value=2, axis=0)


setup_seed(42)



bs = 5000;
wdir = './freshly_trained_nets/'

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);

#make residual tower of convolutional blocks
def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=1,depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_blocks * word_size * 2,));
  rs = Reshape((2 * num_blocks, word_size))(inp);
  perm = Permute((2,1))(rs);
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  conv0 = BatchNormalization()(conv0);
  conv0 = Activation('gelu')(conv0)
  #add residual blocks
  shortcut = conv0;
  for i in range(depth):
    conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    conv1 = BatchNormalization()(conv1);
    conv1 = Activation('gelu')(conv1);
    R21 = Add()([shortcut, conv1]);
    conv2 = Conv1D(num_filters, kernel_size=(ks+2), padding='same',kernel_regularizer=l2(reg_param))(R21);
    conv2 = BatchNormalization()(conv2);
    conv2 = Activation('gelu')(conv2);
    R22 = Add()([shortcut, conv1,conv2]);
    conv3 = Conv1D(num_filters, kernel_size=(ks+4), padding='same',kernel_regularizer=l2(reg_param))(R22);
    conv3 = BatchNormalization()(conv3);
    conv3 = Activation('gelu')(conv3);
    R23 = Add()([shortcut, conv1,conv2,conv3]);
    conv4 = Conv1D(num_filters, kernel_size=(ks+6), padding='same',kernel_regularizer=l2(reg_param))(R23);
    conv4 = BatchNormalization()(conv4);
    conv4 = Activation('gelu')(conv4);
    R24 = Add()([shortcut, conv1,conv2,conv3,conv4]);
    conv5 = Conv1D(num_filters, kernel_size=(ks+8), padding='same',kernel_regularizer=l2(reg_param))(R24);
    conv5 = BatchNormalization()(conv5);
    conv5 = Activation('gelu')(conv5);
    shortcut = Add()([shortcut, conv5]); 
    ks +=2;
  #add prediction head
  flat1 = Flatten()(shortcut);
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
  dense1 = BatchNormalization()(dense1);
  dense1 = Activation('gelu')(dense1);
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
  dense2 = BatchNormalization()(dense2);
  dense2 = Activation('gelu')(dense2);
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
  model = Model(inputs=inp, outputs=out);
  model.summary()
  return(model);

def train_speck_distinguisher(num_epochs, num_rounds=7, depth=1):
    #create the network
    net = make_resnet(depth=depth, reg_param=10**-5);
    net.compile(optimizer='adam',loss='mse',metrics=['acc']);
    #generate training and validation data
    X, Y = sp.make_train_data(10**7,num_rounds);
    X_eval, Y_eval = sp.make_train_data(10**6, num_rounds);
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    net.save('./saved_model/' + str(num_rounds)+ 'DV3_sc_distinguisher.h5')
    return(net, h);
