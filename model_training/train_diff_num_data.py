import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
# import matplotlib
# import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib
from model import Model
#from prediction_model import dataprep
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def tom_fit2mlp(df, data_cols_conv):
    df['batchsize']       = df['batchsize']
    df['elements_matrix'] = df['matsize'] ** 2
    df['elements_kernel'] = df['kernelsize'] ** 2
    df['channels_in']     = df['channels_in']
    df['channels_out']    = df['channels_out']
    df['strides']         = df['strides']
    df['padding']         = df['padding']
    df['act_None']        = df['activation_fct']
    df['use_bias']        = df['use_bias']
    df['opt_SGD']         = 0
    df['opt_Adadelta']    = 0
    df['opt_Adagrad']     = 0
    df['opt_Momentum']    = 0
    df['opt_None']        = 1
    df['opt_Adam']        = 0
    df['opt_RMSProp']     = 0
    df['act_relu']        = 0
    df['act_tanh']        = 0
    df['act_sigmoid']     = 0
    df['bandwidth']       = 616
    df['cores']           = 4352
    df['clock']           = 1545
    df['timeUsed_median'] = df['time_median']
    return df

def data_split(dataframe, data_cols, split_per = 0.2, use_target = 'timeUsed_median'):
    x_train, x_test, y_train, y_test = train_test_split(dataframe[data_cols], dataframe[use_target],
        test_size=split_per, random_state=None, shuffle=True)
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test  = x_test.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    x_train[use_target] = y_train
    x_test[use_target] = y_test
    x_test = x_test.sort_values(by=[use_target])
    x_test  = x_test.reset_index(drop=True)
    return x_train, x_test

def data_prep(df_train, df_test, data_cols, use_target = 'timeUsed_median'):
    scaler = StandardScaler()
    scaler.fit(df_train[data_cols])
    train_scale = scaler.transform(df_train[data_cols])
    test_scale = scaler.transform(df_test[data_cols])
    return train_scale, test_scale, df_train[use_target], df_test[use_target], scaler


def train_diff_num_data(data_path, predict_device, operanion_type, num_data, data_cols_conv, use_target, log_dir):
    data_path = os.path.join(data_path, predict_device, operanion_type)
    train_filename = os.path.join(data_path, 'train_data_%s.csv' % num_data)
    test_filename  = os.path.join(data_path, 'test_data_20000.csv')
    #get the data
    df_train = pd.read_csv(train_filename)
    df_test  = pd.read_csv(test_filename)
    df_train = tom_fit2mlp(df_train, data_cols_conv)
    df_test = tom_fit2mlp(df_test, data_cols_conv)

    train_x, test_x, train_y, test_y, scaler = data_prep(df_train, df_test, data_cols_conv, use_target = use_target)

    #model setting
    num_neurons = [32, 64, 128, 256]#[32,64,128,128]#
    lr_initial = 0.1#05#0.1
    lr_decay_step = 400
    tf.reset_default_graph()
    data_dim = train_x.shape[1]

    # tensor input
    inputs = tf.placeholder(tf.float32, shape=(None, data_dim), name='model_input')
    targets = tf.placeholder(tf.float32, shape=(None), name='model_targets')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    reg_constant = .00001
    dropout_rate = 0.2
    batch_size = 128
    epochs = 5000

    model_name = predict_device+'_'+operanion_type+'_'+num_data
    model = Model(inputs,targets,learning_rate,reg_constant,dropout_rate,
                num_neurons,lr_initial,lr_decay_step,batch_size,model_name,log_dir)

    model.prediction
    model.train_op

    model.train(train_x, train_y, test_x, test_y, epochs)




data_cols_conv = ['batchsize','elements_matrix','elements_kernel',
        'channels_in','channels_out','strides', 'padding', 'activation_fct', 'use_bias']
use_target = 'timeUsed_median'
data_path = '../data_generator/goldan_values/inference/'
predict_device = '2080ti'#'excluding_1080ti'
operanion_type = 'convolution'
log_dir = 'train_diff_num_data_log'

train_diff_num_data(data_path, predict_device, operanion_type, '10000', data_cols_conv, use_target, log_dir)
train_diff_num_data(data_path, predict_device, operanion_type, '20000', data_cols_conv, use_target, log_dir)
train_diff_num_data(data_path, predict_device, operanion_type, '30000', data_cols_conv, use_target, log_dir)
train_diff_num_data(data_path, predict_device, operanion_type, '40000', data_cols_conv, use_target, log_dir)
train_diff_num_data(data_path, predict_device, operanion_type, '50000', data_cols_conv, use_target, log_dir)
train_diff_num_data(data_path, predict_device, operanion_type, '60000', data_cols_conv, use_target, log_dir)
train_diff_num_data(data_path, predict_device, operanion_type, '70000', data_cols_conv, use_target, log_dir)
train_diff_num_data(data_path, predict_device, operanion_type, '80000', data_cols_conv, use_target, log_dir)





