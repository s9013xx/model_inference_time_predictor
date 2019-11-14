
import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

parser = argparse.ArgumentParser('Train Model Parser')
# Benchmarks parameters
parser.add_argument('--conv', action="store_true", default=False, help='Train convolution model')
parser.add_argument('--fc', action="store_true", default=False, help='Train fully connected model')
parser.add_argument('--pool', action="store_true", default=False, help='Train pooling model')
# General parameters
parser.add_argument('--graph_file', type=str, default='', help='Model file to store results')
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')

args = parser.parse_args()

def main(_):
    ########### Training convolution model ##########
    if args.conv:
        # step1 get the dataset 
        df_conv_data = pd.read_csv('./data_generator/goldan_values/fc_goldan_values_%s.csv' % args.device)

    ########### Training fully connected model ##########
    if args.conv:

    ########### Training pooling model ##########
    if args.conv:

if __name__ == '__main__':
    tf.app.run()

import os 
import sys
import time
import shutil
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

col_name    = ['image size', 'channel', 'kernel_size', 'batch', 'units', 'strides', 'time']
feature_col = ['image size', 'channel', 'kernel_size', 'batch', 'units', 'strides']
#col_name    = ['image size', 'channel', 'kernel_size', 'batch', 'units', 'strides', 'i3', 'i5', 'i7', 'time']
#feature_col = ['image size', 'channel', 'kernel_size', 'batch', 'units', 'strides', 'i3', 'i5', 'i7']

def read_flags():
    parser = argparse.ArgumentParser(description='Tom research')
    #Basic setting
    parser.add_argument('--data_path', '-dp', default=os.path.join(os.getcwd(),  'csv', 'inf', 'i7'), 
                        type=str, help='data path of the dataset')
    parser.add_argument('--training_data', '-trd', default='conv_train_all_cpu.csv', 
                        type=str, help='training filename of the dataset')
    parser.add_argument('--validation_data', '-vald', default='conv_test_all_cpu.csv', 
                        type=str, help='validation filename of the dataset')        
    parser.add_argument('--batch_size', '-b', default=1024, 
                        type=int, help='batch size of val and train')  
    parser.add_argument('--epochs', '-e', default=1, 
                        type=int, help='epoch while training')
    parser.add_argument('--learning_rate', '-lr', default=0.001, 
                        type=float, help='learning rate of dnn model')
    parser.add_argument('--scheduler_step', '-sstep', default=1000, 
                        type=int, help='scheduler_step of dnn model')
    parser.add_argument('--scheduler_gamma', '-sg', default=0.5, 
                        type=float, help='scheduler_gamma of dnn model')  
    
    parser.add_argument('--loss_function_name', '-lfn', default='tom_method', 
                        choices=['mse', 'mae', 'rmse', 'tom_method'], help='loss function name while training')
    parser.add_argument('--training_method', '-tm', default= 'dnn',
                        choices=['reg', 'dnn'], help='training method of dataset')
    parser.add_argument('--optimizer_method', '-opt', default= 'adam',
                        choices=['adam', 'sgd'], help='optimizer method of dataset')
    parser.add_argument('--reg_poly', '-rp', default= 5,
                       type=int, help='regression poly size while train')  

    parser.add_argument('--out_graphname', '-ogn', default= "",
                       type=str, help='name of output graph')  
    parser.add_argument('--out_csvname', '-ocn', default= "",
                       type=str, help='name of output csv')  

    args = parser.parse_args()
    args.training_data = os.path.join(args.data_path, args.training_data)
    args.validation_data = os.path.join(args.data_path, args.validation_data)
    if not args.out_graphname:
        args.out_graphname = os.path.join(os.getcwd(), 'graph.pb')
    if not args.out_csvname:
        args.out_csvname = os.path.join(os.getcwd(), 'output.csv')
    return args

def do_preprocess(df, new_time_name = 'new_time', new_norm_name = 'new_time_norm'):
    ###################################################TBD
    # choose your pre process
    ###################################################
    df[new_time_name] = np.log(df[['time']]) #(df[['time']] * 100 ) ** (1/2)
    time_min = df[new_time_name].min()
    time_max = df[new_time_name].max()
    df[new_norm_name] = ((df[new_time_name] - time_min) / (time_max - time_min))*10
    return 

def define_ft(df_train, df_val, fc = feature_col, prec = ['new_time']):
    feature_num = len(feature_col)
    tr_np_feature = df_train[fc].to_numpy().reshape(-1, feature_num)
    tr_np_time    = df_train[prec].to_numpy().reshape(-1, 1)

    val_np_feature = df_val[fc].to_numpy().reshape(-1, feature_num)
    val_np_time    = df_val[['time']].to_numpy().reshape(-1, 1)
    return tr_np_feature, tr_np_time, val_np_feature, val_np_time
   
def train_val_regression(df_train, df_val, tr_np_feature, tr_np_time, val_np_feature, val_np_time, poly = 5, prec = ['new_time'], csv_name = "output.csv"):
    scaler = StandardScaler()
    scaler.fit(tr_np_feature)
    tr_np_feature_scale  = scaler.transform(tr_np_feature) # do feature processing
    val_np_feature_scale = scaler.transform(val_np_feature) # do feature processing
    
    poly = PolynomialFeatures(poly) # poly regression extension
    tr_np_feature_scale_new  =  poly.fit_transform(tr_np_feature_scale)
    val_np_feature_scale_new =  poly.fit_transform(val_np_feature_scale)

    reg = LinearRegression().fit(tr_np_feature_scale_new, tr_np_time)# training
    
    pre_val_time = reg.predict(val_np_feature_scale_new)# validation
    if prec == ['new_time']:
        pre_real_time =  np.exp(pre_val_time)#(pre_val_time **2) / 100   #process_back to real elasped time 
    real_time = df_val[['time']].to_numpy().reshape(-1, 1)
    print("pre is ", pre_real_time) # print predition time 
    print("real is ", real_time)# print the real answer time 
    
    abs_error = np.abs(real_time - pre_real_time) # calculate abs error 
    re_error = abs_error / real_time # calculate re error
    mean_abs_error = np.mean(abs_error)
    mean_re_error  = np.mean(re_error)
    print("[val] mean re error: {:.4f} , mean abs error: {:.8f}".format(mean_re_error, mean_abs_error)) #print result
    df_val['pre'] = pre_real_time
    df_val.to_csv(csv_name, index=None)

if __name__ == '__main__':
    flags = read_flags()

    # step1 get the dataset 
    df_train = pd.read_csv(flags.training_data)
    df_val = pd.read_csv(flags.validation_data)

    # step2 do preporcess
    do_preprocess(df_train)
    do_preprocess(df_val)
    print(df_train.shape, df_val.shape)
    
    # step3 define the features and target 
    prec = ['new_time']
    tr_np_feature, tr_np_time, val_np_feature, val_np_time = define_ft(df_train, df_val, prec = prec)
    training_data_size = tr_np_feature.shape[0]
    device = 'cuda'
    # step4 train the data
    if flags.training_method == 'dnn':
        #step 4.1 create tf model
       #step 4.1 create tf model
        xs = tf.placeholder(tf.float32, [None, len(feature_col)], name='feature')
        ys = tf.placeholder(tf.float32, [None, 1])
        xs_nom = tf.layers.batch_normalization(xs)     
        l1 = tf.layers.dense(inputs=xs_nom, units=1024, activation=tf.nn.relu)
        l1_nom = tf.layers.batch_normalization(l1)
        l2 = tf.layers.dense(inputs=l1_nom, units=1024, activation=None)
        l2_nom = tf.layers.batch_normalization(l2)
        l3 = tf.layers.dense(inputs=l2_nom, units=1024, activation=None)
        l3_nom = tf.layers.batch_normalization(l3)
        prediction = tf.layers.dense(inputs=l3_nom, units=1, activation=None, name="prediction")
        accuracy_operation = tf.reduce_mean(tf.abs(prediction-ys)/ys)
        abs_haha = tf.reduce_mean(tf.abs(prediction-ys))

        #step 4.2  choose the loss function
        if flags.loss_function_name   == 'tom_method':
            loss_op = tf.reduce_mean(tf.reduce_sum(tf.abs(prediction-ys)/ys, reduction_indices=[1]))
        elif flags.loss_function_name == 'rmse':
            loss_op = tf.sqrt(tf.losses.mean_squared_error(ys, prediction))
        elif flags.loss_function_name == 'mse':
            loss_op = tf.losses.mean_squared_error(ys, prediction)
        elif flags.loss_function_name == 'mae':
            loss_op = tf.losses.absolute_difference(ys, prediction)
        
        # step 4.2.2 choose the optimizer
        tf_lr = tf.placeholder(tf.float32, shape=[])
        lr = flags.learning_rate
        if flags.optimizer_method == 'adam':
            train_op = tf.train.AdamOptimizer(learning_rate=tf_lr).minimize(loss_op)
        else:
            learning_value = [ flags.learning_rate * ((flags.scheduler_gamma) ** i) for i in range(flags.epochs //flags.scheduler_step) ]
            boundaries     = [ flags.scheduler_step*(i+1) for i in range(flags.epochs // flags.scheduler_step) ]
            train_op = tf.train.MomentumOptimizer(learning_rate=tf_lr, momentum=0.9).minimize(loss_op)
            print("boundaries", boundaries)
            print("lr ", learning_value)

        #step4.3 do train
        re_min = sys.float_info.max
        abs_min = sys.float_info.max
        b_count = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(flags.epochs):
                total_loss = 0 
                if flags.optimizer_method == 'sgd':
                    if (i+1) >= boundaries[b_count]:
                        lr = learning_value[b_count]
                        b_count += 1
                    #print("{} : =========>Now learning rate is {:.10f}".format(i,lr))
                for offset in range(0, training_data_size, flags.batch_size):
                    end = offset + flags.batch_size
                    _, loss_value, pre_y = sess.run([train_op, loss_op, prediction], 
                        feed_dict={xs: tr_np_feature[offset:end], ys: tr_np_time[offset:end], tf_lr: lr})
                    total_loss += loss_value
                
                #step4.4 validate it 
                num_examples = val_np_feature.shape[0]
                total_re = 0
                total_abse = 0
                count  = 0
                out_time = np.array([])

                for offset in range(0, num_examples, flags.batch_size):
                    batch_x, batch_y = val_np_feature[offset:offset+flags.batch_size], val_np_time[offset:offset+flags.batch_size]
                    pre_y = sess.run(prediction, feed_dict={xs: batch_x, ys: batch_y})
                    if prec == ['new_time']:
                        ###################################################TBD
                        # choose your post process
                        ###################################################
                        pre_y = np.exp(np.array(pre_y))#(np.array(pre_y).astype(float) ** 2) / 100
                    abs_error = np.abs(val_np_time[offset:offset+flags.batch_size] - pre_y) # calculate abs error 
                    re_error  = abs_error / val_np_time[offset:offset+flags.batch_size]
                    total_abse += np.sum(abs_error)
                    total_re   += np.sum(re_error)
                    out_time = np.append(out_time, pre_y)
                mean_re_error, mean_abse =  total_re / num_examples, total_abse / num_examples

                # update the min error
                if re_min >= mean_re_error:
                    re_min  = mean_re_error
                    abs_min = mean_abse
                    df_val['pre'] = out_time
                    df_val.to_csv(flags.out_csvname, index=None)
                    # store the mode
                    # TBD !!!!
                    graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['prediction/BiasAdd'])
                    tf.io.write_graph(graph, '.', flags.out_graphname, as_text=False)

                print("EPOCH {} ...".format(i+1))
                print("training loss: {:.10f}".format(total_loss / training_data_size))
                print("[val] mean relative error: {:.3f}, abs error: {:.3f} ** [min re error: {:.3f} , min abs errror {:.8f}]" \
                    .format(mean_re_error, mean_abse, re_min, abs_min))

    elif flags.training_method == 'reg':
        train_val_regression(df_train, df_val, tr_np_feature, tr_np_time, val_np_feature, val_np_time, 
        poly = flags.reg_poly, prec = prec, csv_name = flags.csv_name)
