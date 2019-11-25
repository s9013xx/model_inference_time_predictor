import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

parser = argparse.ArgumentParser('Train Model Parser')
# Benchmarks parameters
parser.add_argument('--conv', action="store_true", default=False, help='Train convolution model')
parser.add_argument('--fc', action="store_true", default=False, help='Train fully connected model')
parser.add_argument('--pool', action="store_true", default=False, help='Train pooling model')
# General parameters
parser.add_argument('--graph_file', type=str, default='', help='Model file to store results')
parser.add_argument('--device', type=str, default='', help='Device name as appearing in logfile')
# Training data parameters
parser.add_argument('--data_number', '-dn', default=100000, type=int, help='number of data')
parser.add_argument('--training_data_number', '-trdn', default=80000, type=int, help='number of training data')
parser.add_argument('--validation_data_number', '-vdn', default=10000, type=int, help='number of training data')
parser.add_argument('--test_data_number', '-tedn', default=10000, type=int, help='number of training data')
# Training parameters
parser.add_argument('--batch_size', '-b', default=4096, type=int, help='batch size of val and train')
parser.add_argument('--epochs', '-e', default=1000, type=int, help='epoch while training')
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='learning rate of dnn model')
parser.add_argument('--scheduler_step', '-sstep', default=1000, type=int, help='scheduler_step of dnn model')
parser.add_argument('--scheduler_gamma', '-sg', default=0.5, type=float, help='scheduler_gamma of dnn model')
parser.add_argument('--loss_function_name', '-lfn', default='rmse', choices=['mse', 'mae', 'rmse', 'tom_method'], help='loss function name while training')
parser.add_argument('--optimizer_method', '-opt', default= 'adam', choices=['adam', 'sgd'], help='optimizer method of dataset')

args = parser.parse_args()

if args.device == '':
    print('you should use --device parameter to specify collect data for which device, ex: --device 2080ti')
    exit()

input_dir = '../data_generator/goldan_values'

# conv_time_col = ['log_time_median']
conv_time_col = ['time_median']

def tensorflow_model_training(np_train_feature, np_train_time, np_validation_feature, np_validation_time, np_test_feature, np_test_time, feature_col, time_col, store_df_validation_feature):
    training_data_size = np_train_feature.shape[0]
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
    abs_value = tf.reduce_mean(tf.abs(prediction-ys))

    #step 4.2  choose the loss function
    if args.loss_function_name   == 'tom_method':
        loss_op = tf.reduce_mean(tf.reduce_sum(tf.abs(prediction-ys)/ys, reduction_indices=[1]))
    elif args.loss_function_name == 'rmse':
        loss_op = tf.sqrt(tf.losses.mean_squared_error(ys, prediction))
    elif args.loss_function_name == 'mse':
        loss_op = tf.losses.mean_squared_error(ys, prediction)
    elif args.loss_function_name == 'mae':
        loss_op = tf.losses.absolute_difference(ys, prediction)
    
    # step 4.2.2 choose the optimizer
    tf_lr = tf.placeholder(tf.float32, shape=[])
    lr = args.learning_rate
    if args.optimizer_method == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate=tf_lr).minimize(loss_op)
    else:
        learning_value = [ args.learning_rate * ((args.scheduler_gamma) ** i) for i in range(args.epochs //args.scheduler_step) ]
        boundaries     = [ args.scheduler_step*(i+1) for i in range(args.epochs // args.scheduler_step) ]
        train_op = tf.train.MomentumOptimizer(learning_rate=tf_lr, momentum=0.9).minimize(loss_op)
        print("boundaries", boundaries)
        print("lr ", learning_value)

    #step4.3 do train
    re_min = sys.float_info.max
    abs_min = sys.float_info.max
    b_count = 0

    log_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(args.epochs):
            total_loss = 0
            if args.optimizer_method == 'sgd':
                if (i+1) >= boundaries[b_count]:
                    lr = learning_value[b_count]
                    b_count += 1
                print("{} : =========>Now learning rate is {:.10f}".format(i,lr))
            for offset in range(0, training_data_size, args.batch_size):
                end = offset + args.batch_size
                if(end > training_data_size):
                    end = training_data_size
                # print('train : %d~%d'%(offset, end))
                _, loss_value, pre_y = sess.run([train_op, loss_op, prediction], 
                    feed_dict={xs: np_train_feature[offset:end], ys: np_train_time[offset:end], tf_lr: lr})
                total_loss += loss_value
            
            #step4.4 validate it 
            num_examples = np_validation_feature.shape[0]
            total_re = 0
            total_abse = 0
            count  = 0
            out_time = np.array([])

            for offset in range(0, num_examples, args.batch_size):
                end = offset + args.batch_size
                if(end > num_examples):
                    end = num_examples
                # print('val : %d~%d'%(offset, end))
                batch_x, batch_y = np_validation_feature[offset:end], np_validation_time[offset:end]
                pre_y = sess.run(prediction, feed_dict={xs: batch_x, ys: batch_y})
                if conv_time_col == ['log_time_median']:
                    ###################################################TBD
                    # choose your post process
                    ###################################################
                    pre_y = np.exp(np.array(pre_y))#(np.array(pre_y).astype(float) ** 2) / 100
                abs_error = np.abs(np_validation_time[offset:end] - pre_y) # calculate abs error 
                re_error  = abs_error / np_validation_time[offset:end]
                total_abse += np.sum(abs_error)
                total_re   += np.sum(re_error)
                out_time = np.append(out_time, pre_y)
            mean_re_error, mean_abse =  total_re / num_examples, total_abse / num_examples

            if re_min >= mean_re_error:
                re_min  = mean_re_error
                abs_min = mean_abse
                store_df_validation_feature['pre'] = out_time
                store_df_validation_feature.to_csv('output_%s.csv' % args.device, index=None)
                # store the mode
                # TBD !!!!
                graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['prediction/BiasAdd'])
                tf.io.write_graph(graph, '.', 'graph.pb', as_text=False)
            
            log_row_data = [i, total_loss, mean_re_error, mean_abse, re_min, abs_min]
            log_list.append(log_row_data)
            
            print("EPOCH {} ...".format(i+1))
            print("training loss: {:.10f}".format(total_loss / training_data_size))
            print("[val] mean relative error: {:.3f}, abs error: {:.3f} ** [min re error: {:.3f} , min abs errror {:.8f}]" \
                .format(mean_re_error, mean_abse, re_min, abs_min))
        
        np_log_array = np.array(log_list)
        result_col = ['epochs', 'total_loss', 'mean_re_error', 'mean_abse', 're_min', 'abs_min']
        log_data = pd.DataFrame(np_log_array, columns=result_col)
        log_data.to_csv('log_%s.csv' % args.device, index=False)


# ---divide---                                                                                                                                                                                                     
# (80000, 15)                                                                                                                                                                                                      
# (10000, 15)                                                                                                                                                                                                      
# (10000, 15)                                                                                                                                                                                                      
# ---filter---                                                                                                                                                                                                     
# (79987, 15)                                                                                                                                                                                                      
# (9998, 15)                                                                                                                                                                                                       
# (9999, 15)                                                                                                                                                                                                       
# ---feature---                                                                                                                                                                                                    
# (79987, 9)                                                                                                                                                                                                       
# (9998, 9)                                                                                                                                                                                                        
# (9999, 9)                                                                                                                                                                                                        
# ---time---                                                                                                                                                                                                       
# (79987, 1)                                                                                                                                                                                                       
# (9998, 1)                                                                                                                                                                                                        
# (9999, 1)  

# This funcion is divide total data to three parts (train, validate, test)
def data_divider(df_ori_data, train_start_index, train_end_index, validate_start_index, validate_end_index, test_start_index, test_end_index, feature_col, time_col):
    # 1. Divide total data to train, validation, test part
    df_train_data = df_ori_data.loc[train_start_index:train_end_index, :]
    df_validate_data = df_ori_data.loc[validate_start_index:validate_end_index, :]
    df_test_data = df_ori_data.loc[test_start_index:test_end_index, :]
    print('---divide---')
    print(df_train_data.shape)
    print(df_validate_data.shape)
    print(df_test_data.shape)
    # 2. Time invalid filter
    df_train_data = df_train_data[df_train_data['time_median']>0]
    df_validate_data = df_validate_data[df_validate_data['time_median']>0]
    df_test_data = df_test_data[df_test_data['time_median']>0]
    print('---filter---')
    print(df_train_data.shape)
    print(df_validate_data.shape)
    print(df_test_data.shape)
    # 3. Get features and translate to numpy format
    df_train_feature = df_train_data.loc[:, feature_col]
    df_validate_feature = df_validate_data.loc[:, feature_col]
    df_test_feature = df_test_data.loc[:, feature_col]
    np_train_feature = df_train_feature[feature_col].to_numpy().reshape(-1, len(feature_col))
    np_validate_feature = df_validate_feature[feature_col].to_numpy().reshape(-1, len(feature_col))
    np_test_feature = df_test_feature[feature_col].to_numpy().reshape(-1, len(feature_col))
    print('---feature---')
    print(df_train_feature.shape)
    print(df_validate_feature.shape)
    print(df_test_feature.shape)
    # 4. Get time and translate to numpy format
    df_train_time = df_train_data.loc[:, time_col]
    df_validate_time = df_validate_data.loc[:, time_col]
    df_test_time = df_test_data.loc[:, time_col]
    np_train_time = df_train_time[time_col].to_numpy().reshape(-1, len(time_col))
    np_validate_time = df_validate_time[time_col].to_numpy().reshape(-1, len(time_col))
    np_test_time = df_test_time[time_col].to_numpy().reshape(-1, len(time_col))
    print('---time---')
    print(df_train_time.shape)
    print(df_validate_time.shape)
    print(df_test_time.shape)

    # store_df_conv_validate_feature = df_validate_data[df_validate_data['time_median']>0]

    return np_train_feature, np_train_time, np_validate_feature, np_validate_time, np_test_feature, np_test_time, df_validate_data

def main(_):
    ########### Training convolution model ##########
    if args.conv:
        # step1 get the dataset 
        conv_feature_col = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias']
        golden_values_col = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 'padding', 'activation_fct', 'use_bias', 'time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']
        for file in glob.glob(os.path.join(input_dir, 'conv_goldan_values_%s_*.csv' % args.device)):
            df_conv_data = pd.read_csv(file, usecols=golden_values_col)
            df_conv_data['log_time_median'] = np.log(df_conv_data[['time_median']])

            np_conv_train_feature, np_conv_train_time               \
                , np_conv_validate_feature, np_conv_validate_time   \
                , np_conv_test_feature, np_conv_test_time           \
                , store_df_conv_validate_feature = data_divider(df_conv_data, 0, 79999, 80000, 89999, 90000, 99999, conv_feature_col, conv_time_col)

            tensorflow_model_training(np_conv_train_feature, np_conv_train_time \
                , np_conv_validate_feature, np_conv_validate_time               \
                , np_conv_test_feature, np_conv_test_time                       \
                , conv_feature_col, conv_time_col                               \
                , store_df_conv_validate_feature)
    
    ########### Training fully connected model ##########
    # if args.conv:

    ########### Training pooling model ##########
    # if args.conv:

        
if __name__ == '__main__':
    tf.app.run()
# print(np.log([0.012]))
# print(np.exp(np.log([0.012])))