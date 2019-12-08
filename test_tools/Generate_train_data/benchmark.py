"""Generates benchmarks for convolutions with different, randomly determined
parameters. Saves results into a pandas dataframe (.pkl)
and a numpy array (.npy)
"""
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import benchmark_conv
import run_benchmark


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


parser = argparse.ArgumentParser('Benchmarking convolutions')

# Benchmarks to perform
parser.add_argument('--testDense', action="store_true", default=False,
                    help='Benchmark fully connected layer/matrix multiplication')
parser.add_argument('--testConv', action="store_true", default=False,
                    help='Benchmark 2D convolution')
# General parameters
parser.add_argument('--backprop_ratio', type=float, default=0.5,
                    help='ratio of iterations with backward pass ([0..1])')
parser.add_argument('--num_gpu', type=int, default=1,
                    help='Number of GPUs to use')
parser.add_argument('--devlist', type=str, default='',
                    help='List of devices to use, overwrites num_gpu if set')
parser.add_argument('--num_val', type=int, default=10000,
                    help='Number of results to compute')
parser.add_argument('--logfile', type=str, default='',
                    help='Text file to store results')
parser.add_argument('--device', type=str, default='',
                    help='Device name as appearing in logfile')
parser.add_argument('--iter_benchmark', type=int, default=10,
                    help='Number of iterations for benchmark')
parser.add_argument('--iter_warmup', type=int, default=5,
                    help='Number of iterations for warm-up')
parser.add_argument('--repetitions', type=int, default=5,
                    help='Number of repetitions of the same experiment')

args = parser.parse_args()


def generate_devlist(devlist, num_gpu):
    """Creates list with devices

    Args:
        devlist: Comma separated list of devices, overwrites num_gpu
        num_gpu: Number of GPUs to be used

    Return:
        devlist: List of devices
        use_gpu: Whether GPUs are used (boolean)
    """
    if devlist=='':
        if num_gpu==0:
            devlist = ['/cpu:0']
            use_gpu = False
        else:
            devlist = ['/gpu:%d' %i for i in range(num_gpu)]
            use_gpu = True
    else:
        use_gpu = ('gpu' in devlist.lower())
        devlist = devlist.split(',')
    print(use_gpu, devlist)
    return devlist, use_gpu


def main(_):
    """Main function that runs all benchmarks"""

    print('start')
    devlist, use_gpu = generate_devlist(args.devlist, args.num_gpu)

    activation_list = [
            'None',
            'tf.nn.relu']


    ########### Benchmark convolution ##########
    if args.testConv:
        if args.logfile == '':
                logfile = str('./docker.csv')
        else:
            logfile = args.logfile

        daniel_col = ['activation_fct', 'batchsize', 'channels_in', 'channels_out', 'gpu', 'kernelsize', 'matsize', 'optimizer', 'padding', 'precision', 'strides', 'timeUsed_max', 'timeUsed_median', 'timeUsed_min', 'timeUsed_std', 'use_bias', 'docker_time_median']
        df = pd.read_pickle('./benchmark_convolution__20180924.pkl')
        df = df[(df['activation_fct'] < 2)]
        df = df.reset_index(drop=True)

        golden_values_data = pd.DataFrame(columns=daniel_col)
        for i in range(len(df.index)):

            print('========== conv', i , '==========')
            activation_fct = df.iloc[i, daniel_col.index('activation_fct')]
            batchsize = df.iloc[i, daniel_col.index('batchsize')]
            channels_in = df.iloc[i, daniel_col.index('channels_in')]
            channels_out = df.iloc[i, daniel_col.index('channels_out')]
            gpu = df.iloc[i, daniel_col.index('gpu')]
            kernelsize = df.iloc[i, daniel_col.index('kernelsize')]
            matsize = df.iloc[i, daniel_col.index('matsize')]
            optimizer = df.iloc[i, daniel_col.index('optimizer')]
            padding = df.iloc[i, daniel_col.index('padding')]
            precision = df.iloc[i, daniel_col.index('precision')]
            strides = df.iloc[i, daniel_col.index('strides')]
            timeUsed_max = df.iloc[i, daniel_col.index('timeUsed_max')]
            timeUsed_median = df.iloc[i, daniel_col.index('timeUsed_median')]
            timeUsed_min = df.iloc[i, daniel_col.index('timeUsed_min')]
            timeUsed_std = df.iloc[i, daniel_col.index('timeUsed_std')]
            if df.iloc[i, daniel_col.index('use_bias')] == True:
                use_bias = 1
            else: 
                use_bias = 0

            # batchsize, matsize, kernelsize, channels_in, channels_out, strides, padding, activation_fct, use_bias = 62, 471, 7, 19, 18, 1, 0, 0, 1

            gpu_index = np.arange(args.num_val)%(len(devlist))

            timeUsed = np.zeros([args.num_val,args.repetitions])

            tprint = time.time()
            
            conv = benchmark_conv.convolution(
                batchsize,
                matsize,
                kernelsize,
                channels_in,
                channels_out,
                strides,
                ('SAME' if padding==1 else 'VALID'),
                activation_list[activation_fct],
                use_bias,
                devlist)

            benchmark_op, benchmark_graph = conv.create_benchmark_op()

            # default iter_warmup is 5, iter_benchmark is 50
            bm_conv = run_benchmark.benchmark(
                    benchmark_op,
                    args.iter_warmup,
                    args.iter_benchmark,
                    benchmark_graph)

            try:
                timeUsed =  bm_conv.run_benchmark()
                print('timeUsed:', timeUsed)
            except:
                print('execption')
                print("Unexpected error:", sys.exc_info()[0])
                print("!!!!!")
                timeUsed = None

            conv_row_data = [activation_fct, batchsize, channels_in, channels_out, gpu, kernelsize, matsize, optimizer, padding, precision, strides, timeUsed_max, timeUsed_median, timeUsed_min, timeUsed_std, use_bias, timeUsed]
            golden_values_data.loc[0] = conv_row_data
            
            if i==0: 
                golden_values_data.to_csv(logfile, index=False)
            else:
                golden_values_data.to_csv(logfile, index=False, mode='a', header=False)

if __name__ == '__main__':
    tf.app.run()
