"""Generates benchmarks for convolutions with different, randomly determined
parameters. Saves results into a pandas dataframe (.pkl)
and a numpy array (.npy)
"""
import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import random
import numpy
from scipy import stats

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

parser = argparse.ArgumentParser('Collect Actual Data Parser')

# General parameters
parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')

args = parser.parse_args()

def run_bench_mark():
    tf.reset_default_graph()
    image = tf.Variable(tf.random_normal([16, 1024, 1024, 3]))
    op = tf.layers.conv2d(image, filters=64, kernel_size=[8, 8], strides=(1, 1), padding='VALID')
    sess = tf.Session()
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # Warm-up run
    for _ in range(args.iter_warmup):
        sess.run(op)
    # Benchmark run
    start_time = time.time()
    for _ in range(args.iter_benchmark):
        sess.run(op)
    
    return ((time.time()-start_time)/(args.iter_benchmark))*1000

def main(_):
    
    gpu_time = run_bench_mark()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    cpu_time = run_bench_mark()

    print('gpu_time:', gpu_time)
    print('cpu_time:', cpu_time)

if __name__ == '__main__':
    tf.app.run()
