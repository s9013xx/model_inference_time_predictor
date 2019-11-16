"""Benchmark convolution"""

import tensorflow as tf
import numpy as np


class convolution(object):
    """Class for gerenating the benchmark operations"""

    def __init__(self,
                 batchsize,
                 matsize,
                 kernelsize,
                 channels_in,
                 channels_out,
                 strides,
                 padding,
                 activation_fct,
                 use_bias,
                 devlist):
        """Initialize convolution

        Args:
            args: Input arguments
            devlist: List of GPUs / CPUs (list)
        """
        print('matsize: %d' % matsize)
        print('kernelsize: %d' % kernelsize)
        print('channels_in: %d' % channels_in)
        print('channels_out: %d' % channels_out)
        print('strides: %d' % strides)
        print('batchsize: %d' % batchsize)
        print('padding: %s' % padding)
        print('use_bias: %d' % use_bias)
        print('devlist: %s' % devlist)
        print('activation_fct: %s' % activation_fct)
        self.matsize = matsize
        self.kernelsize = kernelsize
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.strides = strides
        self.batchsize = batchsize
        self.padding = padding
        self.use_bias = use_bias
        self.devlist = devlist
        self.activation_fct = activation_fct


    def create_benchmark_op(self):
        """Create benchmark operation using tf.layer

        Returns:
            conv.op: Operation for convolution
            g: TensorFlow graph
        """

        # datatype = eval('tf.float%d' %(self.precision))
        act = eval(self.activation_fct)

        # tf.reset_default_graph()
        g = tf.Graph()
        run_metadata = tf.RunMetadata()
        with g.as_default():
            for dev in self.devlist:
                with tf.device(dev):
                    matA = tf.Variable(
                            tf.ones([
                                    self.batchsize,
                                    self.matsize,
                                    self.matsize,
                                    self.channels_in]))
                    conv = tf.layers.conv2d(
                            inputs=matA,
                            filters=self.channels_out,
                            kernel_size=[self.kernelsize,self.kernelsize],
                            strides=(self.strides, self.strides),
                            # strides=self.strides,
                            padding=self.padding,
                            activation = act,
                            use_bias=self.use_bias)

        # return conv.op, g
        return conv, g