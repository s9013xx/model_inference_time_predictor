from keras.utils import plot_model
import re
import pandas as pd
import argparse
from keras.datasets import mnist

parser = argparse.ArgumentParser('Create Model Parser')
parser.add_argument('--model', type=str, default='', help='Device name as appearing in logfile')
args = parser.parse_args()
batches = [1, 2, 4, 8, 16, 32, 64, 128]


#load the MNIST dataset from keras datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if args.model == 'vgg16':
    from keras.applications.vgg16 import VGG16
    model = VGG16()
elif args.model == 'vgg19':
    from keras.applications.vgg19 import VGG19
    model = VGG19()
elif args.model == 'resnet50': 
    from keras.applications.resnet50 import ResNet50
    model = ResNet50()
elif args.model == 'inceptionv3':
    from keras.applications.inception_v3 import InceptionV3
    model = InceptionV3()
elif args.model == 'lenet':
    from keras.models import Sequential
    from keras.layers import Dense,Flatten
    from keras.layers.convolutional import Conv2D,MaxPooling2D



    model = Sequential()
    model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    LetNet_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])

	model.evaluate(test_x,test_y,batch_size=20,verbose=2)
elif args.model =='alexnet':
    from keras.models import Sequential
    from keras.layers import Dense,Flatten,Dropout
    from keras.layers.convolutional import Conv2D,MaxPooling2D
    model = Sequential()
    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='softmax'))