import os 

os.system("python actual_inference.py --device i3 --cpu --model vgg16")
os.system("python actual_inference.py --device i3 --cpu --model vgg19")
os.system("python actual_inference.py --device i3 --cpu --model resnet50")
os.system("python actual_inference.py --device i3 --cpu --model inceptionv3")
os.system("python actual_inference.py --device i3 --cpu --model lenet")
os.system("python actual_inference.py --device i3 --cpu --model alexnet")
