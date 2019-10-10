import numpy as np
import matplotlib.pyplot as plt
from mnist.loader import MNIST

mnist = MNIST('./dataset/MNIST')
x_train, y_train = mnist.load_training() #60000 samples
x_test, y_test = mnist.load_testing()    #10000 samples
