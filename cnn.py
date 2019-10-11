import numpy as np
import matplotlib.pyplot as plt
import mnist
import scipy.misc

X_train = mnist.train_images()
Y_train = mnist.train_labels()

scipy.misc.toimage(scipy.misc.imresize(X_train[0,:,:] * -1 + 256, 10.))

X_test = mnist.test_images()
Y_test = mnist.test_labels()





