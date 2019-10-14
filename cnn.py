import numpy as np
import matplotlib.pyplot as plt
import mnist

def convolution(X, F, step_size):
    return X


#60000 samples
X_train = mnist.train_images()
Y_train = mnist.train_labels()

#10000 sample
X_test = mnist.test_images()
Y_test = mnist.test_labels()

X_train = X_train / 255
X_test = X_test / 255

F1 = np.random.rand(5,5)

convolution(X_train[0], F1, 1)
print(F1)






