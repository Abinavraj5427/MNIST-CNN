import numpy as np
import matplotlib.pyplot as plt
import mnist

def convolution(X, F, step_size):
    C = np.zeros((X.shape[0], X.shape[1]-F.shape[0]+1, X.shape[2]-F.shape[1]+1))
    for r in range(28- (F.shape[0]-1)):
        for c in range(28-(F.shape[1]-1)):
            C[:,r,c] = (X[:,r:r+F.shape[0],c:c+F.shape[1]] * F).sum()
    return C
            


#60000 samples
X_train = mnist.train_images()
Y_train = mnist.train_labels()

#10000 sample
X_test = mnist.test_images()
Y_test = mnist.test_labels()

X_train = X_train / 255
X_test = X_test / 255

F1 = np.random.rand(5,5)
#print(X_train[0:2].shape)
C1 = convolution(X_train[0:2], F1, 1)
#print(C1.shape)






