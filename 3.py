import numpy as np
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Tạo kernel 3x3 với các weight random, bias random
    # conv2d 1
kernel_conv1 = np.random.randn(3,3,1,8)
bias_conv1 = np.random.randn(8)
    # conv2d 2
kernel_conv2 = np.random.randn(3,3,8,16)
bias_conv2 = np.random.randn(16)
    # conv2d 3
kernel_conv3 = np.random.randn(3,3,16,16)
bias_conv3 = np.random.randn(16)
    # dense
kernel_dense=np.random.randn(144,10)
bias_dense=np.random.randn(10)

# Hàm conv2d
def conv2d(X, kernel_conv2d,bias_conv2d):
    feature_map = np.zeros((26,26))
    for x in range(26):
        for y in range(26):
            filter=X[x:x+3,y:y+3]
            feature_map[x,y]=np.sum(filter*kernel_conv2d)+bias_conv2d
    return feature_map

# print output qua hàm conv2d của ảnh đầu tiên
# X=X_train[0]
# print(conv2d(X,kernel_conv2d,bias_conv2d))

# Hàm relu
def relu(a):
    return(np.maximum(a,0))

# Hàm maxpooling
def maxpooling(a):
    maxpool=np.zeros((13,13))
    for i in range(13):
        for j in range(13):
            maxpool[i,j]=np.max(a[2*i:2*i+2,2*j:2*j+2])
    return maxpool

# Hàm flatten
def flatten(a):
    height = len(a)
    width = len(a[0])
    vector = []
    for i in range(height):
        for j in range(width):
            vector.append(a[i,j])
    return vector

# Hàm dense
def dense(a,kernel_dense,bias_dense,so):
    z = []
    b = len(a)
    for i in range(so):
        total=0
        for j in range(b):
            total+=a[j]*kernel_dense[j,i]
        z.append(total+bias_dense[i])
    return z

# Hàm dropout
def bernouli(p):
    m = np.random.rand()
    if m>=p:
        return 1
    else:
        return 0
def dropout(a,p):
    z=[]
    for i in a:
        z.append((i*bernouli(p))/(1-p))
    return z

# Hàm softmax:
def softmax(a):
    tong=0
    for i in a:
        tong += np.exp(i-max(a))
    xs = []
    for i in a:
        xs.append(np.exp(i-max(a))/tong)
    return xs

