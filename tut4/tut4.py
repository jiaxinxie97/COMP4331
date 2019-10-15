import numpy as np
import pickle, gzip
import matplotlib.pyplot as plt
import time 


#MLP
# Useful functions 
def initalize_weights_relu(size_layer, size_next_layer):
    np.random.seed(5)
    # Method presented in "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classfication"
    # He et Al. 2015
    epsilon = np.sqrt(2.0 / (size_layer * size_next_layer) )
    # Weigts from Normal distribution mean = 0, std = epsion
    w = epsilon * (np.random.randn(size_next_layer, size_layer))
    return w.transpose()
def load_mnist():
    # Import MNIST data
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    # Training data, only
    X = valid_set[0]
    y = valid_set[1]

    # change y [1D] to Y [2D] sparse array coding class
    n_examples = len(y)
    labels = np.unique(y)
    Y = np.zeros((n_examples, len(labels)))
    for ix_label in range(len(labels)):
        # Find examples with with a Label = lables(ix_label)
        ix_tmp = np.where(y == labels[ix_label])[0]
        Y[ix_tmp, ix_label] = 1

    return X, Y, labels, y
# Training with 400 epochs
epochs = 400
loss = np.zeros([epochs,1])
# Load data
X, Y, labels, y = load_mnist()
tic = time.time()
# size_layers = [784, 100, 10]

# Randomly initialize weights
w1 = initalize_weights_relu(784, 100)
w2 = initalize_weights_relu(100, 10)

for ix in range(epochs):
    n_examples = X.shape[0]
    # Forward pass: compute y_hat    
    a1 = X
    z2 = a1.dot(w1)
    a2 = np.maximum(z2, 0)
    z3 = a2.dot(w2)
    a3 = np.maximum(z3, 0)
    Y_hat = a3
    
    # Compute loss
    loss[ix] = (0.5) * np.square(Y_hat - Y).mean()
    # Backprop to compute gradients of w1 and w2 with respect to loss
    d3 = Y_hat - Y
    grad2 = a2.T.dot(d3) / n_examples
    d2_tmp = d3.dot(w2.T)
    d2 = d2_tmp.copy()
    d2[z2 <= 0] = 0 #d2 = d2 * derivate of ReLU function
    grad1 = a1.T.dot(d2) / n_examples
    
    # Update weights
    w1 = w1 - grad1
    w2 = w2 - grad2

print(str(time.time() - tic) + ' s')
    
# Ploting loss vs epochs
plt.figure()
ix = np.arange(epochs)
plt.plot(ix, loss)
plt.show()
# Training Accuracy
y_tmp = np.argmax(Y_hat, axis=1)
y_hat = labels[y_tmp]
acc = np.mean(1 * (y_hat == y))
print('MLP Training Accuracy: ' + str(acc*100))


#CNN
#useful function
def convolution(image, filt, bias, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions

    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions

    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

    out = np.zeros((n_f,out_dim,out_dim))

    # convolve the filter over every part of the image, adding the bias at each step.
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return out

def maxpool(image, f=2, s=2):
    '''
    Downsample `image` using kernel size `f` and stride `s`
    '''
    n_c, h_prev, w_prev = image.shape

    h = int((h_prev - f)/s)+1
    w = int((w_prev - f)/s)+1

    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s = 1, pool_f = 2, pool_s = 2):
    '''
    Make predictions with trained filters/weights.
    '''
    conv1 = convolution(image, f1, b1, conv_s) # convolution operation
    conv1[conv1<=0] = 0 #relu activation

    conv2 = convolution(conv1, f2, b2, conv_s) # second convolution operation
    conv2[conv2<=0] = 0 # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_f, pool_s) # maxpooling operation
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer

    z = w3.dot(fc) + b3 # first dense layer
    z[z<=0] = 0 # pass through ReLU non-linearity

    out = w4.dot(z) + b4 # second dense layer
    probs = softmax(out) # predict class probabilities with the softmax activation function

    return np.argmax(probs), np.max(probs)

n_examples = X.shape[0]
# CNN test
#load pretrained model
save_path = './params.pkl'
params, cost = pickle.load(open(save_path, 'rb'))
[f1, f2, w3, w4, b1, b2, b3, b4] = params


corr = 0
test_data=X[:100]
for i in range(len(test_data)):
    image=test_data[i].reshape((1,28,28))
    #print ('image:',image.shape)
    pred, prob = predict(image, f1, f2, w3, w4, b1, b2, b3, b4)
    if pred==y[i]:
        corr+=1


print("test Accuracy: %.2f" % (float(corr/len(test_data)*100)))
