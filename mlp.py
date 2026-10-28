import numpy as np

def init_params():
    w1=np.random.rand(2,2)
    b1=np.random.rand(1,2)

    return [w1,b1]

def feed_forward(w1,b1,x):
    z=x@w1+b1
    return z

def sigmoid(z):
    return 1/(1+np.exp(-z))


X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])
y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

w1,b1=init_params()
w2,b2=np.random.rand(2,1),np.random.rand(1,1)
w3,b3=


lr=1.0
epochs=10000




m=X.shape[0]


for i in range(epochs):
    z1=feed_forward(w1,b1,X)
    A1=sigmoid(z1)
    z2=feed_forward(w2,b2,A1)
    A2=sigmoid(z2)
    

    #backpropagation
    delta2=2/m*(A2-y)*A2*(1-A2)
    dw2=A1.T @ delta2
    db2=np.sum(delta2,axis=0,keepdims=True)

    delta1=(delta2@w2.T)*A1*(1-A1)
    dw1=X.T @ delta1
    db1=np.sum(delta1,axis=0,keepdims=True) 

    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

    loss=np.mean((A2-y)**2)

A1 = sigmoid(X @ w1 + b1)
predictions = sigmoid(A1 @ w2 + b2)

print("\nExpected outputs:")
print(y)

print("\nPredictions:")
print(predictions)

print("\nPredictions rounded to 0 or 1:")
print((predictions >= 0.5).astype(int))
    


