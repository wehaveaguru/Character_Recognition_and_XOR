import numpy as np

np.random.seed(42)

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
w2,b2=init_params()
w3,b3=np.random.rand(2,1),np.random.rand(1,1)


lr=2
epochs=10000




m=X.shape[0]


for i in range(epochs):
    z1=feed_forward(w1,b1,X)
    A1=sigmoid(z1)
    z2=feed_forward(w2,b2,A1)
    A2=sigmoid(z2)
    z3=feed_forward(w3,b3,A2)
    A3=sigmoid(z3)
    

    #backpropagation

    delta3=2/m*(A3-y)*A3*(1-A3)
    dw3=A2.T @ delta3
    db3=np.sum(delta3,axis=0,keepdims=True)

    delta2=(delta3 @ w3.T)*A2*(1-A2)
    dw2=A1.T @ delta2
    db2=np.sum(delta2,axis=0,keepdims=True)

    delta1=(delta2@w2.T)*A1*(1-A1)
    dw1=X.T @ delta1
    db1=np.sum(delta1,axis=0,keepdims=True) 

    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    w3 -= lr * dw3
    b3 -= lr * db3

    loss=np.mean((A3-y)**2)

A1 = sigmoid(X @ w1 + b1)
A2 = sigmoid(A1 @ w2 + b2)
predictions = sigmoid(A2 @ w3 + b3)

print("\nExpected outputs:")
print(y)

print("\nPredictions:")
print(predictions)

print("\nPredictions rounded to 0 or 1:")
print((predictions >= 0.5).astype(int))
    


