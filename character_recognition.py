
import numpy as np
import pandas as pd


class CR:

    def __init__(self):
        self.w1 = np.random.randn(784, 128) * np.sqrt(2 / 784)
        self.b1 = np.zeros((1, 128))

        self.w2 = np.random.randn(128, 64) * np.sqrt(2 / 128)
        self.b2 = np.zeros((1, 64))

        self.w3 = np.random.randn(64, 10) * np.sqrt(2 / 64)
        self.b3 = np.zeros((1, 10))

    def feed_forward(self, X):
        self.z1 = X @ self.w1 + self.b1
        self.A1 = self.reLU(self.z1)

        self.z2 = self.A1 @ self.w2 + self.b2
        self.A2 = self.reLU(self.z2)

        self.z3 = self.A2 @ self.w3 + self.b3
        self.A3 = self.softmax(self.z3)

    def reLU(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def one_hot(self, y):
        return np.eye(10)[y]

    def reLU_derivative(self, Z):
        return (Z > 0).astype(float)

    def backpropagate(self, y, m, X):
        one_hot_y = self.one_hot(y)

        
        delta3 = (self.A3 - one_hot_y) / m

        dw3 = self.A2.T @ delta3
        db3 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = (delta3 @ self.w3.T) * self.reLU_derivative(self.z2)
        dw2 = self.A1.T @ delta2
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ self.w2.T) * self.reLU_derivative(self.z1)
        dw1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        lr = 0.2

        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w3 -= lr * dw3
        self.b3 -= lr * db3

    def get_prediction(self, A):
        return np.argmax(A, axis=1)

    def get_accuracy(self, preds, Y):
        return np.mean(preds == Y)

    def get_loss(self, y):
        m = len(y)
        correct_probs = self.A3[np.arange(m), y]
        return -np.mean(np.log(correct_probs + 1e-12))


# Load datasets
train_df = pd.read_csv("train_small.csv", header=None)
test_df = pd.read_csv("test_small.csv", header=None)

X_train = train_df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
y_train = train_df.iloc[:, 0].to_numpy(dtype=np.int64)

X_test = test_df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
y_test = test_df.iloc[:, 0].to_numpy(dtype=np.int64)


# Initialize once: preserve weights between epochs
network = CR()

epochs = 100
m = len(X_train)

for epoch in range(epochs):
    network.feed_forward(X_train)
    loss = network.get_loss(y_train)
    network.backpropagate(y_train, m, X_train)

    if (epoch + 1) % 10 == 0:
        predictions = network.get_prediction(network.A3)
        accuracy = network.get_accuracy(predictions, y_train)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {loss:.4f} | "
            f"Training accuracy: {accuracy * 100:.2f}%"
        )



network.feed_forward(X_test)
test_predictions = network.get_prediction(network.A3)
test_accuracy = network.get_accuracy(test_predictions, y_test)
test_loss = network.get_loss(y_test)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


example = test_df.iloc[:1, 1:].to_numpy(dtype=np.float32) / 255.0

network.feed_forward(example)

print("Predicted digit:", network.get_prediction(network.A3))
print("Actual digit:", test_df.iloc[0, 0])